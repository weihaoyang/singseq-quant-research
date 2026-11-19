"""Decentralized benchmark ranking from signed private-evaluation receipts.

The ranking is reconstructed locally from requests, receipts and trusted
evaluator keys. No factor execution, upload or central leaderboard is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from quant_sdk.private_evaluation import verify_receipt

LEADERBOARD_FORMAT = "qbench-leaderboard/1"
LEADERBOARD_VERSION = "1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _key_fingerprint(path: Path) -> str:
    from cryptography.hazmat.primitives import serialization

    raw = path.read_bytes()
    if raw.startswith(b"-----BEGIN"):
        key = serialization.load_pem_public_key(raw)
        raw = key.public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    else:
        import base64

        raw = base64.b64decode(raw, validate=True)
    return hashlib.sha256(raw).hexdigest()


def _trusted_keys(directory: str | Path) -> list[tuple[Path, str]]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("trustedKeys must be a directory")
    keys = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        try:
            keys.append((path, _key_fingerprint(path)))
        except Exception:
            continue
    if not keys:
        raise ValueError("trustedKeys contains no readable Ed25519 public keys")
    return keys


def _receipt_pairs(directory: str | Path) -> list[tuple[Path, Path]]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("receipts must be a directory")
    pairs = []
    for request_path in root.rglob("*.request.json"):
        stem = request_path.name[:-len(".request.json")]
        receipt_path = request_path.with_name(f"{stem}.receipt.json")
        if receipt_path.is_file():
            pairs.append((request_path, receipt_path))
    for request_path in root.rglob("request.json"):
        receipt_path = request_path.with_name("receipt.json")
        if receipt_path.is_file():
            pairs.append((request_path, receipt_path))
    return sorted(set(pairs), key=lambda item: (str(item[0]), str(item[1])))


def _verified_receipt(
    request_path: Path,
    receipt_path: Path,
    keys: Sequence[tuple[Path, str]],
) -> tuple[dict[str, Any], dict[str, Any], str] | None:
    try:
        receipt = _read_json(receipt_path)
    except Exception:
        return None
    claimed = str(
        (receipt.get("evaluator") or {}).get("publicKeyFingerprint") or "")
    ordered = [item for item in keys if item[1] == claimed]
    ordered.extend(item for item in keys if item[1] != claimed)
    for key_path, fingerprint in ordered:
        checked = verify_receipt(
            request_file=request_path,
            receipt_file=receipt_path,
            trusted_key=key_path,
        )
        if checked.get("valid"):
            return receipt, checked, fingerprint
    return None


def _group_key(
    receipt: dict[str, Any],
) -> str:
    commitments = receipt.get("dataCommitments") or []
    digests = sorted(
        str(item.get("sha256") or "")
        for item in commitments
        if isinstance(item, dict) and item.get("sha256")
    )
    binding = {
        "benchmarkId": str(receipt.get("benchmarkId") or ""),
        "benchmarkVersion": str(receipt.get("benchmarkVersion") or ""),
        "benchmarkRuleSha256": str(receipt.get("benchmarkRuleSha256") or ""),
        "evaluationWindow": receipt.get("evaluationWindow"),
        "dataCommitments": digests,
    }
    digest = hashlib.sha256(
        json.dumps(
            binding,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"qbench-group:{digest}"


def _score(output: dict[str, Any]) -> float | None:
    holdout = output.get("holdout") or {}
    ic = _number(holdout.get("meanRankIc"))
    days = _number(holdout.get("icDays"))
    coverage = _number(holdout.get("coverage"))
    if ic is None or days is None or coverage is None:
        return None
    return abs(ic) * math.sqrt(max(days, 0.0)) * math.sqrt(
        min(max(coverage, 0.0), 1.0)
    )


def build_leaderboard(
    *,
    receipts_dir: str | Path,
    trusted_keys_dir: str | Path,
    benchmark_id: str | None = None,
) -> dict[str, Any]:
    keys = _trusted_keys(trusted_keys_dir)
    key_fingerprints = {fingerprint for _, fingerprint in keys}
    pairs = _receipt_pairs(receipts_dir)
    if not pairs:
        raise ValueError("no request/receipt pairs found")
    valid_receipts = 0
    invalid_receipts: list[dict[str, str]] = []
    observations: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    groups: dict[str, dict[str, Any]] = {}

    for request_path, receipt_path in pairs:
        verified = _verified_receipt(request_path, receipt_path, keys)
        if verified is None:
            invalid_receipts.append({
                "request": str(request_path),
                "receipt": str(receipt_path),
                "reason": "untrusted_or_invalid_receipt",
            })
            continue
        receipt, checked, evaluator_fingerprint = verified
        result = receipt.get("result") or {}
        current_benchmark = str(receipt.get("benchmarkId") or "")
        if benchmark_id and current_benchmark != benchmark_id:
            continue
        valid_receipts += 1
        artifact = str(receipt.get("artifactSha256") or "")
        commitments = receipt.get("dataCommitments") or []
        group_key = _group_key(receipt)
        groups.setdefault(group_key, {
            "benchmarkId": current_benchmark,
            "benchmarkVersion": receipt.get("benchmarkVersion"),
            "benchmarkRuleSha256": receipt.get("benchmarkRuleSha256"),
            "evaluationWindow": receipt.get("evaluationWindow"),
            "dataCommitments": commitments,
        })
        for output in result.get("outputs") or []:
            if not isinstance(output, dict):
                continue
            name = str(output.get("name") or "").strip()
            if not name:
                continue
            observation_key = (
                group_key, artifact, name, str(checked["evaluatorFingerprint"]))
            holdout = output.get("holdout") or {}
            selection = output.get("selection") or {}
            row = {
                "factorName": name,
                "artifactSha256": artifact,
                "selectionChecksPassed": bool(
                    output.get("selectionChecksPassed")
                ),
                "score": _score(output),
                "selectionRankIc": _number(selection.get("meanRankIc")),
                "selectionIcIr": _number(selection.get("icIr")),
                "holdoutRankIc": _number(holdout.get("meanRankIc")),
                "holdoutIcIr": _number(holdout.get("icIr")),
                "holdoutIcDays": _number(holdout.get("icDays")),
                "holdoutCoverage": _number(holdout.get("coverage")),
                "evaluatorFingerprint": evaluator_fingerprint,
                "completedAt": receipt.get("completedAt"),
                "receiptSha256": receipt.get("receiptSha256"),
            }
            previous = observations.get(observation_key)
            if previous is None or str(row["completedAt"]) > str(
                previous["completedAt"]):
                observations[observation_key] = {
                    **row,
                    "groupKey": group_key,
                }

    aggregated: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observations.values():
        aggregated[
            (str(row["groupKey"]), str(row["artifactSha256"]), str(row["factorName"]))
        ].append(row)

    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (group_key, artifact, name), rows in aggregated.items():
        passed = sum(1 for row in rows if row["selectionChecksPassed"])
        holdout_ics = [
            float(row["holdoutRankIc"])
            for row in rows
            if row["holdoutRankIc"] is not None
        ]
        scores = [
            float(row["score"])
            for row in rows
            if row["score"] is not None
        ]
        grouped_rows[group_key].append({
            "factorName": name,
            "artifactSha256": artifact,
            "qualified": passed > len(rows) / 2,
            "evaluatorCount": len(rows),
            "selectionChecksPassedVotes": passed,
            "conflictingOutcomes": 0 < passed < len(rows),
            "medianHoldoutRankIc": (
                statistics.median(holdout_ics) if holdout_ics else None
            ),
            "medianScore": statistics.median(scores) if scores else None,
            "evaluators": sorted({
                str(row["evaluatorFingerprint"]) for row in rows
            }),
            "latestCompletedAt": max(
                str(row["completedAt"]) for row in rows
            ),
            "receipts": sorted({
                str(row["receiptSha256"]) for row in rows
            }),
        })

    output_groups = []
    for group_key, group in groups.items():
        rows = grouped_rows.get(group_key, [])
        ranked = sorted(
            rows,
            key=lambda row: (
                not row["qualified"],
                -abs(float(row["medianScore"] or 0.0)),
                -abs(float(row["medianHoldoutRankIc"] or 0.0)),
                row["factorName"],
            ),
        )
        for index, row in enumerate(ranked, 1):
            row["rank"] = index
        output_groups.append({
            **group,
            "groupKey": group_key,
            "rows": ranked,
        })

    return {
        "format": LEADERBOARD_FORMAT,
        "version": LEADERBOARD_VERSION,
        "status": "LOCAL_VERIFIED",
        "benchmark": benchmark_id,
        "source": {
            "receipts": len(pairs),
            "validReceipts": valid_receipts,
            "invalidReceipts": invalid_receipts,
            "trustedKeyFingerprints": sorted(key_fingerprints),
        },
        "groups": output_groups,
        "submitted": False,
        "published": False,
        "note": "This ranking is reconstructed from locally verified signed receipts and is not an automatic platform submission.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify signed receipts and rebuild a benchmark ranking")
    parser.add_argument("--receipts", required=True)
    parser.add_argument("--trusted-keys", required=True)
    parser.add_argument("--benchmark")
    args = parser.parse_args(argv)
    result = build_leaderboard(
        receipts_dir=args.receipts,
        trusted_keys_dir=args.trusted_keys,
        benchmark_id=args.benchmark,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
