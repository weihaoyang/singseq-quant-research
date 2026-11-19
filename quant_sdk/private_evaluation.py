"""Portable private-evaluation request and signed evaluator receipt protocol.

This module never uploads data or factor source. A data custodian or secure
runner may use the request contract, evaluate inside its own boundary, and
return a signed receipt that binds the benchmark, artifact, data commitments
and result.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from quant_sdk.benchmarks import benchmark_manifest

REQUEST_FORMAT = "qprivate-request/1"
RECEIPT_FORMAT = "qprivate-receipt/1"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write(path: str | Path, value: dict[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def _compact_date(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        text = text.replace("-", "")
    if not re.fullmatch(r"\d{8}", text):
        raise ValueError(f"{label} must be YYYYMMDD or YYYY-MM-DD")
    return text


def create_request(
    *,
    output: str | Path,
    benchmark_id: str,
    data_commitments: Sequence[dict[str, Any]],
    artifact_sha256: str,
    source_disclosure: str = "none",
    result_disclosure: str = "summary",
    expires_hours: int = 168,
    evaluation_window: dict[str, Any] | None = None,
    nonce: str | None = None,
) -> dict[str, Any]:
    manifest = benchmark_manifest(benchmark_id)
    if source_disclosure not in {"none", "on_accept", "public"}:
        raise ValueError("sourceDisclosure must be none/on_accept/public")
    if result_disclosure not in {"summary", "evidence", "full"}:
        raise ValueError("resultDisclosure must be summary/evidence/full")
    digest = str(artifact_sha256 or "").strip().lower()
    if not len(digest) == 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("artifactSha256 must be a 64-character hex digest")
    commitments = []
    for index, item in enumerate(data_commitments):
        if not isinstance(item, dict) or not item.get("sha256"):
            raise ValueError(f"data commitment {index + 1} is missing sha256")
        commitments.append({
            "name": str(item.get("name") or f"dataset-{index + 1}"),
            "sha256": str(item["sha256"]),
            "rows": item.get("rows"),
            "assets": item.get("assets"),
            "dates": item.get("dates"),
            "window": item.get("window"),
        })
    if not commitments:
        raise ValueError("at least one data commitment is required")
    normalized_window = None
    if evaluation_window is not None:
        if not isinstance(evaluation_window, dict):
            raise ValueError("evaluationWindow must be an object")
        normalized_window = {
            "start": _compact_date(evaluation_window.get("start"), "window.start"),
            "end": _compact_date(evaluation_window.get("end"), "window.end"),
        }
        if normalized_window["start"] > normalized_window["end"]:
            raise ValueError("evaluationWindow start must not be after end")
    now = datetime.now(UTC).replace(microsecond=0)
    request_id = nonce or base64.urlsafe_b64encode(os.urandom(18)).decode("ascii")
    body = {
        "format": REQUEST_FORMAT,
        "requestId": request_id,
        "createdAt": now.isoformat(),
        "expiresAt": (now + timedelta(hours=int(expires_hours))).isoformat(),
        "benchmarkId": manifest["id"],
        "benchmarkVersion": manifest["version"],
        "benchmarkRuleSha256": _sha256(_canonical_json(manifest)),
        "artifactSha256": digest,
        "dataCommitments": commitments,
        "evaluationWindow": normalized_window,
        "privacy": {
            "sourceDisclosure": source_disclosure,
            "resultDisclosure": result_disclosure,
            "sourceBoundary": "software_isolation",
            "operatorCanReadArtifactSource": True,
            "confidentialCompute": False,
            "dataLeavesCustodian": True,
            "automaticSubmission": False,
            "secureComputeRequired": True,
        },
    }
    request = {**body, "requestSha256": _sha256(_canonical_json(body))}
    target = _write(output, request)
    return {
        "status": "REQUEST_CREATED",
        "path": str(target),
        **request,
    }


def create_receipt(
    *,
    request_file: str | Path,
    result_file: str | Path,
    output: str | Path,
    signing_key_file: str | Path,
    evaluator: str,
    secure_environment: dict[str, Any] | None = None,
    completed_at: datetime | str | None = None,
) -> dict[str, Any]:
    request = json.loads(
        Path(request_file).expanduser().read_text(encoding="utf-8"))
    result_path = Path(result_file).expanduser().resolve()
    result_bytes = result_path.read_bytes()
    try:
        result_value = json.loads(result_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("receipt result must be a UTF-8 JSON file") from error
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as error:  # pragma: no cover - dependency error
        raise RuntimeError("receipt signing requires cryptography") from error
    key_path = Path(signing_key_file).expanduser().resolve()
    private_key = serialization.load_pem_private_key(
        key_path.read_bytes(), password=None,
    )
    if not isinstance(private_key, Ed25519PrivateKey):
        raise ValueError("evaluator key must be an Ed25519 private key")
    public = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    if completed_at is None:
        completed = datetime.now(UTC).replace(microsecond=0)
    elif isinstance(completed_at, datetime):
        completed = completed_at.astimezone(UTC).replace(microsecond=0)
    else:
        completed = datetime.fromisoformat(
            str(completed_at).replace("Z", "+00:00")
        ).astimezone(UTC).replace(microsecond=0)
    body = {
        "format": RECEIPT_FORMAT,
        "requestSha256": request.get("requestSha256"),
        "requestId": request.get("requestId"),
        "benchmarkId": request.get("benchmarkId"),
        "benchmarkVersion": request.get("benchmarkVersion"),
        "benchmarkRuleSha256": request.get("benchmarkRuleSha256"),
        "evaluationWindow": request.get("evaluationWindow"),
        "artifactSha256": request.get("artifactSha256"),
        "dataCommitments": request.get("dataCommitments"),
        "resultSha256": _sha256(_canonical_json(result_value)),
        "result": result_value,
        "privacy": request.get("privacy"),
        "evaluator": {
            "name": str(evaluator or "").strip() or "private-evaluator",
            "publicKey": base64.b64encode(public).decode("ascii"),
            "publicKeyFingerprint": _sha256(public),
            "secureEnvironment": secure_environment or {
                "type": "NOT_DECLARED",
            },
        },
        "completedAt": completed.isoformat(),
    }
    signature = private_key.sign(_canonical_json(body))
    receipt = {
        **body,
        "signature": {
            "algorithm": "ed25519",
            "value": base64.b64encode(signature).decode("ascii"),
        },
        "receiptSha256": _sha256(_canonical_json(body)),
    }
    target = _write(output, receipt)
    return {
        "status": "RECEIPT_CREATED",
        "path": str(target),
        **receipt,
    }


def verify_receipt(
    *,
    request_file: str | Path,
    receipt_file: str | Path,
    trusted_key: str | Path,
) -> dict[str, Any]:
    request = json.loads(
        Path(request_file).expanduser().read_text(encoding="utf-8"))
    receipt = json.loads(
        Path(receipt_file).expanduser().read_text(encoding="utf-8"))
    errors: list[str] = []
    request_body = dict(request)
    request_digest = request_body.pop("requestSha256", None)
    if request_digest != _sha256(_canonical_json(request_body)):
        errors.append("request_hash_mismatch")
    body = dict(receipt)
    body.pop("signature", None)
    body.pop("receiptSha256", None)
    if receipt.get("receiptSha256") != _sha256(_canonical_json(body)):
        errors.append("receipt_hash_mismatch")
    if receipt.get("requestSha256") != request_digest:
        errors.append("request_binding_mismatch")
    if receipt.get("benchmarkId") != request.get("benchmarkId"):
        errors.append("benchmark_binding_mismatch")
    if receipt.get("benchmarkVersion") != request.get("benchmarkVersion"):
        errors.append("benchmark_version_binding_mismatch")
    if receipt.get("benchmarkRuleSha256") != request.get("benchmarkRuleSha256"):
        errors.append("benchmark_rule_binding_mismatch")
    if receipt.get("evaluationWindow") != request.get("evaluationWindow"):
        errors.append("evaluation_window_binding_mismatch")
    if receipt.get("artifactSha256") != request.get("artifactSha256"):
        errors.append("artifact_binding_mismatch")
    if receipt.get("dataCommitments") != request.get("dataCommitments"):
        errors.append("data_binding_mismatch")
    result = receipt.get("result")
    if _sha256(_canonical_json(result)) != receipt.get("resultSha256"):
        errors.append("result_hash_mismatch")
    expires_at = request.get("expiresAt")
    completed_at = receipt.get("completedAt")
    if not completed_at:
        errors.append("receipt_completed_at_missing")
    if expires_at:
        expires = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
        if completed_at:
            completed = datetime.fromisoformat(
                str(completed_at).replace("Z", "+00:00"))
            if completed > expires:
                errors.append("receipt_completed_after_expiry")
    trusted_bytes = Path(trusted_key).expanduser().read_bytes()
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        if trusted_bytes.startswith(b"-----BEGIN"):
            key = serialization.load_pem_public_key(trusted_bytes)
            trusted_bytes = key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        else:
            trusted_bytes = base64.b64decode(trusted_bytes, validate=True)
        signature = base64.b64decode(
            str((receipt.get("signature") or {}).get("value") or ""),
            validate=True,
        )
        Ed25519PublicKey.from_public_bytes(trusted_bytes).verify(
            signature, _canonical_json(body),
        )
        signature_valid = True
    except Exception:  # noqa: BLE001 - all failures are untrusted receipts
        signature_valid = False
    if not signature_valid:
        errors.append("evaluator_signature_invalid")
    evaluator = receipt.get("evaluator") or {}
    trusted_fingerprint = _sha256(trusted_bytes) if len(trusted_bytes) == 32 else None
    if evaluator.get("publicKeyFingerprint") != trusted_fingerprint:
        errors.append("evaluator_key_mismatch")
    return {
        "status": "VERIFIED" if not errors else "INVALID",
        "valid": not errors,
        "requestId": request.get("requestId"),
        "benchmarkId": request.get("benchmarkId"),
        "benchmarkVersion": request.get("benchmarkVersion"),
        "benchmarkRuleSha256": request.get("benchmarkRuleSha256"),
        "evaluationWindow": request.get("evaluationWindow"),
        "artifactSha256": request.get("artifactSha256"),
        "evaluator": evaluator.get("name"),
        "evaluatorFingerprint": evaluator.get("publicKeyFingerprint"),
        "signatureValid": signature_valid,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Private evaluation protocol")
    sub = parser.add_subparsers(dest="command", required=True)
    request = sub.add_parser("request")
    request.add_argument("--output", required=True)
    request.add_argument("--benchmark", required=True)
    request.add_argument("--data-commitments", required=True)
    request.add_argument("--artifact-sha256", required=True)
    request.add_argument("--source-disclosure", default="none")
    request.add_argument("--result-disclosure", default="summary")
    request.add_argument("--expires-hours", type=int, default=168)
    request.add_argument("--start")
    request.add_argument("--end")
    receipt = sub.add_parser("receipt")
    receipt.add_argument("--request", required=True)
    receipt.add_argument("--result", required=True)
    receipt.add_argument("--output", required=True)
    receipt.add_argument("--signing-key", required=True)
    receipt.add_argument("--evaluator", required=True)
    receipt.add_argument("--secure-environment")
    verify = sub.add_parser("verify")
    verify.add_argument("--request", required=True)
    verify.add_argument("--receipt", required=True)
    verify.add_argument("--trusted-key", required=True)
    args = parser.parse_args(argv)
    if args.command == "request":
        result = create_request(
            output=args.output, benchmark_id=args.benchmark,
            data_commitments=json.loads(args.data_commitments),
            artifact_sha256=args.artifact_sha256,
            source_disclosure=args.source_disclosure,
            result_disclosure=args.result_disclosure,
            expires_hours=args.expires_hours,
            evaluation_window=(
                {"start": args.start, "end": args.end}
                if args.start or args.end else None
            ),
        )
    elif args.command == "receipt":
        result = create_receipt(
            request_file=args.request, result_file=args.result,
            output=args.output, signing_key_file=args.signing_key,
            evaluator=args.evaluator,
            secure_environment=(
                json.loads(args.secure_environment)
                if args.secure_environment else None
            ),
        )
    else:
        result = verify_receipt(
            request_file=args.request, receipt_file=args.receipt,
            trusted_key=args.trusted_key,
        )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result.get("valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
