"""Standalone verifier for portable ``.qalpha`` artifacts.

The verifier uses only the Python standard library. It checks the manifest,
every embedded member digest, and optional Ed25519 signatures without loading
the Quant SDK, executing the archive, or executing factor source.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

PACK_FORMAT = "qalpha/1"
MANIFEST_NAME = "manifest.json"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_FILES = 10_000
MAX_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
_PUBLICATION_AUDIT_CHECKS = (
    "staticSourcePassed",
    "runtimeCausalityPassed",
    "selectionObserved",
    "holdoutObserved",
    "holdoutSameDirection",
    "holdoutExcessAfterCostPositive",
    "benchmarkObserved",
    "capacityObserved",
    "leakagePassed",
)

_Q = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = (-121665 * pow(121666, _Q - 2, _Q)) % _Q
_I = pow(2, (_Q - 1) // 4, _Q)
_IDENTITY = (0, 1, 1, 0)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_body(manifest: dict[str, Any]) -> dict[str, Any]:
    body = dict(manifest)
    body.pop("manifestSha256", None)
    body.pop("signature", None)
    return body


def _safe_member(name: str) -> str:
    member = str(PurePosixPath(name))
    parts = PurePosixPath(member).parts
    if not member or member.startswith("/") or ".." in parts:
        raise ValueError(f"unsafe member path: {name}")
    return member


def _recover_x(y: int, sign: int) -> int | None:
    if y >= _Q:
        return None
    x2 = (y * y - 1) * pow(_D * y * y + 1, _Q - 2, _Q)
    x = pow(x2 % _Q, (_Q + 3) // 8, _Q)
    if (x * x - x2) % _Q:
        x = (x * _I) % _Q
    if (x * x - x2) % _Q or x == 0 and sign:
        return None
    if (x & 1) != sign:
        x = _Q - x
    return x


def _decode_point(encoded: bytes) -> tuple[int, int, int, int] | None:
    if len(encoded) != 32:
        return None
    value = int.from_bytes(encoded, "little")
    sign = value >> 255
    y = value & ((1 << 255) - 1)
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, (x * y) % _Q)


def _encode_point(point: tuple[int, int, int, int]) -> bytes:
    x, y, z, _ = point
    inverse = pow(z, _Q - 2, _Q)
    x = (x * inverse) % _Q
    y = (y * inverse) % _Q
    encoded = y | ((x & 1) << 255)
    return encoded.to_bytes(32, "little")


def _point_add(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x1, y1, z1, t1 = left
    x2, y2, z2, t2 = right
    a = ((y1 - x1) * (y2 - x2)) % _Q
    b = ((y1 + x1) * (y2 + x2)) % _Q
    c = (2 * _D * t1 * t2) % _Q
    d = (2 * z1 * z2) % _Q
    e = b - a
    f = d - c
    g = d + c
    h = b + a
    return (
        (e * f) % _Q,
        (g * h) % _Q,
        (f * g) % _Q,
        (e * h) % _Q,
    )


def _scalar_mult(scalar: int, point: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    result = _IDENTITY
    current = point
    while scalar:
        if scalar & 1:
            result = _point_add(result, current)
        current = _point_add(current, current)
        scalar >>= 1
    return result


def _point_equal(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    x1, y1, z1, _ = left
    x2, y2, z2, _ = right
    return (x1 * z2 - x2 * z1) % _Q == 0 and (y1 * z2 - y2 * z1) % _Q == 0


def _base_point() -> tuple[int, int, int, int]:
    encoded = bytes.fromhex(
        "5866666666666666666666666666666666666666666666666666666666666666"
    )
    point = _decode_point(encoded)
    if point is None:
        raise RuntimeError("ed25519 base point is unavailable")
    return point


def _verify_ed25519(public_key: bytes, signature: bytes, message: bytes) -> bool:
    if len(public_key) != 32 or len(signature) != 64:
        return False
    scalar = int.from_bytes(signature[32:], "little")
    if scalar >= _L:
        return False
    public_point = _decode_point(public_key)
    signature_point = _decode_point(signature[:32])
    if public_point is None or signature_point is None:
        return False
    if _encode_point(signature_point) != signature[:32]:
        return False
    if _point_equal(_scalar_mult(8, public_point), _IDENTITY):
        return False
    challenge = int.from_bytes(
        hashlib.sha512(signature[:32] + public_key + message).digest(),
        "little",
    ) % _L
    expected = _point_add(
        signature_point,
        _scalar_mult(challenge, public_point),
    )
    return _point_equal(_scalar_mult(scalar, _base_point()), expected)


def _signature_result(
    manifest: dict[str, Any],
    trusted_key: bytes | None,
) -> tuple[bool, str | None, str | None]:
    signature = manifest.get("signature")
    if signature is None:
        return False, None, None
    if not isinstance(signature, dict) or signature.get("algorithm") != "ed25519":
        return False, "unsupported_signature", None
    try:
        public_key = base64.b64decode(str(signature["publicKey"]), validate=True)
        value = base64.b64decode(str(signature["value"]), validate=True)
        fingerprint = _sha256(public_key)
        if trusted_key is not None and trusted_key != public_key:
            return (
                _verify_ed25519(
                    public_key,
                    value,
                    _canonical_json(_manifest_body(manifest)),
                ),
                "trusted_key_mismatch",
                fingerprint,
            )
        valid = _verify_ed25519(
            public_key,
            value,
            _canonical_json(_manifest_body(manifest)),
        )
        return valid, None if valid else "signature_mismatch", fingerprint
    except (KeyError, TypeError, ValueError):
        return False, "signature_mismatch", None


def _parse_trusted_key(value: str | None) -> bytes | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    encoded = candidate.read_bytes().strip() if candidate.is_file() else value.encode("ascii")
    try:
        return base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ValueError("trusted key must be raw base64") from error


def _audit_binding_errors(
    audit: dict[str, Any],
    summary: dict[str, Any],
    pack_status: Any,
) -> list[str]:
    errors: list[str] = []
    if audit.get("format") != summary.get("format"):
        errors.append("audit_format_mismatch")
    if audit.get("status") != summary.get("status"):
        errors.append("audit_status_mismatch")
    if audit.get("validated") != summary.get("validated"):
        errors.append("audit_validation_mismatch")
    if pack_status != audit.get("status"):
        errors.append("pack_audit_status_mismatch")
    status = str(audit.get("status") or "").upper()
    validated = audit.get("validated")
    if status not in {
        "LOCAL_PROOF_OF_ALPHA", "RESEARCH_ONLY", "REJECTED_LEAKAGE",
    }:
        errors.append("audit_status_invalid")
    if not isinstance(validated, bool) or validated != (
        status == "LOCAL_PROOF_OF_ALPHA"
    ):
        errors.append("audit_validation_invalid")
    if status == "LOCAL_PROOF_OF_ALPHA":
        checks = audit.get("checks")
        if not isinstance(checks, dict) or checks.get("leakagePassed") is not True:
            errors.append("audit_leakage_gate_missing")
        if not isinstance(audit.get("costSensitivity"), list) or not audit.get(
            "costSensitivity"
        ):
            errors.append("audit_cost_sensitivity_missing")
        errors.extend(_audit_publication_errors(audit))
    return errors


def _audit_publication_errors(audit: dict[str, Any]) -> list[str]:
    """Validate the complete publication-grade evidence, not just self-report."""
    errors: list[str] = []
    checks = audit.get("checks")
    if not isinstance(checks, dict):
        errors.append("audit_checks_missing")
    else:
        for key in _PUBLICATION_AUDIT_CHECKS:
            if checks.get(key) is not True:
                errors.append(f"audit_check_failed:{key}")
    if audit.get("validated") is not True:
        errors.append("audit_not_validated")
    if audit.get("submitted") not in {None, False}:
        errors.append("audit_submitted_state_invalid")
    if audit.get("published") not in {None, False}:
        errors.append("audit_published_state_invalid")
    configuration = audit.get("configuration")
    label = (
        configuration.get("label")
        if isinstance(configuration, dict)
        and isinstance(configuration.get("label"), dict)
        else {}
    )
    if label.get("version") != "next-open-adjusted/1":
        errors.append("audit_label_mismatch")
    if label.get("matching") != "next-open":
        errors.append("audit_label_not_next_open")
    split = audit.get("split")
    if not isinstance(split, dict) or not all(
        split.get(key)
        for key in (
            "selectionStart", "selectionEnd",
            "holdoutStart", "holdoutEnd",
        )
    ):
        errors.append("audit_split_incomplete")
    point_in_time = audit.get("pointInTime")
    point_in_time = point_in_time if isinstance(point_in_time, dict) else {}
    if point_in_time.get("status") != "OBSERVED":
        errors.append("audit_point_in_time_missing")
    if point_in_time.get("signalTime") != "session-close:t":
        errors.append("audit_signal_time_invalid")
    if point_in_time.get("executionTime") != "open:t+1":
        errors.append("audit_execution_time_invalid")
    out_of_sample = audit.get("outOfSample")
    selection = (
        out_of_sample.get("selection")
        if isinstance(out_of_sample, dict)
        and isinstance(out_of_sample.get("selection"), dict)
        else {}
    )
    holdout = (
        out_of_sample.get("holdout")
        if isinstance(out_of_sample, dict)
        and isinstance(out_of_sample.get("holdout"), dict)
        else {}
    )
    for name, value in (("selection", selection), ("holdout", holdout)):
        mean = value.get("meanIc")
        days = value.get("days")
        if not isinstance(mean, (int, float)) or not math.isfinite(float(mean)):
            errors.append(f"audit_{name}_ic_missing")
        if not isinstance(days, int) or days < 10:
            errors.append(f"audit_{name}_history_insufficient")
    benchmark = audit.get("benchmark")
    benchmark = benchmark if isinstance(benchmark, dict) else {}
    if benchmark.get("status") != "OBSERVED":
        errors.append("audit_market_benchmark_observed_required")
    if benchmark.get("grade") != "MARKET":
        errors.append("audit_market_benchmark_missing")
    if not str(benchmark.get("basis") or "").startswith("next-open:"):
        errors.append("audit_benchmark_not_next_open")
    excess = benchmark.get("excessAnnualized")
    if not isinstance(excess, (int, float)) or not math.isfinite(float(excess)) or float(excess) <= 0.0:
        errors.append("audit_benchmark_excess_not_positive")
    capacity = audit.get("capacity")
    capacity = capacity if isinstance(capacity, dict) else {}
    if capacity.get("status") != "OK":
        errors.append("audit_capacity_not_ok")
    costs = audit.get("costSensitivity")
    names = {
        str(item.get("name"))
        for item in costs or []
        if isinstance(item, dict) and item.get("name")
    } if isinstance(costs, list) else set()
    if len(names) < 3:
        errors.append("audit_cost_sensitivity_insufficient")
    if not {"normal", "high"}.issubset(names):
        errors.append("audit_cost_sensitivity_baselines_missing")
    return errors


def _family_binding_errors(
    family: dict[str, Any],
    summary: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if family.get("format") != summary.get("format"):
        errors.append("family_format_mismatch")
    if family.get("status") != summary.get("status"):
        errors.append("family_status_mismatch")
    if family.get("validated") != summary.get("validated"):
        errors.append("family_validation_mismatch")
    if family.get("familySha256") != summary.get("familySha256"):
        errors.append("family_identity_mismatch")
    matrix = family.get("returnMatrix")
    matrix = matrix if isinstance(matrix, dict) else {}
    packed_matrix = summary.get("returnMatrix")
    packed_matrix = packed_matrix if isinstance(packed_matrix, dict) else {}
    if matrix.get("shape") != packed_matrix.get("shape"):
        errors.append("family_matrix_shape_mismatch")
    if str(matrix.get("sha256") or "") != str(packed_matrix.get("sha256") or ""):
        errors.append("family_matrix_hash_mismatch")
    return errors


def _publication_manifest_errors(
    audit: dict[str, Any] | None,
    publication: Any,
    family: dict[str, Any] | None = None,
) -> list[str]:
    profile = publication if isinstance(publication, dict) else {}
    eligible = profile.get("eligible") is True
    status = str((audit or {}).get("status") or "").upper()
    errors: list[str] = []
    if eligible:
        if status != "LOCAL_PROOF_OF_ALPHA":
            errors.append("publication_eligible_without_proof")
        elif audit is not None:
            errors.extend(_audit_publication_errors(audit))
        if profile.get("status") != "PUBLISHABLE":
            errors.append("publication_status_mismatch")
        if profile.get("missing") not in (None, []):
            errors.append("publication_missing_fields")
        if profile.get("policyErrors") not in (None, []):
            errors.append("publication_policy_errors_present")
        if profile.get("automaticPublication") is not False:
            errors.append("publication_must_remain_explicit")
        candidate_family = profile.get("candidateFamily")
        if not isinstance(candidate_family, dict) or family is None:
            errors.append("publication_candidate_family_missing")
        else:
            if family.get("status") != "PASS":
                errors.append("publication_candidate_family_not_sealed")
            if family.get("validated") is not True:
                errors.append("publication_candidate_family_not_validated")
            pbo = family.get("pbo") if isinstance(family.get("pbo"), dict) else {}
            if pbo.get("gateStatus") != "PASS":
                errors.append("publication_candidate_family_pbo_gate_failed")
            if (
                family.get("selectedCandidateSurvivedMultipleTesting")
                is not True
            ):
                errors.append("publication_selected_candidate_not_significant")
    elif status == "LOCAL_PROOF_OF_ALPHA":
        errors.append("publication_profile_not_promoted")
    return errors


def verify(path: str | Path, *, trusted_key: str | None = None) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise ValueError(f"Evidence Pack not found: {target}")
    errors: list[str] = []
    expected_key = _parse_trusted_key(trusted_key)
    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
        if len(names) > MAX_ARCHIVE_FILES:
            raise ValueError(
                f"Evidence Pack has too many files: {len(names)}"
            )
        if MANIFEST_NAME not in names:
            raise ValueError("Evidence Pack has no manifest.json")
        manifest_info = archive.getinfo(MANIFEST_NAME)
        if manifest_info.file_size > MAX_MANIFEST_BYTES:
            raise ValueError("Evidence Pack manifest is too large")
        manifest = json.loads(
            archive.read(MANIFEST_NAME).decode("utf-8")
        )
        if manifest.get("format") != PACK_FORMAT:
            errors.append("unsupported_format")
        expected_manifest = manifest.get("manifestSha256")
        actual_manifest = _sha256(_canonical_json(_manifest_body(manifest)))
        if not expected_manifest or expected_manifest != actual_manifest:
            errors.append("manifest_hash_mismatch")
        signature_valid, signature_error, signer_fingerprint = _signature_result(
            manifest,
            expected_key,
        )
        if signature_error:
            errors.append(signature_error)
        listed: set[str] = set()
        total_uncompressed = manifest_info.file_size
        for item in manifest.get("files") or []:
            member = _safe_member(str(item.get("path") or ""))
            listed.add(member)
            if member not in names:
                errors.append(f"missing:{member}")
                continue
            info = archive.getinfo(member)
            total_uncompressed += info.file_size
            if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("Evidence Pack uncompressed size exceeds the safety limit")
            digest = hashlib.sha256()
            with archive.open(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != item.get("sha256"):
                errors.append(f"hash_mismatch:{member}")
            if info.file_size != item.get("bytes"):
                errors.append(f"size_mismatch:{member}")
        unlisted = sorted(set(names) - listed - {MANIFEST_NAME})
        if unlisted:
            errors.append("unlisted:" + ",".join(unlisted))
        audit: dict[str, Any] | None = None
        audit_identity = manifest.get("audit")
        if isinstance(audit_identity, dict):
            audit_member = _safe_member(str(audit_identity.get("path") or ""))
            if audit_member not in names:
                errors.append(f"missing:{audit_member}")
            else:
                audit_bytes = archive.read(audit_member)
                if _sha256(audit_bytes) != audit_identity.get("sha256"):
                    errors.append(f"hash_mismatch:{audit_member}")
                if len(audit_bytes) != audit_identity.get("bytes"):
                    errors.append(f"size_mismatch:{audit_member}")
                try:
                    parsed_audit = json.loads(audit_bytes.decode("utf-8"))
                    if not isinstance(parsed_audit, dict):
                        raise ValueError("audit report is not an object")
                    audit = parsed_audit
                    errors.extend(_audit_binding_errors(
                        parsed_audit,
                        audit_identity,
                        manifest.get("status"),
                    ))
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    errors.append("audit_report_invalid")
        family: dict[str, Any] | None = None
        family_identity = manifest.get("familyEvidence")
        if isinstance(family_identity, dict):
            family_member = _safe_member(
                str(family_identity.get("path") or "")
            )
            if family_member not in names:
                errors.append(f"missing:{family_member}")
            else:
                family_bytes = archive.read(family_member)
                if _sha256(family_bytes) != family_identity.get("sha256"):
                    errors.append(f"hash_mismatch:{family_member}")
                if len(family_bytes) != family_identity.get("bytes"):
                    errors.append(f"size_mismatch:{family_member}")
                try:
                    parsed_family = json.loads(family_bytes.decode("utf-8"))
                    if not isinstance(parsed_family, dict):
                        raise ValueError("family report is not an object")
                    family = parsed_family
                    errors.extend(_family_binding_errors(
                        parsed_family,
                        family_identity,
                    ))
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    errors.append("family_report_invalid")
        errors.extend(_publication_manifest_errors(
            audit,
            manifest.get("publication"),
            family,
        ))
    publisher_verified = bool(
        signature_valid
        and expected_key is not None
        and signer_fingerprint == _sha256(expected_key)
    )
    status = (
        "INVALID"
        if errors
        else "TRUSTED"
        if publisher_verified
        else "SIGNED_UNTRUSTED"
        if signature_valid
        else "VERIFIED_INTEGRITY"
    )
    return {
        "format": "qalpha-verify/1",
        "status": status,
        "valid": not errors,
        "path": str(target),
        "sha256": _sha256_file(target),
        "name": manifest.get("name"),
        "version": manifest.get("version"),
        "factor": manifest.get("factor"),
        "data": manifest.get("data"),
        "packStatus": manifest.get("status"),
        "audit": audit,
        "auditSummary": audit_identity,
        "familyEvidence": family,
        "familyEvidenceSummary": family_identity,
        "researchContext": manifest.get("researchContext"),
        "publication": manifest.get("publication"),
        "claimScope": "ARCHIVE_INTEGRITY_AND_EMBEDDED_EVIDENCE_ONLY",
        "independentEvaluation": False,
        "evaluation": manifest.get("evaluation"),
        "signatureValid": signature_valid,
        "signerFingerprint": signer_fingerprint,
        "publisherVerified": publisher_verified,
        "trustedKeySupplied": bool(trusted_key),
        "errors": errors,
        "reproduction": {
            "status": "REQUIRES_QUANT_RUNTIME",
            "command": "quant-agent evidence.reproduce --file <pack.qalpha> --data <panel> --json",
        },
        "executedFactorSource": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a self-contained Quant Alpha Evidence Pack",
    )
    parser.add_argument("pack")
    parser.add_argument("--trusted-key")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = verify(args.pack, trusted_key=args.trusted_key)
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        result = {
            "format": "qalpha-verify/1",
            "status": "INVALID",
            "valid": False,
            "path": str(args.pack),
            "errors": [f"{type(error).__name__}: {error}"],
        }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print(f"Quant Alpha Evidence Pack: {result.get('name') or '<unknown>'}")
        print(f"Status: {result['status']}")
        print(f"SHA-256: {result.get('sha256') or '<unavailable>'}")
        if result.get("signerFingerprint"):
            print(f"Signer fingerprint: {result['signerFingerprint']}")
        print(
            "Publisher trust: "
            + ("verified by supplied key" if result.get("publisherVerified") else "not verified")
        )
        for error in result.get("errors") or []:
            print(f"ERROR: {error}")
        if result.get("valid"):
            print("Integrity and optional signature checks passed.")
            print("Factor reproduction requires the local Quant runtime and the bound data slice.")
    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
