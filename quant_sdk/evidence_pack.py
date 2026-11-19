"""Portable, content-addressed `.qalpha` research artifacts.

The pack contains the factor definition and evidence metadata, never the
researcher's market database.  Verification recomputes every embedded digest
and reports missing or modified files instead of trusting the manifest.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from quant_sdk.labels import LABEL_VERSION

PACK_FORMAT = "qalpha/1"
MANIFEST_NAME = "manifest.json"
SELF_VERIFY_README = "VERIFY.txt"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_FILES = 10_000
MAX_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
_PUBLICATION_CONTEXT = (
    "pointInTime", "costModel", "benchmark", "selectionProtocol",
    "outOfSample", "capacity", "leakageChecks",
)
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
FAMILY_AUDIT_FORMAT = "qalpha-family-audit/1"


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return format(float(value), ".12g") if math.isfinite(value) else None
    return str(value)


def _canonical_table_bytes(frame: pd.DataFrame, *, value_columns: list[str] | None = None) -> bytes:
    if not {"ts_code", "trade_date"}.issubset(frame.columns):
        raise ValueError("canonical table requires ts_code and trade_date")
    work = frame.copy()
    work["ts_code"] = work["ts_code"].astype(str)
    work["trade_date"] = work["trade_date"].astype(str).str.replace("-", "", regex=False)
    work = work.sort_values(["ts_code", "trade_date"], kind="mergesort")
    columns = (
        [str(name) for name in value_columns]
        if value_columns is not None
        else [str(name) for name in work.columns if name not in {"ts_code", "trade_date"}]
    )
    digest_input = {
        "columns": columns,
        "rows": [
            {
                "ts_code": str(row["ts_code"]),
                "trade_date": str(row["trade_date"]),
                "values": [_canonical_scalar(row.get(name)) for name in columns],
            }
            for row in work.to_dict(orient="records")
        ],
    }
    return _canonical_json(digest_input)


def _data_slice_identity(
    path: Path, start: str | None, end: str | None, benchmark_code: str,
) -> dict[str, Any]:
    from quant_sdk.research_workflow import _compact_date, _load_frame

    if not start or not end:
        raise ValueError("绑定数据切片需要 start 和 end")
    compact_start = _compact_date(start, "start")
    compact_end = _compact_date(end, "end")
    frame = _load_frame(
        path, compact_start, compact_end, benchmark_code=benchmark_code,
    )
    identity = _frame_slice_identity(frame)
    identity["benchmarkResolution"] = frame.attrs.get("benchmarkResolution")
    return identity


def _infer_data_window(path: Path) -> tuple[str, str]:
    import duckdb

    from quant_sdk.research_workflow import _relation_for_duckdb

    connection = duckdb.connect(
        str(path), read_only=True,
    ) if path.suffix.lower() in {".duckdb", ".db"} else duckdb.connect()
    try:
        relation, _ = _relation_for_duckdb(connection, path)
        columns = {
            str(row[0])
            for row in connection.execute(
                f"DESCRIBE SELECT * FROM {relation}"
            ).fetchall()
        }
        date_column = next(
            (name for name in ("trade_date", "date", "datetime", "time", "timestamp")
             if name in columns),
            None,
        )
        if date_column is None:
            raise ValueError("data has no trade_date/date column")
        quoted = '"' + date_column.replace('"', '""') + '"'
        minimum, maximum = connection.execute(
            f"SELECT min(replace(CAST({quoted} AS VARCHAR), '-', '')), "
            f"max(replace(CAST({quoted} AS VARCHAR), '-', '')) FROM {relation}"
        ).fetchone()
        if not minimum or not maximum:
            raise ValueError("data has no usable date range")
        return str(minimum)[:8], str(maximum)[:8]
    finally:
        connection.close()


def _frame_slice_identity(frame: pd.DataFrame) -> dict[str, Any]:
    canonical = _canonical_table_bytes(frame)
    return {
        "sha256": _sha256_bytes(canonical),
        "canonicalPrecision": 12,
        "rows": len(frame),
        "assets": int(frame["ts_code"].astype(str).nunique()),
        "dates": int(frame["trade_date"].astype(str).nunique()),
        "minDate": str(frame["trade_date"].astype(str).min()),
        "maxDate": str(frame["trade_date"].astype(str).max()),
        "columns": [str(column) for column in frame.columns],
    }


def _result_identity(content: bytes, name: str) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "name": name,
        "sha256": _sha256_bytes(content),
        "bytes": len(content),
    }
    if name.lower().endswith(".csv"):
        try:
            from io import BytesIO

            frame = pd.read_csv(BytesIO(content))
            if {"ts_code", "trade_date"}.issubset(frame.columns):
                value_columns = [
                    str(column) for column in frame.columns
                    if column not in {"ts_code", "trade_date"}
                ]
                canonical = _canonical_table_bytes(frame, value_columns=value_columns)
                identity.update({
                    "digestKind": "canonical-factor-csv",
                    "canonicalSha256": _sha256_bytes(canonical),
                    "canonicalPrecision": 12,
                    "valueColumns": value_columns,
                    "rows": len(frame),
                })
        except (OSError, ValueError, pd.errors.ParserError):
            identity["digestKind"] = "file"
    else:
        identity["digestKind"] = "file"
    return identity


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_identity() -> dict[str, Any]:
    packages = {
        name: _package_version(name)
        for name in (
            "quant-sdk", "pandas", "numpy", "duckdb", "polars",
            "pyarrow", "TA-Lib", "ta-cn", "MyTT",
            "pandas-ta-classic", "ta", "quantalib", "talipp", "pyti",
        )
    }
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: version for name, version in packages.items() if version},
    }


def _builtin_catalog_identity(factor_name: str | None) -> dict[str, Any] | None:
    if not factor_name:
        return None
    path = Path(__file__).with_name("factor_capability_manifest.json")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    item = next(
        (
            value for value in payload.get("factors") or []
            if isinstance(value, dict) and value.get("name") == factor_name
        ),
        None,
    )
    if item is None:
        return None
    return {
        "catalogId": payload.get("catalogId") or "quant-research-builtin",
        "catalogSha256": hashlib.sha256(raw).hexdigest(),
        "catalogFactorSha256": hashlib.sha256(
            json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "minimumHistory": item.get("minimumHistory"),
        "executionValidated": item.get("executionValidated"),
        "source": item.get("source"),
        "library": item.get("library"),
    }


def _used_capability_packs(
    graph: dict[str, Any] | None,
    factor_name: str | None,
) -> list[dict[str, Any]]:
    """Bind referenced Factor/Operator Packs to their installed content."""
    factors: set[str] = {factor_name} if factor_name else set()
    operators: set[str] = set()
    if isinstance(graph, dict):
        for node in graph.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            if str(node.get("op") or "").lower() == "factor":
                name = str(node.get("factor") or "").strip()
                if name:
                    factors.add(name)
            elif str(node.get("op") or "").lower() == "operator":
                name = str(node.get("operator") or "").strip().lower()
                if name:
                    operators.add(name)
    if not factors and not operators:
        return []

    from quant_sdk.capabilities import (
        installed_factor_specs,
        installed_operator_specs,
    )

    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for spec in installed_factor_specs():
        name = str(spec.get("name") or "")
        if name not in factors:
            continue
        pack = spec.get("pack") or {}
        key = (
            str(pack.get("kind") or "factor-pack"),
            str(pack.get("id") or ""),
            str(pack.get("version") or ""),
        )
        entry = selected.setdefault(key, {
            **pack,
            "factors": [],
            "operators": [],
        })
        entry["factors"].append(name)
    for spec in installed_operator_specs():
        name = str(spec.get("name") or "").lower()
        if name not in operators:
            continue
        pack = spec.get("pack") or {}
        key = (
            str(pack.get("kind") or "operator-pack"),
            str(pack.get("id") or ""),
            str(pack.get("version") or ""),
        )
        entry = selected.setdefault(key, {
            **pack,
            "factors": [],
            "operators": [],
        })
        entry["operators"].append(name)
    return [
        {
            **entry,
            "factors": sorted(set(entry["factors"])),
            "operators": sorted(set(entry["operators"])),
        }
        for _, entry in sorted(selected.items())
    ]


def _context_provided(value: Any) -> bool:
    if not isinstance(value, dict):
        return value is not None
    return str(value.get("status") or "").upper() != "NOT_PROVIDED"


def _publication_profile(
    *,
    factor_identity: dict[str, Any],
    data_identity: dict[str, Any] | None,
    audit_identity: dict[str, Any] | None,
    family_identity: dict[str, Any] | None,
    research_context: dict[str, Any],
    capability_packs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Describe whether this pack is publication-grade without publishing it."""
    missing: list[str] = []
    if not factor_identity:
        missing.append("factor")
    if data_identity is None:
        missing.append("data")
    if audit_identity is None:
        missing.append("audit")
    if family_identity is None:
        missing.append("candidateFamily")
    for key in _PUBLICATION_CONTEXT:
        if not _context_provided(research_context.get(key)):
            missing.append(key)
    audit_status = (
        str(audit_identity.get("status") or "")
        if isinstance(audit_identity, dict)
        else "NOT_PROVIDED"
    )
    audit_config = (
        audit_identity.get("configuration")
        if isinstance(audit_identity, dict)
        and isinstance(audit_identity.get("configuration"), dict)
        else {}
    )
    audit_label = (
        audit_config.get("label")
        if isinstance(audit_config.get("label"), dict)
        else {}
    )
    if str(audit_label.get("version") or "") != LABEL_VERSION:
        missing.append("label")
    policy_errors: list[str] = []
    if audit_status == "LOCAL_PROOF_OF_ALPHA":
        policy_errors = _audit_summary_publication_errors(audit_identity)
    if family_identity is None:
        policy_errors.append("candidate_family_missing")
    else:
        if family_identity.get("status") != "PASS":
            policy_errors.append("candidate_family_not_sealed")
        if family_identity.get("validated") is not True:
            policy_errors.append("candidate_family_not_validated")
        if family_identity.get("pboGateStatus") != "PASS":
            policy_errors.append("candidate_family_pbo_gate_failed")
        if family_identity.get("selectedCandidateSurvivedMultipleTesting") is not True:
            policy_errors.append("selected_candidate_not_significant")
    eligible = (
        audit_status == "LOCAL_PROOF_OF_ALPHA"
        and not missing
        and not policy_errors
    )
    return {
        "status": "PUBLISHABLE" if eligible else (
            audit_status if audit_identity is not None else "PRIVATE_RESEARCH"
        ),
        "eligible": eligible,
        "claimScope": "LOCAL_SELF_AUDITED_EVIDENCE",
        "independentEvaluationRequired": eligible,
        "missing": sorted(set(missing)),
        "policyErrors": policy_errors,
        "candidateFamily": family_identity,
        "automaticPublication": False,
        "labelVersion": audit_label.get("version"),
        "sourceDisclosure": (
            "graph_only"
            if factor_identity.get("kind") == "graph"
            else "source_included"
            if "source" in factor_identity
            else "catalog_reference"
        ),
        "capabilityPacks": capability_packs,
    }


def _audit_publication_errors(audit: dict[str, Any]) -> list[str]:
    """Return stable codes for an incomplete publication-grade audit."""
    errors: list[str] = []
    status = str(audit.get("status") or "").upper()
    if status != "LOCAL_PROOF_OF_ALPHA":
        return ["audit_not_publishable"]
    if audit.get("validated") is not True:
        errors.append("audit_not_validated")
    if audit.get("executionBoundary") not in {None, "local"}:
        errors.append("audit_execution_boundary_invalid")
    if audit.get("submitted") not in {None, False}:
        errors.append("audit_submitted_state_invalid")
    if audit.get("published") not in {None, False}:
        errors.append("audit_published_state_invalid")

    checks = audit.get("checks")
    if not isinstance(checks, dict):
        errors.append("audit_checks_missing")
    else:
        for key in _PUBLICATION_AUDIT_CHECKS:
            if checks.get(key) is not True:
                errors.append(f"audit_check_failed:{key}")

    configuration = audit.get("configuration")
    label = (
        configuration.get("label")
        if isinstance(configuration, dict)
        and isinstance(configuration.get("label"), dict)
        else {}
    )
    if str(label.get("version") or "") != LABEL_VERSION:
        errors.append("audit_label_mismatch")
    if str(label.get("matching") or "") != "next-open":
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


def _audit_summary_publication_errors(
    audit_identity: dict[str, Any],
) -> list[str]:
    """Evaluate the compact audit summary without trusting its status alone."""
    errors: list[str] = []
    checks = audit_identity.get("checks")
    if not isinstance(checks, dict):
        return ["audit_checks_missing"]
    for key in _PUBLICATION_AUDIT_CHECKS:
        if checks.get(key) is not True:
            errors.append(f"audit_check_failed:{key}")
    configuration = audit_identity.get("configuration")
    label = (
        configuration.get("label")
        if isinstance(configuration, dict)
        and isinstance(configuration.get("label"), dict)
        else {}
    )
    if str(label.get("version") or "") != LABEL_VERSION:
        errors.append("audit_label_mismatch")
    if str(label.get("matching") or "") != "next-open":
        errors.append("audit_label_not_next_open")
    split = audit_identity.get("split")
    if not isinstance(split, dict) or not all(
        split.get(key)
        for key in (
            "selectionStart", "selectionEnd",
            "holdoutStart", "holdoutEnd",
        )
    ):
        errors.append("audit_split_incomplete")
    point_in_time = audit_identity.get("pointInTime")
    point_in_time = point_in_time if isinstance(point_in_time, dict) else {}
    if point_in_time.get("status") != "OBSERVED":
        errors.append("audit_point_in_time_missing")
    if point_in_time.get("signalTime") != "session-close:t":
        errors.append("audit_signal_time_invalid")
    if point_in_time.get("executionTime") != "open:t+1":
        errors.append("audit_execution_time_invalid")
    out_of_sample = audit_identity.get("outOfSample")
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
    benchmark = audit_identity.get("benchmark")
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
    capacity = audit_identity.get("capacity")
    capacity = capacity if isinstance(capacity, dict) else {}
    if capacity.get("status") != "OK":
        errors.append("audit_capacity_not_ok")
    costs = audit_identity.get("costSensitivity")
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


def _publication_manifest_errors(
    audit: dict[str, Any] | None,
    publication: Any,
) -> list[str]:
    """Bind the manifest publication projection to its embedded audit."""
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
        family = profile.get("candidateFamily")
        if not isinstance(family, dict):
            errors.append("publication_candidate_family_missing")
        else:
            if family.get("status") != "PASS":
                errors.append("publication_candidate_family_not_sealed")
            if family.get("validated") is not True:
                errors.append("publication_candidate_family_not_validated")
            if family.get("pboGateStatus") != "PASS":
                errors.append("publication_candidate_family_pbo_gate_failed")
            if family.get("selectedCandidateSurvivedMultipleTesting") is not True:
                errors.append("publication_selected_candidate_not_significant")
    elif status == "LOCAL_PROOF_OF_ALPHA":
        errors.append("publication_profile_not_promoted")
    return errors


def _safe_member(name: str) -> str:
    member = str(PurePosixPath(name))
    if not member or member.startswith("/") or ".." in PurePosixPath(member).parts:
        raise ValueError(f"非法 .qalpha 成员路径: {name}")
    return member


def _zip_add(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    info = zipfile.ZipInfo(_safe_member(name), date_time=(2020, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, content)


def _audit_summary(
    audit: dict[str, Any],
    *,
    member: str,
    content: bytes,
) -> dict[str, Any]:
    """Validate a local audit and return the manifest's evidence binding."""
    if not isinstance(audit, dict):
        raise ValueError("审计报告必须是 JSON object")
    if audit.get("format") != "qalpha-audit/1":
        raise ValueError("审计报告必须使用 qalpha-audit/1")
    status = str(audit.get("status") or "").upper()
    allowed = {"LOCAL_PROOF_OF_ALPHA", "RESEARCH_ONLY", "REJECTED_LEAKAGE"}
    if status not in allowed:
        raise ValueError(f"不支持的审计状态: {status or '<missing>'}")
    validated = audit.get("validated")
    if not isinstance(validated, bool):
        raise ValueError("审计报告 validated 必须是布尔值")
    if validated != (status == "LOCAL_PROOF_OF_ALPHA"):
        raise ValueError("审计 status 与 validated 不一致")
    if audit.get("executionBoundary") not in {None, "local"}:
        raise ValueError("审计报告不是本地执行证据")
    if audit.get("submitted") not in {None, False}:
        raise ValueError("审计报告不得绑定已提交状态")
    if audit.get("published") not in {None, False}:
        raise ValueError("审计报告不得绑定已发布状态")
    checks = audit.get("checks")
    if status == "LOCAL_PROOF_OF_ALPHA":
        if not isinstance(checks, dict) or checks.get("leakagePassed") is not True:
            raise ValueError("LOCAL_PROOF_OF_ALPHA 必须包含通过的泄漏检查")
        costs = audit.get("costSensitivity")
        if not isinstance(costs, list) or not costs:
            raise ValueError("LOCAL_PROOF_OF_ALPHA 必须包含成本敏感性")
        policy_errors = _audit_publication_errors(audit)
        if policy_errors:
            raise ValueError(
                "LOCAL_PROOF_OF_ALPHA 发布证据不完整: "
                + ", ".join(policy_errors)
            )
    factor = audit.get("factor") if isinstance(audit.get("factor"), dict) else {}
    split = audit.get("split") if isinstance(audit.get("split"), dict) else {}
    out_of_sample = (
        audit.get("outOfSample")
        if isinstance(audit.get("outOfSample"), dict)
        else {}
    )
    costs = audit.get("costSensitivity")
    return {
        "path": member,
        "sha256": _sha256_bytes(content),
        "bytes": len(content),
        "format": audit.get("format"),
        "status": status,
        "validated": validated,
        "window": audit.get("window"),
        "configuration": audit.get("configuration"),
        "pointInTime": audit.get("pointInTime"),
        "factor": {
            "name": factor.get("name"),
            "factorName": factor.get("factorName"),
            "direction": factor.get("direction"),
        },
        "split": split,
        "checks": checks,
        "outOfSample": {
            "selection": out_of_sample.get("selection"),
            "holdout": out_of_sample.get("holdout"),
            "protocol": out_of_sample.get("protocol"),
        },
        "costSensitivity": costs if isinstance(costs, list) else [],
        "benchmark": audit.get("benchmark"),
        "capacity": audit.get("capacity"),
    }


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
    packed_matrix = (
        summary.get("returnMatrix")
        if isinstance(summary.get("returnMatrix"), dict)
        else {}
    )
    if matrix.get("shape") != packed_matrix.get("shape"):
        errors.append("family_matrix_shape_mismatch")
    if str(matrix.get("sha256") or "") != str(
        packed_matrix.get("sha256") or ""
    ):
        errors.append("family_matrix_hash_mismatch")
    return errors


def _audit_research_context(audit: dict[str, Any], member: str) -> dict[str, Any]:
    runtime = (
        audit.get("runtimeCausality")
        if isinstance(audit.get("runtimeCausality"), dict)
        else {}
    )
    static = (
        audit.get("staticAudit")
        if isinstance(audit.get("staticAudit"), dict)
        else {}
    )
    costs = audit.get("costSensitivity")
    cost_names = (
        [str(item.get("name")) for item in costs if isinstance(item, dict)]
        if isinstance(costs, list)
        else []
    )
    return {
        "pointInTime": audit.get("pointInTime") or {
            "status": "NOT_PROVIDED",
        },
        "costModel": {
            "status": "EMBEDDED",
            "source": member,
            "cases": cost_names,
        },
        "benchmark": audit.get("benchmark") or {"status": "NOT_PROVIDED"},
        "selectionProtocol": {
            "status": "EMBEDDED",
            "source": member,
            "split": audit.get("split"),
            "configuration": audit.get("configuration"),
        },
        "outOfSample": audit.get("outOfSample") or {"status": "NOT_PROVIDED"},
        "capacity": audit.get("capacity") or {"status": "NOT_PROVIDED"},
        "leakageChecks": {
            "status": runtime.get("status"),
            "staticSourceAudit": static.get("status"),
            "runtimeCausality": runtime,
            "checks": audit.get("checks"),
            "source": member,
        },
    }


def _validate_audit_binding(
    audit: dict[str, Any],
    *,
    factor_name: str | None,
    data_identity: dict[str, Any] | None,
) -> None:
    window = audit.get("window")
    if data_identity is not None and isinstance(window, dict):
        packed_window = data_identity.get("window") or {}
        audit_start = _compact_date(window.get("start"), "audit.window.start")
        audit_end = _compact_date(window.get("end"), "audit.window.end")
        data_start = _compact_date(
            packed_window.get("start"), "data.window.start"
        )
        data_end = _compact_date(packed_window.get("end"), "data.window.end")
        if (audit_start, audit_end) != (data_start, data_end):
            raise ValueError("审计报告的数据窗口与 Evidence Pack 不一致")
    if factor_name:
        audited = (audit.get("factor") or {}).get("factorName")
        if audited and str(audited) != str(factor_name):
            raise ValueError("审计报告的 factorName 与 Evidence Pack 不一致")


def _family_audit_summary(
    family: dict[str, Any],
    *,
    member: str,
    content: bytes,
) -> dict[str, Any]:
    """Validate a family audit and return its bounded manifest projection."""
    if family.get("format") != FAMILY_AUDIT_FORMAT:
        raise ValueError(f"候选族报告必须使用 {FAMILY_AUDIT_FORMAT}")
    status = str(family.get("status") or "").upper()
    if status not in {"PASS", "BLOCKED"}:
        raise ValueError(f"不支持的候选族状态: {status or '<missing>'}")
    if family.get("executionBoundary") != "local":
        raise ValueError("候选族报告不是本地执行证据")
    if family.get("submitted") not in {None, False}:
        raise ValueError("候选族报告不得绑定已提交状态")
    if family.get("published") not in {None, False}:
        raise ValueError("候选族报告不得绑定已发布状态")
    if not isinstance(family.get("validated"), bool):
        raise ValueError("候选族报告 validated 必须是布尔值")
    matrix = family.get("returnMatrix")
    matrix = matrix if isinstance(matrix, dict) else {}
    shape = matrix.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not all(isinstance(value, int) and value >= 0 for value in shape)
    ):
        raise ValueError("候选族报告缺少 returnMatrix.shape")
    matrix_hash = str(matrix.get("sha256") or "")
    if matrix_hash and (
        len(matrix_hash) != 64
        or any(char not in "0123456789abcdef" for char in matrix_hash.lower())
    ):
        raise ValueError("候选族 returnMatrix.sha256 无效")
    pbo = family.get("pbo") if isinstance(family.get("pbo"), dict) else {}
    multiple = (
        family.get("multipleTesting")
        if isinstance(family.get("multipleTesting"), dict)
        else {}
    )
    return {
        "path": member,
        "sha256": _sha256_bytes(content),
        "bytes": len(content),
        "format": FAMILY_AUDIT_FORMAT,
        "familyId": family.get("familyId"),
        "familySha256": family.get("familySha256"),
        "status": status,
        "validated": family.get("validated"),
        "window": family.get("window"),
        "selectedCandidateId": family.get("selectedCandidateId"),
        "selectedCandidateSurvivedMultipleTesting": (
            family.get("selectedCandidateSurvivedMultipleTesting") is True
        ),
        "pboGateStatus": pbo.get("gateStatus"),
        "candidateCount": family.get("candidateCount"),
        "observedCandidateCount": family.get("observedCandidateCount"),
        "candidates": [
            {
                "id": item.get("id"),
                "factorName": item.get("factorName"),
                "direction": item.get("direction"),
                "sourceSha256": item.get("sourceSha256"),
                "status": item.get("status"),
            }
            for item in family.get("candidates") or []
            if isinstance(item, dict)
        ],
        "returnMatrix": {
            "shape": shape,
            "sha256": matrix_hash or None,
            "dates": matrix.get("dates"),
            "candidateIds": matrix.get("candidateIds"),
        },
        "pbo": {
            "status": pbo.get("status"),
            "gateStatus": pbo.get("gateStatus"),
            "pbo": pbo.get("pbo"),
            "nPartitions": pbo.get("n_partitions"),
            "nStrategies": pbo.get("n_strategies"),
            "nObservations": pbo.get("n_obs"),
            "gateMax": pbo.get("gateMax"),
        },
        "multipleTesting": {
            "status": multiple.get("status"),
            "method": multiple.get("method"),
            "alpha": multiple.get("alpha"),
            "pValues": multiple.get("pValues"),
            "rejected": multiple.get("rejected"),
            "threshold": multiple.get("threshold"),
        },
        "checks": family.get("checks"),
        "disclosure": family.get("disclosure"),
    }


def _validate_family_binding(
    family: dict[str, Any],
    *,
    factor_name: str | None,
    factor_identity: dict[str, Any],
    data_identity: dict[str, Any] | None,
) -> None:
    window = family.get("window")
    if data_identity is not None and isinstance(window, dict):
        packed_window = data_identity.get("window") or {}
        family_start = _compact_date(window.get("start"), "family.window.start")
        family_end = _compact_date(window.get("end"), "family.window.end")
        data_start = _compact_date(
            packed_window.get("start"), "data.window.start"
        )
        data_end = _compact_date(packed_window.get("end"), "data.window.end")
        if (family_start, family_end) != (data_start, data_end):
            raise ValueError("候选族报告的数据窗口与 Evidence Pack 不一致")
    selected = str(family.get("selectedCandidateId") or "").strip()
    candidates = [
        item for item in family.get("candidates") or []
        if isinstance(item, dict) and str(item.get("id") or "").strip() == selected
    ]
    if not selected or len(candidates) != 1:
        raise ValueError("候选族报告缺少唯一 selectedCandidateId")
    selected_candidate = candidates[0]
    source_hash = str(factor_identity.get("sourceSha256") or "")
    selected_source = str(selected_candidate.get("sourceSha256") or "")
    selected_name = str(selected_candidate.get("factorName") or "").strip()
    if factor_name and selected_name:
        if selected_name != str(factor_name):
            raise ValueError("候选族 selectedCandidate 与 Evidence Pack 因子不一致")
    elif source_hash and selected_source != source_hash:
        raise ValueError("候选族 selectedCandidate 源码与 Evidence Pack 不一致")


def create_pack(
    output_path: str | Path,
    *,
    name: str,
    version: str = "1",
    source_file: str | Path | None = None,
    source_text: str | None = None,
    graph: dict[str, Any] | None = None,
    factor_name: str | None = None,
    data_path: str | Path | None = None,
    data_start: str | None = None,
    data_end: str | None = None,
    benchmark_code: str = "000300.SH",
    result_file: str | Path | None = None,
    audit_file: str | Path | None = None,
    family_audit_file: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    signing_key_file: str | Path | None = None,
    signer: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic `.qalpha` pack from local files and evidence."""
    factor_name = str(factor_name or "").strip() or None
    source_path = Path(source_file).expanduser().resolve() if source_file else None
    if source_path is not None and not source_path.is_file():
        raise ValueError(f"因子源码不存在: {source_path}")
    if source_text is None and source_path is not None:
        source_text = source_path.read_text(encoding="utf-8")
    if not source_text and not graph and not factor_name:
        raise ValueError("Evidence Pack 需要源码、graph 或 factor_name")

    embedded: dict[str, bytes] = {}
    factor_identity: dict[str, Any] = {}
    if source_text:
        source_bytes = str(source_text).replace("\r\n", "\n").encode("utf-8")
        embedded["factor/source.py"] = source_bytes
        factor_identity.update({
            "kind": "source",
            "source": "factor/source.py",
            "sourceSha256": _sha256_bytes(source_bytes),
        })
    if graph is not None:
        graph_bytes = json.dumps(
            graph, ensure_ascii=False, sort_keys=True, indent=2,
        ).encode("utf-8")
        embedded["factor/graph.json"] = graph_bytes
        factor_identity.update({
            "kind": "graph" if not source_text else "source_and_graph",
            "graph": "factor/graph.json",
            "graphSha256": _sha256_bytes(graph_bytes),
        })
    if factor_name:
        factor_identity["catalogName"] = factor_name
        factor_identity.setdefault("kind", "catalog")
        catalog_identity = _builtin_catalog_identity(factor_name)
        if catalog_identity is not None:
            factor_identity["catalog"] = catalog_identity
    if not factor_identity:
        raise ValueError("无法建立因子身份")
    capability_packs = _used_capability_packs(graph, factor_name)

    embedded[SELF_VERIFY_README] = (
        b"Portable Quant Alpha Evidence Pack\n"
        b"\n"
        b"This archive is data-only and is never executed as a program.\n"
        b"\n"
        b"Verify it with the separately distributed, standard-library-only\n"
        b"`qalpha-verify.py`:\n"
        b"\n"
        b"  python qalpha-verify.py <artifact>.qalpha --json\n"
        b"\n"
        b"The verifier checks the manifest, every embedded digest, and the\n"
        b"optional Ed25519 signature. When `evidence/audit.json` is present,\n"
        b"the verifier also exposes its OOS and cost-sensitivity tables without\n"
        b"executing factor source. When `evidence/family-audit.json` is present,\n"
        b"it exposes the sealed PBO and multiple-testing evidence. It streams\n"
        b"member hashes and rejects\n"
        b"manifests over 16 MiB, archives over 10,000 files, or more than 4 GiB\n"
        b"uncompressed content. Factor reproduction still requires the Quant\n"
        b"runtime and the market-data slice identified by the manifest.\n"
    )

    data_identity: dict[str, Any] | None = None
    if data_path:
        path = Path(data_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"研究数据不存在: {path}")
        if not data_start or not data_end:
            data_start, data_end = _infer_data_window(path)
        data_identity = {
            "name": path.name,
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
            "window": {"start": data_start, "end": data_end},
            "slice": _data_slice_identity(
                path, data_start, data_end, str(benchmark_code),
            ),
            "embedded": False,
        }

    result_identity: dict[str, Any] | None = None
    if result_file:
        path = Path(result_file).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"结果文件不存在: {path}")
        content = path.read_bytes()
        member = f"results/{path.name}"
        embedded[member] = content
        result_identity = {
            **_result_identity(content, path.name),
            "path": member,
        }

    audit_identity: dict[str, Any] | None = None
    audit_payload: dict[str, Any] | None = None
    if audit_file:
        path = Path(audit_file).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"审计报告不存在: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"审计报告不是有效 JSON: {path}") from error
        audit_payload = value if isinstance(value, dict) else {}
        _validate_audit_binding(
            audit_payload,
            factor_name=factor_name,
            data_identity=data_identity,
        )
        audit_member = "evidence/audit.json"
        audit_bytes = (
            json.dumps(
                audit_payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        embedded[audit_member] = audit_bytes
        audit_identity = _audit_summary(
            audit_payload,
            member=audit_member,
            content=audit_bytes,
        )

    family_identity: dict[str, Any] | None = None
    family_payload: dict[str, Any] | None = None
    if family_audit_file:
        path = Path(family_audit_file).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"候选族报告不存在: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"候选族报告不是有效 JSON: {path}") from error
        family_payload = value if isinstance(value, dict) else {}
        _validate_family_binding(
            family_payload,
            factor_name=factor_name,
            factor_identity=factor_identity,
            data_identity=data_identity,
        )
        family_member = "evidence/family-audit.json"
        family_bytes = (
            json.dumps(
                family_payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        embedded[family_member] = family_bytes
        family_identity = _family_audit_summary(
            family_payload,
            member=family_member,
            content=family_bytes,
        )

    created_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    supplied_metadata = metadata or {}
    research_context = {
        key: supplied_metadata.get(key, {"status": "NOT_PROVIDED"})
        for key in (
            "pointInTime", "costModel", "benchmark", "selectionProtocol",
            "outOfSample", "capacity", "leakageChecks",
        )
    }
    if audit_payload is not None and audit_identity is not None:
        for key, value in _audit_research_context(
            audit_payload, audit_identity["path"],
        ).items():
            # Audit-derived evidence always wins. Caller metadata cannot promote
            # a pack by self-reporting a stronger PIT or benchmark status.
            research_context[key] = value
    publication = _publication_profile(
        factor_identity=factor_identity,
        data_identity=data_identity,
        audit_identity=audit_identity,
        family_identity=family_identity,
        research_context=research_context,
        capability_packs=capability_packs,
    )
    files = [
        {"path": member, "sha256": _sha256_bytes(content), "bytes": len(content)}
        for member, content in sorted(embedded.items())
    ]
    manifest: dict[str, Any] = {
        "format": PACK_FORMAT,
        "name": str(name).strip(),
        "version": str(version).strip(),
        "createdAt": created_at,
        "status": (
            audit_identity["status"]
            if audit_identity is not None
            else "LOCAL_UNVERIFIED"
        ),
        "executionBoundary": "local_only",
        "factor": factor_identity,
        "data": data_identity,
        "result": result_identity,
        "audit": audit_identity,
        "familyEvidence": family_identity,
        "verification": {
            "format": "qalpha-external-verify/1",
            "verifier": "qalpha-verify.py",
            "command": "python qalpha-verify.py <artifact>.qalpha --json",
            "standardLibraryOnly": True,
            "packIsExecutable": False,
            "executesFactorSource": False,
        },
        "runtime": _runtime_identity(),
        "evaluation": {
            "runtimeLabelVersion": LABEL_VERSION,
            "auditLabel": (
                ((audit_payload or {}).get("configuration") or {}).get("label")
                if isinstance(audit_payload, dict) else None
            ),
            "alphaGraphVersion": (
                str(graph.get("version") or "alpha-graph-1")
                if graph is not None else None
            ),
            "capabilityPacks": capability_packs,
        },
        "files": files,
        "metadata": supplied_metadata,
        "researchContext": research_context,
        "publication": publication,
        "reproduce": {
            "requiresLocalDataSha256": data_identity["sha256"] if data_identity else None,
            "command": (
                "quant-agent evidence.verify --file <pack.qalpha> --json"
            ),
            "note": "数据不随包分发；只记录本地数据身份。",
        },
        "signature": None,
    }
    if not manifest["name"] or not manifest["version"]:
        raise ValueError("Evidence Pack 的 name/version 不能为空")
    manifest["manifestSha256"] = _sha256_bytes(_canonical_json(_manifest_body(manifest)))
    if signing_key_file:
        key_path = Path(signing_key_file).expanduser().resolve()
        if not key_path.is_file():
            raise ValueError(f"签名私钥不存在: {key_path}")
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
        except ImportError as error:  # pragma: no cover - dependency error
            raise RuntimeError("签名需要安装 cryptography") from error
        private_key = serialization.load_pem_private_key(
            key_path.read_bytes(), password=None,
        )
        if not isinstance(private_key, Ed25519PrivateKey):
            raise ValueError("签名私钥必须是 Ed25519 私钥")
        signature = private_key.sign(_canonical_json(_manifest_body(manifest)))
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        manifest["signature"] = {
            "algorithm": "ed25519",
            "publicKey": base64.b64encode(public_key).decode("ascii"),
            "publicKeyFingerprint": _sha256_bytes(public_key),
            "value": base64.b64encode(signature).decode("ascii"),
            "signer": str(signer or "").strip() or None,
            "trustModel": "self-signed; a trust anchor is required to prove publisher identity",
        }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    target = Path(output_path).expanduser().resolve()
    if target.suffix.lower() != ".qalpha":
        target = target.with_suffix(".qalpha")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with zipfile.ZipFile(temporary, "w") as archive:
        _zip_add(archive, MANIFEST_NAME, manifest_bytes)
        for member, content in sorted(embedded.items()):
            _zip_add(archive, member, content)
    os.replace(temporary, target)
    return {
        "status": "PACKED",
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": _sha256_file(target),
        "manifestSha256": manifest["manifestSha256"],
        "format": PACK_FORMAT,
        "factor": factor_identity,
        "data": data_identity,
        "audit": audit_identity,
        "familyEvidence": family_identity,
        "publication": publication,
        "signed": bool(manifest.get("signature")),
        "externallyVerifiable": True,
        "verificationCommand": "python qalpha-verify.py <artifact>.qalpha --json",
    }


def generate_signing_key(output_path: str | Path) -> dict[str, Any]:
    """Create an Ed25519 private PEM for local Evidence Pack signing."""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as error:  # pragma: no cover - dependency error
        raise RuntimeError("需要安装 cryptography 才能生成签名密钥") from error
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError(f"拒绝覆盖已有签名私钥: {target}")
    private_key = Ed25519PrivateKey.generate()
    target.write_bytes(private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return {
        "status": "CREATED",
        "privateKey": str(target),
        "publicKey": base64.b64encode(public_key).decode("ascii"),
        "algorithm": "ed25519",
    }


def _manifest_body(manifest: dict[str, Any]) -> dict[str, Any]:
    body = dict(manifest)
    body.pop("manifestSha256", None)
    body.pop("signature", None)
    return body


def _trusted_public_key(value: str | Path | None) -> bytes | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    content = (
        path.read_bytes().strip()
        if path.is_file()
        else str(value).strip().encode("ascii")
    )
    if content.startswith(b"-----BEGIN"):
        try:
            from cryptography.hazmat.primitives import serialization

            key = serialization.load_pem_public_key(content)
            return key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        except Exception as error:
            raise ValueError("trusted key PEM cannot be decoded") from error
    try:
        return base64.b64decode(content, validate=True)
    except Exception as error:
        raise ValueError("trusted key must be raw base64 or a PEM public key") from error


def _verify_signature(
    manifest: dict[str, Any],
) -> tuple[bool, str | None, str | None]:
    signature = manifest.get("signature")
    if signature is None:
        return False, None, None
    if not isinstance(signature, dict) or signature.get("algorithm") != "ed25519":
        return False, "unsupported_signature", None
    fingerprint = None
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        public_bytes = base64.b64decode(str(signature["publicKey"]), validate=True)
        fingerprint = _sha256_bytes(public_bytes)
        value = base64.b64decode(str(signature["value"]), validate=True)
        Ed25519PublicKey.from_public_bytes(public_bytes).verify(
            value, _canonical_json(_manifest_body(manifest)),
        )
        return True, None, fingerprint
    except ImportError:
        return False, "cryptography_missing", fingerprint
    except Exception:  # noqa: BLE001 - any verification failure is untrusted
        return False, "signature_mismatch", fingerprint


def verify_pack(
    path: str | Path, *, trusted_key: str | Path | None = None,
    data_path: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
    benchmark_code: str = "000300.SH",
) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise ValueError(f"Evidence Pack 不存在: {target}")
    errors: list[str] = []
    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
        if len(names) > MAX_ARCHIVE_FILES:
            raise ValueError(
                f"Evidence Pack 文件数量超过安全上限: {len(names)}")
        if MANIFEST_NAME not in names:
            raise ValueError("Evidence Pack 缺少 manifest.json")
        manifest_info = archive.getinfo(MANIFEST_NAME)
        if manifest_info.file_size > MAX_MANIFEST_BYTES:
            raise ValueError("Evidence Pack manifest 超过安全大小上限")
        manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        if manifest.get("format") != PACK_FORMAT:
            errors.append("unsupported_format")
        expected_manifest = manifest.get("manifestSha256")
        body = _manifest_body(manifest)
        actual_manifest = _sha256_bytes(_canonical_json(body))
        if not expected_manifest or expected_manifest != actual_manifest:
            errors.append("manifest_hash_mismatch")
        signature_valid, signature_error, signer_fingerprint = _verify_signature(manifest)
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
                raise ValueError("Evidence Pack 解压后大小超过安全上限")
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
        audit_report: dict[str, Any] | None = None
        audit_summary = manifest.get("audit")
        if isinstance(audit_summary, dict):
            audit_member = _safe_member(str(audit_summary.get("path") or ""))
            if audit_member not in names:
                errors.append(f"missing:{audit_member}")
            else:
                audit_bytes = archive.read(audit_member)
                if _sha256_bytes(audit_bytes) != audit_summary.get("sha256"):
                    errors.append(f"hash_mismatch:{audit_member}")
                if len(audit_bytes) != audit_summary.get("bytes"):
                    errors.append(f"size_mismatch:{audit_member}")
                try:
                    parsed_audit = json.loads(audit_bytes.decode("utf-8"))
                    if not isinstance(parsed_audit, dict):
                        raise ValueError("audit report is not an object")
                    audit_report = parsed_audit
                    errors.extend(_audit_binding_errors(
                        parsed_audit,
                        audit_summary,
                        manifest.get("status"),
                    ))
                    errors.extend(_publication_manifest_errors(
                        parsed_audit,
                        manifest.get("publication"),
                    ))
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    errors.append("audit_report_invalid")
        family_report: dict[str, Any] | None = None
        family_summary = manifest.get("familyEvidence")
        if isinstance(family_summary, dict):
            family_member = _safe_member(str(family_summary.get("path") or ""))
            if family_member not in names:
                errors.append(f"missing:{family_member}")
            else:
                family_bytes = archive.read(family_member)
                if _sha256_bytes(family_bytes) != family_summary.get("sha256"):
                    errors.append(f"hash_mismatch:{family_member}")
                if len(family_bytes) != family_summary.get("bytes"):
                    errors.append(f"size_mismatch:{family_member}")
                try:
                    parsed_family = json.loads(family_bytes.decode("utf-8"))
                    if not isinstance(parsed_family, dict):
                        raise ValueError("family report is not an object")
                    family_report = parsed_family
                    errors.extend(_family_binding_errors(
                        parsed_family, family_summary,
                    ))
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    errors.append("family_report_invalid")
        data_matches = None
        data_file_matches = None
        if data_path:
            source = manifest.get("data") or {}
            expected_slice = (source.get("slice") or {}).get("sha256")
            window = source.get("window") or {}
            local_path = Path(data_path).expanduser().resolve()
            try:
                local = _data_slice_identity(
                    local_path, start or window.get("start"),
                    end or window.get("end"), str(benchmark_code),
                )
                data_file_matches = _sha256_file(local_path) == source.get("sha256")
                data_matches = bool(
                    expected_slice
                    and local["sha256"] == expected_slice
                )
                if not data_matches:
                    errors.append("data_slice_mismatch")
            except Exception as error:  # noqa: BLE001 - reported in verification result
                data_matches = False
                errors.append(f"data_unavailable:{type(error).__name__}")
    expected_key = _trusted_public_key(trusted_key)
    publisher_verified = bool(
        signature_valid
        and expected_key is not None
        and signer_fingerprint == _sha256_bytes(expected_key)
    )
    if expected_key is not None and not publisher_verified:
        errors.append("trusted_key_mismatch")
    status = (
        "INVALID" if errors
        else "TRUSTED" if publisher_verified
        else "SIGNED_UNTRUSTED" if signature_valid
        else "VERIFIED_INTEGRITY"
    )
    return {
        "status": status,
        "valid": not errors,
        "path": str(target),
        "sha256": _sha256_file(target),
        "format": manifest.get("format"),
        "name": manifest.get("name"),
        "version": manifest.get("version"),
        "manifestSha256": manifest.get("manifestSha256"),
        "factor": manifest.get("factor"),
        "data": manifest.get("data"),
        "audit": audit_report,
        "auditSummary": audit_summary,
        "familyEvidence": family_report,
        "familyEvidenceSummary": family_summary,
        "researchContext": manifest.get("researchContext"),
        "publication": manifest.get("publication"),
        "evaluation": manifest.get("evaluation"),
        "signed": signature_valid,
        "signatureValid": signature_valid,
        "claimScope": "ARCHIVE_INTEGRITY_AND_EMBEDDED_EVIDENCE_ONLY",
        "independentEvaluation": False,
        "signerFingerprint": signer_fingerprint,
        "publisherVerified": publisher_verified,
        "trustedKeySupplied": expected_key is not None,
        "dataMatches": data_matches,
        "dataFileShaMatches": data_file_matches,
        "errors": errors,
    }


def _trusted_key_inventory(values: Sequence[str | Path]) -> dict[str, Path]:
    keys: dict[str, Path] = {}
    for value in values:
        path = Path(value).expanduser().resolve()
        candidates = (
            sorted(item for item in path.rglob("*") if item.is_file())
            if path.is_dir()
            else [path]
        )
        for candidate in candidates:
            try:
                raw = _trusted_public_key(candidate)
            except (OSError, ValueError):
                continue
            if raw:
                keys[_sha256_bytes(raw)] = candidate
    return keys


def _pack_paths(values: Sequence[str | Path]) -> list[Path]:
    paths: dict[str, Path] = {}
    for value in values:
        root = Path(value).expanduser().resolve()
        if root.is_file() and root.suffix.lower() == ".qalpha":
            paths[str(root)] = root
            continue
        if not root.is_dir():
            raise ValueError(f"Evidence Pack 路径不存在: {root}")
        for path in root.rglob("*.qalpha"):
            if path.is_file():
                resolved = path.resolve()
                paths[str(resolved)] = resolved
    return [paths[key] for key in sorted(paths)]


def build_index(
    roots: Sequence[str | Path],
    *,
    trusted_keys: Sequence[str | Path] = (),
    query: str | None = None,
    factor_kind: str | None = None,
    data_sha256: str | None = None,
    trusted_only: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Index local `.qalpha` artifacts without uploading or executing them."""
    files = _pack_paths(roots)
    key_inventory = _trusted_key_inventory(trusted_keys)
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    duplicates: list[dict[str, str]] = []
    seen_pack_hashes: set[str] = set()
    normalized_query = str(query or "").strip().lower()
    normalized_kind = str(factor_kind or "").strip().lower()
    normalized_data = str(data_sha256 or "").strip().lower()

    for path in files:
        try:
            checked = verify_pack(path)
            pack_sha = str(checked["sha256"])
            if pack_sha in seen_pack_hashes:
                duplicates.append({"path": str(path), "sha256": pack_sha})
                continue
            seen_pack_hashes.add(pack_sha)
            if not checked.get("valid"):
                invalid.append({
                    "path": str(path),
                    "sha256": pack_sha,
                    "errors": checked.get("errors") or [],
                })
                continue
            signer = str(checked.get("signerFingerprint") or "")
            trusted_key = key_inventory.get(signer)
            if checked.get("signed") and trusted_key is not None:
                checked = verify_pack(path, trusted_key=trusted_key)
            factor = checked.get("factor") or {}
            data = checked.get("data") or {}
            kind = str(factor.get("kind") or "").lower()
            slice_sha = str((data.get("slice") or {}).get("sha256") or "").lower()
            if normalized_kind and kind != normalized_kind:
                continue
            if normalized_data and slice_sha != normalized_data:
                continue
            if trusted_only and not checked.get("publisherVerified"):
                continue
            if normalized_query:
                searchable = " ".join([
                    str(checked.get("name") or ""),
                    str(checked.get("version") or ""),
                    str(factor.get("catalogName") or ""),
                    kind,
                    json.dumps(
                        checked.get("researchContext") or {},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                ]).lower()
                if normalized_query not in searchable:
                    continue
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
            publisher_status = (
                "TRUSTED" if checked.get("publisherVerified")
                else "SIGNED_UNTRUSTED" if checked.get("signed")
                else "UNSIGNED"
            )
            rows.append({
                "id": f"qalpha:{pack_sha}",
                "path": str(path),
                "packSha256": pack_sha,
                "manifestSha256": checked.get("manifestSha256"),
                "name": checked.get("name"),
                "version": checked.get("version"),
                "createdAt": manifest.get("createdAt"),
                "status": publisher_status,
                "integrityValid": True,
                "signed": bool(checked.get("signed")),
                "publisherVerified": bool(checked.get("publisherVerified")),
                "signerFingerprint": checked.get("signerFingerprint"),
                "factor": factor,
                "data": data,
                "result": manifest.get("result"),
                "audit": manifest.get("audit"),
                "researchContext": checked.get("researchContext"),
                "publication": manifest.get("publication"),
                "evaluation": manifest.get("evaluation"),
                "metadata": manifest.get("metadata") or {},
            })
        except Exception as error:  # noqa: BLE001 - one bad pack cannot hide the index
            invalid.append({
                "path": str(path),
                "errors": [f"{type(error).__name__}: {error}"],
            })

    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        data = row.get("data") or {}
        slice_sha = str((data.get("slice") or {}).get("sha256") or "")
        if not slice_sha:
            continue
        group = groups.setdefault(slice_sha, {
            "dataSliceSha256": slice_sha,
            "window": data.get("window"),
            "rows": data.get("slice", {}).get("rows"),
            "assets": data.get("slice", {}).get("assets"),
            "dates": data.get("slice", {}).get("dates"),
            "factors": [],
        })
        group["factors"].append({
            "id": row["id"],
            "name": row["name"],
            "version": row["version"],
            "factorKind": (row.get("factor") or {}).get("kind"),
            "publisherStatus": row["status"],
        })
    for group in groups.values():
        group["factors"].sort(key=lambda item: (
            str(item.get("name") or ""),
            str(item.get("version") or ""),
            str(item.get("id") or ""),
        ))

    result = {
        "format": "qalpha-index/1",
        "status": "LOCAL_VERIFIED",
        "source": {
            "roots": [str(Path(root).expanduser().resolve()) for root in roots],
            "packs": len(files),
            "uniquePacks": len(rows),
            "invalid": len(invalid),
            "duplicates": duplicates,
            "trustedKeyFingerprints": sorted(key_inventory),
        },
        "filters": {
            "query": query,
            "factorKind": factor_kind,
            "dataSha256": data_sha256,
            "trustedOnly": bool(trusted_only),
        },
        "rows": sorted(rows, key=lambda item: (
            str(item.get("name") or ""),
            str(item.get("version") or ""),
            str(item.get("packSha256") or ""),
        )),
        "groups": sorted(groups.values(), key=lambda item: item["dataSliceSha256"]),
        "invalid": invalid,
        "uploaded": False,
        "submitted": False,
        "published": False,
        "note": "Indexed locally. Signature validity and publisher trust remain separate claims.",
    }
    if output_path:
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        result["output"] = str(target)
    return result


def _capability_reproduction_status(
    expected: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare Evidence Pack capability identities with the active local packs."""
    if not expected:
        return {
            "ready": True,
            "required": [],
            "observed": [],
            "issues": [],
        }
    from quant_sdk.capabilities import (
        installed_factor_specs,
        installed_operator_specs,
    )

    observed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for spec in installed_factor_specs():
        pack = spec.get("pack") or {}
        key = (
            str(pack.get("kind") or "factor-pack"),
            str(pack.get("id") or ""),
            str(pack.get("version") or ""),
        )
        entry = observed.setdefault(key, {
            "kind": key[0],
            "id": key[1],
            "version": key[2],
            "contentSha256": pack.get("contentSha256"),
            "factors": [],
            "operators": [],
        })
        entry["factors"].append(str(spec.get("name") or ""))
    for spec in installed_operator_specs():
        pack = spec.get("pack") or {}
        key = (
            str(pack.get("kind") or "operator-pack"),
            str(pack.get("id") or ""),
            str(pack.get("version") or ""),
        )
        entry = observed.setdefault(key, {
            "kind": key[0],
            "id": key[1],
            "version": key[2],
            "contentSha256": pack.get("contentSha256"),
            "factors": [],
            "operators": [],
        })
        entry["operators"].append(str(spec.get("name") or "").lower())
    for entry in observed.values():
        entry["factors"] = sorted(set(entry["factors"]))
        entry["operators"] = sorted(set(entry["operators"]))

    issues: list[dict[str, Any]] = []
    for item in expected:
        if not isinstance(item, dict):
            issues.append({"code": "INVALID_REQUIREMENT"})
            continue
        kind = str(item.get("kind") or "factor-pack")
        pack_id = str(item.get("id") or "")
        version = str(item.get("version") or "")
        key = (kind, pack_id, version)
        current = observed.get(key)
        if current is None:
            alternatives = [
                value for value in observed.values()
                if value["kind"] == kind and value["id"] == pack_id
            ]
            issues.append({
                "code": (
                    "CAPABILITY_VERSION_NOT_ACTIVE"
                    if alternatives else "CAPABILITY_NOT_INSTALLED"
                ),
                "kind": kind,
                "id": pack_id,
                "requiredVersion": version,
                "activeVersions": sorted(
                    value["version"] for value in alternatives
                ),
            })
            continue
        expected_hash = str(item.get("contentSha256") or "")
        current_hash = str(current.get("contentSha256") or "")
        if expected_hash and expected_hash != current_hash:
            issues.append({
                "code": "CAPABILITY_CONTENT_MISMATCH",
                "kind": kind,
                "id": pack_id,
                "version": version,
                "expectedContentSha256": expected_hash,
                "activeContentSha256": current_hash or None,
            })
        missing_factors = sorted(
            set(str(value) for value in item.get("factors") or [])
            - set(current.get("factors") or [])
        )
        missing_operators = sorted(
            set(str(value).lower() for value in item.get("operators") or [])
            - set(current.get("operators") or [])
        )
        if missing_factors or missing_operators:
            issues.append({
                "code": "CAPABILITY_ENTRY_MISSING",
                "kind": kind,
                "id": pack_id,
                "version": version,
                "factors": missing_factors,
                "operators": missing_operators,
            })
    return {
        "ready": not issues,
        "required": expected,
        "observed": [
            observed[key] for key in sorted(observed)
        ],
        "issues": issues,
    }


def reproduce_pack(
    path: str | Path, *, data_path: str | Path,
    start: str | None = None, end: str | None = None,
    benchmark_code: str = "000300.SH",
) -> dict[str, Any]:
    """Re-run a graph/catalog factor against local data and compare factor values."""
    from quant_sdk.alpha_graph import evaluate_graph_file, graph_from_factor_names

    target = Path(path).expanduser().resolve()
    checked = verify_pack(
        target, data_path=data_path, start=start, end=end,
        benchmark_code=benchmark_code,
    )
    if not checked["valid"]:
        return {
            "status": "INVALID",
            "reproduced": False,
            "verification": checked,
        }
    with zipfile.ZipFile(target) as archive:
        manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        factor = manifest.get("factor") or {}
        result = manifest.get("result") or {}
        graph_member = factor.get("graph")
        source_member = factor.get("source")
        source_text = None
        if graph_member and graph_member in archive.namelist():
            graph = json.loads(archive.read(graph_member).decode("utf-8"))
        elif factor.get("catalogName"):
            graph = graph_from_factor_names([str(factor["catalogName"])])
        elif source_member and source_member in archive.namelist():
            graph = None
            source_text = archive.read(source_member).decode("utf-8")
        else:
            raise ValueError(
                "Evidence Pack 缺少可执行源码、graph 或 catalogName。")
        result_member = result.get("path")
        expected_digest = result.get("canonicalSha256")
        if not result_member or result_member not in archive.namelist() or not expected_digest:
            raise ValueError("Evidence Pack 没有可比较的 canonical factor CSV。")
        expected_columns = list(result.get("valueColumns") or [])

    capability_status = _capability_reproduction_status(
        ((manifest.get("evaluation") or {}).get("capabilityPacks") or []),
    )
    if not capability_status["ready"]:
        return {
            "status": "CAPABILITY_MISMATCH",
            "reproduced": False,
            "path": str(target),
            "verification": checked,
            "capabilityPacks": capability_status,
        }

    window = (manifest.get("data") or {}).get("window") or {}
    check_start = _compact_date(start or window.get("start"), "start")
    check_end = _compact_date(end or window.get("end"), "end")
    if graph is not None:
        with tempfile.TemporaryDirectory(prefix="qalpha-replay-") as directory:
            summary = evaluate_graph_file(
                data_path=str(Path(data_path).expanduser().resolve()),
                start=check_start,
                end=check_end,
                graph=graph,
                output_dir=directory,
                benchmark_code=str(benchmark_code),
            )
            actual_path = Path(directory) / "factor_values.csv"
            actual_content = actual_path.read_bytes()
    else:
        from quant_sdk.local import LocalFactorRunner
        from quant_sdk.research_workflow import _load_frame

        frame = _load_frame(
            data_path, check_start, check_end,
            benchmark_code=str(benchmark_code),
        )
        runner = LocalFactorRunner(None)
        summary = runner.compute(str(source_text or ""), frame, price_col="close")
        work = runner.last_work
        actual_path = Path(result_member)
        actual_content = pd.DataFrame({
            "ts_code": work["ts_code"],
            "trade_date": work["trade_date"],
            "factor_value": work["factor_value"],
        }).to_csv(index=False).encode("utf-8")
    original_value_columns = list(
        _result_identity(actual_content, actual_path.name).get("valueColumns") or []
    )
    column_mapping: dict[str, str] = {}
    if (
        expected_columns
        and original_value_columns
        and original_value_columns != expected_columns
        and len(original_value_columns) == len(expected_columns)
    ):
        from io import BytesIO

        actual_frame = pd.read_csv(BytesIO(actual_content))
        column_mapping = dict(zip(original_value_columns, expected_columns, strict=True))
        actual_frame = actual_frame.rename(columns=column_mapping)
        actual_content = actual_frame.to_csv(index=False).encode("utf-8")
    actual_identity = _result_identity(actual_content, actual_path.name)
    actual_digest = actual_identity.get("canonicalSha256")
    reproduced = bool(actual_digest and actual_digest == expected_digest)
    return {
        "status": "REPRODUCED" if reproduced else "MISMATCH",
        "reproduced": reproduced,
        "path": str(target),
        "dataPath": str(Path(data_path).expanduser().resolve()),
        "window": {"start": check_start, "end": check_end},
        "expectedCanonicalSha256": expected_digest,
        "actualCanonicalSha256": actual_digest,
        "expectedValueColumns": expected_columns,
        "originalValueColumns": original_value_columns,
        "actualValueColumns": actual_identity.get("valueColumns"),
        "valueColumnMapping": column_mapping,
        "rows": actual_identity.get("rows"),
        "summary": summary,
        "verification": checked,
        "capabilityPacks": capability_status,
    }


def _compact_date(value: str | None, label: str) -> str:
    from quant_sdk.research_workflow import _compact_date as compact

    return compact(value or "", label)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or verify a .qalpha Evidence Pack")
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("pack")
    pack.add_argument("--output", required=True)
    pack.add_argument("--name", required=True)
    pack.add_argument("--version", default="1")
    pack.add_argument("--source-file")
    pack.add_argument("--graph-file")
    pack.add_argument("--factor-name")
    pack.add_argument("--data")
    pack.add_argument("--start")
    pack.add_argument("--end")
    pack.add_argument("--benchmark-code", default="000300.SH")
    pack.add_argument("--result-file")
    pack.add_argument("--audit")
    pack.add_argument("--family-audit")
    pack.add_argument("--metadata")
    pack.add_argument("--signing-key")
    pack.add_argument("--signer")
    verify = sub.add_parser("verify")
    verify.add_argument("--file", required=True)
    verify.add_argument("--trusted-key")
    verify.add_argument("--data")
    verify.add_argument("--start")
    verify.add_argument("--end")
    verify.add_argument("--benchmark-code", default="000300.SH")
    reproduce = sub.add_parser("reproduce")
    reproduce.add_argument("--file", required=True)
    reproduce.add_argument("--data", required=True)
    reproduce.add_argument("--start")
    reproduce.add_argument("--end")
    reproduce.add_argument("--benchmark-code", default="000300.SH")
    index = sub.add_parser("index")
    index.add_argument("--root", action="append", required=True)
    index.add_argument("--trusted-key", action="append", default=[])
    index.add_argument("--query")
    index.add_argument("--factor-kind")
    index.add_argument("--data-sha256")
    index.add_argument("--trusted-only", action="store_true")
    index.add_argument("--output")
    keygen = sub.add_parser("keygen")
    keygen.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "keygen":
        print(json.dumps(generate_signing_key(args.output), ensure_ascii=False))
        return 0
    if args.command == "verify":
        result = verify_pack(
            args.file, trusted_key=args.trusted_key, data_path=args.data,
            start=args.start, end=args.end, benchmark_code=args.benchmark_code,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["valid"] else 1
    if args.command == "reproduce":
        result = reproduce_pack(
            args.file, data_path=args.data, start=args.start, end=args.end,
            benchmark_code=args.benchmark_code,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["reproduced"] else 1
    if args.command == "index":
        result = build_index(
            args.root,
            trusted_keys=args.trusted_key,
            query=args.query,
            factor_kind=args.factor_kind,
            data_sha256=args.data_sha256,
            trusted_only=args.trusted_only,
            output_path=args.output,
        )
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    graph = None
    if args.graph_file:
        graph = json.loads(Path(args.graph_file).read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata) if args.metadata else {}
    result = create_pack(
        args.output, name=args.name, version=args.version,
        source_file=args.source_file, graph=graph,
        factor_name=args.factor_name, data_path=args.data,
        data_start=args.start, data_end=args.end,
        benchmark_code=args.benchmark_code,
        result_file=args.result_file, audit_file=args.audit,
        family_audit_file=args.family_audit,
        metadata=metadata,
        signing_key_file=args.signing_key,
        signer=args.signer,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

