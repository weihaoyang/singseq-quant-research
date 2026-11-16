"""Summarize reference-factor execution status from a baseline batch."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from factors_catalog.factor_registry import FACTOR_DEFINITIONS


def summarize(root: str | Path) -> dict:
    directory = Path(root)
    current = directory / "current.json"
    if not current.is_file():
        current = directory / "status.json"
    state = json.loads(current.read_text(encoding="utf-8"))
    rows = list(state.get("rows") or [])
    statuses = Counter(str(row.get("evaluation_status") or "UNKNOWN") for row in rows)
    families = Counter(str(row.get("family") or "unknown") for row in rows)
    origins = Counter(
        "registry"
        if str(row.get("factor_name") or "") in FACTOR_DEFINITIONS
        else "generated"
        for row in rows
    )
    missing = [
        {
            "factor_name": row.get("factor_name"),
            "family": row.get("family"),
            "status": row.get("evaluation_status"),
            "reason": row.get("reason"),
        }
        for row in rows
        if row.get("evaluation_status") != "COMPLETE"
    ]
    generic_reasons = [
        {
            "factor_name": row.get("factor_name"),
            "reason": row.get("reason"),
        }
        for row in rows
        if "当前公开数据及已登记实现无法计算此因子" in str(
            row.get("reason") or ""
        )
    ]
    availability_path = directory / "r441-registry-availability.json"
    availability_summary = None
    if availability_path.is_file():
        availability = json.loads(
            availability_path.read_text(encoding="utf-8")
        )
        availability_rows = list(
            (availability.get("rows") or {}).values()
        )
        availability_summary = {
            "total": availability.get("total"),
            "computed": availability.get("computed"),
            "statuses": dict(
                sorted(
                    Counter(
                        str(row.get("status") or "UNKNOWN")
                        for row in availability_rows
                    ).items()
                )
            ),
            "non_complete": [
                {
                    "factor_name": row.get("factor_name"),
                    "status": row.get("status"),
                    "missing_inputs": row.get("missing_inputs") or [],
                    "reason": row.get("reason"),
                    "coverage": row.get("coverage"),
                }
                for row in availability_rows
                if row.get("status") != "COMPLETE"
            ],
        }
    audit_passed = (
        state.get("status") == "COMPLETE"
        and not generic_reasons
    )
    return {
        "batch_status": state.get("status"),
        "contract_id": state.get("contract_id"),
        "progress": state.get("progress"),
        "total": len(rows),
        "statuses": dict(sorted(statuses.items())),
        "families": dict(sorted(families.items())),
        "origins": dict(sorted(origins.items())),
        "registry_availability": availability_summary,
        "generic_reason_rows": generic_reasons,
        "audit_passed": audit_passed,
        "non_complete": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = summarize(args.root)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
