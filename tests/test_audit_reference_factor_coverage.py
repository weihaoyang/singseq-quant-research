from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_reference_factor_coverage import summarize


class ReferenceFactorCoverageAuditTests(unittest.TestCase):
    def test_summary_preserves_non_complete_reasons(self) -> None:
        state = {
            "status": "COMPLETE",
            "contract_id": "contract",
            "progress": {"completed": 2, "total": 2},
            "rows": [
                {
                    "factor_name": "return_20",
                    "family": "momentum",
                    "evaluation_status": "COMPLETE",
                },
                {
                    "factor_name": "close_drive_strength_30m",
                    "family": "intraday",
                    "evaluation_status": "UNAVAILABLE",
                    "reason": "30-minute source missing",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            Path(temporary, "current.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            Path(
                temporary,
                "r441-registry-availability.json",
            ).write_text(
                json.dumps(
                    {
                        "total": 2,
                        "computed": 1,
                        "rows": {
                            "return_20": {
                                "factor_name": "return_20",
                                "status": "COMPLETE",
                                "reason": "real values",
                            },
                            "close_drive_strength_30m": {
                                "factor_name": "close_drive_strength_30m",
                                "status": "PARTIAL",
                                "missing_inputs": [],
                                "reason": (
                                    "minute_source_published_through=20260410; "
                                    "uncovered_from=20260411"
                                ),
                                "coverage": {
                                    "published_through": "20260410"
                                },
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = summarize(temporary)
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["statuses"], {"COMPLETE": 1, "UNAVAILABLE": 1})
        self.assertEqual(report["origins"], {"generated": 1, "registry": 1})
        self.assertTrue(report["audit_passed"])
        self.assertEqual(report["generic_reason_rows"], [])
        self.assertEqual(
            report["registry_availability"]["statuses"],
            {"COMPLETE": 1, "PARTIAL": 1},
        )
        self.assertEqual(
            report["registry_availability"]["non_complete"][0][
                "coverage"
            ]["published_through"],
            "20260410",
        )
        self.assertEqual(
            report["non_complete"][0]["reason"],
            "30-minute source missing",
        )

    def test_generic_fallback_reason_fails_the_audit(self) -> None:
        state = {
            "status": "COMPLETE",
            "rows": [
                {
                    "factor_name": "unknown_factor",
                    "family": "event",
                    "evaluation_status": "UNAVAILABLE",
                    "reason": (
                        "当前公开数据及已登记实现无法计算此因子，"
                        "需要其声明的基本面、资金流或分钟输入。"
                    ),
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            Path(temporary, "current.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            report = summarize(temporary)
        self.assertFalse(report["audit_passed"])
        self.assertEqual(
            report["generic_reason_rows"][0]["factor_name"],
            "unknown_factor",
        )


if __name__ == "__main__":
    unittest.main()
