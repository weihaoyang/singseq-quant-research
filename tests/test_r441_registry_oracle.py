from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from factors_catalog.factor_registry import FACTOR_DEFINITIONS
from factors_catalog.r441_data import load_reference_factor_context
from factors_catalog.r441_panel import (
    MINUTE_FACTOR_NAMES,
    compute_registry_arrays,
    compute_registry_arrays_parallel,
    compute_registry_panel,
    registry_availability_report,
)
from factors_catalog.r441_registry import (
    RegistryFactorInputs,
    compute_registry_factor_snapshot,
)


def _price_window(length: int = 128) -> dict[str, object]:
    random = np.random.default_rng(41)
    prices = 100.0 + np.cumsum(random.normal(0.0, 0.8, length))
    turnover = 10_000_000.0 * (1.0 + random.random(length))
    flows = random.normal(0.0, 1.0, length)
    return {
        "prices": prices.tolist(),
        "turnover": turnover.tolist(),
        "opens": (prices * 0.999).tolist(),
        "highs": (prices * 1.01).tolist(),
        "lows": (prices * 0.99).tolist(),
        "benchmark_prices": prices.tolist(),
        "stock_flows": flows.tolist(),
        "sector_flows": flows.tolist(),
        "sector_closes": prices.tolist(),
        "market_closes": prices.tolist(),
        "sector_up_counts": [50.0] * length,
        "sector_total_counts": [100.0] * length,
        "leader_prices": prices.tolist(),
        "laggard_prices": prices.tolist(),
        "sentiment_series": [0.5] * length,
        "board_break_series": [0.2] * length,
        "gap_series": [0.0] * length,
        "reversal_series": [0.5] * length,
        "event_direction_series": [1.0] * length,
        "event_intensity_series": [0.5] * length,
        "event_confidence_series": [0.8] * length,
        "event_age_minutes": [0.0] * length,
        "event_impact_series": [0.1] * length,
        "event_sector_hits": [1.0] * length,
        "event_symbol_hits": [1.0] * length,
        "regulatory_event_series": [0.0] * length,
        "shareholder_event_series": [0.1] * length,
        "policy_event_series": [0.1] * length,
        "hotword_series": [0.2] * length,
    }


class RegistryOracleTests(unittest.TestCase):
    def test_daily_only_run_does_not_invent_missing_side_inputs(self) -> None:
        window = _price_window()
        snapshot = compute_registry_factor_snapshot(
            RegistryFactorInputs(
                prices=window["prices"],
                turnover=window["turnover"],
                opens=window["opens"],
                highs=window["highs"],
                lows=window["lows"],
                benchmark_prices=window["benchmark_prices"],
                enable_numba_kernels=False,
            )
        )
        self.assertIn("momentum_5", snapshot)
        self.assertIn("volatility_20", snapshot)
        self.assertNotIn("stock_net_inflow_5_20", snapshot)
        self.assertNotIn("event_surprise_score_5", snapshot)
        self.assertNotIn("cash_to_assets_ttm_proxy_20", snapshot)

    def test_comprehensive_context_emits_every_non_minute_registry_name(self) -> None:
        window = _price_window()
        fundamentals = {
            name: 0.1
            for name, definition in FACTOR_DEFINITIONS.items()
            if "financial_" in " ".join(definition.dependencies)
        }
        snapshot = compute_registry_factor_snapshot(
            RegistryFactorInputs(
                **window,
                fundamental_factors=fundamentals,
                enable_numba_kernels=False,
            )
        )
        minute_names = {
            name
            for name, definition in FACTOR_DEFINITIONS.items()
            if any(
                dependency.startswith(("bars_", "vwap_"))
                or dependency in {
                    "turnover_30m",
                    "opens_30m",
                    "closes_30m",
                }
                for dependency in definition.dependencies
            )
        }
        self.assertEqual(len(minute_names), 6)
        self.assertEqual(set(snapshot), set(FACTOR_DEFINITIONS) - minute_names)

    def test_event_factors_require_their_specific_source_series(self) -> None:
        window = _price_window()
        top_list_event_fields = {
            "event_direction_series",
            "event_intensity_series",
            "event_confidence_series",
            "event_impact_series",
            "event_symbol_hits",
        }
        top_list_only = {
            key: value
            for key, value in window.items()
            if key not in {
                "event_age_minutes",
                "event_sector_hits",
                "regulatory_event_series",
                "shareholder_event_series",
                "policy_event_series",
                "hotword_series",
            }
        }
        snapshot = compute_registry_factor_snapshot(
            RegistryFactorInputs(
                **top_list_only,
                enable_numba_kernels=False,
            )
        )
        self.assertTrue(top_list_event_fields.issubset(
            top_list_only
        ))
        self.assertIn("event_surprise_score_5", snapshot)
        self.assertIn("event_consistency_20d", snapshot)
        self.assertNotIn("event_recency_decay_30m", snapshot)
        self.assertNotIn("event_sector_diffusion_5d", snapshot)
        self.assertNotIn("regulatory_risk_penalty_20d", snapshot)
        self.assertNotIn("shareholder_alignment_60d", snapshot)
        self.assertNotIn("policy_alignment_score_10d", snapshot)
        self.assertNotIn("crowding_event_conflict_5d", snapshot)

    def test_event_surprise_requires_all_three_top_list_components(self) -> None:
        window = _price_window()
        for omitted in (
            "event_direction_series",
            "event_intensity_series",
            "event_confidence_series",
        ):
            values = {
                key: value
                for key, value in window.items()
                if key != omitted
            }
            snapshot = compute_registry_factor_snapshot(
                RegistryFactorInputs(
                    **values,
                    enable_numba_kernels=False,
                )
            )
            self.assertNotIn(
                "event_surprise_score_5",
                snapshot,
                omitted,
            )

    def test_panel_and_array_execution_align(self) -> None:
        dates = [str(20200101 + index) for index in range(80)]
        rows = []
        for code, offset in (("000001.SZ", 0.0), ("000002.SZ", 1.0)):
            for index, day in enumerate(dates):
                price = 100.0 + index + offset
                rows.append(
                    {
                        "ts_code": code,
                        "trade_date": day,
                        "open": price - 0.2,
                        "high": price + 0.5,
                        "low": price - 0.5,
                        "close": price,
                        "amount": 10_000_000.0 + index,
                        "benchmark_close": 3000.0 + index,
                    }
                )
        prepared = pd.DataFrame(rows).sort_values(
            ["ts_code", "trade_date"], kind="mergesort"
        ).reset_index(drop=True)
        panel, names = compute_registry_panel(
            prepared, enable_numba_kernels=False
        )
        with tempfile.TemporaryDirectory() as temporary:
            array_names = compute_registry_arrays(
                prepared,
                temporary,
                enable_numba_kernels=False,
                checkpoint_identity="test-contract",
            )
            self.assertEqual(set(names), set(array_names))
            for name in names:
                import hashlib

                path = (
                    Path(temporary)
                    / f"{hashlib.sha256(name.encode()).hexdigest()}.npy"
                )
                values = np.load(path)
                np.testing.assert_allclose(
                    values,
                    panel[name].to_numpy(dtype=np.float32),
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )

    def test_parallel_entry_falls_back_inline_without_posix_fork(self) -> None:
        dates = [str(20200101 + index) for index in range(80)]
        prepared = pd.DataFrame(
            [
                {
                    "ts_code": code,
                    "trade_date": day,
                    "open": 10.0 + index,
                    "high": 10.5 + index,
                    "low": 9.5 + index,
                    "close": 10.0 + index,
                    "amount": 1_000_000.0,
                    "benchmark_close": 3000.0 + index,
                }
                for code in ("000001.SZ", "000002.SZ")
                for index, day in enumerate(dates)
            ]
        ).sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(
            drop=True
        )
        with tempfile.TemporaryDirectory() as temporary:
            names = compute_registry_arrays_parallel(
                prepared,
                temporary,
                workers=2,
                checkpoint_identity="portable-entry",
            )
            checkpoint = json.loads(
                (
                    Path(temporary) / "r441-registry.checkpoint.json"
                ).read_text(encoding="utf-8")
            )
        self.assertIn("momentum_5", names)
        self.assertTrue(checkpoint["complete"])
        self.assertIn(checkpoint["execution_mode"], {"inline", "process-fork"})

    def test_registry_availability_names_missing_event_fields(self) -> None:
        prepared = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": f"202001{index + 1:02d}",
                    "open": 10.0 + index,
                    "high": 10.5 + index,
                    "low": 9.5 + index,
                    "close": 10.0 + index,
                    "amount": 1_000_000.0,
                    "benchmark_close": 3000.0 + index,
                    "event_direction": 1.0,
                    "event_intensity": 0.5,
                    "event_confidence": 0.8,
                    "event_impact": 0.4,
                    "event_symbol_hits": 1.0,
                }
                for index in range(30)
            ]
        )
        report = registry_availability_report(
            prepared,
            computed_names={
                "event_surprise_score_5",
                "event_consistency_20d",
                *MINUTE_FACTOR_NAMES,
            },
            start="20200101",
            end="20260630",
            minute_source_root="E:/minute_data",
        )
        self.assertEqual(report["status_counts"]["PARTIAL"], 6)
        recency = report["rows"]["event_recency_decay_30m"]
        self.assertEqual(recency["status"], "UNAVAILABLE")
        self.assertEqual(
            recency["missing_inputs"],
            ["event_age_minutes"],
        )
        self.assertIn(
            "event_age_minutes",
            recency["reason"],
        )
        self.assertNotIn(
            "event_symbol_hits",
            report["rows"]["event_sector_diffusion_5d"][
                "missing_inputs"
            ],
        )
        self.assertEqual(
            report["rows"]["event_sector_diffusion_5d"]["missing_inputs"],
            ["event_sector_hits"],
        )


class ReferenceContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = duckdb.connect()
        self.addCleanup(self.connection.close)
        dates = [20200101 + index for index in range(80)]
        self.panel = pd.DataFrame(
            [
                {
                    "ts_code": code,
                    "trade_date": str(day),
                    "open": 10.0 + index * 0.03 - 0.01,
                    "high": 10.0 + index * 0.03 + 0.02,
                    "low": 10.0 + index * 0.03 - 0.02,
                    "close": 10.0 + index * 0.03,
                    "amount": 1_000_000.0 + index,
                    "benchmark_close": 3000.0 + index,
                }
                for code in ("000001.SZ", "000002.SZ")
                for index, day in enumerate(dates)
            ]
        ).sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(
            drop=True
        )
        connection = self.connection
        connection.execute(
            "CREATE TABLE index_daily("
            "ts_code VARCHAR,trade_date VARCHAR,close DOUBLE,open DOUBLE,"
            "pre_close DOUBLE)"
        )
        connection.execute(
            "CREATE TABLE moneyflow("
            "ts_code VARCHAR,trade_date VARCHAR,net_mf_amount DOUBLE)"
        )
        for index, day in enumerate(dates):
            connection.execute(
                "INSERT INTO index_daily VALUES (?,?,?,?,?)",
                ["000300.SH", str(day), 3000.0 + index, 2999.5 + index, 2999.0 + index],
            )
            for code in ("000001.SZ", "000002.SZ"):
                connection.execute(
                    "INSERT INTO moneyflow VALUES (?,?,?)",
                    [code, str(day), 10.0 + index % 5],
                )
        connection.execute(
            "CREATE TABLE industry_membership("
            "ts_code VARCHAR,start_date VARCHAR,end_date VARCHAR,"
            "industry_code VARCHAR)"
        )
        connection.execute(
            "INSERT INTO industry_membership VALUES "
            "('000001.SZ','20200101','20201231','801010.SI'),"
            "('000002.SZ','20200101','20201231','801020.SI')"
        )
        connection.execute(
            "CREATE TABLE sw_daily("
            "ts_code VARCHAR,trade_date VARCHAR,close DOUBLE)"
        )
        for index, day in enumerate(dates):
            connection.execute(
                "INSERT INTO sw_daily VALUES (?,?,?),(?,?,?)",
                ["801010.SI", str(day), 1000.0 + index, "801020.SI", str(day), 2000.0 + index],
            )
        connection.execute(
            'CREATE TABLE limit_list_d('
            'trade_date VARCHAR,"limit" VARCHAR,open_times DOUBLE)'
        )
        for index, day in enumerate(dates):
            connection.execute(
                "INSERT INTO limit_list_d VALUES (?,?,?)",
                [str(day), "U" if index % 2 else "D", float(index % 3 == 0)],
            )
        connection.execute(
            "CREATE TABLE top_list("
            "ts_code VARCHAR,trade_date VARCHAR,net_amount DOUBLE,"
            "amount DOUBLE,amount_rate DOUBLE)"
        )
        for day in dates[::5]:
            connection.execute(
                "INSERT INTO top_list VALUES (?,?,?,?,?)",
                ["000001.SZ", str(day), 100.0, 1000.0, 10.0],
            )
        connection.execute(
            "CREATE TABLE trade_cal(cal_date VARCHAR,is_open INTEGER)"
        )
        for day in dates:
            connection.execute(
                "INSERT INTO trade_cal VALUES (?,1)", [str(day)]
            )

    def test_context_joins_available_market_flow_sector_and_event_sources(self) -> None:
        context = load_reference_factor_context(
            self.connection,
            self.panel,
            start="20200101",
            end="20200320",
        )
        pd.testing.assert_frame_equal(
            context.panel[["ts_code", "trade_date"]],
            self.panel[["ts_code", "trade_date"]],
        )
        self.assertEqual(context.metadata["moneyflow"], "AVAILABLE")
        self.assertEqual(
            context.metadata["sector_daily"],
            "AVAILABLE_DERIVED_PIT_EQUAL_WEIGHT",
        )
        self.assertEqual(context.metadata["top_list_events"], "AVAILABLE")
        generated, names = compute_registry_panel(
            context.panel,
            fundamental_by_symbol_date=context.fundamental_by_symbol_date,
            enable_numba_kernels=False,
        )
        self.assertIn("stock_net_inflow_5_20", names)
        self.assertIn("sector_rs_20", names)
        self.assertIn("event_surprise_score_5", names)
        self.assertNotIn("cash_to_assets_ttm_proxy_20", names)
        self.assertTrue(np.isfinite(generated.to_numpy()).any())

    def test_supplemental_financial_files_extend_empty_base_tables(self) -> None:
        periods = [
            ("2019-03-31", "2019-04-30"),
            ("2019-06-30", "2019-08-30"),
            ("2019-09-30", "2019-10-30"),
            ("2019-12-31", "2020-01-10"),
        ]
        income = []
        balance = []
        cashflow = []
        for index, (period, announced) in enumerate(periods, start=1):
            for symbol in ("000001.SZ", "000002.SZ"):
                income.append(
                    {
                        "ts_code": symbol,
                        "end_date": period.replace("-", ""),
                        "ann_date": announced.replace("-", ""),
                        "f_ann_date": announced.replace("-", ""),
                        "report_type": "1",
                        "revenue": 100.0 + index,
                        "operate_profit": 10.0 + index,
                        "n_income_attr_p": 8.0 + index,
                        "oper_cost": 60.0,
                        "ebitda": 20.0 + index,
                    }
                )
                balance.append(
                    {
                        "ts_code": symbol,
                        "end_date": period.replace("-", ""),
                        "ann_date": announced.replace("-", ""),
                        "f_ann_date": announced.replace("-", ""),
                        "report_type": "1",
                        "total_assets": 500.0 + index,
                        "total_liab": 200.0,
                        "money_cap": 80.0,
                        "accounts_receiv": 30.0,
                        "inventories": 40.0,
                        "fix_assets": 100.0,
                        "total_cur_assets": 180.0,
                        "total_cur_liab": 90.0,
                        "undistr_porfit": 50.0,
                    }
                )
                cashflow.append(
                    {
                        "ts_code": symbol,
                        "end_date": period.replace("-", ""),
                        "ann_date": announced.replace("-", ""),
                        "f_ann_date": announced.replace("-", ""),
                        "report_type": "1",
                        "n_cashflow_act": 25.0 + index,
                        "n_cash_flows_fnc_act": -5.0,
                    }
                )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pd.DataFrame(income).to_parquet(
                directory / "income_20191231.parquet", index=False
            )
            pd.DataFrame(balance).to_parquet(
                directory / "balancesheet_20191231.parquet", index=False
            )
            pd.DataFrame(cashflow).to_parquet(
                directory / "cashflow_20191231.parquet", index=False
            )
            context = load_reference_factor_context(
                self.connection,
                self.panel,
                start="20200101",
                end="20200120",
                supplemental_financial_root=directory,
            )
        self.assertEqual(
            context.metadata["fundamentals"]["status"], "AVAILABLE"
        )
        self.assertEqual(
            context.metadata["fundamentals"]["latest_reported_period"],
            "20191231",
        )
        self.assertGreater(
            len(context.fundamental_by_symbol_date["000001.SZ"]), 0
        )


if __name__ == "__main__":
    unittest.main()
