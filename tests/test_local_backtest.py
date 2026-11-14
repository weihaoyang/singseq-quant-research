"""Local canonical backtest runs the real engine on a researcher-owned panel."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from quant_sdk.local_backtest import run_local_backtest


def _panel_df() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    codes = [f"{600000 + i}.SH" for i in range(12)]
    dates = [f"2023{(d // 28) + 1:02d}{(d % 28) + 1:02d}" for d in range(40)]
    rows = []
    for code in codes:
        px = 10.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, len(dates)))
        for i, d in enumerate(dates):
            rows.append({
                "ts_code": code, "trade_date": d,
                "open": float(px[i]), "high": float(px[i] * 1.01),
                "low": float(px[i] * 0.99), "close": float(px[i]),
                "vol": 1_000_000.0, "amount": float(px[i] * 1_000_000.0),
                "adj_close": float(px[i]), "benchmark_ret": 0.0002,
            })
    return pd.DataFrame(rows)


class LocalBacktestTests(unittest.TestCase):
    def test_optional_risk_borrow_and_universe_columns_reach_engine_aligned(self):
        from quant_sdk.local_backtest import panel_from_frame
        data = _panel_df()
        data['industry_id'] = data.ts_code.map(lambda code: 'sector-' + code[-4])
        for column in ('circ_mv', 'pb', 'turnover_rate', 'cash_dividend', 'borrow_rate'):
            data[column] = np.arange(len(data), dtype=float) + 1
        data['pool_membership'] = 1
        data['shortable'] = 0
        data.loc[0, ['pool_membership', 'shortable']] = np.nan
        panel = panel_from_frame(data.sample(frac=1, random_state=19))
        self.assertEqual(panel.industry_id[0, 0], data.industry_id.iloc[0])
        for column in ('circ_mv', 'pb', 'turnover_rate', 'cash_dividend', 'borrow_rate'):
            self.assertEqual(getattr(panel, column)[0, 0], 1)
            self.assertEqual(getattr(panel, column)[1, 0], 2)
        self.assertFalse(panel.pool_membership[0, 0])
        self.assertFalse(panel.shortable[0, 0])
        default = panel_from_frame(_panel_df())
        self.assertIsNone(default.circ_mv)
        self.assertIsNone(default.shortable)

    def test_short_research_requires_real_borrow_inventory(self):
        with self.assertRaisesRegex(RuntimeError, 'shortable'):
            run_local_backtest("def compute_factor(df):\n    return df.close\n", _panel_df(), allow_short=True)

    def test_full_report_keeps_benchmark_groups_configuration_and_approximations(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'report.html'
            result = run_local_backtest('def compute_factor(df):\n    return df.close\n',
                                        _panel_df(), n_groups=3, costs=False, tearsheet_path=str(target))
            self.assertEqual(len(result['curves']['groups']), 3)
            self.assertEqual(len(result['curves']['benchmark']), result['dates'])
            self.assertGreater(result['curves']['benchmark'][-1], 1)
            html = target.read_text(encoding='utf-8')
            for label in ('第 3 组净值', '基准净值', '回测配置', '数据假设与近似',
                          'Friction breakdown', 'Blocked orders', 'costs_enabled'):
                self.assertIn(label, html)

    def test_custom_daily_table_keeps_optional_engine_columns(self):
        import duckdb
        import tempfile
        from pathlib import Path
        from quant_sdk.local_backtest import load_panel
        with tempfile.TemporaryDirectory() as directory:
            data = _panel_df()
            data['shortable'] = 0
            data['industry_id'] = 'sector-A'
            data['circ_mv'] = 123.0
            database = Path(directory) / 'input.duckdb'
            with duckdb.connect(str(database)) as con:
                con.register('input_frame', data)
                con.execute('CREATE TABLE daily_raw AS SELECT * FROM input_frame')
            loaded = load_panel(database, '20230101', '20230228')
            self.assertEqual(set(loaded.industry_id), {'sector-A'})
            self.assertTrue((loaded.circ_mv == 123).all())
            self.assertTrue((loaded.shortable == 0).all())

    def test_missing_adjusted_window_cannot_produce_a_flat_success_report(self):
        data = _panel_df()
        data["adj_close"] = np.nan
        with self.assertRaisesRegex(ValueError, "复权收盘"):
            run_local_backtest("def compute_factor(df):\n    return df['close']\n", data)

    def test_csv_keeps_finite_values_and_binds_provenance_to_content(self):
        import tempfile
        import hashlib
        import json
        from pathlib import Path
        from quant_sdk.research import run_research
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = _panel_df()
            data.attrs.update(package_version="test-version", package_hash="a" * 64)
            data.to_parquet(root / "data.parquet")
            result = run_research(factor_source="def compute_factor(df):\n    return -df.groupby('ts_code')['close'].pct_change(5)\n",
                data_path=str(root / "data.parquet"), start="20230101", end="20230212", out_csv=str(root / "factor.csv"))
            output = pd.read_csv(root / "factor.csv")
            self.assertTrue(np.isfinite(output.factor_value).all())
            self.assertEqual(len(output), result["factor_rows"])
            self.assertGreater(result["nonfinite_factor_rows"], 0)
            receipt = json.loads((root / "factor.csv.provenance.json").read_text())
            self.assertEqual(receipt["csv_sha256"], hashlib.sha256((root / "factor.csv").read_bytes()).hexdigest())
            self.assertEqual(receipt["package_version"], "test-version")

    def test_factor_series_is_aligned_by_index_even_when_returned_in_reverse(self):
        from quant_sdk.research import _compute_factor_series
        data = _panel_df()
        source = "def compute_factor(df):\n    return df['close'].iloc[::-1]\n"
        np.testing.assert_allclose(_compute_factor_series(source, data), data.close)
        with self.assertRaisesRegex(RuntimeError, "索引"):
            _compute_factor_series("def compute_factor(df):\n    return df['close'].set_axis(range(1,len(df)+1))\n", data)

    def test_ret5_is_stable_under_input_permutation_and_respects_stock_boundaries(self):
        from quant_sdk.factor_templates import find
        source = find("RET5")["code"]
        data = _panel_df()
        first = run_local_backtest(source, data, with_novelty=True)
        second = run_local_backtest(source, data.sample(frac=1, random_state=17), with_novelty=True)
        self.assertEqual(first["curves"], second["curves"])
        from quant_sdk.research import _compute_factor_series
        ordered = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        values = _compute_factor_series(source, ordered)
        expected = -ordered.groupby("ts_code")["close"].pct_change(5, fill_method=None)
        np.testing.assert_allclose(values, expected.to_numpy(), equal_nan=True)

    def test_runs_canonical_engine_locally(self):
        res = run_local_backtest(
            "def compute_factor(df):\n"
            "    return -df.groupby('ts_code')['close'].pct_change(5)\n",
            _panel_df(), with_novelty=True)
        self.assertEqual(res["status"], "LOCAL_UNVERIFIED")
        self.assertEqual(res["matching"], "next-day-open")
        self.assertFalse(res["allow_short"])
        self.assertIn("sharpe", res["metrics"])
        self.assertGreater(res["dates"], 0)
        curves = res["curves"]
        self.assertEqual(len(curves["dates"]), res["dates"])
        self.assertEqual(len(curves["ls"]), res["dates"])
        self.assertEqual(len(curves["drawdown"]), res["dates"])
        self.assertEqual(curves["dates"][0], sorted(curves["dates"])[0])

    def test_missing_benchmark_uses_labelled_zero_proxy(self):
        # The engine deliberately falls back to a zero-return benchmark when
        # benchmark_ret is absent, and records that approximation explicitly
        # instead of trusting an unlabelled proxy.
        res = run_local_backtest(
            "def compute_factor(df):\n    return df['close']\n",
            _panel_df().drop(columns=["benchmark_ret"]))
        self.assertEqual(res["status"], "LOCAL_UNVERIFIED")
        self.assertIn("benchmark_ret=0", " ".join(res.get("approximations", [])))

    def test_novelty_short_panel_is_explicitly_unavailable(self):
        # The backtest remains useful on a small research panel even when no
        # same-date Zoo comparison can meet the novelty minimum.
        with patch(
            "evaluation.novelty.novelty",
            side_effect=ValueError("novelty: no comparable zoo factor had enough names per date"),
        ):
            res = run_local_backtest(
                "def compute_factor(df):\n    return df['close']\n",
                _panel_df(), with_novelty=True)
        self.assertEqual(res["status"], "LOCAL_UNVERIFIED")
        self.assertEqual(res["novelty"]["status"], "UNAVAILABLE")
        self.assertIsNone(res.get("scoring"))

    def test_benchmark_array_is_copied_before_the_trailing_value_is_filled(self):
        # pandas 3 (Copy-on-Write) hands back read-only arrays from ``to_numpy``,
        # and the engine fills the trailing benchmark value with 0.0. Writing into
        # that buffer raised "assignment destination is read-only" for every
        # researcher on pandas 3 / numpy 2, so the engine must copy first — and it
        # must never mutate a caller-supplied array either.
        from quant_sdk.local_backtest import _benchmark_series

        dates = ["20230101", "20230102", "20230103"]
        work = pd.DataFrame({"trade_date": dates, "benchmark_ret": [0.01, 0.02, np.nan]})
        filled = _benchmark_series(work, dates, None)
        self.assertTrue(filled.flags.writeable)
        self.assertEqual(filled[-1], 0.0)

        frozen = np.array([0.01, 0.02, 0.03])
        frozen.setflags(write=False)
        out = _benchmark_series(work, dates, frozen)
        self.assertTrue(out.flags.writeable)
        np.testing.assert_allclose(out, [0.01, 0.02, 0.03])
        np.testing.assert_allclose(frozen, [0.01, 0.02, 0.03])


if __name__ == "__main__":
    unittest.main()
