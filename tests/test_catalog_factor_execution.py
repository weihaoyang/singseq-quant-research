from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quant_sdk.catalog_factors import compute_named_factor, runnable_factor_names
from quant_sdk.research import run_named_factor


def _panel() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2024-01-02", periods=90)
    for code_index in range(12):
        base = 10.0 + code_index
        for day_index, day in enumerate(dates):
            close = base * (1.0 + 0.0015 * day_index + 0.002 * np.sin(day_index / 4 + code_index))
            rows.append({
                "ts_code": f"{code_index:06d}.SZ",
                "trade_date": day.strftime("%Y%m%d"),
                "list_date": "20100101",
                "open": close * 0.997,
                "high": close * 1.012,
                "low": close * 0.988,
                "close": close,
                "adj_close": close,
                "vol": 100000 + code_index * 100 + day_index,
                "amount": close * (100000 + code_index * 100 + day_index),
                "benchmark_ret": 0.0005 * np.cos(day_index / 7),
            })
    return pd.DataFrame(rows)


class CatalogFactorExecutionTests(unittest.TestCase):
    def test_real_implemented_families_are_selectable_and_causal(self) -> None:
        frame = _panel()
        names = runnable_factor_names(frame)
        self.assertIn("price_momentum_20", names)
        self.assertIn("roc_20", names)
        self.assertIn("alpha_006", names)
        self.assertIn("return_20", names)
        self.assertIn("parkinson_20", names)

        for name in ("price_momentum_20", "roc_20", "alpha_006", "return_20", "parkinson_20"):
            values = compute_named_factor(name, frame)
            self.assertEqual(len(values), len(frame))
            self.assertTrue(np.isfinite(values.to_numpy(dtype=float)).any(), name)

    def test_metadata_without_formula_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "尚无可运行实现"):
            compute_named_factor("cashflow_profit_match_ttm", _panel())

    def test_named_factor_runs_through_canonical_local_engine(self) -> None:
        frame = _panel()
        with tempfile.TemporaryDirectory(prefix="catalog-factor-") as tmp:
            panel_path = Path(tmp) / "panel.csv"
            output_path = Path(tmp) / "factor.csv"
            frame.to_csv(panel_path, index=False)
            result = run_named_factor(
                factor_name="price_momentum_20", data_path=str(panel_path),
                start="20240102", end="20240507", out_csv=str(output_path),
                tearsheet_path=str(Path(tmp) / "tearsheet.html"), groups=3)
            self.assertEqual(result["status"], "LOCAL_UNVERIFIED")
            self.assertEqual(result["factor_name"], "price_momentum_20")
            self.assertEqual(result["execution_boundary"], "catalog_implementation_local")
            self.assertEqual(result["factor_rows"], len(frame))
            self.assertGreater(result["metrics"]["n_ic_observations"], 0)
            self.assertGreater(len(set(result["curves"]["ls"])), 1)
            exported = pd.read_csv(output_path)
            self.assertEqual(list(exported.columns), ["ts_code", "trade_date", "factor_value"])
            self.assertGreater(int(exported["factor_value"].notna().sum()), 0)


if __name__ == "__main__":
    unittest.main()
