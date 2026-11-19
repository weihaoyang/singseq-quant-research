"""Unit tests for Alphalens factor tear sheet (quant_sdk.alphalens_tear_sheet)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quant_sdk.alphalens_tear_sheet import (
    calc_factor_metrics,
    generate_factor_tear_sheet,
    prepare_factor_data,
)


class AlphalensTearSheetTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        assets = ["000001.SZ", "600519.SH", "000858.SZ", "000333.SZ", "601318.SH"]
        idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])

        # Create momentum factor: today's return correlated with forward return
        self.factor = pd.Series(np.random.randn(len(idx)), index=idx, name="alpha")

        # Create price panel
        self.prices = pd.DataFrame(
            np.cumprod(1 + np.random.randn(30, 5) * 0.02, axis=0) * 100,
            index=dates,
            columns=assets,
        )

    def test_prepare_factor_data(self):
        fd = prepare_factor_data(self.factor, self.prices, periods=(1, 5), quantiles=3, max_loss=0.4)
        self.assertIsInstance(fd, pd.DataFrame)
        self.assertIn("1D", fd.columns)
        self.assertIn("5D", fd.columns)
        self.assertIn("factor_quantile", fd.columns)

    def test_calc_factor_metrics(self):
        fd = prepare_factor_data(self.factor, self.prices, periods=(1, 5), quantiles=3, max_loss=0.4)
        metrics = calc_factor_metrics(fd)

        self.assertIn("mean_ic", metrics)
        self.assertIn("1D", metrics["mean_ic"])
        self.assertIn("ic_ir", metrics)
        self.assertIn("quantile_returns", metrics)
        self.assertIn("monotonicity_score", metrics)
        self.assertIn("mean_rank_autocorrelation", metrics)

    def test_generate_tear_sheet_and_html(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_html = Path(tmp_dir) / "factor_sheet.html"
            metrics = generate_factor_tear_sheet(
                self.factor, self.prices, periods=(1, 5), quantiles=3, max_loss=0.4, output_html=out_html
            )

            self.assertTrue(out_html.exists())
            content = out_html.read_text(encoding="utf-8")
            self.assertIn("Alphalens 因子质量透视表", content)
            self.assertIn("qChart", content)


if __name__ == "__main__":
    unittest.main()
