"""Unit tests for PyFolio & Empyrical risk tear sheet (quant_sdk.pyfolio_tear_sheet)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quant_sdk.pyfolio_tear_sheet import (
    calc_pyfolio_risk_metrics,
    generate_pyfolio_tear_sheet,
    get_drawdown_table,
)


class PyfolioTearSheetTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        self.returns = pd.Series(np.random.randn(100) * 0.01 + 0.001, index=dates)
        self.benchmark = pd.Series(np.random.randn(100) * 0.01, index=dates)

    def test_calc_pyfolio_risk_metrics(self):
        metrics = calc_pyfolio_risk_metrics(self.returns, self.benchmark)
        self.assertIn("annual_return", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("var_95", metrics)
        self.assertIn("cvar_95", metrics)
        self.assertIn("alpha", metrics)
        self.assertIn("beta", metrics)

    def test_get_drawdown_table(self):
        dd_list = get_drawdown_table(self.returns, top=3)
        self.assertIsInstance(dd_list, list)
        self.assertLessEqual(len(dd_list), 3)
        if dd_list:
            item = dd_list[0]
            self.assertIn("drawdown_pct", item)
            self.assertIn("peak_date", item)
            self.assertIn("valley_date", item)

    def test_generate_pyfolio_tear_sheet_and_html(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_html = Path(tmp_dir) / "risk_sheet.html"
            res = generate_pyfolio_tear_sheet(self.returns, self.benchmark, output_html=out_html)

            self.assertTrue(out_html.exists())
            html = out_html.read_text(encoding="utf-8")
            self.assertIn("PyFolio & Empyrical 投资组合全息风险透视表", html)
            self.assertIn("underwaterChart", html)
            self.assertIn("历史前 5 大深幅回撤期", html)


if __name__ == "__main__":
    unittest.main()
