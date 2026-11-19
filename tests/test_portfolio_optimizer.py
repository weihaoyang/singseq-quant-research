import unittest

import numpy as np
import pandas as pd

from quant_sdk.portfolio_optimizer import (
    evaluate_portfolio,
    generate_portfolio_report_html,
    optimize_black_litterman,
    optimize_hrp,
    optimize_max_sharpe,
    optimize_min_volatility,
)


class TestPortfolioOptimizer(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        # 3 assets with synthetic price trends
        p1 = 100 * np.cumprod(1 + np.random.normal(0.001, 0.015, size=len(dates)))
        p2 = 50 * np.cumprod(1 + np.random.normal(0.0005, 0.01, size=len(dates)))
        p3 = 20 * np.cumprod(1 + np.random.normal(0.0015, 0.02, size=len(dates)))
        self.prices = pd.DataFrame({"600519.SH": p1, "000858.SZ": p2, "601318.SH": p3}, index=dates)

    def test_max_sharpe(self):
        weights = optimize_max_sharpe(self.prices)
        self.assertIsInstance(weights, dict)
        self.assertEqual(len(weights), 3)
        total_w = sum(weights.values())
        self.assertAlmostEqual(total_w, 1.0, places=2)
        for _sym, w in weights.items():
            self.assertGreaterEqual(w, 0.0)

    def test_min_volatility(self):
        weights = optimize_min_volatility(self.prices)
        self.assertIsInstance(weights, dict)
        total_w = sum(weights.values())
        self.assertAlmostEqual(total_w, 1.0, places=2)

    def test_hrp(self):
        weights = optimize_hrp(self.prices)
        self.assertIsInstance(weights, dict)
        total_w = sum(weights.values())
        self.assertAlmostEqual(total_w, 1.0, places=2)

    def test_black_litterman(self):
        views = {"600519.SH": 0.20, "000858.SZ": 0.05}
        weights = optimize_black_litterman(self.prices, investor_views=views)
        self.assertIsInstance(weights, dict)
        total_w = sum(weights.values())
        self.assertAlmostEqual(total_w, 1.0, places=2)

    def test_evaluate_and_html(self):
        weights = {"600519.SH": 0.5, "000858.SZ": 0.3, "601318.SH": 0.2}
        perf = evaluate_portfolio(weights, self.prices)
        self.assertIn("expected_return", perf)
        self.assertIn("volatility", perf)
        self.assertIn("sharpe_ratio", perf)
        html = generate_portfolio_report_html(weights, perf)
        self.assertIn("Optimal Weights Allocation", html)
        self.assertIn("600519.SH", html)


if __name__ == "__main__":
    unittest.main()
