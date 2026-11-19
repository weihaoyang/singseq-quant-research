"""Intraday engine: next-bar fill, T+1 lock, limit-down no-sell, real costs."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from server.engine.friction import FrictionConfig
from server.engine.intraday import run_intraday


def _bars(days_prices):
    rows = []
    for day, prices in days_prices:
        for i, p in enumerate(prices):
            rows.append({
                "ts_code": "000001.SZ", "trade_date": day,
                "trade_time": f"{day}093{i}:00",
                "open": float(p), "high": float(p) * 1.001,
                "low": float(p) * 0.999, "close": float(p),
                "vol": 1e6, "amount": 1e8,
                "up_limit": round(float(p) * 1.1, 2),
                "down_limit": round(float(p) * 0.9, 2),
            })
    return pd.DataFrame(rows)


class IntradayEngineTests(unittest.TestCase):
    def test_t1_lock_then_round_trip_next_day(self):
        bars = _bars([("20230103", [10, 10.1, 10.2, 10.3, 10.4]),
                      ("20230104", [10.5, 10.4, 10.3, 10.2, 10.1])])
        factor = np.array([1, -1, 0, 0, 0, 1, -1, 0, 0, 0], dtype=float)
        res = run_intraday(bars, factor, base_shares=0)
        self.assertGreaterEqual(res.blocked["t1_locked"], 1)   # same-day sell blocked
        self.assertGreaterEqual(res.round_trips, 1)            # next-day sell allowed
        self.assertGreater(res.costs["commission"], 0.0)
        self.assertIsNotNone(res.metrics["sharpe"])

    def test_limit_down_blocks_sell(self):
        rows = [
            ("20230103", [10.0]),  # flat at down limit
        ]
        bars = pd.DataFrame([
            {"ts_code": "000001.SZ", "trade_date": "20230103",
             "trade_time": "202301030931:00", "open": 10.0, "high": 10.0,
             "low": 10.0, "close": 10.0, "vol": 1e6, "amount": 1e8,
             "up_limit": 11.0, "down_limit": 10.0},
            {"ts_code": "000001.SZ", "trade_date": "20230103",
             "trade_time": "202301030932:00", "open": 10.0, "high": 10.0,
             "low": 10.0, "close": 10.0, "vol": 1e6, "amount": 1e8,
             "up_limit": 11.0, "down_limit": 10.0},
            {"ts_code": "000001.SZ", "trade_date": "20230103",
             "trade_time": "202301030933:00", "open": 10.2, "high": 10.3,
             "low": 10.1, "close": 10.2, "vol": 1e6, "amount": 1e8,
             "up_limit": 11.0, "down_limit": 9.0},
        ])
        factor = np.array([-1, 0, 0], dtype=float)
        res = run_intraday(bars, factor, base_shares=10_000)
        self.assertGreaterEqual(res.blocked["limit_down_sell"], 1)

    def test_short_is_refused(self):
        bars = _bars([("20230103", [10, 10.1, 10.2])])
        with self.assertRaises(ValueError):
            run_intraday(bars, np.zeros(len(bars)), allow_short=True)

    def test_returns_use_previous_equity_and_observed_bar_frequency(self):
        bars = _bars([("20230103", [10.0, 11.0, 12.1, 13.31, 14.641])])
        res = run_intraday(
            bars, np.zeros(len(bars)), base_shares=10_000,
            config=FrictionConfig(initial_cash=1_000.0))
        self.assertAlmostEqual(res.returns[0], 0.0)
        self.assertAlmostEqual(
            res.returns[1], res.equity_curve[1] / res.equity_curve[0] - 1.0)
        self.assertEqual(res.periods_per_year, 252 * 5)
        self.assertEqual(res.metrics["annualized"],
                         round(float(np.prod(1.0 + res.returns) ** (252 * 5 / len(res.returns)) - 1.0), 6))

    def test_turnover_is_executed_notional_not_cost_amount(self):
        bars = _bars([("20230103", [10.0, 10.1]),
                      ("20230104", [10.2, 10.3])])
        factor = np.array([1.0, 0.0, -1.0, 0.0])
        res = run_intraday(
            bars, factor, order_shares=1_000_000,
            config=FrictionConfig(
                initial_cash=20_000_000.0, commission_rate=0.0003,
                impact_coeff=0.0, max_participation=1.0))
        self.assertGreater(res.turnover, 0.5)
        self.assertGreater(res.turnover,
                           sum(res.costs.values()) / FrictionConfig().initial_cash)
        self.assertEqual(
            res.turnover_definition,
            "realized traded notional / mean finite equity")


if __name__ == "__main__":
    unittest.main()
