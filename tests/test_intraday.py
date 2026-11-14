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


if __name__ == "__main__":
    unittest.main()
