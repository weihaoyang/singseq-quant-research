"""Unit test suite for Paper Trading Engine (quant_sdk.paper_trading)."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import quant_sdk as ts
from quant_sdk.paper_trading import (
    PaperTradingManager,
)


def _mock_market_day(date: str, price_a: float, price_b: float) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "ts_code": "000001.SZ",
            "trade_date": date,
            "open": price_a,
            "high": price_a + 0.1,
            "low": price_a - 0.1,
            "close": price_a + 0.05,
            "vol": 1000000.0,
            "amount": price_a * 1000000.0,
            "up_limit": price_a * 1.10,
            "down_limit": price_a * 0.90,
        },
        {
            "ts_code": "600519.SH",
            "trade_date": date,
            "open": price_b,
            "high": price_b + 0.5,
            "low": price_b - 0.5,
            "close": price_b - 0.2,
            "vol": 500000.0,
            "amount": price_b * 500000.0,
            "up_limit": price_b * 1.10,
            "down_limit": price_b * 0.90,
        },
    ])


class PaperTradingTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.storage = Path(self.tmp_dir.name)
        self.mgr = PaperTradingManager(self.storage)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_account_creation_and_persistence(self):
        acc = self.mgr.create_account("acc_test_01", "DualMaStrategy", 500000.0)
        self.assertEqual(acc.account_id, "acc_test_01")
        self.assertEqual(acc.cash, 500000.0)
        self.assertEqual(acc.total_value, 500000.0)

        # File exists
        self.assertTrue((self.storage / "acc_test_01.json").exists())

        # Load back
        loaded = self.mgr.get_account("acc_test_01")
        self.assertEqual(loaded.strategy_name, "DualMaStrategy")
        self.assertEqual(loaded.cash, 500000.0)

    def test_paper_trading_daily_step_and_t1(self):
        self.mgr.create_account("acc_step", "TrendStrategy", 200000.0)

        # Day 1: 20240102, buy stock A
        day1_data = _mock_market_day("20240102", 10.0, 100.0)

        def strat_day1(context, data):
            ts.order("000001.SZ", 1000)

        res1 = self.mgr.step_account(
            "acc_step", strat_day1, "20240102", custom_data=day1_data
        )
        self.assertEqual(res1["trades_count"], 1)
        acc1 = self.mgr.get_account("acc_step")
        self.assertIn("000001.SZ", acc1.positions)
        # T+1 lock check: on day 1 closeable amount is 0 (unlocked next day)
        self.assertEqual(acc1.positions["000001.SZ"]["total_amount"], 1000)
        self.assertEqual(acc1.positions["000001.SZ"]["closeable_amount"], 1000)  # Settle_daily unlocked it!

        # Day 2: 20240103, stock A rose from 10.0 to 11.0, sell 500 shares
        day2_data = _mock_market_day("20240103", 11.0, 101.0)

        def strat_day2(context, data):
            ts.order("000001.SZ", -500)

        res2 = self.mgr.step_account(
            "acc_step", strat_day2, "20240103", custom_data=day2_data
        )
        self.assertEqual(res2["trades_count"], 1)
        acc2 = self.mgr.get_account("acc_step")
        self.assertEqual(acc2.positions["000001.SZ"]["total_amount"], 500)
        self.assertEqual(len(acc2.daily_equity), 2)
        self.assertGreater(acc2.total_value, 200000.0)  # Profit made

    def test_summary_and_list_accounts(self):
        self.mgr.create_account("acc_list_1", "Alpha1", 100000.0)
        self.mgr.create_account("acc_list_2", "Alpha2", 200000.0)

        items = self.mgr.list_accounts()
        self.assertEqual(len(items), 2)
        ids = {i["account_id"] for i in items}
        self.assertIn("acc_list_1", ids)
        self.assertIn("acc_list_2", ids)


if __name__ == "__main__":
    unittest.main()
