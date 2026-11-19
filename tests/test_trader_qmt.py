"""Unit test suite for Broker Live Execution Gateway (quant_sdk.trader)."""
import unittest

from quant_sdk.trader import (
    MockQmtTrader,
    QmtTrader,
    TraderRiskController,
)


class TraderGatewayTests(unittest.TestCase):
    def test_risk_controller_limits(self):
        risk = TraderRiskController(
            max_order_value=100000.0,
            max_orders_per_second=2.0,
            blacklist={"000002.SZ"},
            allow_st=False,
        )

        # 1. Normal order passes
        ok, reason = risk.validate_order(
            "000001.SZ", "BUY", 1000, 10.0, available_cash=20000.0, available_shares=0
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "")

        # 2. Lot size failure (not multiple of 100 on BUY)
        ok, reason = risk.validate_order(
            "000001.SZ", "BUY", 150, 10.0, available_cash=20000.0, available_shares=0
        )
        self.assertFalse(ok)
        self.assertIn("100 shares", reason)

        # 3. Blacklist rejection
        ok, reason = risk.validate_order(
            "000002.SZ", "BUY", 100, 10.0, available_cash=20000.0, available_shares=0
        )
        self.assertFalse(ok)
        self.assertIn("blacklist", reason)

        # 4. ST stock rejection
        ok, reason = risk.validate_order(
            "*ST0001.SZ", "BUY", 100, 10.0, available_cash=20000.0, available_shares=0
        )
        self.assertFalse(ok)
        self.assertIn("ST stocks", reason)

        # 5. Single order notional cap
        ok, reason = risk.validate_order(
            "000001.SZ", "BUY", 20000, 10.0, available_cash=500000.0, available_shares=0  # 200k > 100k
        )
        self.assertFalse(ok)
        self.assertIn("exceeds cap", reason)

    def test_mock_qmt_trader_lifecycle(self):
        trader = MockQmtTrader("MOCK_TEST_ACCOUNT", initial_cash=200000.0)
        trader.connect()
        self.assertTrue(trader.is_connected())

        trader.set_mock_price("000001.SZ", 10.0)

        # Buy 2000 shares
        o_buy = trader.order("000001.SZ", 2000, price=10.0)
        self.assertEqual(o_buy.status, "FILLED")
        self.assertEqual(o_buy.filled_amount, 2000)

        # Check positions & asset
        positions = trader.query_positions()
        self.assertIn("000001.SZ", positions)
        self.assertEqual(positions["000001.SZ"].volume, 2000)

        asset = trader.query_asset()
        self.assertAlmostEqual(asset["total_asset"], 200000.0, delta=100.0)

        # Sell 1000 shares
        o_sell = trader.order("000001.SZ", -1000, price=10.0)
        self.assertEqual(o_sell.status, "FILLED")
        self.assertEqual(trader.query_positions()["000001.SZ"].volume, 1000)

        # Target percent
        trader.order_target_percent("000001.SZ", 0.0, price=10.0)
        self.assertNotIn("000001.SZ", trader.query_positions())

    def test_qmt_trader_import_diagnostics_when_not_installed(self):
        trader = QmtTrader(r"D:\MiniQmt\userdata_mini", "12345678")
        # In environment without xtquant, calling connect() must raise clear diagnostic error
        try:
            import xtquant  # noqa: F401
        except ImportError:
            with self.assertRaises(ImportError) as ctx:
                trader.connect()
            self.assertIn("xtquant", str(ctx.exception))
            self.assertIn("MockQmtTrader", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
