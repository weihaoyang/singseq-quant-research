"""Unit tests for Nautilus Trader integration (quant_sdk.nautilus_adapter)."""
import unittest

import pandas as pd

from quant_sdk.nautilus_adapter import (
    NAUTILUS_AVAILABLE,
    AshareNautilusEngine,
    convert_df_to_nautilus_bars,
    create_ashare_instrument,
)


@unittest.skipUnless(NAUTILUS_AVAILABLE, "nautilus_trader is not available")
class NautilusAdapterTests(unittest.TestCase):
    def setUp(self):
        self.mock_data = pd.DataFrame([
            {"ts_code": "000001.SZ", "trade_date": "20240102", "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2, "vol": 50000},
            {"ts_code": "000001.SZ", "trade_date": "20240103", "open": 10.2, "high": 10.8, "low": 10.1, "close": 10.6, "vol": 60000},
            {"ts_code": "000001.SZ", "trade_date": "20240104", "open": 10.6, "high": 10.7, "low": 10.3, "close": 10.4, "vol": 45000},
        ])

    def test_instrument_creation(self):
        inst_sz = create_ashare_instrument("000001.SZ")
        self.assertEqual(inst_sz.id.value, "000001.XSHE")
        self.assertEqual(int(inst_sz.lot_size), 100)
        self.assertEqual(inst_sz.quote_currency.code, "CNY")

        inst_sh = create_ashare_instrument("600519.SH")
        self.assertEqual(inst_sh.id.value, "600519.XSHG")
        self.assertEqual(int(inst_sh.lot_size), 100)

    def test_bar_conversion(self):
        inst = create_ashare_instrument("000001.SZ")
        bars = convert_df_to_nautilus_bars(self.mock_data, inst)
        self.assertEqual(len(bars), 3)
        self.assertEqual(float(bars[0].open), 10.0)
        self.assertEqual(float(bars[0].close), 10.2)
        self.assertEqual(int(bars[0].volume), 50000)

    def test_ashare_nautilus_engine_setup(self):
        engine = AshareNautilusEngine(initial_cash=500_000.0, trader_id="TEST-NAUTILUS")
        self.assertEqual(engine.initial_cash, 500_000.0)

        # Register and add data
        count = engine.add_data("000001.SZ", self.mock_data)
        self.assertEqual(count, 3)

        # Run engine (without strategy, empty loop)
        res = engine.run()
        self.assertIn("trader_id", res)
        self.assertEqual(res["engine"], "NautilusTrader")


if __name__ == "__main__":
    unittest.main()
