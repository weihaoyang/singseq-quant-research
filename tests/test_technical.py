import unittest

import numpy as np
import pandas as pd

from quant_sdk.technical import (
    CROSS,
    MA,
    add_technical_indicators,
    atr,
    bollinger_bands,
    kdj,
    macd,
    rsi,
)


class TestTechnical(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        close = 100 * np.cumprod(1 + np.random.normal(0.001, 0.015, size=n))
        high = close * 1.02
        low = close * 0.98
        open_ = close * 0.99
        volume = np.random.uniform(1000, 5000, size=n)

        self.df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
        )

    def test_talib_functions(self):
        c = self.df["close"]
        h = self.df["high"]
        l_ = self.df["low"]

        # MACD
        m, s, hist = macd(c)
        self.assertEqual(len(m), len(c))

        # RSI
        r = rsi(c, timeperiod=14)
        self.assertEqual(len(r), len(c))

        # Bollinger Bands
        up, mid, low_b = bollinger_bands(c, timeperiod=20)
        self.assertEqual(len(up), len(c))
        # Check band ordering where not NaN
        valid = ~np.isnan(up)
        self.assertTrue(np.all(up[valid] >= low_b[valid]))

        # ATR
        a = atr(h, l_, c, timeperiod=14)
        self.assertEqual(len(a), len(c))

    def test_mytt_kdj_and_cross(self):
        c = self.df["close"]
        h = self.df["high"]
        l_ = self.df["low"]
        k, d, j = kdj(c, h, l_)
        self.assertEqual(len(k), len(c))

        ma5 = MA(c, 5)
        ma10 = MA(c, 10)
        golden_cross = CROSS(ma5, ma10)
        self.assertEqual(len(golden_cross), len(c))

    def test_add_technical_indicators(self):
        enriched = add_technical_indicators(self.df)
        expected_cols = [
            "macd", "macd_signal", "macd_hist",
            "rsi_6", "rsi_12", "rsi_24",
            "boll_upper", "boll_mid", "boll_lower",
            "atr_14", "kdj_k", "kdj_d", "kdj_j"
        ]
        for col in expected_cols:
            self.assertIn(col, enriched.columns)


if __name__ == "__main__":
    unittest.main()
