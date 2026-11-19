import unittest
from unittest.mock import patch

import pandas as pd

from quant_sdk.akshare_connector import (
    _normalize_symbol,
    get_macro_indicator,
    get_northbound_money_flow,
    get_realtime_quotes,
    get_stock_hist,
)


class TestAkshareConnector(unittest.TestCase):
    def test_normalize_symbol(self):
        self.assertEqual(_normalize_symbol("600519.SH"), "600519")
        self.assertEqual(_normalize_symbol("000858.SZ"), "000858")
        self.assertEqual(_normalize_symbol("sh600000"), "600000")
        self.assertEqual(_normalize_symbol("000001"), "000001")

    @patch("akshare.stock_zh_a_spot_em")
    def test_get_realtime_quotes(self, mock_spot):
        mock_df = pd.DataFrame({
            "代码": ["600519", "000858"],
            "名称": ["贵州茅台", "五粮液"],
            "最新价": [1700.0, 150.0],
            "涨跌幅": [1.5, -0.5],
            "今开": [1680.0, 151.0],
            "最高": [1710.0, 152.0],
            "最低": [1675.0, 149.0],
            "昨收": [1675.0, 150.75],
            "成交量": [10000, 20000],
            "成交额": [17000000, 3000000],
            "市盈率-动态": [28.5, 20.1],
            "市净率": [8.2, 5.1],
            "总市值": [2100000000000, 580000000000],
        })
        mock_spot.return_value = mock_df

        res = get_realtime_quotes(symbols=["600519.SH"])
        self.assertEqual(len(res), 1)
        self.assertEqual(res.iloc[0]["symbol"], "600519.SH")
        self.assertEqual(res.iloc[0]["price"], 1700.0)

    @patch("akshare.stock_zh_a_hist_em")
    def test_get_stock_hist(self, mock_hist):
        mock_df = pd.DataFrame({
            "日期": ["2024-01-02", "2024-01-03"],
            "开盘": [10.0, 10.5],
            "收盘": [10.5, 11.0],
            "最高": [10.8, 11.2],
            "最低": [9.9, 10.3],
            "成交量": [1000, 1500],
            "成交额": [10500, 16500],
            "换手率": [1.2, 1.8],
        })
        mock_hist.return_value = mock_df

        res = get_stock_hist("600519.SH", start_date="20240101", end_date="20240105")
        self.assertEqual(len(res), 2)
        self.assertIn("close", res.columns)
        self.assertIsInstance(res.index, pd.DatetimeIndex)

    @patch("akshare.stock_hsgt_hist_em")
    def test_get_northbound_money_flow(self, mock_hsgt):
        mock_df = pd.DataFrame({
            "日期": ["2024-01-02"],
            "沪股通净流入": [15.2],
            "深股通净流入": [10.5],
            "北向资金净流入": [25.7],
        })
        mock_hsgt.return_value = mock_df
        res = get_northbound_money_flow()
        self.assertEqual(len(res), 1)
        self.assertIn("total_net_inflow", res.columns)

    @patch("akshare.macro_china_cpi")
    def test_get_macro_indicator(self, mock_cpi):
        mock_cpi.return_value = pd.DataFrame({"月份": ["2024.01"], "全国": [100.2]})
        res = get_macro_indicator("cpi")
        self.assertFalse(res.empty)


if __name__ == "__main__":
    unittest.main()
