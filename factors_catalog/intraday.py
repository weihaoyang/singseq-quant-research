"""
factors_catalog/intraday.py
--------------------------------------------------
日内微观高频研产同构因子 (Intraday & Microstructural Factors)
支持 1min/5min 及日内量价计算
"""

from __future__ import annotations
from typing import Any, Sequence
import math
import polars as pl
import numpy as np

from factor_base import BaseFactor, FactorMetadata


class IntradayVWAPBiasFactor(BaseFactor):
    """日内 VWAP 偏离度因子 (Intraday Price Bias from VWAP)

    反映价格在日内相对于主力成交均价的偏离程度：
    - 正偏离过大 -> 日内超买/短期动量衰竭 (反转信号)
    - 负偏离过大 -> 日内超卖/反弹机会

    公式: (price[-1] - intraday_vwap) / intraday_vwap
    其中 intraday_vwap = sum(amount) / sum(volume)
    """

    def __init__(self):
        meta = FactorMetadata(
            name="intraday_vwap_bias",
            family="intraday",
            higher_is_better=False,  # 均值回归反转偏好
            dependencies=("prices_1m", "vol_1m", "amount_1m"),
            description="Price deviation percentage relative to intraday cumulative VWAP.",
            window_size=30          # 默认 30 分钟窗口或日内累计
        )
        super().__init__(meta)

    def compute_batch(self, lf: pl.LazyFrame) -> pl.DataFrame:
        """Polars 批量日内计算"""
        # lf 需包含 ts_code, trade_time (或 trade_date+time), close, vol, amount
        # 按股票和交易日分组累加
        cum_amount = pl.col("amount").cum_sum().over(["ts_code", "trade_date"])
        cum_vol = pl.col("vol").cum_sum().over(["ts_code", "trade_date"])
        vwap = (cum_amount / cum_vol).fill_null(pl.col("close"))

        expr = ((pl.col("close") - vwap) / vwap).fill_null(0.0).alias(self.name)

        return (
            lf.with_columns(expr)
            .select(["ts_code", "trade_date", "trade_time", self.name])
            .collect()
        )

    def compute_single(
        self,
        prices: Sequence[float],
        turnover: Sequence[float] | None = None,
        volumes: Sequence[float] | None = None,
        **kwargs: Any
    ) -> float:
        """生产端单资产日内微秒级计算"""
        if not prices:
            return 0.0
        latest_price = prices[-1]

        if turnover is not None and volumes is not None and len(turnover) == len(volumes) and len(turnover) > 0:
            tot_amt = sum(turnover)
            tot_vol = sum(volumes)
            if tot_vol > 0:
                vwap = tot_amt / tot_vol
                return float((latest_price - vwap) / vwap)

        # 若无细分成交量，退化为收盘价均值比较
        avg_price = sum(prices) / len(prices)
        if avg_price == 0.0:
            return 0.0
        return float((latest_price - avg_price) / avg_price)


class ParkinsonVolatility20Factor(BaseFactor):
    """帕金森微观真实波动率因子 (Parkinson Realized Volatility)

    基于最高价与最低价的高频极差估计波动率，效率是普通收盘价标准差的 5 倍！
    对微观流动性挤压和异动极其敏感。

    公式: sqrt( 1 / (4 * ln(2) * N) * sum( (ln(high_i / low_i))^2 ) )
    """

    def __init__(self):
        meta = FactorMetadata(
            name="parkinson_volatility_20",
            family="volatility",
            higher_is_better=False,  # 波动率越小越平稳
            dependencies=("highs_1d", "lows_1d"),
            description="Parkinson extreme-value realized volatility over 20 bars.",
            window_size=20
        )
        super().__init__(meta)

    def compute_batch(self, lf: pl.LazyFrame) -> pl.DataFrame:
        """Polars 批量向量化计算"""
        factor_const = 1.0 / (4.0 * math.log(2.0))
        log_hl_sq = (pl.col("high") / pl.col("low")).log() ** 2
        roll_sum = log_hl_sq.rolling_sum(window_size=20).over("ts_code")
        expr = ((roll_sum * factor_const / 20.0).sqrt()).fill_null(0.0).alias(self.name)

        return (
            lf.with_columns(expr)
            .select(["ts_code", "trade_date", self.name])
            .collect()
        )

    def compute_single(
        self,
        prices: Sequence[float] | None = None,
        turnover: Sequence[float] | None = None,
        highs: Sequence[float] | None = None,
        lows: Sequence[float] | None = None,
        **kwargs: Any
    ) -> float:
        """生产端单资产微观切片计算"""
        if highs is None or lows is None or len(highs) < 20 or len(lows) < 20:
            return 0.0
        h_win = highs[-20:]
        l_win = lows[-20:]
        factor_const = 1.0 / (4.0 * math.log(2.0))

        sum_sq = 0.0
        for h, l in zip(h_win, l_win):
            if h > 0 and l > 0 and h >= l:
                ratio = math.log(h / l)
                sum_sq += ratio * ratio

        var = (sum_sq * factor_const) / 20.0
        return float(math.sqrt(var))
