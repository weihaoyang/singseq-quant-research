"""
factors_catalog/momentum.py
--------------------------------------------------
动量与趋势家族研产同构因子
严格对齐 quant-clean/quant_runtime/domain/factors/r441_daily_formulas.py
"""

from __future__ import annotations
from typing import Any, Sequence
import polars as pl
import numpy as np

from factor_base import BaseFactor, FactorMetadata


class PriceMomentum20Factor(BaseFactor):
    """20日价格动量因子 (Twenty-day raw price momentum)

    生产对应: quant-clean/r441_daily_formulas.py: price_momentum_20
    公式: (close[-1] - close[-21]) / close[-21]
    """

    def __init__(self):
        meta = FactorMetadata(
            name="price_momentum_20",
            family="momentum",
            higher_is_better=True,
            dependencies=("prices_1d",),
            description="Twenty-day raw price momentum.",
            window_size=21
        )
        super().__init__(meta)

    def compute_batch(self, lf: pl.LazyFrame) -> pl.DataFrame:
        """Polars 批量向量化计算"""
        # 要求 lf 包含 ts_code, trade_date, close, 且按 ts_code, trade_date 升序排列
        expr = (
            (pl.col("close") - pl.col("close").shift(20).over("ts_code"))
            / pl.col("close").shift(20).over("ts_code")
        ).fill_null(0.0).alias(self.name)

        return (
            lf.with_columns(expr)
            .select(["ts_code", "trade_date", self.name])
            .collect()
        )

    def compute_single(
        self,
        prices: Sequence[float],
        turnover: Sequence[float] | None = None,
        **kwargs: Any
    ) -> float:
        """生产端单资产滑动窗口实时切片计算 (与 quant-clean r441_daily_formulas.py 源码 100% 完全一致)"""
        if len(prices) < 21:
            return 0.0
        base = prices[-21]
        if base == 0.0:
            return 0.0
        return float((prices[-1] - base) / base)


class MAGap20Factor(BaseFactor):
    """20日均线乖离率因子 (Distance between close and 20-day moving average)

    生产对应: quant-clean/r441_daily_formulas.py: ma_gap_20
    公式: (close[-1] - mean(close[-20:])) / mean(close[-20:])
    """

    def __init__(self):
        meta = FactorMetadata(
            name="ma_gap_20",
            family="trend",
            higher_is_better=True,
            dependencies=("prices_1d",),
            description="Distance between close and 20-day moving average.",
            window_size=20
        )
        super().__init__(meta)

    def compute_batch(self, lf: pl.LazyFrame) -> pl.DataFrame:
        """Polars 批量向量化计算"""
        ma20 = pl.col("close").rolling_mean(window_size=20).over("ts_code")
        expr = ((pl.col("close") - ma20) / ma20).fill_null(0.0).alias(self.name)

        return (
            lf.with_columns(expr)
            .select(["ts_code", "trade_date", self.name])
            .collect()
        )

    def compute_single(
        self,
        prices: Sequence[float],
        turnover: Sequence[float] | None = None,
        **kwargs: Any
    ) -> float:
        """生产端单资产滑动窗口计算 (与 quant-clean r441_daily_formulas.py 完全一致)"""
        if len(prices) < 20:
            return 0.0
        window = prices[-20:]
        avg = sum(window) / 20.0
        if avg == 0.0:
            return 0.0
        return float((prices[-1] - avg) / avg)
