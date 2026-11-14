"""
factor_base.py
--------------------------------------------------
Vectorized and bounded-history contracts for factor implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence
import polars as pl
import numpy as np


@dataclass(frozen=True)
class FactorMetadata:
    """因子元数据，与 quant-clean 的 FactorDefinition 保持 100% 结构一致"""
    name: str
    family: str               # 'momentum', 'trend', 'volatility', 'flow', 'liquidity', 'intraday'
    higher_is_better: bool     # True: 因子值越大越好 (正向选股); False: 因子值越小越好 (做空或风险规避)
    dependencies: tuple[str, ...] # ('prices_1d', 'turnover_1d', 'prices_1m', etc.)
    description: str
    window_size: int = 20      # 滑动窗口大小 (如20日或20个Bar)


class BaseFactor(ABC):
    """研产一体化因子基类

    必须同时提供两套数学严格等价的算式实现：
    1. compute_batch: 基于 Polars 的批量多资产多时间向量化计算（用于 5 年海量历史回测与 IC 分析）
    2. compute_single: 基于纯 Python / Numba 切片的单资产流式计算（供 quant-clean 生产实盘微秒级打分）
    """

    def __init__(self, metadata: FactorMetadata):
        self.meta = metadata

    @property
    def name(self) -> str:
        return self.meta.name

    @property
    def family(self) -> str:
        return self.meta.family

    @property
    def higher_is_better(self) -> bool:
        return self.meta.higher_is_better

    @property
    def dependencies(self) -> tuple[str, ...]:
        return self.meta.dependencies

    @property
    def description(self) -> str:
        return self.meta.description

    @property
    def window_size(self) -> int:
        return self.meta.window_size

    @abstractmethod
    def compute_batch(self, lf: pl.LazyFrame) -> pl.DataFrame:
        """投研端：基于 Polars LazyFrame 的批量向量化计算

        参数:
            lf: 包含 ts_code, trade_date (或 trade_time), 以及行情字段的 Polars 惰性数据流
        返回:
            包含 [ts_code, trade_date, factor_value] 的 Polars DataFrame
        """
        pass

    @abstractmethod
    def compute_single(
        self,
        prices: Sequence[float],
        turnover: Sequence[float] | None = None,
        **kwargs: Any
    ) -> float:
        """生产端：单股票/单标的最新滑动窗口实时切片打分

        参数:
            prices: 该股票历史收盘价序列 (最新价格在 prices[-1])
            turnover: 该股票历史成交额序列 (最新在 turnover[-1])
            **kwargs: 其他如 opens, highs, lows, vwap, stock_flows 等
        返回:
            float: 该股票当前的标准化因子分值，供 quant-clean 实盘信号引擎使用
        """
        pass
