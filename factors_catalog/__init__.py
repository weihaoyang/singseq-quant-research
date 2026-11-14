"""Formula-based factor families for local quantitative research."""

from .momentum import PriceMomentum20Factor, MAGap20Factor
from .intraday import IntradayVWAPBiasFactor, ParkinsonVolatility20Factor
from .factor_zoo import compute_factor_zoo
from .ohlc_volatility import compute_ohlc_volatility
from .alpha158 import compute_alpha158
from .alpha101_operators import (
    compute_alpha101_selected,
    cross_sectional_rank,
    time_series_delta,
    time_series_correlation,
    linear_decay,
    ts_max,
    ts_min,
    ts_rank,
)
from .moneyflow_signals import compute_moneyflow_signals
from .daily_basic_signals import compute_daily_basic_signals
from .industry_momentum import compute_industry_momentum

ALL_FACTORS = {
    "price_momentum_20": PriceMomentum20Factor(),
    "ma_gap_20": MAGap20Factor(),
    "intraday_vwap_bias": IntradayVWAPBiasFactor(),
    "parkinson_volatility_20": ParkinsonVolatility20Factor(),
}

__all__ = [
    "PriceMomentum20Factor",
    "MAGap20Factor",
    "IntradayVWAPBiasFactor",
    "ParkinsonVolatility20Factor",
    "ALL_FACTORS",
    "compute_factor_zoo",
    "compute_ohlc_volatility",
    "compute_alpha158",
    "compute_alpha101_selected",
    "compute_moneyflow_signals",
    "compute_daily_basic_signals",
    "compute_industry_momentum",
    "cross_sectional_rank",
    "time_series_delta",
    "time_series_correlation",
    "linear_decay",
    "ts_max",
    "ts_min",
    "ts_rank",
]
