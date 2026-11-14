"""WorldQuant Alpha 101 Operators and Formulaic Signals.

Ported from BDC2026 quantitative research lab (factor_operators.py & experiment_alpha101_*.py).
Provides both fundamental building blocks (operators) and concrete, tested Alpha 101 formulas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# =========================================================================
# 1. Fundamental Point-in-Time Factor Operators
# =========================================================================

def cross_sectional_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    """Return same-date percentile ranks without cross-date lookahead."""
    return values.groupby(dates, sort=False).rank(method="average", pct=True)


def time_series_delta(values: pd.Series, stocks: pd.Series, periods: int = 1) -> pd.Series:
    """Return each stock's current value minus its value `periods` rows ago."""
    if periods < 1:
        raise ValueError("periods must be positive")
    return values.groupby(stocks, sort=False).diff(periods)


def time_series_correlation(
    left: pd.Series, right: pd.Series, stocks: pd.Series, window: int, *, min_periods: int | None = None
) -> pd.Series:
    """Return per-stock rolling correlation aligned to original index."""
    if window < 2:
        raise ValueError("window must be at least two")
    minimum = window if min_periods is None else min_periods
    frame = pd.DataFrame({"left": left, "right": right, "stock": stocks})
    result = pd.Series(np.nan, index=left.index, dtype=float)
    for _, group in frame.groupby("stock", sort=False):
        result.loc[group.index] = group["left"].rolling(window, min_periods=minimum).corr(group["right"])
    return result


def linear_decay(values: pd.Series, stocks: pd.Series, window: int) -> pd.Series:
    """Return trailing linear-weighted moving average with weights 1..window."""
    if window < 1:
        raise ValueError("window must be positive")
    weights = np.arange(1.0, window + 1.0)
    weights /= weights.sum()
    return values.groupby(stocks, sort=False).transform(
        lambda group: group.rolling(window, min_periods=window).apply(lambda x: float(x @ weights), raw=True)
    )


def ts_max(values: pd.Series, stocks: pd.Series, window: int) -> pd.Series:
    """Rolling maximum per stock."""
    return values.groupby(stocks, sort=False).transform(lambda g: g.rolling(window, min_periods=window).max())


def ts_min(values: pd.Series, stocks: pd.Series, window: int) -> pd.Series:
    """Rolling minimum per stock."""
    return values.groupby(stocks, sort=False).transform(lambda g: g.rolling(window, min_periods=window).min())


def ts_rank(values: pd.Series, stocks: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank per stock over the trailing window."""
    return values.groupby(stocks, sort=False).transform(lambda g: g.rolling(window, min_periods=window).rank(pct=True))


# =========================================================================
# 2. Selected Alpha 101 Formula Implementations
# =========================================================================

def compute_alpha101_selected(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compute selected high-signal WorldQuant Alpha 101 factors.

    Required columns:
        - stock_id (or ts_code / code)
        - trade_date (or date)
        - open, high, low, close, volume (or vol), amount

    Returns:
        (panel_with_factors, list_of_factor_names)
    """
    panel = df.copy()

    # Normalize column names
    col_map = {}
    if "ts_code" in panel.columns and "stock_id" not in panel.columns:
        col_map["ts_code"] = "stock_id"
    if "code" in panel.columns and "stock_id" not in panel.columns:
        col_map["code"] = "stock_id"
    if "date" in panel.columns and "trade_date" not in panel.columns:
        col_map["date"] = "trade_date"
    if "vol" in panel.columns and "volume" not in panel.columns:
        col_map["vol"] = "volume"
    if col_map:
        panel = panel.rename(columns=col_map)

    panel = panel.sort_values(["stock_id", "trade_date"]).reset_index(drop=True)
    factors: list[str] = []

    stocks = panel["stock_id"]
    dates = panel["trade_date"]
    close = panel["close"]
    open_ = panel["open"]
    high = panel["high"]
    low = panel["low"]
    volume = panel["volume"]

    # Returns
    ret_1 = close.groupby(stocks, sort=False).pct_change()

    # 1. Alpha#101: Linear Decay Return (10d)
    panel["alpha_linear_decay_return_10"] = linear_decay(ret_1, stocks, 10)
    factors.append("alpha_linear_decay_return_10")

    # 2. Alpha#101: Open-Volume Rank Correlation
    open_rank = cross_sectional_rank(open_, dates)
    vol_rank = cross_sectional_rank(volume, dates)
    panel["alpha_open_volume_corr"] = -time_series_correlation(open_rank, vol_rank, stocks, 10)
    factors.append("alpha_open_volume_corr")

    # 3. Alpha#101: Price-Volume Delta
    delta_close = time_series_delta(close, stocks, 1)
    delta_vol = time_series_delta(volume, stocks, 1)
    panel["alpha_price_volume_delta"] = -delta_close * np.sign(delta_vol)
    factors.append("alpha_price_volume_delta")

    # 4. Alpha#6: -1 * correlation(open, volume, 10)
    panel["alpha_006"] = -time_series_correlation(open_, volume, stocks, 10)
    factors.append("alpha_006")

    # 5. Alpha#12: sign(delta(volume, 1)) * (-1 * delta(close, 1))
    panel["alpha_012"] = np.sign(delta_vol) * (-delta_close)
    factors.append("alpha_012")

    # 6. Alpha#44: -1 * correlation(high, rank(volume), 5)
    panel["alpha_044"] = -time_series_correlation(high, vol_rank, stocks, 5)
    factors.append("alpha_044")

    # 7. Alpha#53: -1 * delta((((close - low) - (high - close)) / (close - low)), 9)
    spread = (close - low).replace(0.0, np.nan)
    hl_ratio = ((close - low) - (high - close)) / spread
    panel["alpha_053"] = -time_series_delta(hl_ratio, stocks, 9)
    factors.append("alpha_053")

    # 8. Alpha#54: (-1 * ((low - close) * (open^5))) / ((low - high) * (close^5))
    lh_diff = (low - high).replace(0.0, np.nan)
    c5 = (close ** 5).replace(0.0, np.nan)
    panel["alpha_054"] = (-1.0 * (low - close) * (open_ ** 5)) / (lh_diff * c5)
    factors.append("alpha_054")

    # 9. Alpha#101: Volume-Weighted Return Momentum (vwap proxy)
    vwap = panel["amount"] / volume.replace(0.0, np.nan)
    panel["alpha_vwap_momentum_5"] = (vwap - close.groupby(stocks, sort=False).shift(5)) / close.groupby(stocks, sort=False).shift(5).replace(0.0, np.nan)
    factors.append("alpha_vwap_momentum_5")

    return panel, factors
