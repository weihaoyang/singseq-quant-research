"""Vectorized OHLCV Factor Zoo.

Ported from BDC2026 quantitative research lab (experiment_nested_factor_zoo.py).
Computes 70+ systematic formulaic alpha factors across multiple lookback windows:
- Multi-window returns (2, 3, 5, 10, 15, 20, 40, 60, 120 days)
- Intraday momentum, overnight gaps, and high-low daily range means
- Realized volatility, downside volatility, idiosyncratic volatility
- Residual momentum across multiple scales
- Extreme return distributions (maxret, minret, positive return fractions)
- Volume and amount moving average ratios (5, 10, 20, 40, 60 days)
- Dual-window volume and amount trends (5/20, 5/60, 20/60, 20/120)
- Price channel breakouts and proximity to rolling 20/60/120-day high/low
- Return skewness and path continuousness
- Amihud illiquidity ratio (20-day rolling)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _group_rolling(
    panel: pd.DataFrame,
    column: str,
    window: int,
    method: str,
) -> pd.Series:
    rolling = panel.groupby("stock_id", sort=False)[column].rolling(
        window,
        min_periods=window,
    )
    values = getattr(rolling, method)()
    return values.reset_index(level=0, drop=True).sort_index()


def compute_factor_zoo(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compute complete Factor Zoo from an OHLCV panel.

    Required columns in df:
        - stock_id (or code / ts_code)
        - trade_date (or date)
        - open, high, low, close, volume (or vol), amount
    Optional columns:
        - residual_ret_1 (if missing, market-residual return is computed on the fly)

    Returns:
        (panel_with_factors, list_of_factor_column_names)
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

    # Ensure correct sorting
    panel = panel.sort_values(["stock_id", "trade_date"]).reset_index(drop=True)
    groups = panel.groupby("stock_id", sort=False)
    factors: list[str] = []

    # 1. Base returns & ratios
    if "ret_1" not in panel.columns:
        panel["ret_1"] = groups["close"].pct_change()

    if "gap_1" not in panel.columns:
        prev_close = groups["close"].shift(1)
        panel["gap_1"] = (panel["open"] - prev_close) / prev_close.replace(0.0, np.nan)

    if "intraday_1" not in panel.columns:
        panel["intraday_1"] = (panel["close"] - panel["open"]) / panel["open"].replace(0.0, np.nan)
        factors.append("intraday_1")

    # If residual_ret_1 missing, demean cross-sectionally by trade_date
    if "residual_ret_1" not in panel.columns:
        mkt_ret = panel.groupby("trade_date")["ret_1"].transform("mean")
        panel["residual_ret_1"] = panel["ret_1"] - mkt_ret

    # 2. Multi-window cumulative returns
    for window in (2, 3, 5, 10, 15, 20, 40, 60, 120):
        name = f"return_{window}"
        panel[name] = groups["close"].pct_change(window)
        factors.append(name)

    # 3. High-low range and rolling mean intraday / gap
    prev_close = groups["close"].shift(1)
    daily_range = (panel["high"] - panel["low"]) / prev_close.replace(0.0, np.nan)

    for window in (3, 5, 10, 20, 40):
        for source, raw in (
            ("intraday", panel["intraday_1"]),
            ("gap", panel["gap_1"]),
            ("range_mean", daily_range),
        ):
            name = f"{source}_{window}"
            panel["_rolling_value"] = raw
            panel[name] = _group_rolling(
                panel,
                "_rolling_value",
                window,
                "mean",
            )
            factors.append(name)
    panel = panel.drop(columns="_rolling_value")

    # 4. Volatility, Idiosyncratic Volatility, Downside Volatility
    for window in (5, 10, 20, 40, 60):
        for source, raw in (("vol", panel["ret_1"]), ("idio_vol", panel["residual_ret_1"])):
            name = f"{source}_{window}"
            panel["_rolling_value"] = raw
            panel[name] = _group_rolling(
                panel,
                "_rolling_value",
                window,
                "std",
            )
            factors.append(name)

        name = f"downside_vol_{window}"
        downside = panel["ret_1"].clip(upper=0.0)
        panel["_rolling_value"] = downside
        panel[name] = _group_rolling(
            panel,
            "_rolling_value",
            window,
            "std",
        )
        factors.append(name)
    panel = panel.drop(columns="_rolling_value")

    # 5. Residual Momentum (Cross-sectional market residual cumulative drift)
    for window in (3, 5, 10, 20, 40, 60):
        name = f"residual_momentum_{window}"
        panel[name] = _group_rolling(
            panel,
            "residual_ret_1",
            window,
            "sum",
        )
        factors.append(name)

    # 6. Extreme returns and positive fraction
    for window in (5, 10, 20, 40):
        for prefix, reducer in (("maxret", "max"), ("minret", "min")):
            name = f"{prefix}_{window}"
            panel[name] = _group_rolling(
                panel,
                "ret_1",
                window,
                reducer,
            )
            factors.append(name)

        name = f"positive_fraction_{window}"
        positive = (panel["ret_1"] > 0.0).astype(float)
        panel["_rolling_value"] = positive
        panel[name] = _group_rolling(
            panel,
            "_rolling_value",
            window,
            "mean",
        )
        factors.append(name)
    panel = panel.drop(columns="_rolling_value")

    # 7. Volume and Amount ratios
    for window in (5, 10, 20, 40, 60):
        for source in ("volume", "amount"):
            name = f"{source}_ratio_{window}"
            rolling = _group_rolling(
                panel,
                source,
                window,
                "mean",
            )
            panel[name] = panel[source] / rolling.replace(0.0, np.nan)
            factors.append(name)

    # 8. Volume and Amount dual-window trends
    for short, long in ((5, 20), (5, 60), (20, 60), (20, 120)):
        for source in ("volume", "amount"):
            name = f"{source}_trend_{short}_{long}"
            short_mean = _group_rolling(
                panel,
                source,
                short,
                "mean",
            )
            long_mean = _group_rolling(
                panel,
                source,
                long,
                "mean",
            )
            panel[name] = short_mean / long_mean.replace(0.0, np.nan)
            factors.append(name)

    # 9. Proximity to rolling highs and lows (Donchian channel breakout)
    for window in (20, 60, 120):
        roll_high = _group_rolling(panel, "high", window, "max")
        roll_low = _group_rolling(panel, "low", window, "min")
        high_name, low_name = f"high_proximity_{window}", f"low_proximity_{window}"
        panel[high_name] = panel["close"] / roll_high.replace(0.0, np.nan)
        panel[low_name] = panel["close"] / roll_low.replace(0.0, np.nan)
        factors.extend([high_name, low_name])

    # 10. Skewness and Continuous Return
    for window in (10, 20, 40):
        skew_name = f"return_skew_{window}"
        panel[skew_name] = _group_rolling(
            panel,
            "ret_1",
            window,
            "skew",
        )
        panel["_rolling_value"] = panel["ret_1"].abs()
        abs_sum = _group_rolling(
            panel,
            "_rolling_value",
            window,
            "sum",
        )
        continuous_name = f"continuous_return_{window}"
        panel[continuous_name] = groups["close"].pct_change(window) / abs_sum.replace(0.0, np.nan)
        factors.extend([skew_name, continuous_name])
    panel = panel.drop(columns="_rolling_value")

    # 11. Amihud illiquidity proxy (20-day)
    panel["_rolling_value"] = panel["ret_1"].abs() / panel[
        "amount"
    ].where(panel["amount"] > 0)
    panel["amihud_20"] = _group_rolling(
        panel,
        "_rolling_value",
        20,
        "mean",
    )
    panel = panel.drop(columns="_rolling_value")
    factors.append("amihud_20")

    factors = list(dict.fromkeys(factors))
    return panel, factors
