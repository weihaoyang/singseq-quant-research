"""Academic and Hedge-Fund Grade OHLC Volatility & Jump Estimators.

Ported from BDC2026 quantitative research lab (experiment_ohlc_volatility_estimators.py).
Computes point-in-time causal continuous volatility estimators:
1. Parkinson (1980): Extreme-value volatility using High/Low range
2. Garman-Klass (1980): Minimum-variance OHLC volatility estimator
3. Rogers-Satchell (1991): Drift-independent OHLC volatility
4. Yang-Zhang (2000): Minimum-variance estimator handling both overnight jumps and intraday drift
5. Residual Yang-Zhang: Market-demeaned idiosyncratic Yang-Zhang volatility
6. Barndorff-Nielsen & Shephard (2004) Realized Jump: Realized Variance minus Bipower Variation
7. Positive Idiosyncratic Semivolatility: Upside volatility proxy
8. Negative Idiosyncratic Semivolatility: Downside / crash risk proxy
9. OHLC Consensus: Equal-weighted rank composite volatility
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ohlc_volatility(df: pd.DataFrame, window: int = 20) -> tuple[pd.DataFrame, list[str]]:
    """Compute OHLC volatility and jump estimators.

    Required columns:
        - stock_id (or code / ts_code)
        - trade_date (or date)
        - open, high, low, close
    Optional columns:
        - residual_ret_1

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
    if col_map:
        panel = panel.rename(columns=col_map)

    panel = panel.sort_values(["stock_id", "trade_date"]).reset_index(drop=True)
    by_stock = panel.groupby("stock_id", sort=False)

    # Safe logs
    h_l = (panel["high"] / panel["low"].replace(0.0, np.nan)).clip(lower=1e-8)
    c_o = (panel["close"] / panel["open"].replace(0.0, np.nan)).clip(lower=1e-8)
    h_o = (panel["high"] / panel["open"].replace(0.0, np.nan)).clip(lower=1e-8)
    h_c = (panel["high"] / panel["close"].replace(0.0, np.nan)).clip(lower=1e-8)
    l_o = (panel["low"] / panel["open"].replace(0.0, np.nan)).clip(lower=1e-8)
    l_c = (panel["low"] / panel["close"].replace(0.0, np.nan)).clip(lower=1e-8)

    log_hl = np.log(h_l)
    log_co = np.log(c_o)
    log_ho = np.log(h_o)
    log_hc = np.log(h_c)
    log_lo = np.log(l_o)
    log_lc = np.log(l_c)

    # 1. Parkinson daily variance
    panel["parkinson_daily"] = log_hl**2 / (4.0 * np.log(2.0))

    # 2. Garman-Klass daily variance
    panel["garman_klass_daily"] = 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2

    # 3. Rogers-Satchell daily variance
    panel["rogers_satchell_daily"] = log_ho * log_hc + log_lo * log_lc

    # Rolling estimates
    factors: list[str] = []
    for source, target in (
        ("parkinson_daily", f"parkinson_{window}"),
        ("garman_klass_daily", f"garman_klass_{window}"),
        ("rogers_satchell_daily", f"rogers_satchell_{window}"),
    ):
        variance = by_stock[source].transform(lambda v: v.rolling(window, min_periods=window).mean())
        panel[target] = np.sqrt(variance.clip(lower=0.0))
        factors.append(target)

    # 4. Yang-Zhang
    prev_close = by_stock["close"].shift(1)
    overnight_ret = np.log((panel["open"] / prev_close.replace(0.0, np.nan)).clip(lower=1e-8))
    open_close_ret = log_co

    panel["_overnight_log"] = overnight_ret
    panel["_open_close_log"] = open_close_ret

    # Market demeaned residual overnight & open-close
    panel["_res_overnight"] = panel["_overnight_log"] - panel.groupby("trade_date")["_overnight_log"].transform("mean")
    panel["_res_open_close"] = panel["_open_close_log"] - panel.groupby("trade_date")["_open_close_log"].transform("mean")

    yz_k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
    overnight_var = by_stock["_overnight_log"].transform(lambda v: v.rolling(window, min_periods=window).var())
    open_close_var = by_stock["_open_close_log"].transform(lambda v: v.rolling(window, min_periods=window).var())
    rs_mean = by_stock["rogers_satchell_daily"].transform(lambda v: v.rolling(window, min_periods=window).mean())

    target_yz = f"yang_zhang_{window}"
    panel[target_yz] = np.sqrt((overnight_var + yz_k * open_close_var + (1.0 - yz_k) * rs_mean).clip(lower=0.0))
    factors.append(target_yz)

    # 5. Residual Yang-Zhang
    res_overnight_var = by_stock["_res_overnight"].transform(lambda v: v.rolling(window, min_periods=window).var())
    res_open_close_var = by_stock["_res_open_close"].transform(lambda v: v.rolling(window, min_periods=window).var())

    target_res_yz = f"residual_yang_zhang_{window}"
    panel[target_res_yz] = np.sqrt((res_overnight_var + yz_k * res_open_close_var + (1.0 - yz_k) * rs_mean).clip(lower=0.0))
    factors.append(target_res_yz)

    # 6. Barndorff-Nielsen & Shephard Idiosyncratic Jump
    if "ret_1" not in panel.columns:
        panel["ret_1"] = by_stock["close"].pct_change()
    if "residual_ret_1" not in panel.columns:
        panel["residual_ret_1"] = panel["ret_1"] - panel.groupby("trade_date")["ret_1"].transform("mean")

    res = panel["residual_ret_1"]
    panel["_res_sq"] = res**2
    prev_res_abs = by_stock["residual_ret_1"].shift(1).abs()
    panel["_res_bipower"] = res.abs() * prev_res_abs

    realised_var = by_stock["_res_sq"].transform(lambda v: v.rolling(window, min_periods=window).mean())
    bipower_var = (np.pi / 2.0) * by_stock["_res_bipower"].transform(lambda v: v.rolling(window, min_periods=window).mean())

    target_jump = f"idiosyncratic_jump_{window}"
    panel[target_jump] = np.sqrt((realised_var - bipower_var).clip(lower=0.0))
    factors.append(target_jump)

    # 7. Positive & Negative Idiosyncratic Semivolatility
    panel["_pos_res_sq"] = res.clip(lower=0.0)**2
    panel["_neg_res_sq"] = res.clip(upper=0.0)**2

    target_pos_semi = f"positive_idiosyncratic_semivol_{window}"
    target_neg_semi = f"negative_idiosyncratic_semivol_{window}"
    panel[target_pos_semi] = np.sqrt(by_stock["_pos_res_sq"].transform(lambda v: v.rolling(window, min_periods=window).mean()))
    panel[target_neg_semi] = np.sqrt(by_stock["_neg_res_sq"].transform(lambda v: v.rolling(window, min_periods=window).mean()))
    factors.extend([target_pos_semi, target_neg_semi])

    # 8. OHLC Consensus (Cross-sectional rank average of Garman-Klass, Rogers-Satchell, and Yang-Zhang)
    gk_rank = panel.groupby("trade_date")[f"garman_klass_{window}"].rank(pct=True)
    rs_rank = panel.groupby("trade_date")[f"rogers_satchell_{window}"].rank(pct=True)
    yz_rank = panel.groupby("trade_date")[f"yang_zhang_{window}"].rank(pct=True)
    panel["ohlc_consensus"] = (gk_rank + rs_rank + yz_rank) / 3.0
    factors.append("ohlc_consensus")

    # Cleanup temporary helper columns
    drop_cols = [c for c in panel.columns if c.startswith("_") or c.endswith("_daily")]
    panel = panel.drop(columns=drop_cols)

    return panel, factors
