"""Industry and Within-Industry Relative Momentum Signals.

Ported from BDC2026 quantitative research lab (experiment_industry_momentum.py).
Computes:
1. Leave-one-out industry momentum: The average momentum of a stock's industry peers,
   strictly excluding the stock itself to prevent self-contamination.
2. Within-industry relative momentum: The stock's excess return above its industry peer average.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_industry_momentum(
    panel: pd.DataFrame,
    industry_col: str = "industry",
    windows: tuple[int, ...] = (5, 10, 20, 60),
) -> tuple[pd.DataFrame, list[str]]:
    """Compute industry peer momentum and stock relative momentum.

    Required columns:
        - stock_id (or ts_code / code)
        - trade_date (or date)
        - close
        - industry (if missing, returns empty factors gracefully)

    Returns:
        (panel_with_signals, list_of_signals)
    """
    df = panel.copy()

    # Column name normalization
    col_map = {}
    if "ts_code" in df.columns and "stock_id" not in df.columns:
        col_map["ts_code"] = "stock_id"
    if "code" in df.columns and "stock_id" not in df.columns:
        col_map["code"] = "stock_id"
    if "date" in df.columns and "trade_date" not in df.columns:
        col_map["date"] = "trade_date"
    if col_map:
        df = df.rename(columns=col_map)

    signals: list[str] = []
    if industry_col not in df.columns:
        return df, signals

    df[industry_col] = df[industry_col].fillna("UNKNOWN").astype(str)
    df = df.sort_values(["stock_id", "trade_date"]).reset_index(drop=True)
    by_stock = df.groupby("stock_id", sort=False)

    keys = [df["trade_date"], df[industry_col]]
    ind_count = df.groupby(keys)["stock_id"].transform("count")

    for w in windows:
        ret_name = f"ret_{w}"
        if ret_name not in df.columns:
            df[ret_name] = by_stock["close"].pct_change(w)

        # Leave-one-out industry mean
        ind_sum = df.groupby(keys)[ret_name].transform("sum")
        mkt_mean = df.groupby("trade_date")[ret_name].transform("mean")
        leave_one_out = (ind_sum - df[ret_name]) / (ind_count - 1).replace(0, np.nan)
        ind_val = leave_one_out.fillna(mkt_mean)

        ind_factor = f"industry_momentum_{w}"
        rel_factor = f"industry_relative_{w}"
        df[ind_factor] = ind_val
        df[rel_factor] = df[ret_name] - ind_val
        signals.extend([ind_factor, rel_factor])

    signals = list(dict.fromkeys(signals))
    return df, signals
