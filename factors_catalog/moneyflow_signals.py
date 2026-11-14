"""Tushare Order-Size Moneyflow Imbalance Signals.

Ported from BDC2026 quantitative research lab (experiment_moneyflow_signals.py).
Computes institutional and retail order-flow imbalance signals from Tushare moneyflow data:
- net_flow_ratio (1d, 5d): Total net moneyflow ratio to classified turnover
- main_flow_ratio (1d, 3d, 5d): Institutional (large + extra-large) order net inflow ratio
- retail_flow_ratio_1: Small retail order net flow ratio
- inverse_retail_flow_1: Contrarian retail sentiment indicator (-retail_flow_ratio)
- main_retail_divergence_1: Divergence between institutional and retail order flows
- main_buy_share_1: Proportion of total turnover driven by institutional buy orders
- main_positive_fraction_5: 5-day consistency of positive institutional net flow
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_moneyflow_signals(
    panel: pd.DataFrame, moneyflow_df: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Compute order-flow imbalance factors.

    If moneyflow_df is None, panel must already contain the relevant columns:
        - buy_sm_amount, sell_sm_amount
        - buy_md_amount, sell_md_amount
        - buy_lg_amount, sell_lg_amount
        - buy_elg_amount, sell_elg_amount
        - net_mf_amount

    Returns:
        (merged_panel, list_of_signals)
    """
    df = panel.copy()

    # Normalize column names in panel
    col_map = {}
    if "ts_code" in df.columns and "stock_id" not in df.columns:
        col_map["ts_code"] = "stock_id"
    if "code" in df.columns and "stock_id" not in df.columns:
        col_map["code"] = "stock_id"
    if "date" in df.columns and "trade_date" not in df.columns:
        col_map["date"] = "trade_date"
    if col_map:
        df = df.rename(columns=col_map)

    # Merge moneyflow if provided separately
    if moneyflow_df is not None:
        mf = moneyflow_df.copy()
        mf_map = {}
        if "ts_code" in mf.columns and "stock_id" not in mf.columns:
            mf_map["ts_code"] = "stock_id"
        if "code" in mf.columns and "stock_id" not in mf.columns:
            mf_map["code"] = "stock_id"
        if "date" in mf.columns and "trade_date" not in mf.columns:
            mf_map["date"] = "trade_date"
        if mf_map:
            mf = mf.rename(columns=mf_map)
        df = df.merge(mf, on=["stock_id", "trade_date"], how="left")

    amount_cols = [
        f"{side}_{size}_amount"
        for side in ("buy", "sell")
        for size in ("sm", "md", "lg", "elg")
    ]

    signals: list[str] = []
    # Check if moneyflow columns exist
    if not any(c in df.columns for c in amount_cols):
        # Graceful fallback: return empty signals if data not loaded
        return df, signals

    for c in amount_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    total = df[amount_cols].sum(axis=1).replace(0.0, np.nan)
    main_buy = df.get("buy_lg_amount", 0.0) + df.get("buy_elg_amount", 0.0)
    main_sell = df.get("sell_lg_amount", 0.0) + df.get("sell_elg_amount", 0.0)
    retail_buy = df.get("buy_sm_amount", 0.0)
    retail_sell = df.get("sell_sm_amount", 0.0)

    main_net = main_buy - main_sell
    retail_net = retail_buy - retail_sell

    df["main_net_amount"] = main_net
    df["total_classified_amount"] = total

    # Signals
    if "net_mf_amount" in df.columns:
        df["net_flow_ratio_1"] = pd.to_numeric(df["net_mf_amount"], errors="coerce") / total
        signals.append("net_flow_ratio_1")

    df["main_flow_ratio_1"] = main_net / total
    df["retail_flow_ratio_1"] = retail_net / total
    df["inverse_retail_flow_1"] = -df["retail_flow_ratio_1"]
    df["main_retail_divergence_1"] = df["main_flow_ratio_1"] - df["retail_flow_ratio_1"]
    df["main_buy_share_1"] = main_buy / total
    signals.extend([
        "main_flow_ratio_1",
        "retail_flow_ratio_1",
        "inverse_retail_flow_1",
        "main_retail_divergence_1",
        "main_buy_share_1",
    ])

    # Rolling aggregations per stock
    df = df.sort_values(["stock_id", "trade_date"]).reset_index(drop=True)
    by_stock = df.groupby("stock_id", sort=False)

    for w in (3, 5):
        m_sum = by_stock["main_net_amount"].transform(lambda v: v.rolling(w, min_periods=w).sum())
        t_sum = by_stock["total_classified_amount"].transform(lambda v: v.rolling(w, min_periods=w).sum())
        name = f"main_flow_ratio_{w}"
        df[name] = m_sum / t_sum.replace(0.0, np.nan)
        signals.append(name)

    # 5d positive fraction
    pos_main = (df["main_net_amount"] > 0.0).astype(float)
    df["main_positive_fraction_5"] = pos_main.groupby(df["stock_id"]).transform(
        lambda v: v.rolling(5, min_periods=5).mean()
    )
    signals.append("main_positive_fraction_5")

    signals = list(dict.fromkeys(signals))
    return df, signals
