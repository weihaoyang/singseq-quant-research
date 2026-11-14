"""Tushare Daily Basic Size, Liquidity, and Valuation Factors.

Ported from BDC2026 quantitative research lab (experiment_daily_basic_factors.py).
Computes fundamental and liquidity factors:
- log_circ_mv: Natural log of circulating market capitalization (Size factor)
- turnover_rate_f: Free float turnover rate (Liquidity proxy)
- volume_ratio: Tushare intraday volume ratio
- pb: Price to book ratio
- bp: Book to market ratio (1 / PB, Value factor)
- earnings_yield: E/P ratio (1 / PE_TTM, Value factor)
- sales_yield: S/P ratio (1 / PS_TTM, Value factor)
- dividend_yield: Dividend yield (DV_TTM, Carry factor)
- free_float_ratio: Free float shares / total float shares
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_daily_basic_signals(
    panel: pd.DataFrame, daily_basic_df: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Compute valuation and liquidity factors.

    If daily_basic_df is None, panel must contain the daily_basic fields.

    Returns:
        (merged_panel, list_of_factors)
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

    if daily_basic_df is not None:
        db = daily_basic_df.copy()
        db_map = {}
        if "ts_code" in db.columns and "stock_id" not in db.columns:
            db_map["ts_code"] = "stock_id"
        if "code" in db.columns and "stock_id" not in db.columns:
            db_map["code"] = "stock_id"
        if "date" in db.columns and "trade_date" not in db.columns:
            db_map["date"] = "trade_date"
        if db_map:
            db = db.rename(columns=db_map)
        df = df.merge(db, on=["stock_id", "trade_date"], how="left")

    factors: list[str] = []

    # Size
    if "circ_mv" in df.columns:
        circ = pd.to_numeric(df["circ_mv"], errors="coerce")
        df["log_circ_mv"] = np.log(circ.clip(lower=1.0))
        factors.append("log_circ_mv")

    # Turnover
    if "turnover_rate_f" in df.columns:
        df["turnover_rate_f"] = pd.to_numeric(df["turnover_rate_f"], errors="coerce")
        factors.append("turnover_rate_f")

    # Volume ratio
    if "volume_ratio" in df.columns:
        df["volume_ratio"] = pd.to_numeric(df["volume_ratio"], errors="coerce")
        factors.append("volume_ratio")

    # Valuation: PB & BP
    if "pb" in df.columns:
        pb = pd.to_numeric(df["pb"], errors="coerce")
        df["bp"] = np.where(pb > 0, 1.0 / pb, np.nan)
        factors.extend(["pb", "bp"])

    # Valuation: Earnings yield (E/P)
    if "pe_ttm" in df.columns:
        pe = pd.to_numeric(df["pe_ttm"], errors="coerce")
        df["earnings_yield"] = np.where(pe > 0, 1.0 / pe, np.nan)
        factors.append("earnings_yield")

    # Valuation: Sales yield (S/P)
    if "ps_ttm" in df.columns:
        ps = pd.to_numeric(df["ps_ttm"], errors="coerce")
        df["sales_yield"] = np.where(ps > 0, 1.0 / ps, np.nan)
        factors.append("sales_yield")

    # Dividend yield
    if "dv_ttm" in df.columns:
        df["dividend_yield"] = pd.to_numeric(df["dv_ttm"], errors="coerce")
        factors.append("dividend_yield")

    # Float ratio
    if "free_share" in df.columns and "float_share" in df.columns:
        free_s = pd.to_numeric(df["free_share"], errors="coerce")
        float_s = pd.to_numeric(df["float_share"], errors="coerce").replace(0.0, np.nan)
        df["free_float_ratio"] = free_s / float_s
        factors.append("free_float_ratio")

    factors = list(dict.fromkeys(factors))
    return df, factors
