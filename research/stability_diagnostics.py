"""Small, null-preserving stability summaries for factor research diagnostics.

These functions summarize dated observations only.  They do not create labels,
fill missing IC values, select a model, or certify executable performance.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _summary(values: pd.Series, *, min_days: int = 1) -> dict:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    observed = values.dropna().to_numpy(dtype=float)
    result = {
        "days": int(len(observed)),
        "null_days": int(values.isna().sum()),
        "mean_rank_ic": None,
        "icir_daily": None,
        "positive_share": None,
        "status": "INSUFFICIENT_DATA" if len(observed) < min_days else "OK",
    }
    if len(observed) < min_days:
        return result
    result["mean_rank_ic"] = float(observed.mean())
    if len(observed) > 1:
        sd = float(observed.std(ddof=1))
        result["icir_daily"] = float(observed.mean() / sd) if sd > 0 else None
    result["positive_share"] = float((observed > 0).mean())
    return result


def _dated(frame: pd.DataFrame, ic_col: str) -> pd.DataFrame:
    required = {"trade_date", ic_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing stability columns: {sorted(missing)}")
    out = frame[["trade_date", ic_col]].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    if out["trade_date"].isna().any():
        raise ValueError("trade_date must use YYYYMMDD values")
    out[ic_col] = pd.to_numeric(out[ic_col], errors="coerce")
    return out.sort_values("trade_date", kind="stable").reset_index(drop=True)


def summarize_by_year(frame: pd.DataFrame, *, ic_col: str = "rank_ic",
                      min_days: int = 20) -> dict[str, dict]:
    """Return one summary per calendar year, preserving null observations."""
    dated = _dated(frame, ic_col)
    return {
        str(int(year)): _summary(group[ic_col], min_days=min_days)
        for year, group in dated.assign(year=dated.trade_date.dt.year).groupby("year", sort=True)
    }


def summarize_by_state(frame: pd.DataFrame, *, state_col: str = "market_state",
                       ic_col: str = "rank_ic", min_days: int = 20) -> dict[str, dict]:
    """Summarize IC by a contemporaneous state label supplied by the caller.

    State labels are deliberately not inferred from the target return.  A state
    may be built from information observable at signal time, such as the
    same-day close-to-close market return.
    """
    if state_col not in frame.columns:
        raise ValueError(f"missing state column: {state_col}")
    dated = frame[["trade_date", ic_col, state_col]].copy()
    dated["trade_date"] = pd.to_datetime(dated["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    if dated["trade_date"].isna().any():
        raise ValueError("trade_date must use YYYYMMDD values")
    dated[ic_col] = pd.to_numeric(dated[ic_col], errors="coerce")
    dated[state_col] = dated[state_col].astype("string").fillna("UNKNOWN")
    dated = dated.sort_values("trade_date", kind="stable").reset_index(drop=True)
    return {
        str(state): _summary(group[ic_col], min_days=min_days)
        for state, group in dated.groupby(state_col, sort=True, dropna=False)
    }


def market_state_from_return(market_return: Iterable[float], *, threshold: float = 0.001) -> pd.Series:
    """Classify same-day observable market returns without looking forward."""
    values = pd.to_numeric(pd.Series(market_return), errors="coerce")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    return pd.Series(np.select(
        [values.gt(threshold), values.lt(-threshold)], ["UP", "DOWN"], default="FLAT"
    ), index=values.index, dtype="string").where(values.notna(), "UNKNOWN")


def compare_neutralized(raw: pd.DataFrame, neutralized: pd.DataFrame | None,
                        *, ic_col: str = "rank_ic", min_days: int = 20) -> dict:
    """Compare raw and explicitly supplied neutralized IC series.

    No null filling or date interpolation is performed.  ``None`` is a clear
    blocked state when the source lacks point-in-time neutralization fields.
    """
    if neutralized is None:
        return {"status": "BLOCKED", "reason": "neutralized IC series was not supplied",
                "raw": _summary(_dated(raw, ic_col)[ic_col], min_days=min_days),
                "neutralized": None}
    raw_dated = _dated(raw, ic_col).rename(columns={ic_col: "raw_ic"})
    neutral_dated = _dated(neutralized, ic_col).rename(columns={ic_col: "neutralized_ic"})
    merged = raw_dated.merge(neutral_dated, on="trade_date", how="outer", validate="one_to_one")
    return {
        "status": "UNVERIFIED",
        "raw": _summary(merged["raw_ic"], min_days=min_days),
        "neutralized": _summary(merged["neutralized_ic"], min_days=min_days),
        "paired_days": int(merged[["raw_ic", "neutralized_ic"]].dropna().shape[0]),
        "null_preserved": True,
    }


__all__ = ["compare_neutralized", "market_state_from_return", "summarize_by_state", "summarize_by_year"]
