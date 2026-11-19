"""Canonical forward-return labels shared by every research surface.

A factor observed at the close of session ``t`` is actionable at the next
session's open.  The canonical label therefore enters at ``open(t+1)`` and
exits at ``open(t+1+horizon)``.  Raw execution prices are converted to the
adjusted-price basis when ``adj_close`` or ``adj_factor`` is available.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

LABEL_VERSION = "next-open-adjusted/1"
REQUIRED_COLUMNS = ("ts_code", "trade_date", "open", "close", "vol")


def price_adjustment_basis(work: pd.DataFrame) -> str:
    """Return the strongest adjusted-price reference present in the frame."""
    if "adj_close" in work:
        adjusted = pd.to_numeric(work["adj_close"], errors="coerce")
        if np.isfinite(adjusted.to_numpy(dtype=float)).any():
            return "adj_close"
    if "adj_factor" in work:
        factor = pd.to_numeric(work["adj_factor"], errors="coerce")
        if np.isfinite(factor.to_numpy(dtype=float)).any():
            return "adj_factor"
    return "close_proxy"


def canonical_forward_returns(
    work: pd.DataFrame,
    horizon: int,
    *,
    date_column: str | None = None,
) -> pd.Series:
    """Build the canonical close-signal / next-open execution label.

    ``date_column`` is accepted for callers that keep a differently named
    identity column.  The returned series remains aligned to ``work.index``.
    """
    if int(horizon) < 1:
        raise ValueError("horizon must be at least one")
    missing = [name for name in REQUIRED_COLUMNS if name not in work.columns]
    if missing:
        raise ValueError(
            "canonical forward returns require columns: " + ", ".join(missing)
        )
    identity = str(date_column or "trade_date")
    if identity not in work.columns:
        raise ValueError(f"canonical forward returns require {identity}")

    raw_open = pd.to_numeric(work["open"], errors="coerce")
    close = pd.to_numeric(work["close"], errors="coerce")
    basis = price_adjustment_basis(work)
    if basis == "adj_close":
        adjusted_close = pd.to_numeric(work["adj_close"], errors="coerce")
    elif basis == "adj_factor":
        adjusted_close = close * pd.to_numeric(work["adj_factor"], errors="coerce")
    else:
        adjusted_close = close

    adj_open = raw_open * adjusted_close / close.replace(0.0, np.nan)
    grouped = work.groupby("ts_code", sort=False)
    entry = adj_open.groupby(work["ts_code"], sort=False).shift(-1)
    exit_price = adj_open.groupby(work["ts_code"], sort=False).shift(-(int(horizon) + 1))
    valid = (
        np.isfinite(entry)
        & np.isfinite(exit_price)
        & entry.gt(0)
        & exit_price.gt(0)
        & grouped["vol"].shift(-1).gt(0)
        & grouped["vol"].shift(-(int(horizon) + 1)).gt(0)
    )
    if "st" in work:
        valid &= ~grouped["st"].shift(-1).fillna(False).astype(bool)
    result = (exit_price / entry - 1.0).where(valid)
    result.name = "target_ret"
    result.attrs["labelVersion"] = LABEL_VERSION
    result.attrs["priceAdjustment"] = basis
    result.attrs["horizon"] = int(horizon)
    return result


def label_contract(work: pd.DataFrame, horizon: int) -> dict[str, Any]:
    """Machine-readable identity for one canonical label evaluation."""
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least one")
    return {
        "version": LABEL_VERSION,
        "signal": "session_close_t",
        "entry": "session_open_t_plus_1",
        "exit": f"session_open_t_plus_{horizon + 1}",
        "matching": "next-open",
        "horizon": horizon,
        "priceAdjustment": price_adjustment_basis(work),
        "stFilter": "next_open_st" if "st" in work else "unavailable",
        "volumeFilter": "positive_entry_and_exit",
    }


__all__ = [
    "LABEL_VERSION",
    "REQUIRED_COLUMNS",
    "canonical_forward_returns",
    "label_contract",
    "price_adjustment_basis",
]
