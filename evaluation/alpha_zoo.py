"""Frozen mainstream factor zoo — the novelty baseline (zoo v1).

Purpose
-------
Factor *crowding* is measured against a fixed set of well-known factors: a
candidate is "novel" when it is orthogonal to everything the crowd already
computes.  This module is the frozen baseline; its name list and formulas are
versioned (``ZOO_VERSION``) so a novelty score is reproducible forever.

Scope / honesty
---------------
Only price/volume factors are included, because the public research package
carries OHLCV + amount and no fundamentals.  Value/quality factors are
deliberately absent rather than faked.  Every formula is the factor's **public
definition** (WorldQuant "101 Formulaic Alphas"; GTJA "191 alphas"; classic
styles), re-implemented here from scratch — no third-party factor-library code
is copied, so there is no licence entanglement.

Contract
--------
``compute(df, names=None) -> dict[str, pd.Series]`` keyed by factor name, each
Series aligned to ``df.index`` (a long panel: ``ts_code, trade_date,
open, high, low, close, vol, amount``).
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd

ZOO_VERSION = "zoo-1"

__all__ = ["ZOO_VERSION", "zoo_names", "compute", "compute_matrix"]

_REQUIRED = ("ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount")


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"zoo panel missing columns: {', '.join(missing)}")
    work = df.copy()
    work["trade_date"] = work["trade_date"].astype(str)
    work["ts_code"] = work["ts_code"].astype(str)
    work = work.sort_values(["ts_code", "trade_date"], kind="mergesort")
    for col in ("open", "high", "low", "close", "vol", "amount"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    return work


def _by_code(work: pd.DataFrame, series: pd.Series) -> pd.core.groupby.SeriesGroupBy:
    return series.groupby(work["ts_code"], sort=False)


def _xs_rank(work: pd.DataFrame, series: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank within each trade date."""
    return series.groupby(work["trade_date"], sort=False).rank(pct=True)


def _roll(work: pd.DataFrame, series: pd.Series, n: int, how: str = "mean") -> pd.Series:
    grouped = _by_code(work, series)
    rolled = getattr(grouped.rolling(n, min_periods=max(2, n // 2)), how)()
    return rolled.reset_index(level=0, drop=True)


def _delta(work: pd.DataFrame, series: pd.Series, n: int = 1) -> pd.Series:
    return _by_code(work, series).diff(n)


def _delay(work: pd.DataFrame, series: pd.Series, n: int = 1) -> pd.Series:
    return _by_code(work, series).shift(n)


def _ts_sum(work: pd.DataFrame, series: pd.Series, n: int) -> pd.Series:
    return _roll(work, series, n, "sum")


# --- classic styles -------------------------------------------------------

def _momentum(work, n):
    return _by_code(work, work["close"]).pct_change(n)


def _reversal(work, n):
    return -_momentum(work, n)


def _volatility(work, n):
    ret = _by_code(work, work["close"]).pct_change()
    return _roll(work, ret, n, "std")


def _turnover_ratio(work, short=5, long=20):
    return _roll(work, work["vol"], short) / _roll(work, work["vol"], long) - 1.0


def _illiquidity(work, n=20):
    ret = _by_code(work, work["close"]).pct_change().abs()
    amt = work["amount"].replace(0.0, np.nan)
    return _roll(work, ret / amt, n)


def _amplitude(work, n=20):
    return _roll(work, (work["high"] - work["low"]) / work["close"], n)


def _rsi(work, n=14):
    delta = _by_code(work, work["close"]).diff()
    up = _roll(work, delta.clip(lower=0.0), n, "mean")
    down = (-_roll(work, delta.clip(upper=0.0), n, "mean"))
    rs = up / down.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _bias(work, n=20):
    ma = _roll(work, work["close"], n)
    return work["close"] / ma - 1.0


# --- formulaic alphas (public definitions, re-implemented) ----------------

def _alpha002(work):
    inner = ((work["close"] - work["low"]) - (work["high"] - work["close"])) / (
        (work["high"] - work["low"]).replace(0.0, np.nan))
    return -_delta(work, inner, 1)


def _alpha003(work):
    prev = _delay(work, work["close"], 1)
    cond_up = work["close"] > prev
    cond_dn = work["close"] < prev
    value = np.where(work["close"].eq(prev), 0.0,
                     work["close"] - np.where(cond_up, np.minimum(work["low"], prev),
                                               np.where(cond_dn, np.maximum(work["high"], prev),
                                                        np.nan)))
    return _ts_sum(work, pd.Series(value, index=work.index), 6)


def _alpha009(work):
    mid = (work["high"] + work["low"]) / 2.0
    prev_mid = (_delay(work, work["high"], 1) + _delay(work, work["low"], 1)) / 2.0
    core = (mid - prev_mid) * (work["high"] - work["low"]) / work["vol"].replace(0.0, np.nan)
    return -_roll(work, core, 7, "mean")


def _alpha012(work):
    return np.sign(_delta(work, work["vol"], 1)) * (-_delta(work, work["close"], 1))


def _alpha013(work):
    rc = _xs_rank(work, work["close"])
    rv = _xs_rank(work, work["vol"])
    cov = _roll(work, rc * rv, 5) - _roll(work, rc, 5) * _roll(work, rv, 5)
    return -_xs_rank(work, cov)


def _alpha033(work):
    return _xs_rank(work, work["open"] / work["close"] - 1.0)


def _gtja028(work, n=9):
    """GTJA 028: stochastic %K smoothed (KDJ-K family)."""
    low_n = _roll(work, work["low"], n, "min")
    high_n = _roll(work, work["high"], n, "max")
    rsv = (work["close"] - low_n) / (high_n - low_n).replace(0.0, np.nan) * 100.0
    sma1 = rsv.ewm(alpha=1.0 / 3.0, adjust=False).mean()
    sma2 = sma1.ewm(alpha=1.0 / 3.0, adjust=False).mean()
    return 3.0 * sma1 - 2.0 * sma2


def _gtja034(work, n=12):
    """GTJA 034: MA(close)/close (short MA premium)."""
    return _roll(work, work["close"], n) / work["close"]


_REGISTRY: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "momentum_20": lambda w: _momentum(w, 20),
    "momentum_60": lambda w: _momentum(w, 60),
    "momentum_120": lambda w: _momentum(w, 120),
    "reversal_5": lambda w: _reversal(w, 5),
    "reversal_20": lambda w: _reversal(w, 20),
    "volatility_20": lambda w: _volatility(w, 20),
    "volatility_60": lambda w: _volatility(w, 60),
    "volume_ratio_5_20": lambda w: _turnover_ratio(w, 5, 20),
    "illiquidity_20": lambda w: _illiquidity(w, 20),
    "amplitude_20": lambda w: _amplitude(w, 20),
    "rsi_14": lambda w: _rsi(w, 14),
    "bias_20": lambda w: _bias(w, 20),
    "alpha002": _alpha002,
    "alpha003": _alpha003,
    "alpha009": _alpha009,
    "alpha012": _alpha012,
    "alpha013": _alpha013,
    "alpha033": _alpha033,
    "gtja028": _gtja028,
    "gtja034": _gtja034,
}


def zoo_names() -> List[str]:
    """Frozen, ordered baseline factor names (part of the versioned contract)."""
    return sorted(_REGISTRY)


def compute(df: pd.DataFrame, names: Iterable[str] | None = None) -> Dict[str, pd.Series]:
    work = _prep(df)
    selected = list(names) if names is not None else zoo_names()
    out: Dict[str, pd.Series] = {}
    for name in selected:
        fn = _REGISTRY.get(name)
        if fn is None:
            raise KeyError(f"unknown zoo factor: {name}")
        values = fn(work)
        if isinstance(values, pd.Series):
            series = values.reindex(work.index)
        else:
            series = pd.Series(np.asarray(values, dtype=float), index=work.index)
        out[name] = series.replace([np.inf, -np.inf], np.nan)
    return out


def compute_matrix(df: pd.DataFrame, names: Iterable[str] | None = None) -> pd.DataFrame:
    """Zoo factors as columns aligned to the panel rows (post-sort)."""
    return pd.DataFrame(compute(df, names))
