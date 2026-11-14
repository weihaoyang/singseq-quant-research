"""Factor attribution: how much of a factor's information is *style* vs residual.

Cross-sectional, covariance-based (hence exactly additive), computed against the
frozen zoo on the same panel — no extra backtests needed.  For each date, regress
the factor on the zoo factors; the fitted part is "style", the residual is
"idiosyncratic".  With forward returns ``y`` and covariance-based IC

    cov(x, y) = cov(fitted, y) + cov(resid, y)

so the style share and the per-style contributions are exact.  (Reported
``ic``/``residual_ic`` are the normalised Pearson correlations for intuition;
the *shares* come from the additive covariance form.)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["ATTR_VERSION", "ic_attribution"]

ATTR_VERSION = "attr-1"


def _forward_returns(work: pd.DataFrame, horizon: int) -> pd.Series:
    close = work["close"].astype(float)
    future = close.groupby(work["ts_code"], sort=False).shift(-horizon)
    return future / close - 1.0


def ic_attribution(panel: pd.DataFrame, factor: Sequence[float], *,
                   forward_returns: Optional[Sequence[float]] = None,
                   zoo_names: Optional[Sequence[str]] = None,
                   horizon: int = 1, min_names: int = 20,
                   max_dates: int = 200) -> Dict[str, object]:
    from evaluation import alpha_zoo

    names = list(zoo_names) if zoo_names is not None else alpha_zoo.zoo_names()
    work = panel.copy()
    work["ts_code"] = work["ts_code"].astype(str)
    work["trade_date"] = work["trade_date"].astype(str)
    work = work.sort_values(["ts_code", "trade_date"], kind="mergesort")
    idx = work.index

    if forward_returns is not None:
        fwd = pd.Series(np.asarray(forward_returns, dtype=float), index=panel.index).reindex(idx)
    else:
        fwd = _forward_returns(work, horizon)

    factor_series = (factor.reindex(idx) if isinstance(factor, pd.Series)
                     else pd.Series(np.asarray(factor, dtype=float), index=panel.index).reindex(idx))
    zoo_all = {k: v.reindex(idx) for k, v in alpha_zoo.compute(panel, names).items()}
    # Drop baselines that have no finite value on this panel (e.g. momentum_120 on
    # a 120-day window); otherwise a single all-NaN column zeroes every date's mask.
    zoo = {k: v for k, v in zoo_all.items() if np.isfinite(np.asarray(v, dtype=float)).any()}
    if not zoo:
        raise ValueError("attribution: no zoo baseline has finite coverage on this panel")

    frame = pd.DataFrame({"d": work["trade_date"].to_numpy(),
                          "y": fwd.to_numpy(dtype=float),
                          "x": factor_series.to_numpy(dtype=float)})
    for k, v in zoo.items():
        frame[k] = np.asarray(v, dtype=float)

    dates = [d for d, sub in frame.groupby("d", sort=False) if len(sub) >= min_names]
    if len(dates) > max_dates:
        picks = np.linspace(0, len(dates) - 1, max_dates).astype(int)
        dates = [dates[i] for i in picks]
    if not dates:
        raise ValueError("attribution: no date had enough names")

    cols = list(zoo)
    total_cov: List[float] = []
    style_cov: List[float] = []
    resid_cov: List[float] = []
    per_style: Dict[str, float] = {c: 0.0 for c in cols}
    total_ic: List[float] = []
    resid_ic: List[float] = []

    for d in dates:
        sub = frame[frame["d"] == d]
        y = sub["y"].to_numpy(dtype=float)
        x = sub["x"].to_numpy(dtype=float)
        Z = sub[cols].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(x) & np.isfinite(Z).all(axis=1)
        if mask.sum() < min_names:
            continue
        yv, xv, Zv = y[mask], x[mask], Z[mask]
        # cross-sectional ranks (robust, scale-free)
        xr = _rank(xv)
        yr = _rank(yv)
        Zr = np.column_stack([_rank(Zv[:, j]) for j in range(Zv.shape[1])])
        X = np.column_stack([np.ones(len(xr)), Zr])
        try:
            coef, *_ = np.linalg.lstsq(X, xr, rcond=None)
        except np.linalg.LinAlgError:
            continue
        fitted = X @ coef
        resid = xr - fitted
        total_cov.append(float(np.cov(xr, yr, ddof=0)[0, 1]))
        style_cov.append(float(np.cov(fitted, yr, ddof=0)[0, 1]))
        resid_cov.append(float(np.cov(resid, yr, ddof=0)[0, 1]))
        for j, c in enumerate(cols):
            per_style[c] += float(coef[j + 1]) * float(np.cov(Zr[:, j], yr, ddof=0)[0, 1])
        if xr.std() > 0 and yr.std() > 0:
            total_ic.append(float(np.corrcoef(xr, yr)[0, 1]))
        if resid.std() > 0 and yr.std() > 0:
            resid_ic.append(float(np.corrcoef(resid, yr)[0, 1]))

    t_cov = float(np.mean(total_cov)) if total_cov else 0.0
    s_cov = float(np.mean(style_cov)) if style_cov else 0.0
    r_cov = float(np.mean(resid_cov)) if resid_cov else 0.0
    denom = abs(t_cov) if abs(t_cov) > 1e-12 else 1.0
    style_share = s_cov / denom if t_cov != 0 else 0.0
    resid_share = r_cov / denom if t_cov != 0 else 0.0
    contributions = {c: round(v / denom, 6) for c, v in per_style.items() if abs(v / denom) > 1e-4}

    return {
        "version": ATTR_VERSION,
        "ic": round(float(np.mean(total_ic)), 6) if total_ic else None,
        "residual_ic": round(float(np.mean(resid_ic)), 6) if resid_ic else None,
        "style_share": round(style_share, 4),
        "residual_share": round(resid_share, 4),
        "style_contributions": dict(sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)),
        "dates_used": len(total_cov),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable").astype(float)
    return order / max(1.0, len(values) - 1.0)
