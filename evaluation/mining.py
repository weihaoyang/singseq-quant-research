"""Systematic factor mining: generate, evaluate and rank candidates.

Generates a bounded grammar of price/volume factors (families x windows),
evaluates each with the same institutional metrics the tearsheet uses (IC, ICIR,
OOS retention, cross-sectional monotonicity) and ranks them by the
**novelty-weighted** competition score — so the miner is nudged away from the
crowded families the zoo already contains.

This is the deterministic, reproducible core; an LLM/GP layer can extend
:func:`candidates` with additional expressions without changing the scoring.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from evaluation import novelty as _novelty
from evaluation import novelty_scoring as _scoring

__all__ = ["MINE_VERSION", "candidates", "mine"]

MINE_VERSION = "mine-1"

_WINDOWS = (5, 10, 20, 60)


def _grp(work: pd.DataFrame, series: pd.Series):
    return series.groupby(work["ts_code"], sort=False)


def _roll(work: pd.DataFrame, series: pd.Series, n: int, how: str = "mean") -> pd.Series:
    rolled = getattr(_grp(work, series).rolling(n, min_periods=max(2, n // 2)), how)()
    return rolled.reset_index(level=0, drop=True)


def candidates(panel: pd.DataFrame, windows: Sequence[int] = _WINDOWS) -> Dict[str, pd.Series]:
    """Bounded deterministic candidate set (name -> factor series aligned to panel)."""
    work = panel.copy()
    work["ts_code"] = work["ts_code"].astype(str)
    work["trade_date"] = work["trade_date"].astype(str)
    work = work.sort_values(["ts_code", "trade_date"], kind="mergesort")
    ret = _grp(work, work["close"]).pct_change()
    out: Dict[str, pd.Series] = {}
    for w in windows:
        out[f"momentum_{w}"] = _grp(work, work["close"]).pct_change(w)
        out[f"reversal_{w}"] = -out[f"momentum_{w}"]
        out[f"volatility_{w}"] = _roll(work, ret, w, "std")
        out[f"volume_ratio_{w}"] = _roll(work, work["vol"], 5) / _roll(work, work["vol"], w) - 1.0
        out[f"amplitude_{w}"] = _roll(work, (work["high"] - work["low"]) / work["close"], w)
        out[f"bias_{w}"] = work["close"] / _roll(work, work["close"], w) - 1.0
        out[f"illiquidity_{w}"] = _roll(work, ret.abs() / work["amount"].replace(0.0, np.nan), w)
    clean: Dict[str, pd.Series] = {}
    for name, series in out.items():
        s = pd.Series(np.asarray(series, dtype=float), index=work.index).replace([np.inf, -np.inf], np.nan)
        clean[name] = s
    return clean


def _rank(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values, kind="stable"), kind="stable").astype(float)


def _ic_series(dates: pd.Series, x: pd.Series, y: pd.Series, min_names: int) -> np.ndarray:
    frame = pd.DataFrame({"d": dates.to_numpy(), "x": x.to_numpy(), "y": y.to_numpy()}).dropna()
    if frame.empty:
        return np.asarray([])
    frame["xr"] = frame.groupby("d", sort=False)["x"].rank(pct=True)
    frame["yr"] = frame.groupby("d", sort=False)["y"].rank(pct=True)
    frame["xy"] = frame["xr"] * frame["yr"]
    frame["x2"] = frame["xr"] ** 2
    frame["y2"] = frame["yr"] ** 2
    g = frame.groupby("d", sort=False)
    m = g[["xr", "yr", "xy", "x2", "y2"]].mean()
    n = g.size()
    cov = m["xy"] - m["xr"] * m["yr"]
    vx = (m["x2"] - m["xr"] ** 2).clip(lower=0.0)
    vy = (m["y2"] - m["yr"] ** 2).clip(lower=0.0)
    ic = (cov / np.sqrt(vx * vy).replace(0.0, np.nan)).dropna()
    return ic[n.reindex(ic.index).fillna(0) >= min_names].to_numpy(dtype=float)


def _metrics(dates: pd.Series, x: pd.Series, y: pd.Series, min_names: int) -> Optional[Dict[str, float]]:
    ic = _ic_series(dates, x, y, min_names)
    ic = ic[np.isfinite(ic)]
    if ic.size < 10 or ic.std() == 0.0:
        return None
    half = ic.size // 2
    ic_in = float(np.mean(np.abs(ic[:half]))) if half else 0.0
    ic_out = float(np.mean(np.abs(ic[half:]))) if half else 0.0
    # cross-sectional monotonicity: |corr(quantile index, quantile mean return)|
    mono_vals = []
    df = pd.DataFrame({"d": dates.to_numpy(), "x": x.to_numpy(), "y": y.to_numpy()}).dropna()
    for _, sub in df.groupby("d", sort=False):
        if len(sub) < 2 * min_names:
            continue
        try:
            q = pd.qcut(sub["x"].rank(method="first"), 5, labels=False)
        except ValueError:
            continue
        means = sub.groupby(q)["y"].mean().sort_index()
        if len(means) < 3:
            continue
        idx = np.arange(len(means), dtype=float)
        if means.std() == 0:
            continue
        mono_vals.append(abs(float(np.corrcoef(idx, means.to_numpy())[0, 1])))
    return {
        "ic_mean": float(np.mean(ic)),
        "icir": float(np.mean(ic) / (np.std(ic) + 1e-12)),
        "oos_retention": ic_out / ic_in if ic_in > 0 else 0.0,
        "monotonicity": float(np.mean(mono_vals)) if mono_vals else 0.0,
    }


def mine(panel: pd.DataFrame, *, forward_returns: Optional[Sequence[float]] = None,
         horizon: int = 1, top_k: int = 20, min_names: int = 20,
         cfg: Optional[_scoring.ScoringConfig] = None) -> List[Dict[str, object]]:
    """Rank candidates by the novelty-weighted competition score."""
    cfg = cfg or _scoring.ScoringConfig()
    work = panel.copy()
    work["ts_code"] = work["ts_code"].astype(str)
    work["trade_date"] = work["trade_date"].astype(str)
    work = work.sort_values(["ts_code", "trade_date"], kind="mergesort")
    idx = work.index
    if forward_returns is not None:
        y = pd.Series(np.asarray(forward_returns, dtype=float), index=panel.index).reindex(idx)
    else:
        close = work["close"].astype(float)
        y = close.groupby(work["ts_code"], sort=False).shift(-horizon) / close - 1.0
    dates = work["trade_date"]

    results: List[Dict[str, object]] = []
    for name, factor in candidates(panel).items():
        metrics = _metrics(dates, factor, y, min_names)
        if metrics is None:
            continue
        try:
            nov = _novelty.novelty(panel, factor).as_dict()
        except ValueError:
            continue
        score = _scoring.final_score(metrics, nov["novelty"], cfg=cfg, zoo_version=nov["zoo_version"])
        results.append({
            "name": name,
            "ic_mean": round(metrics["ic_mean"], 6),
            "icir": round(metrics["icir"], 6),
            "oos_retention": round(metrics["oos_retention"], 4),
            "monotonicity": round(metrics["monotonicity"], 4),
            "novelty": nov["novelty"],
            "most_correlated": nov["most_correlated"],
            "gate_passed": score["gate_passed"],
            "final_score": score["final_score"],
            "quality": score["quality"],
        })
    results.sort(key=lambda r: (r["final_score"], r["ic_mean"]), reverse=True)
    return results[:top_k]
