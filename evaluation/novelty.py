"""Factor novelty (anti-crowding) vs the frozen mainstream zoo.

A factor is **novel** when it is orthogonal to everything the crowd already
computes (AlphaAgent, arXiv 2502.16789).  Two complementary measures are
reported, both cross-sectional and multi-day:

* ``novelty`` (primary) — ``1 - max_k mean_t |Spearman_t(f, zoo_k)|``: distance
  from the single most-crowded baseline factor.
* ``novelty_r2`` — ``1 - mean_t R^2`` of regressing ``f`` on the whole zoo each
  day: the share of variance the crowd cannot explain.

Everything is computed on cross-sectional **ranks** per trading date, so the
result is scale-free and PIT-safe (no look-ahead across dates).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from evaluation import alpha_zoo

__all__ = ["NoveltyResult", "novelty"]

_MIN_NAMES_DEFAULT = 20


@dataclass(frozen=True)
class NoveltyResult:
    zoo_version: str
    novelty: float                        # 1 - max_k mean_t |corr|
    novelty_r2: float                     # 1 - mean_t R^2
    max_corr: float                       # max_k mean_t |corr|
    most_correlated: str                  # zoo factor name
    per_factor_corr: Dict[str, float]     # zoo name -> mean_t |corr|
    dates_used: int
    coverage: float                       # mean valid names per date
    top_correlated: List[Tuple[str, float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "zoo_version": self.zoo_version,
            "novelty": round(self.novelty, 6),
            "novelty_r2": round(self.novelty_r2, 6),
            "max_corr": round(self.max_corr, 6),
            "most_correlated": self.most_correlated,
            "per_factor_corr": {k: round(v, 6) for k, v in self.per_factor_corr.items()},
            "dates_used": self.dates_used,
            "coverage": round(self.coverage, 4),
            "top_correlated": [(k, round(v, 6)) for k, v in self.top_correlated],
        }


def _grouped_corr(dates: pd.Series, x: pd.Series, y: pd.Series,
                  min_n: int) -> pd.Series:
    """Per-date Pearson correlation of two cross-sectional rank series."""
    frame = pd.DataFrame({"d": dates.to_numpy(), "x": x.to_numpy(), "y": y.to_numpy()}).dropna()
    if frame.empty:
        return pd.Series(dtype=float)
    frame["x2"] = frame["x"] * frame["x"]
    frame["y2"] = frame["y"] * frame["y"]
    frame["xy"] = frame["x"] * frame["y"]
    g = frame.groupby("d", sort=False)
    means = g[["x", "y", "xy", "x2", "y2"]].mean()
    counts = g.size()
    cov = means["xy"] - means["x"] * means["y"]
    vx = (means["x2"] - means["x"] ** 2).clip(lower=0.0)
    vy = (means["y2"] - means["y"] ** 2).clip(lower=0.0)
    denom = np.sqrt(vx * vy)
    corr = (cov / denom.replace(0.0, np.nan)).dropna()
    return corr[counts.reindex(corr.index).fillna(0) >= min_n]


def _per_date_r2(dates: pd.Series, target_rank: pd.Series,
                 zoo_ranks: pd.DataFrame, min_n: int, max_dates: int = 250) -> float:
    frame = pd.DataFrame({"d": dates.to_numpy(), "y": target_rank.to_numpy()})
    zoo_ranks = zoo_ranks.loc[:, zoo_ranks.notna().any(axis=0)]
    if zoo_ranks.shape[1] == 0:
        return float("nan")
    frame = pd.concat([frame, zoo_ranks.reset_index(drop=True)], axis=1)
    valid_dates = [d for d, sub in frame.groupby("d", sort=False) if len(sub) >= min_n]
    if not valid_dates:
        return float("nan")
    if len(valid_dates) > max_dates:
        idx = np.linspace(0, len(valid_dates) - 1, max_dates).astype(int)
        valid_dates = [valid_dates[i] for i in idx]
    r2s: List[float] = []
    zoo_cols = list(zoo_ranks.columns)
    for d in valid_dates:
        sub = frame[frame["d"] == d]
        y = sub["y"].to_numpy(dtype=float)
        X = sub[zoo_cols].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if mask.sum() < min_n:
            continue
        yv = y[mask]
        Xv = X[mask]
        Xv = np.column_stack([np.ones(len(Xv)), Xv])
        try:
            coef, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = yv - Xv @ coef
        ss_tot = float(np.sum((yv - yv.mean()) ** 2))
        if ss_tot <= 0.0:
            continue
        r2s.append(1.0 - float(np.sum(resid ** 2)) / ss_tot)
    return float(np.mean(r2s)) if r2s else float("nan")


def novelty(panel: pd.DataFrame, factor: Sequence[float] | pd.Series, *,
            zoo_names: Optional[Iterable[str]] = None,
            min_names: int = _MIN_NAMES_DEFAULT) -> NoveltyResult:
    """Compute the novelty of ``factor`` against the frozen zoo.

    ``panel`` is the long OHLCV+amount panel; ``factor`` is aligned to it
    (either a Series indexed like the panel or a positional sequence).
    """
    names = list(zoo_names) if zoo_names is not None else alpha_zoo.zoo_names()
    work = panel[["ts_code", "trade_date"]].copy()
    work["ts_code"] = work["ts_code"].astype(str)
    work["trade_date"] = work["trade_date"].astype(str)
    order = np.arange(len(panel))
    work = work.assign(_order=order).sort_values(["ts_code", "trade_date"], kind="mergesort")

    if isinstance(factor, pd.Series):
        candidate = factor.reindex(work.index)
    else:
        candidate = pd.Series(np.asarray(factor, dtype=float), index=panel.index).reindex(work.index)

    zoo_raw = alpha_zoo.compute(panel, names)
    zoo = pd.DataFrame({k: v.reindex(work.index) for k, v in zoo_raw.items()})

    dates = work["trade_date"]
    cand_rank = candidate.groupby(dates, sort=False).rank(pct=True)
    zoo_ranks = zoo.groupby(dates, sort=False).rank(pct=True)

    per_corr: Dict[str, float] = {}
    used_counts: List[int] = []
    for name in names:
        corr = _grouped_corr(dates, cand_rank, zoo_ranks[name], min_names)
        if not corr.empty:
            used_counts.append(int(corr.shape[0]))
        per_corr[name] = float(np.nanmean(np.abs(corr.to_numpy()))) if not corr.empty else float("nan")

    valid = {k: v for k, v in per_corr.items() if np.isfinite(v)}
    if not valid:
        raise ValueError("novelty: no comparable zoo factor had enough names per date")
    most = max(valid, key=lambda k: valid[k])
    max_corr = valid[most]

    ranked = sorted(valid.items(), key=lambda kv: kv[1], reverse=True)
    group_sizes = cand_rank.groupby(dates, sort=False).count()
    coverage = float(group_sizes.mean()) if len(group_sizes) else 0.0
    dates_used = int((group_sizes >= min_names).sum())
    r2 = _per_date_r2(dates, cand_rank, zoo_ranks, min_names)

    return NoveltyResult(
        zoo_version=alpha_zoo.ZOO_VERSION,
        novelty=float(1.0 - max_corr),
        novelty_r2=float(1.0 - r2) if np.isfinite(r2) else float("nan"),
        max_corr=float(max_corr),
        most_correlated=most,
        per_factor_corr=valid,
        dates_used=dates_used,
        coverage=coverage,
        top_correlated=ranked[:5],
    )
