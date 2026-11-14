"""Performance metrics + threaded rank kernels (S3, engine-pure numpy).

Hot-path ranking is done with a single ``argsort`` per row plus a scatter
(``np.put_along_axis``) and the rows are processed by a ThreadPoolExecutor
(GIL is released inside numpy sorting) - measured 16.6 ms vs 129.9 ms for
the naive double-argsort over a [484, 5219] panel.

Everything here is numpy + stdlib only (H7 purity).
"""
from __future__ import annotations

import os
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_WORKERS = int(os.environ.get("S3_RANK_WORKERS", "20"))


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #
def _rank_row(x: np.ndarray) -> np.ndarray:
    """Average-tie ranks of a 1-D row (NaNs get NaN)."""
    n = x.shape[0]
    finite = np.isfinite(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if not finite.any():
        return out
    vals = x[finite]
    m = vals.shape[0]
    order = np.argsort(vals, kind="stable")
    sorted_vals = vals[order]
    # tie groups, fully vectorised (bincount over group ids)
    if m > 1:
        new_grp = np.empty(m, dtype=bool)
        new_grp[0] = True
        np.not_equal(sorted_vals[1:], sorted_vals[:-1], out=new_grp[1:])
    else:
        new_grp = np.ones(1, dtype=bool)
    gid_s = np.cumsum(new_grp, dtype=np.int64) - 1
    counts = np.bincount(gid_s)
    rank_sum = np.bincount(gid_s, weights=np.arange(1, m + 1, dtype=np.float64))
    avg = rank_sum / counts
    ranks = np.empty(m, dtype=np.float64)
    ranks[order] = avg[gid_s]
    out[finite] = ranks
    return out


def rank_rows(matrix: np.ndarray, workers: int = _WORKERS) -> np.ndarray:
    """Row-wise average-tie ranks, threaded over rows. Shape preserved."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("rank_rows expects a 2-D matrix")
    T, N = matrix.shape
    out = np.empty((T, N), dtype=np.float64)
    if T == 0:
        return out
    if workers <= 1 or T <= 4:
        for t in range(T):
            out[t] = _rank_row(matrix[t])
        return out
    with ThreadPoolExecutor(max_workers=min(workers, T)) as pool:
        futures = [pool.submit(_rank_row, matrix[t]) for t in range(T)]
        for t, fut in enumerate(futures):
            out[t] = fut.result()
    return out


def rank_normalize(ranks: np.ndarray) -> np.ndarray:
    """Map *per-row* ranks to ``[0, 1]``: ``(r - 1) / (n_valid - 1)``.

    ``n_valid`` is the number of finite (i.e. ranked) entries **in that
    row**, not the full panel width - otherwise a partially eligible
    cross-section would collapse into the lower quintiles.
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    n_valid = np.isfinite(ranks).sum(axis=-1, keepdims=True).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (ranks - 1.0) / np.maximum(n_valid - 1.0, 1.0)
    out[n_valid.reshape(-1) <= 1] = np.nan
    return out


# --------------------------------------------------------------------------- #
# IC
# --------------------------------------------------------------------------- #
def _row_ic(f: np.ndarray, fwd: np.ndarray) -> float:
    mask = np.isfinite(f) & np.isfinite(fwd)
    if mask.sum() < 3:
        return np.nan
    a = f[mask]
    b = fwd[mask]
    a = a - a.mean()
    b = b - b.mean()
    da = np.sqrt((a * a).sum())
    db = np.sqrt((b * b).sum())
    if da <= 0 or db <= 0:
        return np.nan
    return float((a * b).sum() / (da * db))


def pearson_rows(a: np.ndarray, b: np.ndarray, min_n: int = 3) -> np.ndarray:
    """Row-wise Pearson correlation of two ``[T, N]`` matrices (NaN-safe).

    Fully vectorised (no Python row loop): invalid pairs are zero-filled,
    which is exact because the per-row means are computed over the same
    valid subset.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    both = np.isfinite(a) & np.isfinite(b)
    cnt = both.sum(axis=1)
    a0 = np.where(both, a, 0.0)
    b0 = np.where(both, b, 0.0)
    sa = a0.sum(axis=1)
    sb = b0.sum(axis=1)
    saa = np.einsum("ij,ij->i", a0, a0)
    sbb = np.einsum("ij,ij->i", b0, b0)
    sab = np.einsum("ij,ij->i", a0, b0)
    c = cnt.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        va = saa - sa * sa / c
        vb = sbb - sb * sb / c
        r = (sab - sa * sb / c) / np.sqrt(va * vb)
    ok = (cnt >= min_n) & (va > 0) & (vb > 0)
    return np.where(ok, r, np.nan)


def normal_ic_series(factor: np.ndarray, forward: np.ndarray,
                     min_n: int = 3) -> np.ndarray:
    """Per-date Pearson (Normal IC) correlation of factor and forward return.

    The pairwise finite mask is applied independently for every cross-section;
    unavailable labels therefore remain ``NaN`` instead of becoming zero.
    """
    return pearson_rows(factor, forward, min_n=min_n)


def cross_sectional_autocorrelation(factor: np.ndarray,
                                    min_n: int = 3) -> np.ndarray:
    """Pearson autocorrelation between adjacent factor cross-sections.

    ``out[t]`` compares the factor exposures at dates ``t - 1`` and ``t``
    over their common finite symbols.  The first date has no predecessor and
    is therefore ``NaN``.  This is a time-series of cross-sectional stability,
    not a pooled panel correlation.
    """
    values = np.asarray(factor, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("cross_sectional_autocorrelation expects a 2-D matrix")
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    if values.shape[0] >= 2:
        out[1:] = pearson_rows(values[1:], values[:-1], min_n=min_n)
    return out


def spearman_ic_rows(factor: np.ndarray, forward: np.ndarray,
                     workers: int = _WORKERS,
                     factor_ranks: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-row Spearman IC between factor ranks and forward returns."""
    fr = rank_rows(factor, workers=workers) if factor_ranks is None \
        else np.asarray(factor_ranks, dtype=np.float64)
    fw = rank_rows(forward, workers=workers)
    return pearson_rows(fr, fw)


def rank_ic_series(factor: np.ndarray, forward: np.ndarray,
                   workers: int = _WORKERS) -> np.ndarray:
    """Alias kept for the ARCHITECTURE 3.4 vocabulary."""
    return spearman_ic_rows(factor, forward, workers=workers)


def ic_decay_map(factor: np.ndarray, prices: np.ndarray,
                 lags: Sequence[int] = (1, 2, 3, 5, 10, 20),
                 workers: int = _WORKERS,
                 factor_ranks: Optional[np.ndarray] = None) -> Dict[int, float]:
    """Mean Spearman IC of factor at t vs return over ``[t, t+lag]``.

    A common finite mask across all lags is used so lag values are
    comparable with each other (only rows present for every lag count).
    """
    T, N = np.asarray(factor).shape
    prices = np.asarray(prices, dtype=np.float64)
    max_lag = int(max(lags))
    if T <= max_lag:
        return {int(l): float("nan") for l in lags}
    n_rows = T - max_lag
    fr_all = rank_rows(factor, workers=workers) if factor_ranks is None \
        else np.asarray(factor_ranks, dtype=np.float64)
    # lag returns restricted to the rows every lag shares
    fwd_by_lag: Dict[int, np.ndarray] = {}
    common = np.ones((n_rows, N), dtype=bool)
    for lag in sorted({int(l) for l in lags if 0 < int(l) < T}):
        fwd = prices[lag:lag + n_rows] / prices[:n_rows] - 1.0
        fwd_by_lag[lag] = fwd
        common &= np.isfinite(fwd)
    common &= np.isfinite(fr_all[:n_rows])
    a = np.where(common, fr_all[:n_rows], np.nan)
    out: Dict[int, float] = {}
    for lag in sorted({int(l) for l in lags}):
        if lag <= 0 or lag >= T:
            out[lag] = float("nan")
            continue
        fwd = fwd_by_lag.get(lag)
        if fwd is None:
            out[lag] = float("nan")
            continue
        fwr = rank_rows(fwd, workers=workers)
        ics = pearson_rows(a, np.where(common, fwr, np.nan))
        ics = ics[np.isfinite(ics)]
        out[lag] = float(ics.mean()) if ics.size else float("nan")
    return out


# --------------------------------------------------------------------------- #
# scalar performance metrics
# --------------------------------------------------------------------------- #
def max_drawdown(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=np.float64)
    if eq.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(eq)
    dd = eq / np.where(peak > 0, peak, np.nan) - 1.0
    dd = dd[np.isfinite(dd)]
    return float(dd.min()) if dd.size else float("nan")


def sharpe_ratio(daily: np.ndarray, periods: int = 252) -> float:
    d = np.asarray(daily, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan")
    sd = float(d.std(ddof=1))
    if sd <= 0:
        return float("nan")
    return float(d.mean() / sd * np.sqrt(periods))


def calmar_ratio(daily: np.ndarray, periods: int = 252) -> float:
    d = np.asarray(daily, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    eq = np.cumprod(1.0 + d)
    mdd = max_drawdown(eq)
    if not np.isfinite(mdd) or mdd >= 0:
        return float("nan")
    ann = float((eq[-1] ** (periods / max(d.size, 1)) - 1.0)) if eq[-1] > 0 else float("nan")
    return float(ann / abs(mdd))


def annualized_return(daily: np.ndarray, periods: int = 252) -> float:
    d = np.asarray(daily, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    eq = float(np.prod(1.0 + d))
    if eq <= 0:
        return float("nan")
    return float(eq ** (periods / d.size) - 1.0)


def win_rate(daily: np.ndarray) -> float:
    d = np.asarray(daily, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    return float((d > 0).mean())


def icir_ratio(ic: np.ndarray) -> float:
    x = np.asarray(ic, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    sd = float(x.std(ddof=1))
    if sd <= 0:
        return float("nan")
    return float(x.mean() / sd * np.sqrt(252))


def cumulative_curve(daily: np.ndarray) -> np.ndarray:
    d = np.asarray(daily, dtype=np.float64)
    curve = np.full(d.shape[0] + 1, np.nan, dtype=np.float64)
    curve[0] = 1.0
    running = 1.0
    for i, value in enumerate(d, start=1):
        if np.isfinite(value):
            running *= 1.0 + float(value)
            curve[i] = running
    return curve


def underwater_curve(daily: np.ndarray) -> np.ndarray:
    eq = cumulative_curve(daily)
    peak = np.maximum.accumulate(eq)
    return eq / np.where(peak > 0, peak, 1.0) - 1.0


def max_drawdown_duration(underwater: np.ndarray) -> int:
    """Return the longest consecutive underwater run in trading observations.

    Missing observations break a run.  The peak observation itself is not
    counted as underwater, so a curve that starts at its high-water mark has
    duration zero.
    """
    values = np.asarray(underwater, dtype=np.float64).ravel()
    longest = current = 0
    for value in values:
        if np.isfinite(value) and value < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def annual_compounded_returns(daily: np.ndarray,
                              dates: np.ndarray) -> Dict[str, float]:
    """Compound finite daily returns by calendar year.

    ``dates`` must align with the first axis of ``daily``.  Years with no
    finite observations are omitted; no zero or forward-filled return is
    introduced.
    """
    values = np.asarray(daily, dtype=np.float64)
    labels = np.asarray(dates).ravel()
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != labels.size:
        raise ValueError("daily must be [series, dates] and align with dates")
    out: Dict[str, float] = {}
    years = np.array([str(int(value))[:4] if np.isfinite(value) else ""
                      for value in labels], dtype=object)
    for year in sorted({str(y) for y in years if y}):
        mask = years == year
        for index, row in enumerate(values):
            sample = row[mask]
            sample = sample[np.isfinite(sample)]
            if sample.size:
                out[f"G{index + 1}:{year}"] = float(np.prod(1.0 + sample) - 1.0)
    return out


def monotonicity_score(group_curves: Sequence[np.ndarray]) -> float:
    """Pearson score of terminal group return versus group rank.

    The score is positive when higher factor groups outperform lower groups;
    insufficient or non-finite terminal values remain unavailable (NaN).
    """
    terminal = []
    for curve in group_curves:
        values = np.asarray(curve, dtype=np.float64).ravel()
        terminal.append(values[-1] - 1.0 if values.size and np.isfinite(values[-1]) else np.nan)
    values = np.asarray(terminal, dtype=np.float64)
    mask = np.isfinite(values)
    if mask.sum() < 2 or np.ptp(values[mask]) <= 0:
        return float("nan")
    ranks = np.arange(values.size, dtype=np.float64)[mask]
    sample = values[mask]
    return float(np.corrcoef(ranks, sample)[0, 1])


def summarize_metrics(ls_daily: np.ndarray, ic: np.ndarray,
                      normal_ic: Optional[np.ndarray] = None,
                      cross_sectional_autocorrelation: Optional[np.ndarray] = None,
                      n_trials: int = 1
                      ) -> Dict[str, float]:
    ls = np.asarray(ls_daily, dtype=np.float64)
    ics = np.asarray(ic, dtype=np.float64)
    normal = (np.asarray(normal_ic, dtype=np.float64)
              if normal_ic is not None else np.asarray([], dtype=np.float64))
    xsec_ac = (np.asarray(cross_sectional_autocorrelation, dtype=np.float64)
               if cross_sectional_autocorrelation is not None
               else np.asarray([], dtype=np.float64))
    ls_finite = ls[np.isfinite(ls)]
    ic_finite = ics[np.isfinite(ics)]
    normal_finite = normal[np.isfinite(normal)]
    xsec_ac_finite = xsec_ac[np.isfinite(xsec_ac)]
    ls_ci = block_bootstrap_ci(ls_finite, seed=0)
    ic_ci = block_bootstrap_ci(ic_finite, seed=0)
    dsr_sharpe, dsr_skew, dsr_kurtosis = _dsr_inputs(ls_finite)
    dsr = deflated_sharpe_probability(
        dsr_sharpe, n_trials=int(n_trials), n_observations=ls_finite.size,
        skew=dsr_skew, kurtosis=dsr_kurtosis)
    return {
        "sharpe": sharpe_ratio(ls_daily),
        "calmar": calmar_ratio(ls_daily),
        "icir": icir_ratio(ic),
        "normal_ic": (float(normal_finite.mean())
                      if normal_finite.size else float("nan")),
        "normal_icir": icir_ratio(normal),
        "cross_sectional_autocorrelation": (
            float(xsec_ac_finite.mean())
            if xsec_ac_finite.size else float("nan")),
        "maxdd": max_drawdown(cumulative_curve(ls_daily)),
        "winrate": win_rate(ls_daily),
        "annualized": annualized_return(ls_daily),
        "n_return_observations": float(ls_finite.size),
        "n_ic_observations": float(ic_finite.size),
        "n_normal_ic_observations": float(normal_finite.size),
        "n_cross_sectional_autocorrelation_observations": float(
            xsec_ac_finite.size),
        "return_mean": float(ls_finite.mean()) if ls_finite.size else float("nan"),
        "ic_mean": float(ic_finite.mean()) if ic_finite.size else float("nan"),
        "return_ci95_low": ls_ci[0],
        "return_ci95_high": ls_ci[1],
        "ic_ci95_low": ic_ci[0],
        "ic_ci95_high": ic_ci[1],
        "prob_positive_return": bootstrap_probability(ls_finite, seed=0),
        "prob_positive_ic": bootstrap_probability(ic_finite, seed=0),
        "deflated_sharpe_probability": dsr,
        # NOTE: both keys carry the *same probability* — the DSR is a probability,
        # not a Sharpe ratio.  Kept under both names for the existing UI columns.
        "deflated_sharpe": dsr,
    }


# --------------------------------------------------------------------------- #
# research validation / multiple-testing controls
# --------------------------------------------------------------------------- #
def walk_forward_slices(n_observations: int, train_size: int,
                        test_size: int, step: Optional[int] = None
                        ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return chronological train/test slices with no future observations.

    The test block always starts after the train block.  ``step`` controls
    rolling movement; the default is an expanding walk-forward origin.
    """
    n = int(n_observations)
    train = int(train_size)
    test = int(test_size)
    if n < 0 or train <= 0 or test <= 0:
        raise ValueError("n_observations, train_size and test_size must be positive")
    stride = test if step is None else int(step)
    if stride <= 0:
        raise ValueError("step must be positive")
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    origin = train
    while origin + test <= n:
        out.append((np.arange(origin, dtype=np.int64),
                    np.arange(origin, origin + test, dtype=np.int64)))
        origin += stride
    return out


def purged_embargo_splits(n_observations: int, n_splits: int = 5,
                          purge: int = 0, embargo: int = 0
                          ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build contiguous CV folds with observations around test blocks removed."""
    n = int(n_observations)
    k = int(n_splits)
    purge = int(purge)
    embargo = int(embargo)
    if n < 1 or k < 2 or purge < 0 or embargo < 0:
        raise ValueError("invalid purged/embargo split parameters")
    edges = np.linspace(0, n, k + 1, dtype=np.int64)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(n, dtype=np.int64)
    for i in range(k):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi <= lo:
            continue
        test_idx = all_idx[lo:hi]
        left = max(0, lo - purge)
        right = min(n, hi + embargo)
        train_mask = np.ones(n, dtype=bool)
        train_mask[left:right] = False
        out.append((all_idx[train_mask], test_idx.copy()))
    return out


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return FDR-adjusted q-values while preserving NaN positions."""
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full(p.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    vals = np.clip(p[finite], 0.0, 1.0)
    order = np.argsort(vals, kind="stable")
    ranked = vals[order]
    m = float(len(ranked))
    q_sorted = ranked * m / np.arange(1.0, m + 1.0)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty_like(ranked)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    out[finite] = q
    return out


def capacity_sensitivity(notional: np.ndarray, daily_amount: np.ndarray,
                         *, impact_coeff: float = 0.05,
                         scales: Sequence[float] = (0.25, 0.5, 1.0, 2.0)
                         ) -> List[Dict[str, float]]:
    """Estimate square-root impact at explicit order-capacity scales.

    Inputs are real order notionals and execution-day traded amounts.  The
    result is a cost fraction of scaled notional, so it is comparable across
    account sizes without inventing a return series.
    """
    q = np.asarray(notional, dtype=np.float64)
    a = np.asarray(daily_amount, dtype=np.float64)
    if q.shape != a.shape:
        raise ValueError("notional and daily_amount shapes must match")
    base = np.isfinite(q) & np.isfinite(a) & (q > 0.0) & (a > 0.0)
    out: List[Dict[str, float]] = []
    for raw_scale in scales:
        scale = float(raw_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("capacity scales must be finite and positive")
        scaled = q * scale
        valid = base & np.isfinite(scaled)
        gross = float(scaled[valid].sum())
        if gross <= 0.0:
            cost_fraction = float("nan")
        else:
            cost = float(np.sum(
                scaled[valid] * float(impact_coeff)
                * np.sqrt(scaled[valid] / a[valid])))
            cost_fraction = cost / gross
        out.append({"scale": scale, "impact_cost_fraction": cost_fraction})
    return out


def _dsr_inputs(returns: np.ndarray) -> tuple[float, float, float]:
    """Per-period Sharpe, sample skew and Pearson kurtosis for the DSR.

    The DSR compares a Sharpe against the expected maximum of ``n_trials`` draws
    **in the same units**, so every input must be per-period.  This module used
    to pass the *annualized* Sharpe, which mixed scales and pinned the
    probability near 1 — so the overfit flag never fired.  The conventions match
    the audited reference (`evaluation/overfit.py::deflated_sharpe_ratio`).
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 3:
        return float("nan"), 0.0, 3.0
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    if std == 0.0:
        return float("nan"), 0.0, 3.0
    centred = arr - mean
    m2 = float(np.mean(centred ** 2))
    if m2 <= 0.0:
        return mean / std, 0.0, 3.0
    m3 = float(np.mean(centred ** 3))
    m4 = float(np.mean(centred ** 4))
    g1 = m3 / (m2 ** 1.5)
    skew = (math.sqrt(n * (n - 1)) / (n - 2) * g1) if n > 2 else 0.0
    excess = m4 / (m2 * m2) - 3.0
    kurt = (((n + 1) * excess + 6.0) * (n - 1) / ((n - 2) * (n - 3)) + 3.0
            if n > 3 else 3.0)
    return mean / std, float(skew), float(kurt)


def deflated_sharpe_probability(observed_sharpe: float, *, n_trials: int,
                                n_observations: int, skew: float = 0.0,
                                kurtosis: float = 3.0) -> float:
    """Probability a Sharpe exceeds the expected maximum of ``n_trials``.

    This is the analytical Deflated Sharpe approximation.  It deliberately
    returns NaN for undersampled inputs instead of manufacturing confidence.
    ``kurtosis`` is Pearson kurtosis (normal = 3).
    """
    sr = float(observed_sharpe)
    trials = int(n_trials)
    obs = int(n_observations)
    if not np.isfinite(sr) or trials < 1 or obs < 2:
        return float("nan")
    if trials == 1:
        expected_max = 0.0
    else:
        logn = math.log(float(trials))
        root = math.sqrt(max(2.0 * logn, 1e-12))
        expected_max = root - (math.log(max(math.log(float(trials)), 1e-12))
                               + math.log(4.0 * math.pi)) / (2.0 * root)
    variance = (1.0 - float(skew) * sr
                + ((float(kurtosis) - 1.0) / 4.0) * sr * sr)
    se = math.sqrt(max(variance, 1e-12) / max(obs - 1, 1))
    z = (sr - expected_max) / se
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def _block_length(n: int) -> int:
    """Conservative moving-block length for daily financial observations."""
    if n < 2:
        return 1
    return max(2, min(20, int(round(math.sqrt(n)))))


def _bootstrap_means(values: np.ndarray, *, seed: int = 0,
                     samples: int = 1000) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.empty(0, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    block = _block_length(x.size)
    starts = rng.integers(0, x.size, size=(int(samples),
                                            int(math.ceil(x.size / block))))
    offsets = np.arange(block, dtype=np.int64)
    idx = (starts[..., None] + offsets).reshape(int(samples), -1) % x.size
    idx = idx[:, :x.size]
    return x[idx].mean(axis=1)


def block_bootstrap_ci(values: np.ndarray, *, seed: int = 0,
                       samples: int = 1000) -> Tuple[float, float]:
    """95% moving-block bootstrap CI for a mean; NaN when undersampled."""
    means = _bootstrap_means(values, seed=seed, samples=samples)
    if means.size < 100:
        return float("nan"), float("nan")
    return (float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)))


def bootstrap_probability(values: np.ndarray, *, seed: int = 0,
                          samples: int = 1000) -> float:
    """Probability that a block-bootstrap mean is positive."""
    means = _bootstrap_means(values, seed=seed, samples=samples)
    if means.size < 100:
        return float("nan")
    return float(np.mean(means > 0.0))
