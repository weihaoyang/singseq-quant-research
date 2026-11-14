"""Versioned, pairwise-complete correlations of daily cross-sectional ranks."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

VERSION = "full-library-rank-correlation-2"
ORTHOGONAL_VERSION = "full-bank-ols-eigen-1"
MIN_NAMES = 30
MIN_DATES = 20


def daily_ranks(values: np.ndarray) -> np.ndarray:
    """Rank stocks independently on each date, retaining missing observations."""
    values = np.where(np.isfinite(values), values, np.nan)
    return rankdata(values, axis=-1, method="average", nan_policy="omit").astype(np.float32)


def cross_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pearson of cross-sectional ranks on each pair's common valid names.

    Inputs are [factor, stock] rank matrices for one date. This is the same
    rank-then-pairwise-complete definition used by evaluation.novelty.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    lm = np.isfinite(left).astype(float)
    rm = np.isfinite(right).astype(float)
    x = np.where(lm, left, 0.0)
    y = np.where(rm, right, 0.0)
    count = lm @ rm.T
    safe_count = np.maximum(count, 1)
    sx = x @ rm.T
    sy = lm @ y.T
    covariance = x @ y.T - sx * sy / safe_count
    vx = x ** 2 @ rm.T - sx ** 2 / safe_count
    vy = lm @ (y ** 2).T - sy ** 2 / safe_count
    denominator = np.sqrt(np.maximum(vx, 0) * np.maximum(vy, 0))
    result = np.full_like(covariance, np.nan)
    np.divide(covariance, denominator, out=result,
              where=(count >= MIN_NAMES) & (denominator > 1e-10))
    return np.clip(result, -1, 1)


def summarize(sum_abs: np.ndarray, sum_signed: np.ndarray, counts: np.ndarray,
              names: list[str], *, exclude: str | None = None) -> dict:
    pairs = []
    for index, name in enumerate(names):
        if name == exclude or counts[index] < MIN_DATES:
            continue
        pairs.append({"name": name, "absolute_correlation": float(sum_abs[index] / counts[index]),
                      "signed_correlation": float(sum_signed[index] / counts[index]),
                      "dates": int(counts[index])})
    pairs.sort(key=lambda item: (-item["absolute_correlation"], item["name"]))
    return {"version": VERSION, "status": "COMPLETE" if pairs else "INSUFFICIENT_DATA",
            "compared_count": len(pairs), "baseline_count": len(names) - int(exclude in names),
            "max_abs_correlation": pairs[0]["absolute_correlation"] if pairs else None,
            "closest_baseline": pairs[0]["name"] if pairs else None,
            "top_correlations": pairs[:10], "minimum_names": MIN_NAMES, "minimum_dates": MIN_DATES}


def residual_information(factor: np.ndarray, baselines: np.ndarray, target: np.ndarray) -> tuple[float, float | None] | None:
    """Same-date least-squares residual against the full bank, then residual IC.

    No return label enters the exposure fit. Complete observations are required,
    with at least twice as many stocks as baseline columns. No missing exposure
    is imputed. A symmetric pseudoinverse handles duplicate/collinear baselines.
    """
    valid = np.isfinite(factor) & np.isfinite(target) & np.isfinite(baselines).all(axis=0)
    if valid.sum() < max(MIN_NAMES, 2 * baselines.shape[0]):
        return None
    x = baselines[:, valid].T.astype(float)
    deviation = x.std(axis=0)
    x = x[:, deviation > 1e-10]
    deviation = deviation[deviation > 1e-10]
    if x.shape[1] == 0:
        return None
    x = (x - x.mean(axis=0)) / deviation
    y = factor[valid].astype(float)
    y -= y.mean()
    total = float(y @ y)
    if total < 1e-10:
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(x.T @ x)
    tolerance = np.finfo(float).eps * max(x.shape)
    keep = eigenvalues > tolerance * eigenvalues[-1]
    basis = eigenvectors[:, keep]
    coefficients = basis @ ((basis.T @ (x.T @ y)) / eigenvalues[keep])
    residual = y - x @ coefficients
    fraction = float(residual @ residual) / total
    explained = max(0.0, min(1.0, 1 - fraction))
    if fraction <= tolerance:
        return explained, None
    ranks = daily_ranks(np.vstack([residual, target[valid]]))
    ic = cross_correlations(ranks[:1], ranks[1:])[0, 0]
    return explained, float(ic) if np.isfinite(ic) else None
