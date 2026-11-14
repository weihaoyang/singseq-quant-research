"""Backtest-overfitting gates: Deflated Sharpe Ratio, PBO (CSCV) and a
multiple-testing haircut.

These answer the question a desk actually asks about a mined factor: *is this
real, or is it the best of many tries?*  They are the institutional bar the
single-shot IC/ICIR does not clear.

References
----------
* Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014).
* Bailey, Borwein, López de Prado & Zhu, "The Probability of Backtest
  Overfitting" (CSCV, 2017).
* Harvey, Liu & Zhu, "…and the Cross-Section of Expected Returns" (2016).

All functions are pure numpy/scipy and return a dict with the inputs they used,
so a reported number is reproducible.
"""
from __future__ import annotations

import math
import numbers
from itertools import combinations
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "OVERFIT_VERSION",
    "sharpe_ratio",
    "deflated_sharpe_ratio",
    "pbo_cscv",
    "benjamini_hochberg",
    "bonferroni_t_threshold",
    "multiple_testing_haircut",
]

OVERFIT_VERSION = "overfit-1"
_EULER = 0.5772156649015329
_EPS = 1e-12
# CSCV evaluates every half-split combination.  Keep malformed or untrusted
# reports from turning a diagnostic call into an unbounded combinatorial job.
_MAX_CSCV_PARTITIONS = 100_000
_MAX_CSCV_CELLS = 5_000_000
_MAX_CSCV_SPLITS = 32


def _positive_integer(value: object) -> int | None:
    """Coerce only an actual positive integer; never round a count."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return None
    result = int(value)
    return result if result >= 1 else None


def _invalid_dsr(*, n_trials: object, n_obs: int, reason: str) -> Dict[str, object]:
    return {
        "dsr": 0.0,
        "sr": 0.0,
        "sr_annualized": 0.0,
        "expected_max_sr": 0.0,
        "n_trials": int(n_trials) if _positive_integer(n_trials) is not None else None,
        "n_obs": int(n_obs),
        "skew": 0.0,
        "kurtosis": 0.0,
        "var_sr": 0.0,
        "var_sr_source": "invalid_input",
        "valid": False,
        "reason": reason,
        "version": OVERFIT_VERSION,
    }


def sharpe_ratio(returns: Sequence[float], *, periods_per_year: int = 252) -> float:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2 or arr.std(ddof=1) == 0.0:
        return 0.0
    return float(arr.mean() / arr.std(ddof=1) * math.sqrt(periods_per_year))


def _expected_max_sharpe(var_sr: float, n_trials: int) -> float:
    if n_trials <= 1 or var_sr <= 0.0:
        return 0.0
    a = stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_sr) * ((1.0 - _EULER) * a + _EULER * b)


def deflated_sharpe_ratio(returns: Sequence[float], n_trials: int, *,
                          var_sr: Optional[float] = None,
                          trial_returns: Optional[Sequence[Sequence[float]]] = None,
                          periods_per_year: int = 252) -> Dict[str, object]:
    """Deflated Sharpe Ratio in [0,1] (probability the true SR exceeds the
    expected max under ``n_trials`` independent trials).

    ``var_sr`` is the variance of the **per-period** trial Sharpes.  Supply it,
    or pass ``trial_returns`` (a list of return series) to estimate it; when
    neither is given a neutral single-strategy assumption ``var_sr = sr**2`` is
    used and reported as ``var_sr_source='assumed'``.
    """
    valid_trials = _positive_integer(n_trials)
    if valid_trials is None:
        return _invalid_dsr(n_trials=n_trials, n_obs=0,
                            reason="n_trials must be a positive integer")
    valid_periods = _positive_integer(periods_per_year)
    if valid_periods is None:
        return _invalid_dsr(n_trials=valid_trials, n_obs=0,
                            reason="periods_per_year must be a positive integer")
    try:
        arr = np.asarray(returns, dtype=float)
    except (TypeError, ValueError):
        return _invalid_dsr(n_trials=valid_trials, n_obs=0,
                            reason="returns must be a one-dimensional numeric series")
    if arr.ndim != 1 or not np.isfinite(arr).all():
        return _invalid_dsr(n_trials=valid_trials, n_obs=int(arr.size),
                            reason="returns must be finite and one-dimensional")
    n = int(arr.size)
    if n < 3 or arr.std(ddof=1) == 0.0:
        return {"dsr": 0.0, "sr": 0.0, "sr_annualized": 0.0, "expected_max_sr": 0.0,
                "n_trials": valid_trials, "n_obs": n, "skew": 0.0, "kurtosis": 0.0,
                "var_sr": 0.0, "var_sr_source": "insufficient_data",
                "valid": False, "reason": "insufficient or zero-variance returns",
                "version": OVERFIT_VERSION}
    sr = float(arr.mean() / arr.std(ddof=1))          # per-period (non-annualised)
    skew = float(stats.skew(arr, bias=False)) if n > 2 else 0.0
    kurt = float(stats.kurtosis(arr, fisher=False, bias=False)) if n > 3 else 3.0
    source = "provided"
    if var_sr is None and trial_returns is not None:
        try:
            trial_arrays = [np.asarray(r, dtype=float) for r in trial_returns]
        except (TypeError, ValueError):
            return _invalid_dsr(n_trials=valid_trials, n_obs=n,
                                reason="trial_returns must be finite one-dimensional series")
        if (len(trial_arrays) != valid_trials or
                any(item.ndim != 1 or item.size != n or
                    not np.isfinite(item).all() for item in trial_arrays)):
            return _invalid_dsr(n_trials=valid_trials, n_obs=n,
                                reason=("trial_returns must contain exactly n_trials "
                                        "aligned finite one-dimensional series"))
        srs = [sharpe_ratio(r, periods_per_year=1) for r in trial_arrays]
        srs = [s for s in srs if np.isfinite(s)]
        var_sr = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
        source = "trial_returns"
    if var_sr is None:
        var_sr = sr * sr
        source = "assumed"
    try:
        var_sr = float(var_sr)
    except (TypeError, ValueError):
        return _invalid_dsr(n_trials=valid_trials, n_obs=n,
                            reason="var_sr must be a finite non-negative number")
    if not math.isfinite(var_sr) or var_sr < 0.0:
        return _invalid_dsr(n_trials=valid_trials, n_obs=n,
                            reason="var_sr must be a finite non-negative number")
    emax = _expected_max_sharpe(var_sr, valid_trials)
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    denom = max(denom, _EPS)
    z = (sr - emax) * math.sqrt(n - 1) / math.sqrt(denom)
    return {
        "dsr": float(stats.norm.cdf(z)),
        "sr": sr,
        "sr_annualized": sr * math.sqrt(valid_periods),
        "expected_max_sr": emax,
        "z": float(z),
        "n_trials": valid_trials,
        "n_obs": n,
        "skew": skew,
        "kurtosis": kurt,
        "var_sr": float(var_sr),
        "var_sr_source": source,
        "valid": True,
        "version": OVERFIT_VERSION,
    }


def _sharpe_rows(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)
    return np.divide(mean, std, out=np.zeros_like(mean), where=std > 0)


def pbo_cscv(returns_matrix: Sequence[Sequence[float]], *, n_splits: int = 16) -> Dict[str, object]:
    """Probability of Backtest Overfitting via CSCV.

    ``returns_matrix`` is shape ``(T, N)`` (T observations, N strategies tried).
    Splits T into ``n_splits`` contiguous blocks, forms every half-IS/half-OOS
    partition, picks the IS-best strategy and asks where it ranks OOS.  PBO is
    the fraction of partitions where the IS-best lands below the OOS median.
    """
    try:
        matrix = np.asarray(returns_matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("returns_matrix must be 2-D (T x N)") from exc
    if matrix.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (T x N)")
    split_count = _positive_integer(n_splits)
    if split_count is None or split_count < 2 or split_count % 2:
        raise ValueError("n_splits must be an even integer >= 2")
    t, n = matrix.shape
    # A CSCV result is evidence only when every strategy has an aligned,
    # finite return series.  Letting NaN/Inf reach argmax/ranking can silently
    # manufacture a PBO value from an invalid candidate matrix.
    if not np.isfinite(matrix).all():
        return {"pbo": None, "n_partitions": 0, "n_strategies": n, "n_obs": t,
                "logits": [], "version": OVERFIT_VERSION,
                "note": "non-finite returns matrix; candidate family evidence invalid"}
    if n < 2 or t < 2 * split_count:
        return {"pbo": None, "n_partitions": 0, "n_strategies": n, "n_obs": t,
                "logits": [], "version": OVERFIT_VERSION,
                "note": "insufficient data for CSCV"}
    if split_count > _MAX_CSCV_SPLITS:
        return {"pbo": None, "n_partitions": 0, "n_strategies": n, "n_obs": t,
                "logits": [], "version": OVERFIT_VERSION,
                "note": "CSCV input exceeds bounded diagnostic resource limits"}
    partitions = math.comb(split_count, split_count // 2)
    if t * n > _MAX_CSCV_CELLS or partitions > _MAX_CSCV_PARTITIONS:
        return {"pbo": None, "n_partitions": 0, "n_strategies": n, "n_obs": t,
                "logits": [], "version": OVERFIT_VERSION,
                "note": "CSCV input exceeds bounded diagnostic resource limits"}
    if not np.all(np.std(matrix, axis=0, ddof=1) > 0.0):
        return {"pbo": None, "n_partitions": 0, "n_strategies": n, "n_obs": t,
                "logits": [], "version": OVERFIT_VERSION,
                "note": "zero-variance candidate series; candidate family evidence invalid"}
    blocks = np.array_split(matrix, split_count, axis=0)
    half = split_count // 2
    logits = []
    below = 0
    partition_count = 0
    for combo in combinations(range(split_count), half):
        is_rows = np.concatenate([blocks[i] for i in combo], axis=0)
        oos_rows = np.concatenate([blocks[i] for i in range(split_count) if i not in combo], axis=0)
        is_sr = _sharpe_rows(is_rows)
        oos_sr = _sharpe_rows(oos_rows)
        if (
            not np.isfinite(is_sr).all()
            or not np.isfinite(oos_sr).all()
            or not np.all(np.std(is_rows, axis=0, ddof=1) > 0.0)
            or not np.all(np.std(oos_rows, axis=0, ddof=1) > 0.0)
        ):
            return {"pbo": None, "n_partitions": 0, "n_strategies": n, "n_obs": t,
                    "logits": [], "version": OVERFIT_VERSION,
                    "note": "zero-variance CSCV fold; candidate family evidence invalid"}
        best = int(np.argmax(is_sr))
        # rank of the IS-best strategy within the OOS distribution (1..N)
        rank = int(np.sum(oos_sr <= oos_sr[best]))
        omega = rank / (n + 1)
        omega = min(max(omega, 1.0 / (n + 1)), 1.0 - 1.0 / (n + 1))
        logit = math.log(omega / (1.0 - omega))
        logits.append(logit)
        if logit <= 0.0:
            below += 1
        partition_count += 1
    return {
        "pbo": below / partition_count,
        "n_partitions": partition_count,
        "n_strategies": n,
        "n_obs": t,
        "logits": logits,
        "version": OVERFIT_VERSION,
    }


def benjamini_hochberg(pvalues: Sequence[float], *, alpha: float = 0.05) -> Dict[str, object]:
    """Benjamini–Hochberg FDR control. Returns rejected indices + thresholds."""
    try:
        p = np.asarray(pvalues, dtype=float)
    except (TypeError, ValueError):
        return {"rejected": [], "threshold": 0.0, "n": 0,
                "valid": False, "reason": "pvalues must be a finite one-dimensional probability series",
                "version": OVERFIT_VERSION}
    if p.ndim != 1 or p.size == 0 or not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        return {"rejected": [], "threshold": 0.0, "n": int(p.size),
                "valid": False, "reason": "pvalues must be finite and within [0, 1]",
                "version": OVERFIT_VERSION}
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError):
        alpha_value = float("nan")
    if not math.isfinite(alpha_value) or not 0.0 < alpha_value <= 1.0:
        return {"rejected": [], "threshold": 0.0, "n": int(p.size),
                "valid": False, "reason": "alpha must be within (0, 1]",
                "version": OVERFIT_VERSION}
    order = np.argsort(p)
    ranked = p[order]
    m = p.size
    crit = alpha_value * (np.arange(1, m + 1) / m)
    passed = ranked <= crit
    if not passed.any():
        return {"rejected": [], "threshold": 0.0, "n": int(m),
                "valid": True, "version": OVERFIT_VERSION}
    k = int(np.max(np.nonzero(passed)[0]))
    threshold = float(ranked[k])
    rejected = [int(i) for i in order[: k + 1]]
    return {"rejected": rejected, "threshold": threshold, "n": int(m),
            "valid": True,
            "version": OVERFIT_VERSION}


def bonferroni_t_threshold(n_trials: int, *, alpha: float = 0.05) -> float:
    """Two-sided Bonferroni t-threshold for ``n_trials`` tests at ``alpha``."""
    n = max(1, int(n_trials))
    return float(stats.norm.ppf(1.0 - alpha / (2.0 * n)))


def multiple_testing_haircut(t_stat: float, n_trials: int, *, alpha: float = 0.05) -> Dict[str, object]:
    """Does a t-stat survive a Bonferroni test over ``n_trials`` attempts?"""
    valid_trials = _positive_integer(n_trials)
    try:
        t_value = float(t_stat)
        alpha_value = float(alpha)
    except (TypeError, ValueError):
        valid_trials = None
        t_value = 0.0
        alpha_value = float(alpha) if isinstance(alpha, (int, float)) else 0.0
    if valid_trials is None or not math.isfinite(t_value) or not math.isfinite(alpha_value) or not 0.0 < alpha_value <= 1.0:
        return {
            "t_stat": t_value,
            "n_trials": valid_trials,
            "alpha": alpha_value,
            "bonferroni_threshold": None,
            "survives": False,
            "valid": False,
            "reason": "t_stat, n_trials and alpha must be finite with n_trials >= 1 and alpha in (0, 1]",
            "version": OVERFIT_VERSION,
        }
    threshold = bonferroni_t_threshold(valid_trials, alpha=alpha_value)
    return {
        "t_stat": t_value,
        "n_trials": valid_trials,
        "alpha": alpha_value,
        "bonferroni_threshold": threshold,
        "survives": abs(t_value) >= threshold,
        "valid": True,
        "version": OVERFIT_VERSION,
    }
