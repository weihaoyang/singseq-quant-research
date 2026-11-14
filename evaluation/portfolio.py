"""Multi-factor portfolio construction and capacity (institutional finishing).

Turns a set of factor scores into a portfolio and answers the two questions a
desk always asks next: *how do I weight them?* and *how much money fits?*

* ``factor_return_series`` — per-date long-short quantile spread for each factor
  (a cheap, interpretable return proxy; no full backtest needed).
* ``risk_covariance`` — annualised covariance of those factor returns.
* ``optimize`` — constrained mean-variance / max-Sharpe weights (scipy).
* ``diversification_ratio`` — weighted-average vol / portfolio vol.
* ``capacity`` — rough AUM the book can hold at a given ADV participation.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import optimize as _opt

__all__ = ["PORTFOLIO_VERSION", "factor_return_series", "risk_covariance",
           "optimize", "diversification_ratio", "capacity"]

PORTFOLIO_VERSION = "portfolio-1"
_TRADING_DAYS = 252


def _forward_returns(work: pd.DataFrame, horizon: int) -> pd.Series:
    close = work["close"].astype(float)
    return close.groupby(work["ts_code"], sort=False).shift(-horizon) / close - 1.0


def factor_return_series(panel: pd.DataFrame, factors: Dict[str, Sequence[float]], *,
                         horizon: int = 1, quantile: float = 0.2,
                         min_names: int = 20) -> pd.DataFrame:
    """Long-short quantile spread per date for each factor.

    On each date sort names by the factor, go long the top ``quantile`` and short
    the bottom ``quantile`` (equal weight) on the forward return.  Rows are dates,
    columns are factor names.
    """
    work = panel.copy()
    work["ts_code"] = work["ts_code"].astype(str)
    work["trade_date"] = work["trade_date"].astype(str)
    work = work.sort_values(["ts_code", "trade_date"], kind="mergesort")
    idx = work.index
    fwd = _forward_returns(work, horizon)
    cols = {}
    for name, values in factors.items():
        series = (values.reindex(idx) if isinstance(values, pd.Series)
                  else pd.Series(np.asarray(values, dtype=float), index=panel.index).reindex(idx))
        frame = pd.DataFrame({"d": work["trade_date"].to_numpy(),
                              "f": np.asarray(series, dtype=float),
                              "y": fwd.to_numpy(dtype=float)}).dropna()
        per_date = []
        for _, sub in frame.groupby("d", sort=False):
            if len(sub) < min_names:
                continue
            k = max(1, int(len(sub) * quantile))
            ordered = sub.sort_values("f")
            short = ordered["y"].iloc[:k].mean()
            long = ordered["y"].iloc[-k:].mean()
            per_date.append((sub["d"].iloc[0], long - short))
        if per_date:
            d, r = zip(*per_date)
            cols[name] = pd.Series(r, index=pd.Index(d, name="trade_date"))
    return pd.DataFrame(cols).sort_index()


def risk_covariance(factor_returns: pd.DataFrame) -> pd.DataFrame:
    return factor_returns.cov() * _TRADING_DAYS


def optimize(factor_returns: pd.DataFrame, *, objective: str = "max_sharpe",
             long_only: bool = False, max_weight: float = 0.5,
             max_gross: float = 1.0, risk_free: float = 0.0) -> Dict[str, object]:
    """Constrained mean-variance weights over the factor-return columns."""
    rets = factor_returns.dropna()
    if rets.shape[1] < 2 or rets.shape[0] < 20:
        raise ValueError("optimize: need >=2 factors and >=20 dates")
    mu = rets.mean().to_numpy() * _TRADING_DAYS
    cov = rets.cov().to_numpy() * _TRADING_DAYS
    cov = cov + np.eye(cov.shape[0]) * 1e-10          # jitter for PSD
    n = mu.size
    lo = 0.0 if long_only else -max_weight
    bounds = [(lo, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(np.abs(w)) - max_gross)}]
    if long_only:
        constraints.append({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)})

    def neg_sharpe(w):
        vol = float(np.sqrt(w @ cov @ w))
        return -((w @ mu - risk_free) / vol) if vol > 0 else 1e9

    def variance(w):
        return float(w @ cov @ w)

    x0 = np.full(n, (1.0 / n) if long_only else 0.0)
    if not long_only:
        x0 = np.full(n, max_gross / n)
    func = neg_sharpe if objective == "max_sharpe" else variance
    res = _opt.minimize(func, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                        options={"maxiter": 300, "ftol": 1e-9})
    weights = res.x if res.success else x0
    weights = np.asarray(weights, dtype=float)
    names = list(rets.columns)
    expected = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    return {
        "version": PORTFOLIO_VERSION,
        "weights": {name: round(float(w), 6) for name, w in zip(names, weights)},
        "expected_return": round(expected, 6),
        "volatility": round(vol, 6),
        "sharpe": round((expected - risk_free) / vol, 6) if vol > 0 else None,
        "converged": bool(res.success),
        "objective": objective,
    }


def diversification_ratio(weights: Dict[str, float], cov: pd.DataFrame) -> float:
    names = [n for n in weights if n in cov.columns]
    w = np.asarray([weights[n] for n in names], dtype=float)
    c = cov.loc[names, names].to_numpy()
    num = float(np.sum(np.abs(w) * np.sqrt(np.diag(c))))
    den = float(np.sqrt(w @ c @ w))
    return num / den if den > 0 else 1.0


def capacity(panel: pd.DataFrame, weights: Dict[str, float], *, adv_col: str = "amount",
             participation: float = 0.1, turnover: float = 1.0) -> Dict[str, float]:
    """Rough one-day AUM capacity: sum(|w|) * participation * median ADV."""
    if adv_col not in panel.columns:
        return {"aum_cny": 0.0, "participation": participation, "median_adv_cny": 0.0}
    adv = pd.to_numeric(panel[adv_col], errors="coerce").dropna()
    median_adv = float(adv.median()) if not adv.empty else 0.0
    gross = float(sum(abs(w) for w in weights.values()))
    aum = gross * participation * median_adv * max(0.0, turnover)
    return {"aum_cny": aum, "participation": participation,
            "median_adv_cny": median_adv, "version": PORTFOLIO_VERSION}
