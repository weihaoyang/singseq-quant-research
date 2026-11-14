"""Intraday (bar-interval) long-only backtest engine.

Minute/hour bars, **next-bar-open** execution, T+1 lock with a tradable base
position, one-word limit locks, lot/tick/participation controls, and the same
friction constants as the daily engine (``FrictionConfig``). Metrics reuse
``engine.metrics``.

Matching convention (same as the daily engine): a signal formed on the close
of bar *t* executes at the open of bar *t+1*; costs are charged on *t+1* with
the date-aware stamp-duty policy.

Single tradable symbol per run (the canonical intraday T0 use case). Long-only
by default; ``allow_short`` is refused to stay consistent with the product
contract.

Purity: numpy + pandas + stdlib.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from . import metrics as M
from .friction import FrictionConfig, one_word_limit_down, one_word_limit_up

__all__ = ["IntradayResult", "run_intraday"]


@dataclass
class IntradayResult:
    timestamps: np.ndarray
    equity_curve: np.ndarray
    returns: np.ndarray
    turnover: float
    trade_count: int
    round_trips: int
    costs: Dict[str, float]
    blocked: Dict[str, int]
    metrics: Dict[str, float]


def _round_tick(price: float, tick: float) -> float:
    if tick > 0 and np.isfinite(price):
        return round(price / tick) * tick
    return price


def _impact(price: float, qty: float, amount: float, coeff: float) -> float:
    if amount <= 0 or qty <= 0 or not np.isfinite(price):
        return 0.0
    frac = min(qty * price / amount, 1.0)
    return price * coeff * float(np.sqrt(frac))


def run_intraday(bars: pd.DataFrame, factor: Any, *,
                 config: Optional[FrictionConfig] = None,
                 buy_threshold: float = 0.0, sell_threshold: float = 0.0,
                 order_shares: int = 10_000, base_shares: int = 0,
                 up_limit_pct: float = 0.10,
                 allow_short: bool = False) -> IntradayResult:
    """Run one intraday long-only backtest over ``bars``.

    ``bars`` needs ``trade_time`` (sortable), ``open/high/low/close/vol/amount``
    and either ``trade_date`` or a date extractable from ``trade_time``.
    ``factor`` is a per-bar signal array aligned to the sorted bars.
    """
    if allow_short:
        raise ValueError("intraday long-short is retired from the product contract")
    cfg = config or FrictionConfig()
    df = bars.sort_values("trade_time").reset_index(drop=True)
    n = int(len(df))
    if n < 3:
        raise ValueError("intraday panel too small (need >= 3 bars)")
    sig = np.asarray(factor, dtype=np.float64)
    if sig.shape[0] != n:
        raise ValueError(f"factor length {sig.shape[0]} != bars {n}")

    o = np.asarray(df["open"], dtype=np.float64)
    h = np.asarray(df["high"], dtype=np.float64)
    l = np.asarray(df["low"], dtype=np.float64)
    c = np.asarray(df["close"], dtype=np.float64)
    vol = np.asarray(df["vol"], dtype=np.float64)
    amount = np.asarray(df["amount"], dtype=np.float64)
    if "up_limit" in df.columns:
        up = np.asarray(df["up_limit"], dtype=np.float64)
        down = np.asarray(df["down_limit"], dtype=np.float64)
    else:
        up = np.round(c * (1.0 + up_limit_pct), 2)
        down = np.round(c * (1.0 - up_limit_pct), 2)
    if "trade_date" in df.columns:
        day = df["trade_date"].astype(str).to_numpy()
    else:
        day = pd.to_datetime(df["trade_time"]).dt.strftime("%Y%m%d").to_numpy()

    up_lock = one_word_limit_up(o, h, l, c, up)
    down_lock = one_word_limit_down(o, h, l, c, down)

    available = float(base_shares)
    frozen = 0.0
    cash = float(cfg.initial_cash)
    lot = float(cfg.min_lot)
    costs = {"stamp": 0.0, "commission": 0.0, "slippage": 0.0}
    blocked = {"limit_up_buy": 0, "limit_down_sell": 0, "untradeable": 0,
               "partial_fill": 0, "t1_locked": 0}
    trade_count = 0
    round_trips = 0
    equity = np.full(n, np.nan, dtype=np.float64)
    equity[0] = cash + (available + frozen) * c[0]

    prev_day = day[0]

    for t in range(n - 1):
        e = t + 1
        if day[e] != prev_day:
            available += frozen
            frozen = 0.0
            prev_day = day[e]
        price = _round_tick(float(o[e]), cfg.tick_size)
        if not np.isfinite(price) or price <= 0 or vol[e] <= 0 or not np.isfinite(amount[e]):
            blocked["untradeable"] += 1
            equity[e] = cash + (available + frozen) * c[e]
            continue

        want = 0.0
        s = float(sig[t]) if np.isfinite(sig[t]) else 0.0
        if s >= buy_threshold:
            want = 1.0
        elif s <= sell_threshold:
            want = -1.0

        if want > 0:
            if up_lock[e]:
                blocked["limit_up_buy"] += 1
            else:
                qty = np.floor(order_shares / lot + 1e-9) * lot
                cap = np.floor(max(amount[e], 0.0) * cfg.max_participation
                               / price / lot + 1e-9) * lot
                qty = float(min(qty, cap))
                if qty <= 0:
                    blocked["partial_fill"] += 1
                else:
                    fill = price + _impact(price, qty, amount[e], cfg.impact_coeff)
                    total = fill * qty * (1.0 + cfg.commission_rate)
                    if cash >= total:
                        cash -= total
                        frozen += qty
                        trade_count += 1
                        costs["commission"] += fill * qty * cfg.commission_rate
                        costs["slippage"] += (fill - price) * qty
                    else:
                        blocked["untradeable"] += 1
        elif want < 0:
            if down_lock[e]:
                blocked["limit_down_sell"] += 1
            elif available < lot:
                # A same-day purchase is frozen until the next trading day.
                blocked["t1_locked"] += 1
            else:
                qty = np.floor(min(order_shares, available) / lot + 1e-9) * lot
                cap = np.floor(max(amount[e], 0.0) * cfg.max_participation
                               / price / lot + 1e-9) * lot
                qty = float(min(qty, cap))
                if qty <= 0:
                    blocked["partial_fill"] += 1
                else:
                    fill = price - _impact(price, qty, amount[e], cfg.impact_coeff)
                    gross = fill * qty
                    stamp_rate = cfg.stamp_duty_rate(int(day[e]) if day[e].isdigit() else 0)
                    stamp = gross * stamp_rate
                    commission = gross * cfg.commission_rate
                    cash += gross - stamp - commission
                    available -= qty
                    trade_count += 1
                    round_trips += 1
                    costs["stamp"] += stamp
                    costs["commission"] += commission
                    costs["slippage"] += (price - fill) * qty
        equity[e] = cash + (available + frozen) * c[e]

    finite = np.isfinite(equity)
    returns = np.full(n, np.nan, dtype=np.float64)
    if finite.sum() > 1:
        eq = equity[finite]
        returns[finite] = np.diff(eq, prepend=eq[0]) / np.where(eq > 0, eq, np.nan)
    summary = {
        "sharpe": M.sharpe_ratio(returns),
        "maxdd": M.max_drawdown(equity[np.isfinite(equity)]),
        "annualized": M.annualized_return(returns),
        "winrate": M.win_rate(returns),
        "total_return": (float(equity[finite][-1] / equity[finite][0] - 1.0)
                         if finite.sum() > 1 else float("nan")),
    }
    return IntradayResult(
        timestamps=df["trade_time"].to_numpy(),
        equity_curve=equity, returns=returns,
        turnover=float(sum(abs(v) for v in costs.values())) / max(cfg.initial_cash, 1.0),
        trade_count=int(trade_count), round_trips=int(round_trips),
        costs={k: round(float(v), 6) for k, v in costs.items()},
        blocked=dict(blocked),
        metrics={k: (None if v is None or not np.isfinite(v) else round(float(v), 6))
                 for k, v in summary.items()},
    )
