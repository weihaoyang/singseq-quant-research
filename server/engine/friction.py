"""Friction model + trading state machine (S3, ARCHITECTURE.md 3.4 / H7).

Real rules only, all date-aware:

* Stamp duty 10 bp -> 5 bp on sell side from **2023-08-27** (real policy
  cut); before that date 10 bp.
* Commission 3 bp per side (configurable).
* Market impact / slippage: ``0.1 * sqrt(order_notional / daily_amount)``
  applied per side on the traded notional (square-root law).
* One-word limit-up (一字涨停): **BUY blocked**, holders keep position
  (position lock, i.e. SELL is allowed but we still cannot increase).
* One-word limit-down (一字跌停): **SELL blocked**, position locked.
* Suspended / ST / < 60d listing age: not tradable (target forced to 0
  for newly-ineligible names, realised only when the bar is tradable).

Purity: numpy + stdlib only (no network/DB/framework imports).

Matching convention (shared with ``engine.backtest``): the caller forms
the target at the close of day *t* and passes the bars of the
**execution day t+1** (next-day open fill).  The state machine itself is
date-agnostic: every rule below is evaluated on whatever ``date`` and
bars it is handed - the stamp-duty policy date, the 一字 locks, the
suspension mask and the daily amount used by the impact model all refer
to the day the order actually trades.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np

STAMP_CUT_DATE = 20230827          # real A-share stamp-duty cut 10bp -> 5bp
STAMP_RATE_BEFORE = 10e-4
STAMP_RATE_AFTER = 5e-4
COMMISSION_RATE_DEFAULT = 3e-4     # 3 bp per side
IMPACT_COEFF_DEFAULT = 0.1         # 0.1 * sqrt(order / daily_amount)
LIMIT_ATOL = 0.011                 # flat-bar tolerance vs real stk_limit


@dataclass
class FrictionConfig:
    commission_rate: float = COMMISSION_RATE_DEFAULT
    impact_coeff: float = IMPACT_COEFF_DEFAULT
    stamp_cut_date: int = STAMP_CUT_DATE
    stamp_rate_before: float = STAMP_RATE_BEFORE
    stamp_rate_after: float = STAMP_RATE_AFTER
    portfolio_value: float = 1e8
    block_limit_moves: bool = True
    # Portfolio accounting and order controls.  Values are deliberately
    # explicit so a replay can be reproduced from its manifest.
    initial_cash: Optional[float] = None
    min_lot: int = 100
    tick_size: float = 0.01
    max_participation: float = 1.0
    max_gross_leverage: float = 2.0
    max_short_leverage: float = 1.0
    margin_requirement: float = 0.5
    # A target is expressed in equity weights, then converted to exchange
    # lots at the execution tick.  The resulting integer-lot book can exceed
    # the ideal weight by a small, deterministic rounding/mark-to-market
    # amount; keep that operational buffer explicit rather than treating it
    # as an implicit loan.  Material breaches still fail closed.
    margin_tolerance_ratio: float = 0.005
    financing_rate_annual: float = 0.03
    borrow_rate_annual: float = 0.08
    allow_short: bool = True
    enforce_risk_limits: bool = True

    def __post_init__(self) -> None:
        if self.initial_cash is None:
            self.initial_cash = float(self.portfolio_value)
        if self.portfolio_value <= 0 or self.initial_cash <= 0:
            raise ValueError("portfolio_value and initial_cash must be positive")
        if self.min_lot < 1:
            raise ValueError("min_lot must be >= 1")
        if self.tick_size < 0:
            raise ValueError("tick_size must be >= 0")
        if not 0 < self.max_participation <= 1:
            raise ValueError("max_participation must be in (0, 1]")
        if self.max_gross_leverage <= 0 or self.max_short_leverage < 0:
            raise ValueError("invalid leverage limits")
        if not 0 < self.margin_requirement <= 1:
            raise ValueError("margin_requirement must be in (0, 1]")
        if not 0 <= self.margin_tolerance_ratio < 1:
            raise ValueError("margin_tolerance_ratio must be in [0, 1)")

    def stamp_duty_rate(self, trade_date: int) -> float:
        """Date-aware stamp duty (sell side), 10bp before cut, 5bp after."""
        try:
            d = int(trade_date)
        except (TypeError, ValueError):
            return self.stamp_rate_after
        return self.stamp_rate_before if d < self.stamp_cut_date else self.stamp_rate_after


def stamp_duty_rate(trade_date: int,
                    cut_date: int = STAMP_CUT_DATE,
                    rate_before: float = STAMP_RATE_BEFORE,
                    rate_after: float = STAMP_RATE_AFTER) -> float:
    """Module-level convenience wrapper (policy lookup by YYYYMMDD)."""
    return FrictionConfig(stamp_cut_date=cut_date, stamp_rate_before=rate_before,
                          stamp_rate_after=rate_after).stamp_duty_rate(trade_date)


def one_word_limit_up(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                      close: np.ndarray, up_limit: np.ndarray,
                      atol: float = LIMIT_ATOL) -> np.ndarray:
    """一字涨停: bar flat at the up-limit price (real stk_limit column)."""
    o = np.asarray(open_, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    u = np.asarray(up_limit, dtype=np.float64)
    flat = (np.abs(h - l) <= atol) & (np.abs(o - h) <= atol) & (np.abs(c - h) <= atol)
    at_limit = np.abs(o - u) <= atol
    return flat & at_limit & np.isfinite(u) & (u > 0)


def one_word_limit_down(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                        close: np.ndarray, down_limit: np.ndarray,
                        atol: float = LIMIT_ATOL) -> np.ndarray:
    """一字跌停: bar flat at the down-limit price."""
    o = np.asarray(open_, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    d = np.asarray(down_limit, dtype=np.float64)
    flat = (np.abs(h - l) <= atol) & (np.abs(o - h) <= atol) & (np.abs(c - h) <= atol)
    at_limit = np.abs(o - d) <= atol
    return flat & at_limit & np.isfinite(d) & (d > 0)


@dataclass
class DayResult:
    date: int
    target: np.ndarray
    executed: np.ndarray          # executed weight delta on the execution day
    blocked_buy: int
    blocked_sell: int
    turnover: float               # 0.5 * sum(|executed|)
    costs: Dict[str, float] = field(default_factory=dict)
    cash_before: float = 0.0
    cash_after: float = 0.0
    equity_before: float = 0.0
    equity_after: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    filled_notional: float = 0.0
    partial_fills: int = 0
    rejected_orders: int = 0
    ledger_rows: List[Dict[str, Any]] = field(default_factory=list)


class TradingStateMachine:
    """Applies targets under real frictions, date by date.

    Loop order (documented, next-day-open matching): signal formed on
    close of day *t* -> target weights ``target_t`` -> bars of the
    **execution day t+1** checked for 一字/停牌/ST -> orders executed at
    the open of *t+1* with costs charged on *t+1* (stamp duty by the
    real execution date) -> forward return
    ``adj_open[t+2] / adj_open[t+1] - 1`` accrues on the held weights.
    """

    def __init__(self, config: Optional[FrictionConfig] = None):
        self.config = config or FrictionConfig()
        self.reset()

    # -- state ------------------------------------------------------------- #
    def reset(self) -> None:
        self.holdings: Optional[np.ndarray] = None
        self.shares: Optional[np.ndarray] = None
        self.avg_cost: Optional[np.ndarray] = None
        # Last authoritative execution/mark price per name.  During a
        # documented suspension the exchange supplies no open; carrying this
        # mark keeps the position frozen until the first valid re-open.
        self.mark_prices: Optional[np.ndarray] = None
        self.cash: float = float(self.config.initial_cash)
        self.equity: float = float(self.config.initial_cash)
        self.previous_date: Optional[int] = None
        self.cum_costs: Dict[str, float] = {
            "stamp": 0.0, "commission": 0.0, "slippage": 0.0,
            "financing": 0.0, "borrow_fee": 0.0, "dividend": 0.0,
        }
        self.blocked_orders: Dict[str, int] = {
            "limit_up_buy": 0, "limit_down_sell": 0,
            "untradeable": 0, "partial_fill": 0,
            "risk_reject": 0,
        }
        self.blocked_sell_fallback: int = 0
        self.turnover_total: float = 0.0
        self.trade_count: int = 0
        self.ledger: List[Dict[str, Any]] = []

    @staticmethod
    def _round_price(price: np.ndarray, tick_size: float) -> np.ndarray:
        p = np.asarray(price, dtype=np.float64).copy()
        if tick_size > 0:
            p = np.round(p / tick_size) * tick_size
        p[~np.isfinite(p) | (p <= 0)] = np.nan
        return p

    def _validate_target(self, target: np.ndarray) -> None:
        cfg = self.config
        gross = float(np.abs(target).sum())
        short = float(np.abs(target[target < 0]).sum())
        if not cfg.allow_short and short > 1e-12:
            raise ValueError("short exposure is disabled by friction configuration")
        if cfg.enforce_risk_limits and gross > cfg.max_gross_leverage + 1e-9:
            raise ValueError(
                f"target gross leverage {gross:.6f} exceeds limit "
                f"{cfg.max_gross_leverage:.6f}")
        if cfg.enforce_risk_limits and short > cfg.max_short_leverage + 1e-9:
            raise ValueError(
                f"target short leverage {short:.6f} exceeds limit "
                f"{cfg.max_short_leverage:.6f}")

    # -- one rebalance ------------------------------------------------------ #
    def rebalance(self, date: int, target: np.ndarray, close: np.ndarray,
                  open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                  up_limit: np.ndarray, down_limit: np.ndarray,
                  amount: np.ndarray, suspended: np.ndarray,
                  tradable: Optional[np.ndarray] = None,
                  up_lock_row: Optional[np.ndarray] = None,
                  down_lock_row: Optional[np.ndarray] = None,
                  cash_dividend: Optional[np.ndarray] = None,
                  shortable: Optional[np.ndarray] = None,
                  borrow_rate: Optional[np.ndarray] = None) -> DayResult:
        cfg = self.config
        raw_target = np.asarray(target, dtype=np.float64)
        n = raw_target.shape[0]
        if raw_target.ndim != 1:
            raise ValueError(f"target shape {raw_target.shape} is not one-dimensional")
        if not np.isfinite(raw_target).all():
            raise ValueError("target contains non-finite values")
        target = raw_target.copy()
        if target.shape != (n,):
            raise ValueError(f"target shape {target.shape} != ({n},)")
        self._validate_target(target)
        prices = self._round_price(open_, cfg.tick_size)
        suspended_arr = np.asarray(suspended, dtype=bool)
        if suspended_arr.shape != (n,):
            raise ValueError(f"suspended shape {suspended_arr.shape} != ({n},)")
        if self.holdings is None:
            self.holdings = np.zeros(n, dtype=np.float64)
            self.shares = np.zeros(n, dtype=np.float64)
            self.avg_cost = np.zeros(n, dtype=np.float64)
            self.mark_prices = np.full(n, np.nan, dtype=np.float64)
        assert self.shares is not None and self.avg_cost is not None
        assert self.mark_prices is not None
        # A suspension row may contain stale copied quotes.  It is not an
        # execution or mark event, so the last valid mark must remain frozen.
        finite_execution = np.isfinite(prices) & ~suspended_arr
        self.mark_prices[finite_execution] = prices[finite_execution]
        mark_prices = self.mark_prices
        missing_mark = (np.abs(self.shares) > 1e-12) & ~np.isfinite(mark_prices)
        if missing_mark.any():
            bad = ",".join(str(i) for i in np.flatnonzero(missing_mark)[:8])
            raise ValueError(
                "held position has no authoritative mark on "
                f"{int(date)}: asset indices {bad}")
        cash_before = float(self.cash)

        # Mark existing positions at the execution open and accrue financing
        # and stock borrow fees for the elapsed calendar days.
        valid_price = np.isfinite(mark_prices)
        market_before = self.shares * np.where(valid_price, mark_prices, 0.0)
        equity_before = float(self.cash + market_before.sum())
        if equity_before <= 0:
            raise ValueError("portfolio equity is non-positive before execution")
        elapsed_days = 1
        if self.previous_date is not None:
            try:
                d0 = datetime.strptime(str(self.previous_date), "%Y%m%d")
                d1 = datetime.strptime(str(int(date)), "%Y%m%d")
                elapsed_days = max(1, (d1 - d0).days)
            except (TypeError, ValueError):
                elapsed_days = 1
        financing = max(-self.cash, 0.0) * cfg.financing_rate_annual * elapsed_days / 365.0
        short_mask = self.shares < 0
        borrow_base = float(np.abs(market_before[short_mask]).sum())
        if borrow_rate is None:
            borrow_fee = borrow_base * cfg.borrow_rate_annual * elapsed_days / 365.0
        else:
            rates = np.asarray(borrow_rate, dtype=np.float64)
            if rates.shape != (n,):
                raise ValueError(f"borrow_rate shape {rates.shape} != ({n},)")
            if (not np.isfinite(rates[short_mask]).all()
                    or (rates[short_mask] < 0).any()):
                raise ValueError(
                    "borrow_rate contains non-finite or negative values")
            borrow_fee = float(np.sum(
                np.abs(market_before[short_mask]) * rates[short_mask]
            )) * elapsed_days / 365.0
        self.cash -= financing + borrow_fee
        dividend_cash = 0.0
        if cash_dividend is not None:
            dividend = np.asarray(cash_dividend, dtype=np.float64)
            if dividend.shape != (n,):
                raise ValueError(f"cash_dividend shape {dividend.shape} != ({n},)")
            if not np.isfinite(dividend).all():
                raise ValueError("cash_dividend contains non-finite values")
            # Ex-date cash flows are signed: shorts owe the lender the same
            # per-share dividend that longs receive.
            dividend_cash = float(np.dot(self.shares, dividend))
            self.cash += dividend_cash
        self.cum_costs["financing"] += financing
        self.cum_costs["borrow_fee"] += borrow_fee
        # Costs use the ledger sign convention: a dividend received by a
        # long is negative cost, while a payment on a short is positive cost.
        self.cum_costs["dividend"] += -dividend_cash
        equity_before = float(self.cash + market_before.sum())
        if equity_before <= 0:
            raise ValueError("portfolio equity is non-positive after financing charges")
        current_weight = np.divide(market_before, equity_before,
                                   out=np.zeros(n), where=np.isfinite(market_before))
        h = current_weight.copy()

        blocked_buy_mask = np.zeros(n, dtype=bool)
        blocked_sell_mask = np.zeros(n, dtype=bool)
        if cfg.block_limit_moves:
            # rows may be pre-computed for the whole window (perf path)
            up_bars = (up_lock_row if up_lock_row is not None
                       else one_word_limit_up(open_, high, low, close, up_limit))
            dn_bars = (down_lock_row if down_lock_row is not None
                       else one_word_limit_down(open_, high, low, close, down_limit))
            blocked_buy_mask |= up_bars
            blocked_sell_mask |= dn_bars
        untradeable = suspended_arr.copy()
        if tradable is not None:
            untradeable |= ~np.asarray(tradable, dtype=bool)
        blocked_buy_mask |= untradeable
        blocked_sell_mask |= untradeable
        non_borrow_sell_block = blocked_sell_mask.copy()
        borrow_block = np.zeros(n, dtype=bool)
        if shortable is not None:
            shortable = np.asarray(shortable, dtype=bool)
            # Borrow availability only constrains opening/increasing a short;
            # reducing or closing an existing short remains executable.  An
            # existing long may still be liquidated to zero; only the new
            # short leg is blocked when borrow is unavailable.
            borrow_block = (~shortable) & (target < 0) & (target < h)
            blocked_sell_mask |= borrow_block

        target_delta = target - h
        desired = target.copy()
        untradeable_attempt = untradeable & (np.abs(target - h) > 1e-12)
        # Cannot increase weight in buy-blocked names -> count the *attempt*.
        buy_attempt = blocked_buy_mask & (target > h)
        np.copyto(desired, h, where=buy_attempt)
        # A long can always be liquidated even when a new short cannot be
        # borrowed; this is a valid close, not a rejected sell attempt.
        close_long_without_borrow = (
            borrow_block & (h > 0) & ~non_borrow_sell_block
        )
        # Cannot decrease weight in other sell-blocked names (position lock).
        sell_attempt = blocked_sell_mask & (target < h) & ~close_long_without_borrow
        np.copyto(desired, h, where=sell_attempt)
        np.copyto(desired, 0.0, where=close_long_without_borrow)

        effective_delta = desired - h
        missing_quote_order = (np.abs(effective_delta) > 1e-12) & ~np.isfinite(prices)
        if missing_quote_order.any():
            bad = ",".join(str(i) for i in np.flatnonzero(missing_quote_order)[:8])
            raise ValueError(
                "missing execution open for order(s) on "
                f"{int(date)}: asset indices {bad}")

        requested_notional = (desired - h) * equity_before
        requested_notional[~np.isfinite(prices)] = 0.0
        requested_shares = np.divide(requested_notional, prices,
                                     out=np.zeros(n), where=np.isfinite(prices) & (prices > 0))
        lot = float(cfg.min_lot)
        # Add a tiny numerical guard before lot flooring.  A target that
        # exactly closes a position can otherwise become one lot short due
        # to a binary division such as 99999.99999999999 / 100.
        requested_shares = np.sign(requested_shares) * (
            np.floor(np.abs(requested_shares) / lot + 1e-9) * lot)
        # Participation limits create a real partial fill rather than a
        # synthetic price or silently dropping the order.
        amt = np.asarray(amount, dtype=np.float64)
        missing_amount_order = (np.abs(effective_delta) > 1e-12) & ~np.isfinite(amt)
        if missing_amount_order.any():
            bad = ",".join(str(i) for i in np.flatnonzero(missing_amount_order)[:8])
            raise ValueError(
                "missing execution amount for order(s) on "
                f"{int(date)}: asset indices {bad}")
        cap = np.divide(np.maximum(amt, 0.0) * cfg.max_participation,
                        prices, out=np.full(n, np.inf), where=np.isfinite(prices) & (prices > 0))
        cap = np.floor(cap / lot + 1e-9) * lot
        fill_shares = np.sign(requested_shares) * np.minimum(np.abs(requested_shares), cap)
        fill_shares[~np.isfinite(fill_shares)] = 0.0
        partial = (np.abs(fill_shares) + 1e-12 < np.abs(requested_shares)) & (np.abs(requested_shares) > 1e-12)
        attempted = (np.abs(target_delta) > 1e-12)
        executed_notional_signed = fill_shares * prices
        # Zero quantity multiplied by a missing quote is still NaN in numpy;
        # missing execution bars must contribute no order or cost.
        executed_notional_signed[~np.isfinite(executed_notional_signed)] = 0.0
        executed = np.divide(executed_notional_signed, equity_before,
                             out=np.zeros(n), where=equity_before > 0)
        notional = np.abs(executed_notional_signed)
        turnover = 0.5 * float(np.abs(executed).sum())
        n_trades = int(np.count_nonzero(np.abs(executed) > 1e-12))

        # Margin is checked against the projected filled book before mutating
        # state.  A rejected risk order is a hard replay failure, not an
        # implicit cash loan or a synthetic fill.  The normal gross=2
        # dollar-neutral target consumes the full 0.5 margin budget; allow
        # the explicitly computed execution costs as a deterministic buffer
        # so commissions/slippage do not reject an otherwise valid target.
        projected_shares = self.shares + fill_shares
        projected_market = projected_shares * np.where(np.isfinite(mark_prices), mark_prices, 0.0)
        projected_long = float(projected_market[projected_market > 0].sum())
        projected_short = float(np.abs(projected_market[projected_market < 0]).sum())
        margin_required = (projected_long + projected_short) * cfg.margin_requirement
        margin_before = float(np.abs(market_before).sum()) * cfg.margin_requirement

        stamp_rate = cfg.stamp_duty_rate(date)
        # stamp only on the sell side (real rule)
        sell_notional = float(notional[executed_notional_signed < 0].sum())
        stamp = sell_notional * stamp_rate
        commission = float(notional.sum()) * cfg.commission_rate
        # square-root impact: 0.1 * sqrt(order / daily_amount), 0 when the
        # bar carries no usable amount (NaN must not poison the sum)
        ok = amt > 0
        if notional.any() and ok.any():
            frac = notional[ok] / amt[ok]
            np.sqrt(frac, out=frac)
            frac *= cfg.impact_coeff
            slippage = float(np.dot(notional[ok], frac))
        else:
            slippage = 0.0

        margin_tolerance = max(
            1e-6,
            equity_before * cfg.margin_tolerance_ratio
            + stamp + commission + slippage,
        )
        # A mark-to-market loss can leave an already-held book above the
        # current equity limit.  In that state, rejecting every rebalance
        # would also reject trades that reduce the outstanding margin gap and
        # make recovery impossible.  Keep the normal equity cap for healthy
        # books; for an inherited breach, cap the projected margin relative to
        # the pre-trade breach instead.  The same explicit tolerance covers
        # lot/price rounding and execution costs in either branch.
        inherited_breach = margin_before > equity_before
        margin_cap = (equity_before + margin_tolerance
                      if not inherited_breach else margin_before + margin_tolerance)
        risk_hold_mask = np.zeros(n, dtype=bool)
        locked_order_risk = bool(buy_attempt.any() or sell_attempt.any())
        if cfg.enforce_risk_limits and margin_required > margin_cap:
            self.blocked_orders["risk_reject"] += 1
            if not inherited_breach and not locked_order_risk:
                raise ValueError(
                    f"projected margin {margin_required:.6f} exceeds equity "
                    f"{equity_before:.6f} on {int(date)} "
                    f"(margin_before={margin_before:.6f}, "
                    f"blocked_buy={int(buy_attempt.sum())}, "
                    f"blocked_sell={int(sell_attempt.sum())})")
            # An already-breached book must not increase its exposure.  A
            # target that would do so is rejected as a whole for this bar;
            # keeping the current shares preserves fail-closed accounting and
            # lets a later, deleveraging signal recover the portfolio.
            risk_hold_mask = attempted.copy()
            fill_shares = np.zeros(n, dtype=np.float64)
            partial = np.zeros(n, dtype=bool)
            executed_notional_signed = np.zeros(n, dtype=np.float64)
            executed = np.zeros(n, dtype=np.float64)
            notional = np.zeros(n, dtype=np.float64)
            turnover = 0.0
            n_trades = 0
            stamp = commission = slippage = 0.0

        # Apply cash, positions and per-name cost basis.  Crossing zero is
        # handled as a close followed by an open, preserving realised P&L.
        realized = 0.0
        for i, delta in enumerate(fill_shares):
            if abs(delta) <= 1e-12 or not np.isfinite(prices[i]):
                continue
            old = float(self.shares[i])
            px = float(prices[i])
            if old > 0 and delta < 0:
                close_qty = min(old, -delta)
                realized += (px - self.avg_cost[i]) * close_qty
            elif old < 0 and delta > 0:
                close_qty = min(-old, delta)
                realized += (self.avg_cost[i] - px) * close_qty
            new = old + delta
            if old == 0 or (old > 0 and new < 0) or (old < 0 and new > 0):
                self.avg_cost[i] = px if abs(new) > 1e-12 else 0.0
            elif old * new > 0 and abs(new) > abs(old):
                self.avg_cost[i] = (abs(old) * self.avg_cost[i] + abs(delta) * px) / abs(new)
            self.shares[i] = new
        self.cash -= float(executed_notional_signed.sum()) + stamp + commission + slippage
        market_after = self.shares * np.where(np.isfinite(mark_prices), mark_prices, 0.0)
        equity_after = float(self.cash + market_after.sum())
        if equity_after <= 0:
            raise ValueError("portfolio equity is non-positive after execution")
        self.equity = equity_after
        # Holdings remain weights for the canonical return loop, while the
        # share ledger remains the source of truth for cash and P&L.
        self.holdings = np.divide(market_after, equity_before,
                                  out=np.zeros(n), where=equity_before > 0)
        unrealized = float(np.sum((mark_prices - self.avg_cost) * self.shares,
                                  where=np.isfinite(mark_prices)))
        self.cum_costs["stamp"] += stamp
        self.cum_costs["commission"] += commission
        self.cum_costs["slippage"] += slippage
        self.blocked_orders["limit_up_buy"] += int(buy_attempt.sum())
        self.blocked_orders["limit_down_sell"] += int(sell_attempt.sum())
        self.blocked_orders["untradeable"] += int(untradeable_attempt.sum())
        self.blocked_orders["partial_fill"] += int(partial.sum())
        self.turnover_total += turnover
        self.trade_count += n_trades

        ledger_rows = []
        attempted_shares = np.divide(target_delta * equity_before, prices,
                                      out=np.zeros(n), where=np.isfinite(prices) & (prices > 0))
        attempted_shares = np.sign(attempted_shares) * (
            np.floor(np.abs(attempted_shares) / lot + 1e-9) * lot)
        rejected_mask = (attempted & (np.abs(fill_shares) <= 1e-12) &
                         (buy_attempt | sell_attempt)) | risk_hold_mask
        for i, delta in enumerate(fill_shares):
            if abs(delta) <= 1e-12 and not partial[i] and not rejected_mask[i]:
                continue
            status = ("REJECTED" if rejected_mask[i] else
                      ("PARTIAL" if partial[i] else "FILLED"))
            ledger_rows.append({
                "order_id": f"{int(date)}:{int(i)}", "date": int(date), "asset": int(i),
                "requested_shares": float(attempted_shares[i] if rejected_mask[i]
                                           else requested_shares[i]),
                "filled_shares": float(delta),
                "price": float(prices[i]) if np.isfinite(prices[i]) else None,
                "notional": float(abs(executed_notional_signed[i])),
                "side": "BUY" if delta > 0 else "SELL",
                "partial": bool(partial[i]),
                "blocked": bool((buy_attempt | sell_attempt | risk_hold_mask)[i]),
                "status": status,
            })
        self.ledger.extend(ledger_rows)
        self.previous_date = int(date)
        return DayResult(
            date=int(date), target=target.copy(), executed=executed,
            blocked_buy=int(buy_attempt.sum()),
            blocked_sell=int(sell_attempt.sum()),
            turnover=turnover,
            costs={"stamp": stamp, "commission": commission, "slippage": slippage,
                   "financing": financing, "borrow_fee": borrow_fee,
                   "dividend": -dividend_cash},
            cash_before=cash_before, cash_after=float(self.cash),
            equity_before=equity_before, equity_after=equity_after,
            realized_pnl=float(realized), unrealized_pnl=unrealized,
            filled_notional=float(notional.sum()), partial_fills=int(partial.sum()),
            # A margin check rejects the rebalance event as one atomic order,
            # even when its per-name ledger rows describe both legs.
            rejected_orders=(int(bool(risk_hold_mask.any())) + int(
                ((buy_attempt | sell_attempt) & ~risk_hold_mask).sum())),
            ledger_rows=ledger_rows,
        )

    # -- reporting ---------------------------------------------------------- #
    def friction_breakdown(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.cum_costs.items()}

    def blocked_totals(self) -> Dict[str, int]:
        return {k: int(v) for k, v in self.blocked_orders.items()}

    def audit_summary(self) -> Dict[str, Any]:
        """Serializable ledger summary bound into the canonical replay result."""
        return {
            "initial_cash": float(self.config.initial_cash),
            "final_cash": float(self.cash),
            "final_equity": float(self.equity),
            "trade_count": int(self.trade_count),
            "ledger_rows": int(len(self.ledger)),
            "turnover_total": float(self.turnover_total),
            "blocked_orders": self.blocked_totals(),
            "costs": self.friction_breakdown(),
            "accounting": {
                "min_lot": int(self.config.min_lot),
                "tick_size": float(self.config.tick_size),
                "max_participation": float(self.config.max_participation),
                "max_gross_leverage": float(self.config.max_gross_leverage),
                "max_short_leverage": float(self.config.max_short_leverage),
                "margin_requirement": float(self.config.margin_requirement),
                "financing_rate_annual": float(self.config.financing_rate_annual),
                "borrow_rate_annual": float(self.config.borrow_rate_annual),
                "allow_short": bool(self.config.allow_short),
            },
        }
