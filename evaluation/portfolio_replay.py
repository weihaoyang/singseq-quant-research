"""Long-only next-open research replay with cash, retained positions and fills.

Adjusted units approximate reinvested corporate actions; not a broker ledger.
Daily opening limits are conservative: at-limit orders never fill. Capacity
uses prior-session amount in CNY, never the execution day's future volume.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def replay_next_open(panel, signals, calendar, *, horizon=5, top_n=50,
                     initial_cash=1_000_000., buy_cost=.002, sell_cost=.002,
                     max_participation=.01, amount_scale=1000., signal_dates=None):
    if horizon < 1 or top_n < 1 or initial_cash <= 0 or min(buy_cost, sell_cost) < 0:
        raise ValueError("invalid replay parameters")
    needed = {"ts_code", "trade_date", "open", "close", "vol", "amount", "adj_factor", "up_limit", "down_limit"}
    if not needed.issubset(panel): raise ValueError(f"missing execution columns: {needed-set(panel)}")
    if panel.duplicated(["ts_code", "trade_date"]).any() or signals.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("duplicate market or signal keys")
    if not 0 < max_participation <= 1: raise ValueError("invalid participation")
    days = sorted(set(map(str, calendar)))
    if not set(signals.trade_date).issubset(days): raise ValueError("signal outside calendar")
    market = {d: g.set_index("ts_code") for d, g in panel.groupby("trade_date")}
    scores = {d: g.set_index("ts_code").score for d, g in signals.groupby("trade_date")}
    first = min(scores) if scores else None
    if first is None: raise ValueError("no signals")
    start = days.index(first)
    schedule = set(signal_dates) if signal_dates is not None else set(days[start::horizon])
    units, marks = {}, {}
    cash, prior_nav = float(initial_cash), float(initial_cash)
    equity, fills, rejected = [], [], []
    # Only start after the first signal; include all later market sessions.
    for i in range(start+1, len(days)):
        date, signal_date = days[i], days[i-1]
        quotes = market.get(date, pd.DataFrame())
        prev = market.get(signal_date, pd.DataFrame())
        def quote(code):
            return quotes.loc[code] if code in quotes.index else None
        def price(code, field):
            q = quote(code)
            if q is not None and np.isfinite(q[field]) and q[field] > 0 and np.isfinite(q.adj_factor) and q.adj_factor > 0:
                return float(q[field] * q.adj_factor)
            return marks.get(code, np.nan)
        nav_open = cash + sum(qty * price(code, "open") for code, qty in units.items())
        filled_notional = costs = 0.
        if signal_date in schedule and signal_date in scores:
            ranking = scores[signal_date].replace([np.inf, -np.inf], np.nan).dropna()
            ranking = ranking.sort_index().sort_values(ascending=False, kind="stable").head(top_n)
            # Equal slots are fixed before observing fills; rejected orders leave cash.
            target_value = nav_open / len(ranking) if len(ranking) else 0.
            targets = {code: target_value for code in ranking.index}
            desired = {}
            for code in sorted(set(units) | set(targets)):
                p = price(code, "open")
                desired[code] = targets.get(code, 0.) / p if np.isfinite(p) and p > 0 else units.get(code, 0.)
            for side in ["sell", "buy"]:
                for code in sorted(desired):
                    current = units.get(code, 0.)
                    delta = desired[code] - current
                    if (side == "sell" and delta >= -1e-12) or (side == "buy" and delta <= 1e-12): continue
                    q = quote(code)
                    reason = None
                    if q is None or not all(np.isfinite(q[c]) and q[c] > 0 for c in ["open", "adj_factor", "vol", "up_limit", "down_limit"]):
                        reason = "missing_quote_limit_or_suspended"
                    elif side == "buy" and q.open >= q.up_limit - .005: reason = "opening_limit_up"
                    elif side == "sell" and q.open <= q.down_limit + .005: reason = "opening_limit_down"
                    if reason:
                        rejected.append({"trade_date":date,"signal_date":signal_date,"ts_code":code,"side":side,"reason":reason})
                        continue
                    p = float(q.open*q.adj_factor)
                    prior_amount = prev.loc[code,"amount"] if code in prev.index else np.nan
                    capacity = max_participation * prior_amount * amount_scale
                    if not np.isfinite(capacity) or capacity <= 0:
                        rejected.append({"trade_date":date,"signal_date":signal_date,"ts_code":code,"side":side,"reason":"missing_prior_capacity"})
                        continue
                    rate = buy_cost if side == "buy" else sell_cost
                    qty = min(abs(delta), capacity / p)
                    if side == "buy":
                        qty = min(qty, cash / (p*(1+rate)))
                        raw_qty = np.floor((qty*q.adj_factor + 1e-8)/100)*100
                        qty = raw_qty / q.adj_factor
                    else:
                        qty = min(qty, current)
                        # Entire odd-lot liquidation allowed; partial reductions in lots.
                        if qty < current - 1e-8: qty = np.floor(qty*q.adj_factor/100)*100/q.adj_factor
                    if qty <= 1e-12: continue
                    value, cost = qty*p, qty*p*rate
                    cash += value-cost if side == "sell" else -value-cost
                    units[code] = current-qty if side == "sell" else current+qty
                    if units[code] < 1e-10: del units[code]
                    marks[code] = p
                    filled_notional += value
                    costs += cost
                    fills.append({"signal_date":signal_date,"trade_date":date,"ts_code":code,"side":side,
                                  "raw_quantity":qty*q.adj_factor,"raw_price":q.open,"notional":value,"cost":cost})
        stale = 0
        for code in units:
            q = quote(code)
            if q is None or not np.isfinite(q.close) or q.close <= 0: stale += 1
            marks[code] = price(code,"close")
        nav = cash + sum(qty*marks[code] for code,qty in units.items())
        if cash < -1e-6 or not np.isfinite(nav) or nav <= 0: raise ValueError("invalid portfolio accounting")
        equity.append({"trade_date":date,"nav":nav,"return":nav/prior_nav-1,"cash":cash,
                       "gross_traded_fraction":filled_notional/nav_open,
                       "cost_total":costs,"holdings":len(units),"stale_holdings":stale})
        prior_nav = nav
    daily = pd.DataFrame(equity)
    if daily.empty: raise ValueError("no execution sessions after signals")
    returns = daily["return"]
    peaks = np.maximum.accumulate(np.r_[initial_cash, daily.nav.to_numpy()])[1:]
    sd = returns.std(ddof=1)
    summary = {"status":"UNVERIFIED", "sessions":len(daily),"total_return":float(daily.nav.iloc[-1]/initial_cash-1),
               "annual_return":float((daily.nav.iloc[-1]/initial_cash)**(252/len(daily))-1),
               "sharpe":float(returns.mean()/sd*np.sqrt(252)) if sd > 0 else None,
               "max_drawdown":float((1-daily.nav.to_numpy()/peaks).max()),
               "mean_gross_traded_fraction":float(daily.gross_traded_fraction.mean()),
               "cost_total":float(daily.cost_total.sum()),"fills":len(fills),"rejected_orders":len(rejected),
               "stale_holding_sessions":int((daily.stale_holdings>0).sum()),
               "terminal_positions":units,"terminal_cash":cash,
               "limitations":["adjusted-unit corporate actions; no explicit cash dividends",
                              "daily bars; conservative opening-limit nonfills",
                              "historical ST/universe and delisting completeness require separate audit",
                              "terminal positions marked to market, not forcibly liquidated"]}
    return summary, daily, pd.DataFrame(fills), pd.DataFrame(rejected)
