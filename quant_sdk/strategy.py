"""Event-driven Strategy Engine & JoinQuant-compatible API.

Provides institutional-grade, zero-leak local strategy authoring and backtesting:
- Standard JoinQuant-compatible execution lifecycle:
    ``initialize(context)``, ``before_trading_start(context)``,
    ``handle_data(context, data)``, and scheduled jobs (``run_daily``, ``run_weekly``, ``run_monthly``)
- Orders & Position management:
    ``order``, ``order_target``, ``order_value``, ``order_target_percent``, ``order_target_value``
- Realistic A-share microstructure & friction:
    - T+1 available position settlement (shares bought on day T unlocked on day T+1)
    - 100-share board lot constraints (fractional shares rounded down on buy)
    - Price limit constraints: buying barred on limit-up, selling barred on limit-down
    - Suspension checks: no trading when security is halted
    - Stamp duty (0.05% on sell) + dual-side commission (min 5 RMB) + custom slippage
- Point-in-time local data access directly against DuckDB or custom panels.
"""
from __future__ import annotations

import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from quant_core import metrics as M

__all__ = [
    "AshareCommission",
    "Context",
    "CurrentData",
    "FixedSlippage",
    "Order",
    "OrderStyle",
    "Portfolio",
    "Position",
    "SecurityData",
    "Strategy",
    "StrategyEngine",
    "StrategyResult",
    "VolumeShareSlippage",
    "attribute_history",
    "cancel_order",
    "get_current_data",
    "get_price",
    "history",
    "order",
    "order_target",
    "order_target_percent",
    "order_target_value",
    "order_value",
    "run_daily",
    "run_monthly",
    "run_strategy",
    "run_weekly",
    "set_benchmark",
    "set_commission",
    "set_option",
    "set_slippage",
]

_ACTIVE_ENGINE: StrategyEngine | None = None


# --------------------------------------------------------------------------- #
# Order & Portfolio Models
# --------------------------------------------------------------------------- #
@dataclass
class OrderStyle:
    """Base class for order style."""
    order_type: Literal["MARKET", "LIMIT"] = "MARKET"
    limit_price: float | None = None


class MarketOrderStyle(OrderStyle):
    def __init__(self):
        super().__init__(order_type="MARKET", limit_price=None)


class LimitOrderStyle(OrderStyle):
    def __init__(self, limit_price: float):
        super().__init__(order_type="LIMIT", limit_price=float(limit_price))


@dataclass
class Order:
    order_id: str
    security: str
    amount: int
    action: Literal["BUY", "SELL"]
    status: Literal["SUBMITTED", "FILLED", "REJECTED", "CANCELLED"] = "SUBMITTED"
    style: OrderStyle = field(default_factory=MarketOrderStyle)
    created_time: str = ""
    filled_time: str = ""
    filled_amount: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage_cost: float = 0.0
    tax: float = 0.0
    reject_reason: str = ""

    @property
    def is_filled(self) -> bool:
        return self.status == "FILLED"


@dataclass
class Position:
    security: str
    total_amount: int = 0
    closeable_amount: int = 0
    today_amount: int = 0
    avg_cost: float = 0.0
    price: float = 0.0
    side: Literal["long", "short"] = "long"

    @property
    def value(self) -> float:
        return float(self.total_amount * self.price)

    @property
    def pnl(self) -> float:
        if self.total_amount == 0:
            return 0.0
        return float((self.price - self.avg_cost) * self.total_amount)

    @property
    def pnl_ratio(self) -> float:
        if self.total_amount == 0 or self.avg_cost <= 0:
            return 0.0
        return float((self.price - self.avg_cost) / self.avg_cost)


@dataclass
class Portfolio:
    initial_cash: float
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    positions_value: float = 0.0
    total_value: float = 0.0
    returns: float = 0.0
    daily_pnl: float = 0.0

    def update_valuation(self):
        self.positions_value = sum(pos.value for pos in self.positions.values())
        self.total_value = self.cash + self.positions_value
        self.returns = (self.total_value - self.initial_cash) / self.initial_cash if self.initial_cash > 0 else 0.0


class GlobalState:
    """Convenience attribute-based state container (e.g. context.g.xxx)."""

    def __getattr__(self, name: str) -> Any:
        return None

    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value


class Context:
    """Strategy runtime context holding portfolio, clock and custom state."""

    def __init__(self, initial_cash: float = 1_000_000.0):
        self.portfolio = Portfolio(
            initial_cash=initial_cash,
            cash=initial_cash,
            total_value=initial_cash,
        )
        self.current_dt: str = ""
        self.benchmark: str = "000300.SH"
        self.universe: list[str] = []
        self.run_params: dict[str, Any] = {}
        self.subportfolios: list[Any] = []
        self.g = GlobalState()
        self._user_vars: dict[str, Any] = {}

    def set_benchmark(self, benchmark: str) -> None:
        self.benchmark = str(benchmark)

    def __getattr__(self, name: str) -> Any:
        if name in self._user_vars:
            return self._user_vars[name]
        raise AttributeError(f"'Context' object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"portfolio", "current_dt", "benchmark", "universe", "run_params", "subportfolios", "g", "_user_vars"}:
            super().__setattr__(name, value)
        else:
            self._user_vars[name] = value


@dataclass
class SecurityData:
    security: str
    last_price: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    amount: float = 0.0
    high_limit: float = float("inf")
    low_limit: float = 0.0
    paused: bool = False
    is_st: bool = False
    day_open: float = 0.0


class CurrentData(dict):
    """Snapshot mapping security -> SecurityData for the current bar."""

    def __getitem__(self, key: str) -> SecurityData:
        if key not in self:
            return SecurityData(security=key)
        return super().__getitem__(key)


# --------------------------------------------------------------------------- #
# Friction & Commission Models
# --------------------------------------------------------------------------- #
class SlippageModel:
    def calculate_slippage(self, order: Order, price: float, volume: float) -> float:
        raise NotImplementedError


class FixedSlippage(SlippageModel):
    def __init__(self, spread: float = 0.002):
        self.spread = float(spread)

    def calculate_slippage(self, order: Order, price: float, _volume: float) -> float:
        half_spread = self.spread / 2.0
        return price * (1.0 + half_spread if order.action == "BUY" else 1.0 - half_spread)


class VolumeShareSlippage(SlippageModel):
    def __init__(self, volume_limit: float = 0.1, spread: float = 0.001):
        self.volume_limit = float(volume_limit)
        self.spread = float(spread)

    def calculate_slippage(self, order: Order, price: float, volume: float) -> float:
        order_share = min(abs(order.amount) / max(volume, 1.0), self.volume_limit)
        impact = self.spread * (1.0 + order_share * 2.0)
        return price * (1.0 + impact if order.action == "BUY" else 1.0 - impact)


class CommissionModel:
    def calculate(self, order: Order, notional: float) -> tuple[float, float]:
        """Returns (commission, stamp_duty_tax)."""
        raise NotImplementedError


class AshareCommission(CommissionModel):
    """Institutional A-share commission model.

    Default:
    - Buy: 0.03% commission (min 5 RMB)
    - Sell: 0.03% commission (min 5 RMB) + 0.05% stamp duty tax (post 2023-08-28 policy)
    """

    def __init__(
        self,
        buy_cost: float = 0.0003,
        sell_cost: float = 0.0003,
        min_cost: float = 5.0,
        stamp_duty: float = 0.0005,
    ):
        self.buy_cost = float(buy_cost)
        self.sell_cost = float(sell_cost)
        self.min_cost = float(min_cost)
        self.stamp_duty = float(stamp_duty)

    def calculate(self, order: Order, notional: float) -> tuple[float, float]:
        rate = self.buy_cost if order.action == "BUY" else self.sell_cost
        comm = max(notional * rate, self.min_cost)
        tax = notional * self.stamp_duty if order.action == "SELL" else 0.0
        return comm, tax


# --------------------------------------------------------------------------- #
# Scheduling & Global Hooks
# --------------------------------------------------------------------------- #
@dataclass
class ScheduledTask:
    func: Callable[[Context], None]
    freq: Literal["daily", "weekly", "monthly"] = "daily"
    day_param: int = 1  # weekday (1=Mon) or monthday
    time_target: str = "open"


def _require_engine() -> StrategyEngine:
    if _ACTIVE_ENGINE is None:
        raise RuntimeError("No active strategy engine running. Call within run_strategy().")
    return _ACTIVE_ENGINE


def set_benchmark(security: str) -> None:
    engine = _require_engine()
    engine.context.benchmark = str(security)


def set_slippage(slippage: SlippageModel) -> None:
    engine = _require_engine()
    engine.slippage = slippage


def set_commission(commission: CommissionModel) -> None:
    engine = _require_engine()
    engine.commission = commission


def set_option(key: str, value: Any) -> None:
    engine = _require_engine()
    engine.options[key] = value


def run_daily(func: Callable[[Context], None], time: str = "open") -> None:
    engine = _require_engine()
    engine.tasks.append(ScheduledTask(func=func, freq="daily", time_target=time))


def run_weekly(func: Callable[[Context], None], weekday: int = 1, time: str = "open") -> None:
    engine = _require_engine()
    engine.tasks.append(ScheduledTask(func=func, freq="weekly", day_param=weekday, time_target=time))


def run_monthly(func: Callable[[Context], None], monthday: int = 1, time: str = "open") -> None:
    engine = _require_engine()
    engine.tasks.append(ScheduledTask(func=func, freq="monthly", day_param=monthday, time_target=time))


# --------------------------------------------------------------------------- #
# Order Placement Functions
# --------------------------------------------------------------------------- #
def order(security: str, amount: int, style: OrderStyle | None = None) -> Order | None:
    return _require_engine().place_order(security, int(amount), style)


def order_target(security: str, amount: int, style: OrderStyle | None = None) -> Order | None:
    engine = _require_engine()
    current_shares = engine.context.portfolio.positions.get(security, Position(security)).total_amount
    delta = int(amount) - current_shares
    if delta == 0:
        return None
    return engine.place_order(security, delta, style)


def order_value(security: str, value: float, style: OrderStyle | None = None) -> Order | None:
    engine = _require_engine()
    price = engine.get_current_price(security)
    if price <= 0:
        return None
    shares = int(value / price)
    return engine.place_order(security, shares, style)


def order_target_value(security: str, value: float, style: OrderStyle | None = None) -> Order | None:
    engine = _require_engine()
    price = engine.get_current_price(security)
    if price <= 0:
        return None
    target_shares = int(value / price)
    return order_target(security, target_shares, style)


def order_target_percent(security: str, percent: float, style: OrderStyle | None = None) -> Order | None:
    engine = _require_engine()
    total_val = engine.context.portfolio.total_value
    target_val = total_val * float(percent)
    return order_target_value(security, target_val, style)


def cancel_order(order_id: str) -> bool:
    return _require_engine().cancel_order(order_id)


def get_current_data() -> CurrentData:
    return _require_engine().current_data


def get_price(
    security: str | list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    count: int | None = None,
    fields: list[str] | str | None = None,
) -> pd.DataFrame | pd.Series:
    return _require_engine().query_price(security, start_date, end_date, count, fields)


def history(count: int, unit: str = "1d", field: str = "close", security_list: list[str] | None = None):
    return _require_engine().query_history(count, unit, field, security_list)


def attribute_history(
    security: str,
    count: int,
    unit: str = "1d",
    fields: list[str] | None = None,
) -> pd.DataFrame:
    return _require_engine().query_attribute_history(security, count, unit, fields)


# --------------------------------------------------------------------------- #
# Strategy Base Class (OOP Modern Style)
# --------------------------------------------------------------------------- #
class Strategy:
    """Object-Oriented base class for modern quantitative strategies."""

    def __init__(self):
        self.context: Context = Context()
        self.engine: Any = None

    # Native Modern Lifecycle Hooks
    def on_init(self, context: Context) -> None:
        self.initialize(context)

    def on_before_trading(self, context: Context) -> None:
        self.before_trading_start(context)

    def on_bar(self, context: Context, data: CurrentData) -> None:
        self.handle_data(context, data)

    def on_order_status(self, context: Context, order: Order) -> None:
        pass

    # Aliases for backward compatibility
    def initialize(self, context: Context) -> None:
        pass

    def before_trading_start(self, context: Context) -> None:
        pass

    def handle_data(self, context: Context, data: CurrentData) -> None:
        pass

    # First-class order methods on strategy instance
    def order(self, security: str, amount: int, style: OrderStyle | None = None) -> Order | None:
        if self.engine is not None:
            return self.engine.place_order(security, int(amount), style)
        return order(security, amount, style)

    def order_target(self, security: str, amount: int, style: OrderStyle | None = None) -> Order | None:
        if self.engine is not None:
            current_shares = self.engine.context.portfolio.positions.get(security, Position(security)).total_amount
            delta = int(amount) - current_shares
            if delta == 0:
                return None
            return self.engine.place_order(security, delta, style)
        return order_target(security, amount, style)

    def order_target_percent(self, security: str, percent: float, style: OrderStyle | None = None) -> Order | None:
        if self.engine is not None:
            total_val = self.engine.context.portfolio.total_value
            target_val = total_val * float(percent)
            price = self.engine.get_current_price(security)
            if price <= 0:
                return None
            target_shares = int(target_val / price)
            return self.order_target(security, target_shares, style)
        return order_target_percent(security, percent, style)

    def order_target_value(self, security: str, value: float, style: OrderStyle | None = None) -> Order | None:
        if self.engine is not None:
            price = self.engine.get_current_price(security)
            if price <= 0:
                return None
            target_shares = int(value / price)
            return self.order_target(security, target_shares, style)
        return order_target_value(security, value, style)

    def order_value(self, security: str, value: float, style: OrderStyle | None = None) -> Order | None:
        if self.engine is not None:
            price = self.engine.get_current_price(security)
            if price <= 0:
                return None
            shares = int(value / price)
            return self.engine.place_order(security, shares, style)
        return order_value(security, value, style)

    def cancel_order(self, order_id: str) -> bool:
        if self.engine is not None:
            return self.engine.cancel_order(order_id)
        return cancel_order(order_id)


# --------------------------------------------------------------------------- #
# Strategy Result Container & Institutional Tearsheet Generator
# --------------------------------------------------------------------------- #
class StrategyResult:
    """Holds backtest performance, trades, curves and produces reports."""

    def __init__(
        self,
        portfolio_values: pd.Series,
        daily_returns: pd.Series,
        benchmark_returns: pd.Series,
        trades: list[dict[str, Any]],
        positions_history: list[dict[str, Any]],
        orders: list[Order],
        benchmark: str,
    ):
        self.portfolio_values = portfolio_values
        self.daily_returns = daily_returns
        self.benchmark_returns = benchmark_returns
        self.trades = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=["trade_id", "order_id", "security", "action", "amount", "price", "commission", "tax", "date"]
        )
        self.positions_history = pd.DataFrame(positions_history) if positions_history else pd.DataFrame()
        self.orders = pd.DataFrame([
            {
                "order_id": o.order_id,
                "security": o.security,
                "amount": o.amount,
                "action": o.action,
                "status": o.status,
                "filled_amount": o.filled_amount,
                "filled_price": o.filled_price,
                "commission": o.commission,
                "tax": o.tax,
                "created_time": o.created_time,
                "reject_reason": o.reject_reason,
            }
            for o in orders
        ]) if orders else pd.DataFrame()
        self.benchmark = benchmark
        self.metrics = self._calculate_metrics()

    def _calculate_metrics(self) -> dict[str, Any]:
        rets = self.daily_returns.to_numpy(dtype=float)
        b_rets = self.benchmark_returns.to_numpy(dtype=float)
        if len(rets) < 2:
            return {}

        total_ret = float(self.portfolio_values.iloc[-1] / self.portfolio_values.iloc[0] - 1.0)
        bench_ret = float(np.prod(1.0 + b_rets) - 1.0) if len(b_rets) > 0 else 0.0

        ann_ret = float(M.annualized_return(rets))
        bench_ann = float(M.annualized_return(b_rets)) if len(b_rets) > 0 else 0.0

        eq_curve = M.cumulative_curve(rets)
        max_dd = float(M.max_drawdown(eq_curve))
        sharpe = float(M.sharpe_ratio(rets))
        calmar = float(M.calmar_ratio(rets))
        vol = float(np.std(rets, ddof=1) * math.sqrt(252)) if len(rets) > 1 else 0.0
        w_rate = float(M.win_rate(rets))

        # Alpha & Beta against benchmark
        if len(b_rets) == len(rets) and np.var(b_rets) > 1e-12:
            cov = float(np.cov(rets, b_rets)[0, 1])
            beta = cov / float(np.var(b_rets))
            alpha = ann_ret - beta * bench_ann
        else:
            alpha, beta = 0.0, 1.0

        # Trade metrics
        total_trades = len(self.trades)
        win_trades, loss_trades = 0, 0
        if not self.trades.empty and "action" in self.trades.columns:
            sells = self.trades[self.trades["action"] == "SELL"]
            total_trades = len(sells)

        return {
            "total_return": total_ret,
            "annualized_return": ann_ret,
            "benchmark_total_return": bench_ret,
            "benchmark_annualized_return": bench_ann,
            "alpha": alpha,
            "beta": beta,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "volatility": vol,
            "win_rate": w_rate,
            "trade_count": total_trades,
            "win_trades": win_trades,
            "loss_trades": loss_trades,
        }

    def summary(self) -> str:
        """Formatted institutional tear-sheet summary."""
        m = self.metrics
        lines = [
            "=" * 70,
            "               QUANT RESEARCH KIT - STRATEGY PERFORMANCE             ",
            "=" * 70,
            f"回测区间:         {self.portfolio_values.index[0]} -> {self.portfolio_values.index[-1]} ({len(self.daily_returns)} 交易日)",
            f"比较基准:         {self.benchmark}",
            "-" * 70,
            f"策略累计收益率:    {m.get('total_return', 0.0)*100:>10.2f}%   | 基准累计收益:   {m.get('benchmark_total_return', 0.0)*100:>10.2f}%",
            f"年化收益率:       {m.get('annualized_return', 0.0)*100:>10.2f}%   | 基准年化收益:   {m.get('benchmark_annualized_return', 0.0)*100:>10.2f}%",
            f"超额年化 (Alpha):  {m.get('alpha', 0.0)*100:>10.2f}%   | 贝塔系数 (Beta): {m.get('beta', 1.0):>10.2f}",
            f"夏普比率 (Sharpe): {m.get('sharpe', 0.0):>10.2f}     | 最大回撤 (MDD): {m.get('max_drawdown', 0.0)*100:>10.2f}%",
            f"卡玛比率 (Calmar): {m.get('calmar', 0.0):>10.2f}     | 年化波动率:     {m.get('volatility', 0.0)*100:>10.2f}%",
            f"日度胜率:         {m.get('win_rate', 0.0)*100:>10.2f}%   | 订单成交总笔数: {len(self.trades):>10d}",
            "=" * 70,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "dates": list(self.portfolio_values.index),
            "portfolio_values": [float(x) for x in self.portfolio_values],
            "benchmark_values": [float(x) for x in (1.0 + self.benchmark_returns).cumprod() * self.portfolio_values.iloc[0]],
            "daily_returns": [float(x) for x in self.daily_returns],
            "trade_count": len(self.trades),
        }

    def generate_html_report(self, output_path: str | None = None) -> str:
        """Produce an interactive standalone HTML tearsheet report."""
        m = self.metrics
        dates = list(self.portfolio_values.index)
        p_vals = [round(float(x), 2) for x in self.portfolio_values]
        b_vals = [round(float(x), 2) for x in (1.0 + self.benchmark_returns).cumprod() * self.portfolio_values.iloc[0]]

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>策略研究回测报告 - Quant Research Kit</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
  .container {{ max-width: 1100px; margin: 0 auto; background: #1e293b; border-radius: 12px; padding: 32px; border: 1px solid #334155; }}
  h1 {{ margin-top: 0; font-size: 24px; color: #38bdf8; display: flex; justify-content: space-between; align-items: center; }}
  .badge {{ background: #0284c7; color: white; font-size: 13px; padding: 4px 10px; border-radius: 6px; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }}
  .card {{ background: #0f172a; padding: 16px; border-radius: 8px; border: 1px solid #334155; }}
  .card-label {{ font-size: 13px; color: #94a3b8; }}
  .card-value {{ font-size: 24px; font-weight: 700; margin-top: 6px; font-family: 'IBM Plex Mono', monospace; }}
  .green {{ color: #34d399; }}
  .red {{ color: #f87171; }}
  .blue {{ color: #38bdf8; }}
  .chart-box {{ background: #0f172a; border-radius: 8px; border: 1px solid #334155; padding: 16px; margin: 24px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 13px; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #334155; }}
  th {{ background: #1e293b; color: #94a3b8; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
<div class="container">
  <h1>
    <span>策略绩效报告 (Strategy Performance Tearsheet)</span>
    <span class="badge">基准: {self.benchmark}</span>
  </h1>
  <p style="color: #94a3b8; font-size: 14px;">区间: {dates[0]} 至 {dates[-1]} ({len(dates)} 交易日) | 初始资金: ¥{self.portfolio_values.iloc[0]:,.2f}</p>

  <div class="grid">
    <div class="card">
      <div class="card-label">策略累计收益率</div>
      <div class="card-value {'green' if m.get('total_return', 0) >= 0 else 'red'}">{m.get('total_return', 0)*100:+.2f}%</div>
    </div>
    <div class="card">
      <div class="card-label">年化收益率 (CAGR)</div>
      <div class="card-value {'green' if m.get('annualized_return', 0) >= 0 else 'red'}">{m.get('annualized_return', 0)*100:+.2f}%</div>
    </div>
    <div class="card">
      <div class="card-label">夏普比率 (Sharpe)</div>
      <div class="card-value blue">{m.get('sharpe', 0):.2f}</div>
    </div>
    <div class="card">
      <div class="card-label">最大回撤 (MaxDD)</div>
      <div class="card-value red">{m.get('max_drawdown', 0)*100:.2f}%</div>
    </div>
    <div class="card">
      <div class="card-label">超额年化收益 (Alpha)</div>
      <div class="card-value {'green' if m.get('alpha', 0) >= 0 else 'red'}">{m.get('alpha', 0)*100:+.2f}%</div>
    </div>
    <div class="card">
      <div class="card-label">贝塔系数 (Beta)</div>
      <div class="card-value blue">{m.get('beta', 1.0):.2f}</div>
    </div>
    <div class="card">
      <div class="card-label">卡玛比率 (Calmar)</div>
      <div class="card-value blue">{m.get('calmar', 0):.2f}</div>
    </div>
    <div class="card">
      <div class="card-label">成交流水笔数</div>
      <div class="card-value blue">{len(self.trades)} 笔</div>
    </div>
  </div>

  <div class="chart-box">
    <h3 style="margin-top: 0; color: #e2e8f0;">净值曲线 vs 基准 ({self.benchmark}) [Chart.js 交互视图]</h3>
    <canvas id="navChart" height="90"></canvas>
  </div>

  <h3 style="color: #e2e8f0; margin-top: 32px;">最新交易流水 (Recent Trades)</h3>
  <table>
    <thead>
      <tr>
        <th>日期</th>
        <th>代码</th>
        <th>操作</th>
        <th>成交数量</th>
        <th>成交均价</th>
        <th>佣金</th>
        <th>印花税</th>
      </tr>
    </thead>
    <tbody>
      {"".join(f"<tr><td>{t.get('date','-')}</td><td>{t.get('security','-')}</td><td style='color: {'#34d399' if t.get('action')=='BUY' else '#f87171'}'>{t.get('action','-')}</td><td>{t.get('amount',0)}</td><td>¥{t.get('price',0):.2f}</td><td>¥{t.get('commission',0):.2f}</td><td>¥{t.get('tax',0):.2f}</td></tr>" for t in self.trades.tail(15).to_dict('records')) if not self.trades.empty else "<tr><td colspan='7' style='text-align: center; color: #64748b;'>无交易记录</td></tr>"}
    </tbody>
  </table>
</div>

<script>
  const ctx = document.getElementById('navChart').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: {dates},
      datasets: [
        {{ label: '策略资产', data: {p_vals}, borderColor: '#38bdf8', borderWidth: 2, fill: false, tension: 0.1, pointRadius: 0 }},
        {{ label: '基准表现', data: {b_vals}, borderColor: '#64748b', borderWidth: 1.5, borderDash: [4, 4], fill: false, tension: 0.1, pointRadius: 0 }}
      ]
    }},
    options: {{
      responsive: true,
      interaction: {{ mode: 'index', intersect: false }},
      scales: {{
        x: {{ grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8', maxTicksLimit: 12 }} }},
        y: {{ grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }} }}
      }}
    }}
  }});
</script>
</body>
</html>
"""
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)
        return html


# --------------------------------------------------------------------------- #
# Core Strategy Execution Engine
# --------------------------------------------------------------------------- #
class StrategyEngine:
    """Institutional Event-Driven execution host."""

    def __init__(
        self,
        strategy_instance: Strategy | None = None,
        initialize: Callable[[Context], None] | None = None,
        handle_data: Callable[[Context, CurrentData], None] | None = None,
        before_trading_start: Callable[[Context], None] | None = None,
        start_date: str = "20240101",
        end_date: str = "20240630",
        initial_cash: float = 1_000_000.0,
        benchmark: str = "000300.SH",
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
        db_path: str = "tushare_data.duckdb",
        custom_data: pd.DataFrame | None = None,
    ):
        self.strategy_instance = strategy_instance
        if self.strategy_instance is not None:
            self.strategy_instance.engine = self

        self.initialize_fn = initialize
        self.handle_data_fn = handle_data
        self.before_trading_start_fn = before_trading_start

        self.start_date = str(start_date).replace("-", "")
        self.end_date = str(end_date).replace("-", "")
        self.benchmark = benchmark
        self.db_path = db_path
        self.custom_data = custom_data

        self.context = Context(initial_cash=float(initial_cash))
        self.context.benchmark = benchmark
        self.context.run_params = {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_cash": initial_cash,
        }

        self.slippage: SlippageModel = slippage or FixedSlippage(spread=0.002)
        self.commission: CommissionModel = commission or AshareCommission()
        self.options: dict[str, Any] = {"use_real_price": True}
        self.tasks: list[ScheduledTask] = []

        self.current_data = CurrentData()
        self.order_queue: list[Order] = []
        self.all_orders: list[Order] = []
        self.all_trades: list[dict[str, Any]] = []
        self.positions_history: list[dict[str, Any]] = []

        # Data cache: {date: {security: SecurityData}}
        self._price_panel: dict[str, dict[str, SecurityData]] = {}
        self._trade_dates: list[str] = []
        self._benchmark_returns: dict[str, float] = {}
        self._raw_df: pd.DataFrame | None = None

    def load_data(self) -> None:
        """Loads price, limit, suspension, and benchmark records."""
        if self.custom_data is not None:
            df = self.custom_data.copy()
            df["trade_date"] = df["trade_date"].astype(str)
            df["ts_code"] = df["ts_code"].astype(str)
        else:
            from data_foundation import catalog
            conn = catalog.connect(self.db_path, read_only=True)
            try:
                # Load daily market data with limits
                query = f"""
                SELECT
                    d.ts_code,
                    d.trade_date,
                    d.open,
                    d.high,
                    d.low,
                    d.close,
                    d.vol,
                    d.amount,
                    lim.up_limit,
                    lim.down_limit
                FROM daily d
                LEFT JOIN stk_limit lim
                    ON d.ts_code = lim.ts_code AND d.trade_date = lim.trade_date
                WHERE d.trade_date >= '{self.start_date}' AND d.trade_date <= '{self.end_date}'
                ORDER BY d.trade_date, d.ts_code
                """
                df = conn.execute(query).df()

                # Load benchmark returns
                bench_query = f"""
                SELECT trade_date, pct_chg / 100.0 as ret
                FROM index_daily
                WHERE ts_code = '{self.benchmark}' AND trade_date >= '{self.start_date}' AND trade_date <= '{self.end_date}'
                ORDER BY trade_date
                """
                b_df = conn.execute(bench_query).df()
                if not b_df.empty:
                    self._benchmark_returns = dict(zip(b_df["trade_date"].astype(str), b_df["ret"].astype(float)))
            finally:
                conn.close()

        self._raw_df = df
        dates = sorted(df["trade_date"].unique().tolist())
        self._trade_dates = [d for d in dates if self.start_date <= d <= self.end_date]

        for _, row in df.iterrows():
            d = str(row["trade_date"])
            sec = str(row["ts_code"])
            if d not in self._price_panel:
                self._price_panel[d] = {}

            up_lim = float(row["up_limit"]) if pd.notna(row.get("up_limit")) else float("inf")
            dn_lim = float(row["down_limit"]) if pd.notna(row.get("down_limit")) else 0.0

            sec_data = SecurityData(
                security=sec,
                last_price=float(row["close"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["vol"]),
                amount=float(row["amount"]),
                high_limit=up_lim,
                low_limit=dn_lim,
                day_open=float(row["open"]),
                paused=False,
            )
            self._price_panel[d][sec] = sec_data

    def get_current_price(self, security: str) -> float:
        d = self.context.current_dt
        if d in self._price_panel and security in self._price_panel[d]:
            return self._price_panel[d][security].open
        return 0.0

    def query_price(
        self,
        security: str | list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        count: int | None = None,
        fields: list[str] | str | None = None,
    ) -> pd.DataFrame | pd.Series:
        if self._raw_df is None:
            return pd.DataFrame()
        df = self._raw_df
        current_dt = self.context.current_dt

        secs = [security] if isinstance(security, str) else security
        df = df[df["ts_code"].isin(secs)]

        if end_date:
            df = df[df["trade_date"] <= end_date]
        elif current_dt:
            df = df[df["trade_date"] <= current_dt]

        if start_date:
            df = df[df["trade_date"] >= start_date]

        if count is not None and count > 0:
            df = df.sort_values("trade_date").groupby("ts_code").tail(count)

        if fields:
            f_list = [fields] if isinstance(fields, str) else list(fields)
            cols = ["trade_date", "ts_code"] + [f for f in f_list if f in df.columns]
            df = df[cols]

        return df

    def query_history(
        self,
        count: int,
        unit: str = "1d",
        field: str = "close",
        security_list: list[str] | None = None,
    ) -> pd.DataFrame:
        if self._raw_df is None:
            return pd.DataFrame()
        current_dt = self.context.current_dt
        sub = self._raw_df[self._raw_df["trade_date"] <= current_dt]
        if security_list:
            sub = sub[sub["ts_code"].isin(security_list)]

        piv = sub.pivot_table(index="trade_date", columns="ts_code", values=field, aggfunc="last")
        return piv.tail(count)

    def query_attribute_history(
        self,
        security: str,
        count: int,
        unit: str = "1d",
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        cols = fields or ["open", "close", "high", "low", "vol"]
        df = self.query_price(security, count=count, fields=cols)
        if not df.empty and "trade_date" in df.columns:
            df = df.set_index("trade_date")
        return df

    def place_order(self, security: str, amount: int, style: OrderStyle | None = None) -> Order | None:
        if amount == 0:
            return None
        action = "BUY" if amount > 0 else "SELL"
        # A-share board lot rule: Buys must be multiples of 100 shares
        if action == "BUY":
            amount = (amount // 100) * 100
            if amount <= 0:
                return None

        o = Order(
            order_id=str(uuid.uuid4())[:8],
            security=security,
            amount=amount,
            action=action,
            style=style or MarketOrderStyle(),
            created_time=self.context.current_dt,
        )
        self.order_queue.append(o)
        self.all_orders.append(o)
        return o

    def cancel_order(self, order_id: str) -> bool:
        for o in self.order_queue:
            if o.order_id == order_id and o.status == "SUBMITTED":
                o.status = "CANCELLED"
                self.order_queue.remove(o)
                return True
        return False

    def _execute_orders(self, current_prices: dict[str, SecurityData]) -> None:
        """Executes queued orders against current bar prices enforcing A-share friction."""
        while self.order_queue:
            o = self.order_queue.pop(0)
            if o.status != "SUBMITTED":
                continue

            sec = o.security
            if sec not in current_prices:
                o.status = "REJECTED"
                o.reject_reason = "No market price data on execution bar"
                continue

            sec_data = current_prices[sec]
            if sec_data.paused:
                o.status = "REJECTED"
                o.reject_reason = "Security is suspended / halted"
                continue

            # Limit price check
            open_p = sec_data.open
            if o.action == "BUY" and open_p >= sec_data.high_limit and sec_data.high_limit > 0:
                o.status = "REJECTED"
                o.reject_reason = "Limit-up reached (涨停无法买入)"
                continue
            if o.action == "SELL" and open_p <= sec_data.low_limit and sec_data.low_limit > 0:
                o.status = "REJECTED"
                o.reject_reason = "Limit-down reached (跌停无法卖出)"
                continue

            # Slippage pricing
            fill_p = self.slippage.calculate_slippage(o, open_p, sec_data.volume)
            pos = self.context.portfolio.positions.get(sec, Position(security=sec))

            if o.action == "BUY":
                notional = abs(o.amount) * fill_p
                comm, tax = self.commission.calculate(o, notional)
                total_cost = notional + comm + tax
                if self.context.portfolio.cash < total_cost:
                    # Scale down to available cash in 100-share multiples
                    avail_cash = self.context.portfolio.cash - comm - tax
                    max_shares = int(avail_cash / fill_p // 100) * 100
                    if max_shares <= 0:
                        o.status = "REJECTED"
                        o.reject_reason = f"Insufficient cash (Required: {total_cost:.2f}, Avail: {self.context.portfolio.cash:.2f})"
                        continue
                    o.amount = max_shares
                    notional = max_shares * fill_p
                    comm, tax = self.commission.calculate(o, notional)
                    total_cost = notional + comm + tax

                # Execute Buy
                self.context.portfolio.cash -= total_cost
                new_total = pos.total_amount + o.amount
                pos.avg_cost = (pos.avg_cost * pos.total_amount + notional) / new_total if new_total > 0 else fill_p
                pos.total_amount = new_total
                pos.today_amount += o.amount  # T+1 lock
                pos.price = fill_p
                self.context.portfolio.positions[sec] = pos

                o.status = "FILLED"
                o.filled_amount = o.amount
                o.filled_price = fill_p
                o.commission = comm
                o.tax = tax
                o.filled_time = self.context.current_dt
                self.all_trades.append({
                    "trade_id": str(uuid.uuid4())[:8],
                    "order_id": o.order_id,
                    "security": sec,
                    "action": "BUY",
                    "amount": o.amount,
                    "price": fill_p,
                    "commission": comm,
                    "tax": tax,
                    "date": self.context.current_dt,
                })

            else:  # SELL
                # Enforce T+1 available sellable shares
                sell_shares = abs(o.amount)
                if sell_shares > pos.closeable_amount:
                    sell_shares = pos.closeable_amount
                    if sell_shares <= 0:
                        o.status = "REJECTED"
                        o.reject_reason = "T+1 constraint: no available closeable shares (可卖持仓不足)"
                        continue
                    o.amount = -sell_shares

                notional = sell_shares * fill_p
                comm, tax = self.commission.calculate(o, notional)
                net_proceeds = notional - comm - tax

                self.context.portfolio.cash += net_proceeds
                pos.total_amount -= sell_shares
                pos.closeable_amount -= sell_shares
                pos.price = fill_p
                if pos.total_amount <= 0:
                    self.context.portfolio.positions.pop(sec, None)
                else:
                    self.context.portfolio.positions[sec] = pos

                o.status = "FILLED"
                o.filled_amount = sell_shares
                o.filled_price = fill_p
                o.commission = comm
                o.tax = tax
                o.filled_time = self.context.current_dt
                self.all_trades.append({
                    "trade_id": str(uuid.uuid4())[:8],
                    "order_id": o.order_id,
                    "security": sec,
                    "action": "SELL",
                    "amount": sell_shares,
                    "price": fill_p,
                    "commission": comm,
                    "tax": tax,
                    "date": self.context.current_dt,
                })

    def run(self) -> StrategyResult:
        """Main event-driven backtest loop."""
        global _ACTIVE_ENGINE
        _ACTIVE_ENGINE = self

        self.load_data()
        if not self._trade_dates:
            raise ValueError(f"No trading dates found between {self.start_date} and {self.end_date}")

        # 1. Initialize
        if self.strategy_instance is not None:
            self.strategy_instance.on_init(self.context)
        elif self.initialize_fn:
            self.initialize_fn(self.context)

        portfolio_values: list[float] = []
        daily_returns: list[float] = []
        recorded_dates: list[str] = []
        benchmark_returns: list[float] = []
        prev_total = self.context.portfolio.total_value

        # 2. Iterate trading dates
        for d in self._trade_dates:
            self.as_of_date = d
            self.context.current_dt = d
            dt_obj = datetime.strptime(d, "%Y%m%d")

            # T+1 unlock: unlock yesterday's bought shares
            for pos in self.context.portfolio.positions.values():
                pos.closeable_amount += pos.today_amount
                pos.today_amount = 0

            day_prices = self._price_panel.get(d, {})
            # Update CurrentData container
            self.current_data.clear()
            self.current_data.update(day_prices)

            # Update position prices to day open for intra-day decisions
            for sec, pos in self.context.portfolio.positions.items():
                if sec in day_prices:
                    pos.price = day_prices[sec].open

            # 3. Pre-open hook
            if self.strategy_instance is not None:
                self.strategy_instance.on_before_trading(self.context)
            elif self.before_trading_start_fn:
                self.before_trading_start_fn(self.context)

            # 4. Trigger scheduled tasks
            for task in self.tasks:
                if task.freq == "daily" or task.freq == "weekly" and dt_obj.isoweekday() == task.day_param or task.freq == "monthly" and dt_obj.day == task.day_param:
                    task.func(self.context)

            # 5. Handle data hook
            if self.strategy_instance is not None:
                self.strategy_instance.on_bar(self.context, self.current_data)
            elif self.handle_data_fn:
                self.handle_data_fn(self.context, self.current_data)

            # 6. Execute queued orders
            self._execute_orders(day_prices)

            # 7. Day close valuation (Mark-to-market at close price)
            for sec, pos in self.context.portfolio.positions.items():
                if sec in day_prices:
                    pos.price = day_prices[sec].close

            self.context.portfolio.update_valuation()
            curr_val = self.context.portfolio.total_value
            d_ret = (curr_val - prev_total) / prev_total if prev_total > 0 else 0.0

            portfolio_values.append(curr_val)
            daily_returns.append(d_ret)
            recorded_dates.append(d)
            benchmark_returns.append(self._benchmark_returns.get(d, 0.0))
            prev_total = curr_val

            # Record position snapshot
            for sec, pos in self.context.portfolio.positions.items():
                self.positions_history.append({
                    "date": d,
                    "security": sec,
                    "amount": pos.total_amount,
                    "price": pos.price,
                    "value": pos.value,
                    "pnl": pos.pnl,
                })

        _ACTIVE_ENGINE = None

        p_series = pd.Series(portfolio_values, index=recorded_dates)
        r_series = pd.Series(daily_returns, index=recorded_dates)
        b_series = pd.Series(benchmark_returns, index=recorded_dates)

        return StrategyResult(
            portfolio_values=p_series,
            daily_returns=r_series,
            benchmark_returns=b_series,
            trades=self.all_trades,
            positions_history=self.positions_history,
            orders=self.all_orders,
            benchmark=self.context.benchmark,
        )


# --------------------------------------------------------------------------- #
# Public Convenient Launcher
# --------------------------------------------------------------------------- #
def run_strategy(
    strategy: Strategy | type[Strategy] | Callable[[Context], None] | None = None,
    *,
    initialize: Callable[[Context], None] | None = None,
    handle_data: Callable[[Context, CurrentData], None] | None = None,
    before_trading_start: Callable[[Context], None] | None = None,
    start_date: str = "20240101",
    end_date: str = "20240630",
    initial_cash: float = 1_000_000.0,
    benchmark: str = "000300.SH",
    slippage: SlippageModel | None = None,
    commission: CommissionModel | None = None,
    db_path: str = "tushare_data.duckdb",
    custom_data: pd.DataFrame | None = None,
) -> StrategyResult:
    """Run an event-driven quantitative strategy in the local sandbox."""
    init_fn = initialize
    handle_fn = handle_data
    before_fn = before_trading_start
    strategy_inst: Strategy | None = None

    if isinstance(strategy, type) and issubclass(strategy, Strategy):
        strategy_inst = strategy()
    elif isinstance(strategy, Strategy):
        strategy_inst = strategy
    elif callable(strategy):
        init_fn = strategy

    engine = StrategyEngine(
        strategy_instance=strategy_inst,
        initialize=init_fn,
        handle_data=handle_fn,
        before_trading_start=before_fn,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        benchmark=benchmark,
        slippage=slippage,
        commission=commission,
        db_path=db_path,
        custom_data=custom_data,
    )
    return engine.run()
