"""Paper Trading Engine (模拟盘引擎) for local automated tracking.

Allows researchers to deploy quantitative strategies to persistent local simulated accounts:
- Daily state machine advancing day-by-day (or on live bar triggers)
- Persistent JSON/SQLite accounts stored in `var/paper_trading/`
- Full A-share microstructure: T+1 lock, 100-share board lot, price limit checks, stamp duty, min commission
- Complete trade log, position snapshots, and daily NAV equity curves
"""
from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_core import metrics as M
from quant_sdk.strategy import (
    AshareCommission,
    CommissionModel,
    Context,
    CurrentData,
    FixedSlippage,
    Order,
    Position,
    SecurityData,
    SlippageModel,
    Strategy,
)

__all__ = [
    "PaperAccount",
    "PaperTradingManager",
    "create_paper_account",
    "get_paper_account",
    "list_paper_accounts",
    "step_paper_account",
]

_DEFAULT_STORAGE_DIR = Path("var") / "paper_trading"


# --------------------------------------------------------------------------- #
# Paper Trading Account State
# --------------------------------------------------------------------------- #
@dataclass
class PaperAccount:
    """Persistent state of a simulated trading account."""

    account_id: str
    strategy_name: str
    initial_cash: float
    cash: float
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    total_value: float = 0.0
    daily_equity: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    orders: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    last_trade_date: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        if self.total_value == 0.0:
            self.total_value = self.cash

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperAccount:
        return cls(
            account_id=data["account_id"],
            strategy_name=data.get("strategy_name", "unnamed_strategy"),
            initial_cash=float(data.get("initial_cash", 1_000_000.0)),
            cash=float(data.get("cash", 1_000_000.0)),
            positions=data.get("positions", {}),
            total_value=float(data.get("total_value", 1_000_000.0)),
            daily_equity=data.get("daily_equity", []),
            trades=data.get("trades", []),
            orders=data.get("orders", []),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            last_trade_date=data.get("last_trade_date", ""),
        )

    def save_to_file(self, file_path: Path | str) -> None:
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now().isoformat()
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, file_path: Path | str) -> PaperAccount:
        target = Path(file_path)
        with open(target, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def execute_order(
        self,
        order: Order,
        sec_data: SecurityData,
        commission_model: CommissionModel,
        slippage_model: SlippageModel,
    ) -> bool:
        """Execute order against current bar quote enforcing A-share friction."""
        sec = order.security
        open_p = sec_data.open

        # 1. Price limit check
        if order.action == "BUY" and open_p >= sec_data.high_limit and sec_data.high_limit > 0:
            order.status = "REJECTED"
            order.reject_reason = "Limit-up reached (涨停无法买入)"
            self._record_order(order)
            return False
        if order.action == "SELL" and open_p <= sec_data.low_limit and sec_data.low_limit > 0:
            order.status = "REJECTED"
            order.reject_reason = "Limit-down reached (跌停无法卖出)"
            self._record_order(order)
            return False

        fill_p = slippage_model.calculate_slippage(order, open_p, sec_data.volume)
        pos = self.positions.get(sec, {
            "security": sec,
            "total_amount": 0,
            "closeable_amount": 0,
            "today_amount": 0,
            "avg_cost": 0.0,
            "price": fill_p,
            "value": 0.0,
            "pnl": 0.0,
        })

        if order.action == "BUY":
            notional = abs(order.amount) * fill_p
            comm, tax = commission_model.calculate(order, notional)
            total_cost = notional + comm + tax

            if self.cash < total_cost:
                avail_cash = self.cash - comm - tax
                max_shares = int(avail_cash / fill_p // 100) * 100
                if max_shares <= 0:
                    order.status = "REJECTED"
                    order.reject_reason = f"Insufficient cash (Required: {total_cost:.2f}, Avail: {self.cash:.2f})"
                    self._record_order(order)
                    return False
                order.amount = max_shares
                notional = max_shares * fill_p
                comm, tax = commission_model.calculate(order, notional)
                total_cost = notional + comm + tax

            self.cash -= total_cost
            new_total = pos["total_amount"] + order.amount
            pos["avg_cost"] = (pos["avg_cost"] * pos["total_amount"] + notional) / new_total if new_total > 0 else fill_p
            pos["total_amount"] = new_total
            pos["today_amount"] += order.amount  # T+1 lock
            pos["price"] = fill_p
            pos["value"] = new_total * fill_p
            pos["pnl"] = (fill_p - pos["avg_cost"]) * new_total
            self.positions[sec] = pos

            order.status = "FILLED"
            order.filled_amount = order.amount
            order.filled_price = fill_p
            order.commission = comm
            order.tax = tax
            order.filled_time = order.created_time
            self._record_trade(order, sec, "BUY", order.amount, fill_p, comm, tax)
            self._record_order(order)
            return True

        else:  # SELL
            sell_shares = abs(order.amount)
            if sell_shares > pos["closeable_amount"]:
                sell_shares = pos["closeable_amount"]
                if sell_shares <= 0:
                    order.status = "REJECTED"
                    order.reject_reason = "T+1 constraint: no available closeable shares (可卖持仓不足)"
                    self._record_order(order)
                    return False
                order.amount = -sell_shares

            notional = sell_shares * fill_p
            comm, tax = commission_model.calculate(order, notional)
            net_proceeds = notional - comm - tax

            self.cash += net_proceeds
            pos["total_amount"] -= sell_shares
            pos["closeable_amount"] -= sell_shares
            pos["price"] = fill_p
            pos["value"] = pos["total_amount"] * fill_p
            pos["pnl"] = (fill_p - pos["avg_cost"]) * pos["total_amount"]

            if pos["total_amount"] <= 0:
                self.positions.pop(sec, None)
            else:
                self.positions[sec] = pos

            order.status = "FILLED"
            order.filled_amount = sell_shares
            order.filled_price = fill_p
            order.commission = comm
            order.tax = tax
            order.filled_time = order.created_time
            self._record_trade(order, sec, "SELL", sell_shares, fill_p, comm, tax)
            self._record_order(order)
            return True

    def settle_daily(self, trade_date: str, day_prices: dict[str, SecurityData]) -> None:
        """Mark-to-market day close valuation and T+1 unlock."""
        # 1. T+1 unlock
        for pos in self.positions.values():
            pos["closeable_amount"] += pos.get("today_amount", 0)
            pos["today_amount"] = 0

        # 2. Mark-to-market at close
        positions_val = 0.0
        for sec, pos in self.positions.items():
            if sec in day_prices:
                pos["price"] = day_prices[sec].close
            pos["value"] = pos["total_amount"] * pos["price"]
            pos["pnl"] = (pos["price"] - pos["avg_cost"]) * pos["total_amount"]
            positions_val += pos["value"]

        prev_total = self.total_value
        self.total_value = self.cash + positions_val
        d_ret = (self.total_value - prev_total) / prev_total if prev_total > 0 else 0.0

        self.last_trade_date = trade_date
        self.daily_equity.append({
            "date": trade_date,
            "total_value": round(self.total_value, 2),
            "cash": round(self.cash, 2),
            "positions_value": round(positions_val, 2),
            "daily_return": round(d_ret, 6),
            "holdings_count": len(self.positions),
        })

    def _record_trade(self, order: Order, sec: str, action: str, amount: int, price: float, comm: float, tax: float):
        self.trades.append({
            "trade_id": str(uuid.uuid4())[:8],
            "order_id": order.order_id,
            "security": sec,
            "action": action,
            "amount": amount,
            "price": round(price, 4),
            "commission": round(comm, 2),
            "tax": round(tax, 2),
            "date": order.created_time,
        })

    def _record_order(self, order: Order):
        self.orders.append({
            "order_id": order.order_id,
            "security": order.security,
            "amount": order.amount,
            "action": order.action,
            "status": order.status,
            "filled_amount": order.filled_amount,
            "filled_price": round(order.filled_price, 4),
            "commission": round(order.commission, 2),
            "tax": round(order.tax, 2),
            "created_time": order.created_time,
            "reject_reason": order.reject_reason,
        })

    def get_performance_summary(self) -> dict[str, Any]:
        """Calculates key institutional metrics from daily equity series."""
        if not self.daily_equity:
            return {
                "account_id": self.account_id,
                "strategy_name": self.strategy_name,
                "initial_cash": self.initial_cash,
                "current_value": self.total_value,
                "total_return": 0.0,
                "annualized_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe": 0.0,
                "win_rate": 0.0,
                "total_trades": len(self.trades),
                "total_days": 0,
                "last_date": self.last_trade_date,
            }

        values = [row["total_value"] for row in self.daily_equity]
        rets = np.array([row["daily_return"] for row in self.daily_equity], dtype=float)

        total_ret = (values[-1] - self.initial_cash) / self.initial_cash if self.initial_cash > 0 else 0.0
        ann_ret = float(M.annualized_return(rets)) if len(rets) > 1 else 0.0
        mdd = float(M.max_drawdown(np.array(values, dtype=float))) if len(values) > 1 else 0.0
        sharpe = float(M.sharpe_ratio(rets)) if len(rets) > 1 else 0.0
        win_rate = float(M.win_rate(rets)) if len(rets) > 1 else 0.0

        return {
            "account_id": self.account_id,
            "strategy_name": self.strategy_name,
            "initial_cash": self.initial_cash,
            "current_value": values[-1],
            "total_return": total_ret,
            "annualized_return": ann_ret,
            "max_drawdown": mdd,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "total_trades": len(self.trades),
            "total_days": len(self.daily_equity),
            "last_date": self.last_trade_date,
        }


# --------------------------------------------------------------------------- #
# Paper Execution Engine
# --------------------------------------------------------------------------- #
class PaperExecutionEngine:
    """Active execution engine during paper trading daily step."""

    def __init__(
        self,
        account: PaperAccount,
        current_prices: dict[str, SecurityData],
        as_of_date: str,
        slippage: SlippageModel,
        commission: CommissionModel,
    ):
        self.account = account
        self.current_prices = current_prices
        self.as_of_date = as_of_date
        self.slippage = slippage
        self.commission = commission
        self.order_queue: list[Order] = []

        self.context = Context(initial_cash=account.initial_cash)
        self.context.portfolio.cash = account.cash
        self.context.portfolio.positions = {
            sec: Position(
                security=sec,
                total_amount=p["total_amount"],
                closeable_amount=p["closeable_amount"],
                today_amount=p.get("today_amount", 0),
                avg_cost=p["avg_cost"],
                price=current_prices[sec].open if sec in current_prices else p["price"],
            )
            for sec, p in account.positions.items()
        }
        self.context.portfolio.update_valuation()
        self.context.current_dt = as_of_date

        self.current_data = CurrentData()
        self.current_data.update(current_prices)

    def get_current_price(self, security: str) -> float:
        if security in self.current_prices:
            return self.current_prices[security].open
        return 0.0

    def place_order(self, security: str, amount: int, style=None) -> Order | None:
        if amount == 0:
            return None
        action = "BUY" if amount > 0 else "SELL"
        if action == "BUY":
            amount = (amount // 100) * 100
            if amount <= 0:
                return None
        o = Order(
            order_id=str(uuid.uuid4())[:8],
            security=security,
            amount=amount,
            action=action,
            created_time=self.as_of_date,
        )
        self.order_queue.append(o)
        return o

    def cancel_order(self, order_id: str) -> bool:
        for o in self.order_queue:
            if o.order_id == order_id and o.status == "SUBMITTED":
                o.status = "CANCELLED"
                return True
        return False


# --------------------------------------------------------------------------- #
# Paper Trading Manager
# --------------------------------------------------------------------------- #
class PaperTradingManager:
    """Manages creation, stepping, and inspection of local paper trading accounts."""

    def __init__(self, storage_dir: Path | str | None = None):
        self.storage_dir = Path(storage_dir or _DEFAULT_STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def account_file(self, account_id: str) -> Path:
        return self.storage_dir / f"{account_id}.json"

    def create_account(
        self,
        account_id: str,
        strategy_name: str = "unnamed_strategy",
        initial_cash: float = 1_000_000.0,
    ) -> PaperAccount:
        file = self.account_file(account_id)
        if file.exists():
            raise FileExistsError(f"Paper trading account {account_id} already exists at {file}")
        account = PaperAccount(
            account_id=account_id,
            strategy_name=strategy_name,
            initial_cash=float(initial_cash),
            cash=float(initial_cash),
        )
        account.save_to_file(file)
        return account

    def get_account(self, account_id: str) -> PaperAccount:
        file = self.account_file(account_id)
        if not file.exists():
            raise FileNotFoundError(f"Paper trading account {account_id} not found at {file}")
        return PaperAccount.load_from_file(file)

    def list_accounts(self) -> list[dict[str, Any]]:
        results = []
        for file in self.storage_dir.glob("*.json"):
            try:
                acc = PaperAccount.load_from_file(file)
                results.append(acc.get_performance_summary())
            except Exception:
                continue
        return results

    def step_account(
        self,
        account_id: str,
        strategy: Strategy | type[Strategy] | Callable[[Context, CurrentData], None],
        as_of_date: str,
        db_path: str = "tushare_data.duckdb",
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
        custom_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Advance paper account by one trading day using current market quotes."""
        account = self.get_account(account_id)
        as_of_date = str(as_of_date).replace("-", "")

        # 1. Load today's quotes from DuckDB or custom_data
        if custom_data is not None:
            df = custom_data[custom_data["trade_date"].astype(str) == as_of_date]
        else:
            from data_foundation import catalog
            conn = catalog.connect(db_path, read_only=True)
            try:
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
                WHERE d.trade_date = '{as_of_date}'
                """
                df = conn.execute(query).df()
            finally:
                conn.close()

        if df.empty:
            raise ValueError(f"No price data found for date {as_of_date}")

        current_prices: dict[str, SecurityData] = {}
        for _, row in df.iterrows():
            sec = str(row["ts_code"])
            up_lim = float(row["up_limit"]) if pd.notna(row.get("up_limit")) else float("inf")
            dn_lim = float(row["down_limit"]) if pd.notna(row.get("down_limit")) else 0.0
            current_prices[sec] = SecurityData(
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

        slip = slippage or FixedSlippage()
        comm = commission or AshareCommission()
        engine = PaperExecutionEngine(account, current_prices, as_of_date, slip, comm)

        import quant_sdk.strategy as strat_mod
        prev_engine = strat_mod._ACTIVE_ENGINE
        strat_mod._ACTIVE_ENGINE = engine
        try:
            if isinstance(strategy, type) and issubclass(strategy, Strategy):
                inst = strategy()
                inst.handle_data(engine.context, engine.current_data)
            elif isinstance(strategy, Strategy):
                strategy.handle_data(engine.context, engine.current_data)
            elif callable(strategy):
                strategy(engine.context, engine.current_data)
        finally:
            strat_mod._ACTIVE_ENGINE = prev_engine

        # 4. Execute orders against PaperAccount
        trades_today = []
        for o in engine.order_queue:
            if o.security in current_prices:
                success = account.execute_order(o, current_prices[o.security], comm, slip)
                if success:
                    trades_today.append(o)

        # 5. Settle daily close mark-to-market
        account.settle_daily(as_of_date, current_prices)
        account.save_to_file(self.account_file(account_id))

        return {
            "account_id": account_id,
            "trade_date": as_of_date,
            "total_value": account.total_value,
            "cash": account.cash,
            "positions_count": len(account.positions),
            "trades_count": len(trades_today),
            "summary": account.get_performance_summary(),
        }


# --------------------------------------------------------------------------- #
# Public Helpers
# --------------------------------------------------------------------------- #
def create_paper_account(account_id: str, strategy_name: str = "unnamed", initial_cash: float = 1_000_000.0) -> PaperAccount:
    return PaperTradingManager().create_account(account_id, strategy_name, initial_cash)


def get_paper_account(account_id: str) -> PaperAccount:
    return PaperTradingManager().get_account(account_id)


def list_paper_accounts() -> list[dict[str, Any]]:
    return PaperTradingManager().list_accounts()


def step_paper_account(
    account_id: str,
    strategy: Any,
    as_of_date: str,
    db_path: str = "tushare_data.duckdb",
    custom_data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    return PaperTradingManager().step_account(
        account_id=account_id,
        strategy=strategy,
        as_of_date=as_of_date,
        db_path=db_path,
        custom_data=custom_data,
    )
