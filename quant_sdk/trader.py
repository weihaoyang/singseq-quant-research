"""Broker Live Execution Gateway & miniQMT (迅投 XtQuant) Connector.

Provides institutional-grade local live trading integration:
- Direct local IPC connection to broker miniQMT terminals (国金, 华泰, 银河, 中信建投等)
- Zero cloud transmission: strategy signals and credentials stay strictly on the user's workstation
- Pre-trade risk control firewall (TraderRiskController):
    - Single-order notional caps
    - Rate-limiting (max orders/sec to prevent broker ban)
    - Lot sizing constraints & ST blacklist filters
- Drop-in MockQmtTrader for headless verification and offline strategy testing.
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

__all__ = [
    "BaseTrader",
    "LiveOrder",
    "LivePosition",
    "MockQmtTrader",
    "QmtTrader",
    "TraderRiskController",
]

logger = logging.getLogger("quant_sdk.trader")


# --------------------------------------------------------------------------- #
# Trade Data Models
# --------------------------------------------------------------------------- #
@dataclass
class LivePosition:
    security: str
    volume: int
    can_use_volume: int
    open_price: float
    market_value: float
    avg_price: float = 0.0

    @property
    def is_empty(self) -> bool:
        return self.volume <= 0


@dataclass
class LiveOrder:
    order_id: str
    security: str
    action: Literal["BUY", "SELL"]
    amount: int
    price: float
    order_type: Literal["MARKET", "LIMIT"] = "LIMIT"
    status: Literal["SUBMITTED", "FILLED", "CANCELLED", "REJECTED"] = "SUBMITTED"
    created_time: str = ""
    filled_amount: int = 0
    filled_price: float = 0.0
    reject_reason: str = ""


# --------------------------------------------------------------------------- #
# Pre-Trade Risk Control Firewall
# --------------------------------------------------------------------------- #
class TraderRiskController:
    """Pre-trade safety firewall preventing catastrophic execution errors."""

    def __init__(
        self,
        max_order_value: float = 1_000_000.0,
        max_daily_notional: float = 10_000_000.0,
        max_orders_per_second: float = 5.0,
        blacklist: set[str] | None = None,
        allow_st: bool = False,
    ):
        self.max_order_value = float(max_order_value)
        self.max_daily_notional = float(max_daily_notional)
        self.max_orders_per_second = float(max_orders_per_second)
        self.blacklist = set(blacklist or set())
        self.allow_st = allow_st

        self._order_timestamps: deque[float] = deque()
        self._daily_traded_notional: float = 0.0
        self._current_trade_date: str = datetime.now().strftime("%Y%m%d")

    def validate_order(
        self,
        security: str,
        action: Literal["BUY", "SELL"],
        amount: int,
        price: float,
        available_cash: float,
        available_shares: int,
    ) -> tuple[bool, str]:
        """Validate order against risk thresholds; returns (is_valid, reject_reason)."""
        now = time.time()
        today = datetime.now().strftime("%Y%m%d")
        if today != self._current_trade_date:
            self._current_trade_date = today
            self._daily_traded_notional = 0.0

        # 1. Rate limiter
        while self._order_timestamps and now - self._order_timestamps[0] > 1.0:
            self._order_timestamps.popleft()
        if len(self._order_timestamps) >= self.max_orders_per_second:
            return False, f"Risk rejection: Order rate limit exceeded ({self.max_orders_per_second}/s)"

        # 2. Blacklist & ST check
        if security in self.blacklist:
            return False, f"Risk rejection: Security {security} is in risk blacklist"
        if not self.allow_st and "ST" in security.upper():
            return False, f"Risk rejection: ST stocks trading disabled for {security}"

        # 3. Lot sizing & amount checks
        if amount <= 0:
            return False, f"Risk rejection: Invalid order amount {amount}"
        if action == "BUY" and amount % 100 != 0:
            return False, f"Risk rejection: Buy order amount {amount} must be a multiple of 100 shares"

        # 4. Position and cash checks
        notional = amount * price
        if notional > self.max_order_value:
            return False, f"Risk rejection: Single order notional {notional:.2f} exceeds cap {self.max_order_value:.2f}"
        if self._daily_traded_notional + notional > self.max_daily_notional:
            return False, f"Risk rejection: Daily notional limit {self.max_daily_notional:.2f} exceeded"

        if action == "BUY" and notional > available_cash:
            return False, f"Risk rejection: Insufficient cash (Required: {notional:.2f}, Available: {available_cash:.2f})"
        if action == "SELL" and amount > available_shares:
            return False, f"Risk rejection: Insufficient sellable shares (Required: {amount}, Available: {available_shares})"

        self._order_timestamps.append(now)
        self._daily_traded_notional += notional
        return True, ""


# --------------------------------------------------------------------------- #
# Trader Base Interface
# --------------------------------------------------------------------------- #
class BaseTrader:
    """Universal Trader interface for Live & Simulation brokers."""

    def connect(self) -> bool:
        raise NotImplementedError

    def is_connected(self) -> bool:
        raise NotImplementedError

    def query_asset(self) -> dict[str, float]:
        """Returns {'cash': float, 'total_asset': float, 'market_value': float, 'frozen_cash': float}."""
        raise NotImplementedError

    def query_positions(self) -> dict[str, LivePosition]:
        raise NotImplementedError

    def query_orders(self) -> list[LiveOrder]:
        raise NotImplementedError

    def order(
        self,
        security: str,
        amount: int,
        price: float | None = None,
        order_type: Literal["MARKET", "LIMIT"] = "LIMIT",
    ) -> LiveOrder:
        raise NotImplementedError

    def order_target(self, security: str, target_amount: int, price: float | None = None) -> LiveOrder | None:
        positions = self.query_positions()
        curr_vol = positions[security].volume if security in positions else 0
        delta = target_amount - curr_vol
        if delta == 0:
            return None
        return self.order(security, delta, price)

    def order_target_percent(self, security: str, percent: float, price: float | None = None) -> LiveOrder | None:
        asset = self.query_asset()
        total_val = asset["total_asset"]
        target_val = total_val * float(percent)
        p = price or self.get_latest_price(security)
        if p <= 0:
            raise ValueError(f"Unable to determine market price for {security}")
        target_shares = int(target_val / p // 100) * 100
        return self.order_target(security, target_shares, p)

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    def get_latest_price(self, security: str) -> float:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# miniQMT (迅投 XtQuant) Real Connector
# --------------------------------------------------------------------------- #
class QmtTrader(BaseTrader):
    """Institutional miniQMT (XtQuant) native broker connector."""

    def __init__(
        self,
        qmt_path: str,
        account_id: str,
        account_type: str = "STOCK",
        session_id: int = 123456,
        risk_controller: TraderRiskController | None = None,
    ):
        self.qmt_path = str(qmt_path)
        self.account_id = str(account_id)
        self.account_type = account_type
        self.session_id = session_id
        self.risk = risk_controller or TraderRiskController()

        self._trader: Any = None
        self._account: Any = None
        self._connected: bool = False

    def connect(self) -> bool:
        try:
            from xtquant.xttrader import XtQuantTrader
            from xtquant.xttype import StockAccount
        except ImportError as e:
            raise ImportError(
                "miniQMT 依赖库 xtquant 未安装。\n"
                "如需直连券商实盘 QMT，请在环境中安装 xtquant (如通过券商终端自带 Python 或 pip install xtquant)。\n"
                "在没有实盘终端的环境中，请使用 MockQmtTrader 进行仿真联调。"
            ) from e

        self._trader = XtQuantTrader(self.qmt_path, self.session_id)
        self._account = StockAccount(self.account_id, self.account_type)
        self._trader.start()

        connect_result = self._trader.connect()
        if connect_result == 0:
            self._connected = True
            logger.info("miniQMT IPC pipe connected successfully to %s", self.qmt_path)
            self._trader.subscribe(self._account)
            return True
        logger.error("miniQMT connection failed with code %s", connect_result)
        self._connected = False
        return False

    def is_connected(self) -> bool:
        return self._connected

    def query_asset(self) -> dict[str, float]:
        if not self._connected or not self._trader:
            raise ConnectionError("QMT trader not connected.")
        asset = self._trader.query_stock_asset(self._account)
        if asset is None:
            return {"cash": 0.0, "total_asset": 0.0, "market_value": 0.0, "frozen_cash": 0.0}
        return {
            "cash": float(asset.cash),
            "total_asset": float(asset.total_asset),
            "market_value": float(asset.market_value),
            "frozen_cash": float(asset.frozen_cash),
        }

    def query_positions(self) -> dict[str, LivePosition]:
        if not self._connected or not self._trader:
            raise ConnectionError("QMT trader not connected.")
        positions = self._trader.query_stock_positions(self._account)
        res = {}
        for p in positions or []:
            res[p.stock_code] = LivePosition(
                security=p.stock_code,
                volume=int(p.volume),
                can_use_volume=int(p.can_use_volume),
                open_price=float(p.open_price),
                market_value=float(p.market_value),
            )
        return res

    def query_orders(self) -> list[LiveOrder]:
        if not self._connected or not self._trader:
            raise ConnectionError("QMT trader not connected.")
        orders = self._trader.query_stock_orders(self._account)
        res = []
        for o in orders or []:
            res.append(LiveOrder(
                order_id=str(o.order_id),
                security=o.stock_code,
                action="BUY" if o.order_type == 23 else "SELL",
                amount=int(o.order_volume),
                price=float(o.price),
                status="FILLED" if o.order_status == 56 else "SUBMITTED",
                filled_amount=int(o.traded_volume),
                filled_price=float(o.traded_price),
            ))
        return res

    def order(
        self,
        security: str,
        amount: int,
        price: float | None = None,
        order_type: Literal["MARKET", "LIMIT"] = "LIMIT",
    ) -> LiveOrder:
        if not self._connected:
            raise ConnectionError("QMT trader not connected.")

        action = "BUY" if amount > 0 else "SELL"
        abs_amount = abs(amount)
        exec_price = price or self.get_latest_price(security)

        # Risk control firewall check
        asset = self.query_asset()
        positions = self.query_positions()
        avail_cash = asset["cash"]
        avail_shares = positions[security].can_use_volume if security in positions else 0

        allowed, reason = self.risk.validate_order(
            security=security,
            action=action,
            amount=abs_amount,
            price=exec_price,
            available_cash=avail_cash,
            available_shares=avail_shares,
        )
        if not allowed:
            logger.warning("Order blocked by risk controller: %s", reason)
            return LiveOrder(
                order_id=str(uuid.uuid4())[:8],
                security=security,
                action=action,
                amount=abs_amount,
                price=exec_price,
                status="REJECTED",
                reject_reason=reason,
            )

        from xtquant import xtconstant
        qmt_order_type = xtconstant.STOCK_BUY if action == "BUY" else xtconstant.STOCK_SELL
        qmt_price_type = xtconstant.FIX_PRICE if order_type == "LIMIT" else xtconstant.LATEST_PRICE

        order_seq = self._trader.order_stock(
            self._account,
            security,
            qmt_order_type,
            abs_amount,
            qmt_price_type,
            exec_price,
            "quant_research_kit",
            "AutoOrder",
        )

        return LiveOrder(
            order_id=str(order_seq),
            security=security,
            action=action,
            amount=abs_amount,
            price=exec_price,
            order_type=order_type,
            status="SUBMITTED",
            created_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def cancel_order(self, order_id: str) -> bool:
        if not self._connected:
            return False
        res = self._trader.cancel_order_stock(self._account, int(order_id))
        return res == 0

    def get_latest_price(self, security: str) -> float:
        try:
            from xtquant import xtdata
            quote = xtdata.get_full_tick([security])
            if security in quote and "lastPrice" in quote[security]:
                return float(quote[security]["lastPrice"])
        except Exception:
            pass
        return 0.0


# --------------------------------------------------------------------------- #
# Mock QMT Trader (In-memory Simulation for Testing & Sandboxing)
# --------------------------------------------------------------------------- #
class MockQmtTrader(BaseTrader):
    """In-memory mock broker engine matching miniQMT interface."""

    def __init__(
        self,
        account_id: str = "MOCK_888888",
        initial_cash: float = 1_000_000.0,
        risk_controller: TraderRiskController | None = None,
    ):
        self.account_id = account_id
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: dict[str, LivePosition] = {}
        self.orders: list[LiveOrder] = []
        self.risk = risk_controller or TraderRiskController()
        self._connected: bool = False
        self._mock_prices: dict[str, float] = {}

    def connect(self) -> bool:
        self._connected = True
        return True

    def is_connected(self) -> bool:
        return self._connected

    def set_mock_price(self, security: str, price: float) -> None:
        self._mock_prices[security] = float(price)

    def get_latest_price(self, security: str) -> float:
        return self._mock_prices.get(security, 10.0)

    def query_asset(self) -> dict[str, float]:
        mkt_val = sum(p.market_value for p in self.positions.values())
        return {
            "cash": self.cash,
            "total_asset": self.cash + mkt_val,
            "market_value": mkt_val,
            "frozen_cash": 0.0,
        }

    def query_positions(self) -> dict[str, LivePosition]:
        # Update valuations
        for sec, p in self.positions.items():
            price = self.get_latest_price(sec)
            p.market_value = p.volume * price
        return {k: v for k, v in self.positions.items() if not v.is_empty}

    def query_orders(self) -> list[LiveOrder]:
        return list(self.orders)

    def order(
        self,
        security: str,
        amount: int,
        price: float | None = None,
        order_type: Literal["MARKET", "LIMIT"] = "LIMIT",
    ) -> LiveOrder:
        if not self._connected:
            raise ConnectionError("Mock trader not connected.")

        action = "BUY" if amount > 0 else "SELL"
        abs_amount = abs(amount)
        exec_price = price or self.get_latest_price(security)

        avail_cash = self.cash
        avail_shares = self.positions[security].can_use_volume if security in self.positions else 0

        allowed, reason = self.risk.validate_order(
            security=security,
            action=action,
            amount=abs_amount,
            price=exec_price,
            available_cash=avail_cash,
            available_shares=avail_shares,
        )

        order_id = str(uuid.uuid4())[:8]
        if not allowed:
            o = LiveOrder(
                order_id=order_id,
                security=security,
                action=action,
                amount=abs_amount,
                price=exec_price,
                status="REJECTED",
                reject_reason=reason,
                created_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            self.orders.append(o)
            return o

        # Execute in mock broker
        notional = abs_amount * exec_price
        comm = max(notional * 0.0003, 5.0)
        tax = notional * 0.0005 if action == "SELL" else 0.0

        if action == "BUY":
            self.cash -= (notional + comm + tax)
            pos = self.positions.get(security, LivePosition(security, 0, 0, exec_price, 0.0))
            new_vol = pos.volume + abs_amount
            pos.open_price = (pos.open_price * pos.volume + notional) / new_vol if new_vol > 0 else exec_price
            pos.volume = new_vol
            pos.can_use_volume += abs_amount
            pos.market_value = new_vol * exec_price
            self.positions[security] = pos
        else:
            self.cash += (notional - comm - tax)
            pos = self.positions[security]
            pos.volume -= abs_amount
            pos.can_use_volume -= abs_amount
            pos.market_value = pos.volume * exec_price
            if pos.volume <= 0:
                self.positions.pop(security, None)

        o = LiveOrder(
            order_id=order_id,
            security=security,
            action=action,
            amount=abs_amount,
            price=exec_price,
            order_type=order_type,
            status="FILLED",
            created_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filled_amount=abs_amount,
            filled_price=exec_price,
        )
        self.orders.append(o)
        return o

    def cancel_order(self, order_id: str) -> bool:
        for o in self.orders:
            if o.order_id == order_id and o.status == "SUBMITTED":
                o.status = "CANCELLED"
                return True
        return False
