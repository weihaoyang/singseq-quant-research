"""Nautilus Trader Institutional Engine Integration Adapter for A-Share Markets.

Provides seamless bridge between local DuckDB A-share market data and
Nautilus Trader's Rust-accelerated, nanosecond-precision event-driven engine.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

try:
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.model.currencies import CNY
    from nautilus_trader.model.data import Bar, BarType
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import InstrumentId, Symbol, TraderId, Venue
    from nautilus_trader.model.instruments import Equity
    from nautilus_trader.model.objects import Money, Price, Quantity
    from nautilus_trader.trading.strategy import Strategy as NautilusStrategy
    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False
    NautilusStrategy = object  # type: ignore

__all__ = [
    "NAUTILUS_AVAILABLE",
    "AshareNautilusEngine",
    "NautilusStrategy",
    "convert_df_to_nautilus_bars",
    "create_ashare_instrument",
]


def create_ashare_instrument(ts_code: str) -> Equity:
    """Create a Nautilus Equity instrument conforming to A-share standards (lot size 100, CNY)."""
    if not NAUTILUS_AVAILABLE:
        raise ImportError("nautilus_trader is not installed.")

    parts = ts_code.split(".")
    code = parts[0]
    market = parts[1].upper() if len(parts) > 1 else "SZ"
    venue_code = "XSHG" if market in {"SH", "XSHG"} else ("XBSE" if market in {"BJ", "XBSE"} else "XSHE")

    venue = Venue(venue_code)
    raw_sym = Symbol(code)
    instrument_id = InstrumentId(raw_sym, venue)

    return Equity(
        instrument_id=instrument_id,
        raw_symbol=raw_sym,
        currency=CNY,
        price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(100),
        ts_event=0,
        ts_init=0,
    )


def convert_df_to_nautilus_bars(df: pd.DataFrame, instrument: Equity, bar_type_str: str | None = None) -> list[Bar]:
    """Convert pandas DataFrame (open, high, low, close, vol, trade_date) into Nautilus Bar objects."""
    if not NAUTILUS_AVAILABLE:
        raise ImportError("nautilus_trader is not installed.")

    b_type_str = bar_type_str or f"{instrument.id.value}-1-DAY-LAST-EXTERNAL"
    bar_type = BarType.from_str(b_type_str)

    bars: list[Bar] = []
    for _, row in df.iterrows():
        d_str = str(row["trade_date"]).replace("-", "")
        # Parse date to nanosecond timestamp
        dt = datetime.strptime(d_str, "%Y%m%d")
        ts_ns = int(dt.timestamp() * 1_000_000_000)

        open_p = Price.from_str(f"{float(row['open']):.2f}")
        high_p = Price.from_str(f"{float(row['high']):.2f}")
        low_p = Price.from_str(f"{float(row['low']):.2f}")
        close_p = Price.from_str(f"{float(row['close']):.2f}")
        vol_q = Quantity.from_int(int(row.get("vol", 10000)))

        bar = Bar(
            bar_type=bar_type,
            open=open_p,
            high=high_p,
            low=low_p,
            close=close_p,
            volume=vol_q,
            ts_event=ts_ns,
            ts_init=ts_ns,
        )
        bars.append(bar)

    # Sort bars monotonically by timestamp (Nautilus invariant)
    bars.sort(key=lambda b: b.ts_event)
    return bars


class AshareNautilusEngine:
    """Nautilus Trader Backtest Engine configured for China A-share markets."""

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        trader_id: str = "A-SHARE-NAUTILUS",
    ):
        if not NAUTILUS_AVAILABLE:
            raise ImportError("nautilus_trader is required for AshareNautilusEngine.")

        self.initial_cash = float(initial_cash)
        self.trader_id = TraderId(trader_id)
        self.config = BacktestEngineConfig(trader_id=self.trader_id)
        self.engine = BacktestEngine(config=self.config)

        # Setup standard Chinese exchanges
        self._setup_venues()
        self.instruments: dict[str, Equity] = {}

    def _setup_venues(self):
        for v_name in ["XSHE", "XSHG", "XBSE"]:
            venue = Venue(v_name)
            self.engine.add_venue(
                venue=venue,
                oms_type=OmsType.HEDGING,
                account_type=AccountType.CASH,
                starting_balances=[Money(self.initial_cash, CNY)],
                base_currency=CNY,
            )

    def register_instrument(self, ts_code: str) -> Equity:
        """Register an A-share stock or fund with the engine."""
        if ts_code in self.instruments:
            return self.instruments[ts_code]

        instrument = create_ashare_instrument(ts_code)
        self.engine.add_instrument(instrument)
        self.instruments[ts_code] = instrument
        return instrument

    def add_data(self, ts_code: str, df: pd.DataFrame) -> int:
        """Load DataFrame into Nautilus engine as Bar stream."""
        instrument = self.register_instrument(ts_code)
        bars = convert_df_to_nautilus_bars(df, instrument)
        self.engine.add_data(bars)
        return len(bars)

    def add_strategy(self, strategy: NautilusStrategy) -> None:
        """Add a Nautilus Actor Strategy to the engine."""
        self.engine.add_strategy(strategy)

    def run(self) -> dict[str, Any]:
        """Execute the Nautilus Rust event-driven backtest loop."""
        self.engine.run()

        # Extract portfolio summary
        reports = self.engine.trader.generate_account_report(Venue("XSHE"))
        return {
            "trader_id": self.trader_id.value,
            "engine": "NautilusTrader",
            "account_report": reports if reports is not None else {},
        }
