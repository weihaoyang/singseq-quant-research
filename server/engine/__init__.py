"""server.engine - pure computation layer (S3).

Rules (ARCHITECTURE.md 1 H2/H7, 2): no network, no DB, no FastAPI, no
frontend imports.  Allowed: numpy / pyarrow / torch + stdlib.

Import cost discipline: ``torch`` (≈1.8 s import) is *not* pulled in by
``import server.engine``; the GPU backend is resolved lazily via
``engine.gpu`` (see :func:`__getattr__`).
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "Panel",
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "FrictionConfig",
    "TradingStateMachine",
    "GPUTensorEngine",
    "MappedPanel",
    "mapped_open",
]

_LAZY = {
    "Panel": ("server.engine.backtest", "Panel"),
    "BacktestConfig": ("server.engine.backtest", "BacktestConfig"),
    "BacktestResult": ("server.engine.backtest", "BacktestResult"),
    "run_backtest": ("server.engine.backtest", "run_backtest"),
    "FrictionConfig": ("server.engine.friction", "FrictionConfig"),
    "TradingStateMachine": ("server.engine.friction", "TradingStateMachine"),
    "GPUTensorEngine": ("server.engine.gpu", "GPUTensorEngine"),
    "MappedPanel": ("server.engine.arrow_store", "MappedPanel"),
    "mapped_open": ("server.engine.arrow_store", "mapped_open"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(target[0])
    value = getattr(mod, target[1])
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
