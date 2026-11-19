"""Decoupled JoinQuant Runtime Adapter.

Allows running legacy JoinQuant strategies on-the-fly without polluting
the core quantitative SDK codebase.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import quant_sdk as ts
from jq2quant.transpiler import transpile


class JQStrategyAdapter:
    """Dynamically compiles and executes transpiled JoinQuant code."""

    @classmethod
    def compile_strategy(cls, jq_code: str, class_name: str = "TranslatedStrategy") -> type[ts.Strategy]:
        """Transpiles JoinQuant code and returns a live Strategy class."""
        translated_code = transpile(jq_code, strategy_class_name=class_name)

        namespace: dict[str, Any] = {
            "ts": ts,
            "quant_sdk": ts,
            "pd": pd,
        }
        exec(translated_code, namespace)  # noqa: S102

        strat_cls = namespace.get(class_name)
        if not strat_cls or not issubclass(strat_cls, ts.Strategy):
            raise RuntimeError(f"Transpiled code did not produce a valid ts.Strategy class '{class_name}'.")
        return strat_cls


def run_jq_strategy(
    code_or_file: str | Path,
    start_date: str = "20230101",
    end_date: str = "20231231",
    initial_cash: float = 1_000_000.0,
    benchmark: str = "000300.SH",
    db_path: str = "tushare_data.duckdb",
    custom_data: pd.DataFrame | None = None,
) -> ts.StrategyResult:
    """Decoupled one-line execution entry point for legacy JoinQuant strategies."""
    src: str
    if isinstance(code_or_file, Path) or (isinstance(code_or_file, str) and "\n" not in code_or_file and Path(code_or_file).exists()):
        with open(code_or_file, encoding="utf-8") as f:
            src = f.read()
    else:
        src = str(code_or_file)

    strat_cls = JQStrategyAdapter.compile_strategy(src)

    return ts.run_strategy(
        strategy=strat_cls,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        benchmark=benchmark,
        db_path=db_path,
        custom_data=custom_data,
    )
