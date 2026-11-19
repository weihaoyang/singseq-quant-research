"""jq2quant: Standalone Open-Source JoinQuant-to-Native Strategy Transpiler.

Decouples legacy JoinQuant proprietary APIs completely from core quantitative engines.
Translates procedural code (initialize, handle_data, g, order_target_percent) into
clean, modern object-oriented Strategy classes.
"""
from __future__ import annotations

from jq2quant.runtime import JQStrategyAdapter, run_jq_strategy
from jq2quant.transpiler import JoinQuantTranspiler, transpile, transpile_file

__version__ = "0.1.0"
__all__ = [
    "JQStrategyAdapter",
    "JoinQuantTranspiler",
    "run_jq_strategy",
    "transpile",
    "transpile_file",
]
