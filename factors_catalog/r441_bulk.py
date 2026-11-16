"""Compiled whole-history execution for the frozen r441 registry oracle.

The scalar registry path is intentionally retained as the reference oracle.
This module compiles its trusted source into numeric kernels once and executes
each stock's complete history per factor, avoiding the original
``stock x date x factor`` Python snapshot fan-out.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import math
import os
import sys
import textwrap
import time
import types
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from factors_catalog.factor_registry import FACTOR_DEFINITIONS
from factors_catalog.r441_oracle import (
    factor_common,
    factor_math,
    factors,
    numba_kernels,
    signal_engine,
)
from factors_catalog.r441_oracle import event_factors

_SIDE_COLUMNS = {
    "stock_flows": "stock_flow",
    "sector_flows": "sector_flow",
    "sector_closes": "sector_close",
    "market_closes": "market_close",
    "sector_up_counts": "sector_up_count",
    "sector_total_counts": "sector_total_count",
    "leader_prices": "leader_close",
    "laggard_prices": "laggard_close",
    "sentiment_series": "sentiment",
    "board_break_series": "board_break",
    "gap_series": "market_gap",
    "reversal_series": "market_reversal",
    "event_direction_series": "event_direction",
    "event_intensity_series": "event_intensity",
    "event_confidence_series": "event_confidence",
    "event_age_minutes": "event_age_minutes",
    "event_impact_series": "event_impact",
    "event_sector_hits": "event_sector_hits",
    "event_symbol_hits": "event_symbol_hits",
    "regulatory_event_series": "regulatory_event",
    "shareholder_event_series": "shareholder_event",
    "policy_event_series": "policy_event",
    "hotword_series": "hotword",
}

_MINUTE_INPUTS = {
    "auction_turnover_imbalance_30m",
    "close_drive_strength_30m",
    "intraday_range_close_location_30m",
    "intraday_trend_consistency_30m",
    "opening_impulse_retention_30m",
    "vwap_dislocation_30m",
}

_BULK_COLUMNS = tuple(
    sorted(
        {
            "amount",
            "benchmark_close",
            "close",
            "high",
            "low",
            "open",
            *_SIDE_COLUMNS.values(),
        }
    )
)

_CORE_PRICE_TURNOVER_NAMES = {
    "trend_acceleration_20",
    "breakout_quality_10",
    "price_momentum_20",
    "price_momentum_60",
    "ma_gap_20",
    "ma_gap_60",
    "price_position_20",
    "pullback_absorption_10",
    "liquidity_sponsorship_20",
    "exit_deterioration_score_5",
    "volatility_compression_20",
    "realized_volatility_20",
    "downside_volatility_20",
    "momentum_efficiency_20",
    "trend_stability_15",
    "breakout_pressure_10",
    "turnover_impulse_5_20",
    "turnover_stability_20",
    "amihud_illiquidity_20",
    "exhaustion_risk_10",
    "abnormal_turnover_z20",
    "large_candle_followthrough_5",
    "distribution_warning_5",
}

_EVENT_INPUTS = {
    "event_surprise_score_5": (
        "event_direction_series",
        "event_intensity_series",
        "event_confidence_series",
    ),
    "event_recency_decay_30m": (
        "event_age_minutes",
        "event_impact_series",
    ),
    "event_consistency_20d": ("event_impact_series",),
    "event_sector_diffusion_5d": (
        "event_symbol_hits",
        "event_sector_hits",
    ),
    "regulatory_risk_penalty_20d": ("regulatory_event_series",),
    "shareholder_alignment_60d": ("shareholder_event_series",),
    "policy_alignment_score_10d": (
        "policy_event_series",
        "sector_closes",
    ),
    "crowding_event_conflict_5d": (
        "sentiment_series",
        "hotword_series",
    ),
}

_COMMON_SPECS = (
    (
        "_price_factor_values",
        ("opens", "closes", "highs", "lows", "turnover"),
        "price",
        1,
    ),
    (
        "_risk_factor_values",
        ("closes", "highs", "lows", "close_returns"),
        "risk",
        2,
    ),
    (
        "_liquidity_factor_values",
        (
            "closes",
            "turnover",
            "close_returns",
            "turnover_returns",
        ),
        "liquidity",
        3,
    ),
    (
        "_relative_factor_values",
        (
            "closes",
            "benchmark_closes",
            "close_returns",
            "benchmark_returns",
        ),
        "relative",
        4,
    ),
)

_LEGACY_ALIASES = {
    "momentum_20": "price_momentum_20",
    "momentum_60": "price_momentum_60",
    "price_vs_ma_20": "ma_gap_20",
    "price_vs_ma_60": "ma_gap_60",
    "compression_ratio_20_60": "volatility_compression_20",
    "realized_vol_20": "realized_volatility_20",
    "downside_vol_20": "downside_volatility_20",
    "turnover_zscore_20": "abnormal_turnover_z20",
    "amihud_liquidity_20": "amihud_illiquidity_20",
    "volatility_20": "realized_volatility_20",
    "auction_gap_strength_1d": "gap_strength_1",
}

_PRECOMPOSITE_COMMON_NAMES = {
    "drawdown_from_high_20",
    "range_ratio_20",
    "flow_proxy_strength_10",
    "turnover_return_corr_20",
    "beta_60",
    "alpha_20_beta_60",
    "trend_quality_20",
    "drawdown_recovery_20",
    "recovery_efficiency_20",
    "support_lift_strength_20",
    "box_ceiling_residence_20",
    "resistance_memory_decay_20",
    "breakout_retention_15",
    "breakout_retest_absorption_20",
    "low_volume_box_support_20",
    "support_test_reclaim_15",
    "false_breakdown_repair_10",
    "compression_release_quality_15",
    "compression_persistence_20",
    "flat_top_higher_low_coil_20",
    "breakout_close_control_10",
    "gap_fill_acceptance_8",
    "bullish_engulfing_hold_10",
    "failed_breakout_recoil_repair_15",
    "breakout_digestion_hold_10",
    "breakout_failure_repair_15",
    "shrink_volume_nonbreak_5_20",
    "pullback_energy_decay_20",
    "volume_price_confirmation_10",
    "highs_stepup_consistency_20",
    "stock_sector_flow_resonance_10",
    "stock_flow_residual_vs_sector_10",
    "stock_defensive_residual_on_sector_selloff_15",
    "relative_drawdown_resilience_15",
    "stock_sector_breadth_lead_10",
    "stock_sector_breadth_capture_10",
    "stock_relative_sponsorship_under_sector_crowding_10",
    "sector_flow_breadth_confirm_5_10",
    "sector_flow_breadth_divergence_crowding_15",
    "sector_flow_conversion_efficiency_20",
    "sector_negative_flow_breadth_resilience_10",
    "outflow_repair_without_inflow_10",
    "outflow_reversal_efficiency_10",
    "distribution_decay_after_inflow_spike_8",
    "serial_outflow_damage_dampening_12",
    "turnover_adjusted_outflow_damage_10",
    "severe_outflow_nonbreak_rate_15",
    "upper_shadow_distribution_pressure_10",
    "failed_reclaim_persistence_15",
    "failed_breakout_scarcity_20",
    "advance_participation_uniformity_20",
    "reaction_high_retake_quality_15",
    "steady_trend_advancement_20",
    "fractal_trend_persistence_20",
    "spectral_trend_coherence_20",
    "state_space_trend_snr_20",
    "path_efficiency_tail_resilience_20",
    "trend_liquidity_persistence_20",
    "flow_shock_half_life_resilience_20",
    "institutional_trend_acceptance_20",
    "institutional_trend_acceptance_clean_20",
    "institutional_trend_false_acceptance_risk_20",
    "weak_trend_reaccumulation_spring_20",
    "innovation_drift_absorption_20",
    "entropy_compression_breakout_potential_20",
    "volatility_of_volatility_contraction_20",
    "asymmetric_autocorr_repair_20",
    "drawdown_recovery_convexity_20",
    "tail_repair_skew_balance_20",
    "gap_failure_containment_10",
    "repair_base_tightness_12",
    "repair_liquidity_retention_10",
    "recrowding_reacceleration_quality_12",
    "panic_followthrough_repair_8",
    "panic_followthrough_decay_12",
    "upper_half_churn_trap_20",
}


def numba_available() -> bool:
    return importlib.util.find_spec("numba") is not None


def warm_registry_bulk() -> None:
    """Compile the complete registry program once in the parent process."""

    if not numba_available():
        raise RuntimeError("r441_bulk_requires_numba")
    _program()


class _SequenceLowering(ast.NodeTransformer):
    """Lower Python-only idioms without changing numerical operation order."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        previous = getattr(self, "_function_name", "")
        self._function_name = node.name
        self.generic_visit(node)
        self._function_name = previous
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Name):
            return ast.copy_location(
                ast.Compare(
                    left=ast.Call(
                        func=ast.Name(id="len", ctx=ast.Load()),
                        args=[ast.Name(id=node.operand.id, ctx=ast.Load())],
                        keywords=[],
                    ),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(0)],
                ),
                node,
            )
        return node

    def visit_Return(self, node: ast.Return) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.List) and not node.value.elts:
            if getattr(self, "_function_name", "") == "_slice_window":
                node.value = ast.Subscript(
                    value=ast.Name(id="values", ctx=ast.Load()),
                    slice=ast.Slice(
                        lower=None,
                        upper=ast.Constant(0),
                        step=None,
                    ),
                    ctx=ast.Load(),
                )
            else:
                node.value = ast.Call(
                    func=ast.Name(
                        id="_empty_float_list",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                )
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        self.generic_visit(node)
        annotation = node.annotation
        if (
            isinstance(annotation, ast.Subscript)
            and isinstance(annotation.value, ast.Name)
            and annotation.value.id == "list"
            and isinstance(node.value, ast.List)
            and not node.value.elts
        ):
            element_type = annotation.slice
            helper = "_empty_float_list"
            if (
                isinstance(element_type, ast.Subscript)
                and isinstance(element_type.value, ast.Name)
                and element_type.value.id == "tuple"
            ):
                helper = "_empty_float_pair_list"
            return ast.copy_location(
                ast.Assign(
                    targets=[node.target],
                    value=ast.Call(
                        func=ast.Name(
                            id=helper,
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    ),
                ),
                node,
            )
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id
            in {"sum", "any", "all", "sorted", "min", "max"}
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], (ast.GeneratorExp, ast.ListComp))
        ):
            generator = node.args[0]
            node.args[0] = ast.copy_location(
                ast.ListComp(
                    elt=generator.elt,
                    generators=generator.generators,
                ),
                generator,
            )
            if node.func.id == "any":
                node.func.id = "_r441_any"
            elif node.func.id == "all":
                node.func.id = "_r441_all"
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "log"
            and len(node.args) == 2
        ):
            node.func.id = "_r441_log"
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in {"min", "max"}
            and len(node.args) == 1
            and len(node.keywords) == 1
        ):
            indices = node.args[0]
            key = node.keywords[0]
            if not (
                isinstance(indices, ast.Call)
                and isinstance(indices.func, ast.Name)
                and indices.func.id == "range"
                and len(indices.args) == 1
                and not indices.keywords
                and key.arg == "key"
                and isinstance(key.value, ast.Lambda)
            ):
                return node
            selector = key.value
            element = selector.body
            if (
                len(selector.args.args) == 1
                and isinstance(element, ast.Subscript)
                and isinstance(element.value, ast.Name)
                and isinstance(element.slice, ast.Name)
                and element.slice.id == selector.args.args[0].arg
            ):
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(
                            id="_first_extreme_index",
                            ctx=ast.Load(),
                        ),
                        args=[
                            element.value,
                            indices.args[0],
                            ast.Constant(node.func.id == "max"),
                        ],
                        keywords=[],
                    ),
                    node,
                )
        return node


class _ArrayReturnLowering(ast.NodeTransformer):
    def __init__(self, count: int, block: str) -> None:
        self.count = count
        self.block = block

    def visit_Return(self, node: ast.Return) -> ast.AST:
        if not isinstance(node.value, ast.Dict):
            return node
        rows: list[ast.stmt] = [
            ast.Assign(
                targets=[
                    ast.Name(id=f"_out_{self.block}", ctx=ast.Store())
                ],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="np", ctx=ast.Load()),
                        attr="empty",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Tuple(
                            elts=[
                                ast.Call(
                                    func=ast.Name(
                                        id="len",
                                        ctx=ast.Load(),
                                    ),
                                    args=[
                                        ast.List(
                                            elts=[
                                                ast.Constant(value=key.value)
                                                for key in node.value.keys
                                            ],
                                            ctx=ast.Load(),
                                        )
                                    ],
                                    keywords=[],
                                ),
                                ast.Constant(
                                    value=np.float64.__name__.replace(
                                        "float64", "float64"
                                    )
                                ),
                            ],
                            ctx=ast.Load(),
                        )
                    ],
                    keywords=[
                        ast.keyword(
                            arg="dtype",
                            value=ast.Attribute(
                                value=ast.Name(
                                    id="np", ctx=ast.Load()
                                ),
                                attr="float64",
                                ctx=ast.Load(),
                            ),
                        )
                    ],
                ),
            )
        ]
        # The tuple dtype placeholder above is harmless for AST parsing and is
        # removed below by constructing a straightforward empty() call.
        rows[0].value = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="empty",
                ctx=ast.Load(),
            ),
            args=[
                ast.Call(
                    func=ast.Name(id="len", ctx=ast.Load()),
                    args=[
                        ast.List(
                            elts=[
                                ast.Constant(value=key.value)
                                for key in node.value.keys
                            ],
                            ctx=ast.Load(),
                        )
                    ],
                    keywords=[],
                )
            ],
            keywords=[
                ast.keyword(
                    arg="dtype",
                    value=ast.Attribute(
                        value=ast.Name(id="np", ctx=ast.Load()),
                        attr="float64",
                        ctx=ast.Load(),
                    ),
                )
            ],
        )
        for index, value in enumerate(node.value.values):
            rows.append(
                ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=ast.Name(
                                id=f"_out_{self.block}",
                                ctx=ast.Load(),
                            ),
                            slice=ast.Constant(value=index),
                            ctx=ast.Store(),
                        )
                    ],
                    value=value,
                )
            )
        rows.append(
            ast.Return(
                value=ast.Name(
                    id=f"_out_{self.block}",
                    ctx=ast.Load(),
                )
            )
        )
        return ast.copy_location(ast.If(test=ast.Constant(True), body=rows, orelse=[]), node)


class _CompositeLowering(ast.NodeTransformer):
    """Turn the frozen composite function into array arithmetic."""

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Or) and node.values:
            return node.values[0]
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "factors"
            and len(node.args) >= 1
        ):
            if len(node.args) == 1:
                node.args.append(ast.Name(id="_zero", ctx=ast.Load()))
            else:
                node.args[1] = ast.Name(id="_zero", ctx=ast.Load())
        if isinstance(node.func, ast.Name):
            if node.func.id == "float" and len(node.args) == 1:
                return node.args[0]
            if node.func.id in {"min", "max"} and len(node.args) in {2, 3, 4}:
                node.func.id = (
                    f"_v{node.func.id}{len(node.args)}"
                )
            elif node.func.id == "_clamp":
                node.func.id = "_vclamp"
        return node


class _CoreReturnLowering(ast.NodeTransformer):
    def visit_Return(self, node: ast.Return) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id
            == "_core_price_turnover_bundle_from_values"
        ):
            node.value = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr="asarray",
                    ctx=ast.Load(),
                ),
                args=[ast.Name(id="values", ctx=ast.Load())],
                keywords=[
                    ast.keyword(
                        arg="dtype",
                        value=ast.Attribute(
                            value=ast.Name(id="np", ctx=ast.Load()),
                            attr="float64",
                            ctx=ast.Load(),
                        ),
                    )
                ],
            )
        return node


def _source_tree(function: Any, transformation: ast.NodeTransformer) -> ast.AST:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    tree = transformation.visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def _compile_helpers(
    *,
    module: Any,
    scope: dict[str, Any],
    numba: Any,
) -> None:
    factor_names = {
        name
        for name, value in vars(module).items()
        if inspect.isfunction(value)
        and getattr(value, "__module__", "") == module.__name__
    }
    for _ in range(12):
        changed = False
        for name in factor_names:
            if name not in scope or isinstance(
                scope[name], numba.core.registry.CPUDispatcher
            ):
                continue
            function = module.__dict__[name]
            params: list[Any] = []
            kwargs: dict[str, Any] = {}
            for parameter in inspect.signature(function).parameters.values():
                if parameter.kind in {
                    parameter.POSITIONAL_ONLY,
                    parameter.POSITIONAL_OR_KEYWORD,
                }:
                    params.append(_compile_sample(parameter.name))
                elif parameter.kind == parameter.KEYWORD_ONLY:
                    kwargs[parameter.name] = (
                        parameter.default
                        if parameter.default is not inspect.Parameter.empty
                        else _compile_sample(parameter.name)
                    )
            try:
                compiled = numba.njit(scope[name], cache=False)
                compiled(*params, **kwargs)
            except Exception:
                continue
            scope[name] = compiled
            changed = True
        if not changed:
            break


def _compile_sample(name: str) -> Any:
    lowered = name.lower()
    if (
        "series" in lowered
        or lowered == "gaps"
        or any(
            token in lowered
            for token in (
                "prices",
                "closes",
                "highs",
                "lows",
                "opens",
                "turnover",
                "flows",
                "leaders",
                "laggards",
                "returns",
                "values",
                "left",
                "right",
                "up_counts",
                "total_counts",
            )
        )
    ):
        return 10.0 + np.linspace(0.0, 2.0, 300)
    if any(
        token in lowered
        for token in (
            "window",
            "size",
            "count",
            "limit",
            "end",
            "index",
            "short",
            "long",
        )
    ):
        return 5
    if "age" in lowered:
        return 5.0
    return 0.3


def _clone_module(module: Any, extra: dict[str, Any]) -> dict[str, Any]:
    import numba
    from numba import types as nb_types
    from numba.typed import List

    @numba.njit(cache=False)
    def empty_float_list() -> Any:
        return List.empty_list(nb_types.float64)

    pair_float_type = nb_types.UniTuple(nb_types.float64, 2)

    @numba.njit(cache=False)
    def empty_float_pair_list() -> Any:
        return List.empty_list(pair_float_type)

    @numba.njit(cache=False)
    def any_float(values: Any) -> bool:
        for value in values:
            if value:
                return True
        return False

    @numba.njit(cache=False)
    def all_float(values: Any) -> bool:
        for value in values:
            if not value:
                return False
        return True

    @numba.njit(cache=False)
    def log_base(value: float, base: float) -> float:
        return math.log(value) / math.log(base)

    tree = _source_tree(module, _SequenceLowering())
    clone_name = module.__name__ + "_bulk"
    clone_module = types.ModuleType(clone_name)
    sys.modules[clone_name] = clone_module
    scope = {
        "__name__": clone_name,
        "__file__": module.__file__,
        "__builtins__": __builtins__,
        "__package__": module.__package__,
        "np": np,
        "_empty_float_list": empty_float_list,
        "_empty_float_pair_list": empty_float_pair_list,
        "_r441_any": any_float,
        "_r441_all": all_float,
        "_r441_log": log_base,
        **extra,
    }
    clone_module.__dict__.update(scope)
    exec(compile(tree, module.__file__, "exec"), scope)
    clone_module.__dict__.update(scope)
    return scope


def _build_common_bulk(
    function: Any,
    *,
    count: int,
    block: str,
    scope: dict[str, Any],
    numba: Any,
) -> Any:
    name = f"_{block}_factor_values_bulk"
    tree = _source_tree(
        function,
        _ArrayReturnLowering(count=count, block=block),
    )
    function_node = next(
        item for item in tree.body if isinstance(item, ast.FunctionDef)
    )
    function_node.name = name
    function_node.args.args.extend(
        function_node.args.kwonlyargs
    )
    function_node.args.kwonlyargs = []
    function_node.args.kw_defaults = []
    ast.fix_missing_locations(tree)
    namespace = dict(scope)
    exec(compile(tree, function.__code__.co_filename, "exec"), namespace)
    return numba.njit(namespace[name], cache=False)


def _common_matrix(
    block: str,
    arrays: tuple[np.ndarray, ...],
    *,
    bulk: Any,
    count: int,
    end_count: int,
    numba: Any,
) -> np.ndarray:
    args = ", ".join(f"a{index}" for index in range(len(arrays)))
    slice_args = ", ".join(
        f"a{index}" for index in range(len(arrays))
    )
    end_arguments = ", ".join(["end"] * end_count)
    source = f"""
def evaluate({args}):
    rows = a0.shape[0]
    output = np.empty((rows, {count}), dtype=np.float64)
    for end in range(1, rows + 1):
        output[end - 1] = bulk({slice_args}, {end_arguments})
    return output
"""
    namespace = {"np": np, "bulk": bulk}
    exec(source, namespace)
    evaluator = numba.njit(
        namespace["evaluate"],
        cache=False,
    )
    return evaluator(*arrays)


def _core_matrix(
    prices: np.ndarray,
    turnover: np.ndarray,
    *,
    bulk: Any,
    count: int,
    numba: Any,
) -> np.ndarray:
    source = f"""
def evaluate(prices, turnover):
    rows = prices.shape[0]
    output = np.empty((rows, {count}), dtype=np.float64)
    for end in range(1, rows + 1):
        begin = max(0, end - 128)
        output[end - 1] = bulk(
            prices[begin:end],
            turnover[begin:end],
        )
    return output
"""
    namespace = {"np": np, "bulk": bulk}
    exec(source, namespace)
    evaluator = numba.njit(
        namespace["evaluate"],
        cache=False,
    )
    return evaluator(prices, turnover)


def _function_positional_parameters(function: Any) -> list[inspect.Parameter]:
    return [
        parameter
        for parameter in inspect.signature(function).parameters.values()
        if parameter.kind
        in {
            parameter.POSITIONAL_ONLY,
            parameter.POSITIONAL_OR_KEYWORD,
        }
    ]


def _function_keyword_parameters(
    function: Any,
) -> list[inspect.Parameter]:
    return [
        parameter
        for parameter in inspect.signature(function).parameters.values()
        if parameter.kind == parameter.KEYWORD_ONLY
    ]


def _direct_input_name(factor_name: str, parameter: str) -> str | None:
    mapping = {
        "prices": "close",
        "closes": "close",
        "closes_1d": "close",
        "highs": "high",
        "lows": "low",
        "opens": "open",
        "opens_1d": "open",
        "turnover": "amount",
        "benchmark_prices": "benchmark_close",
        "sector_closes": "sector_close",
        "market_closes": "market_close",
        "stock_flows": "stock_flow",
        "up_count_series": "sector_up_count",
        "total_count_series": "sector_total_count",
        "sector_up_counts": "sector_up_count",
        "sector_total_counts": "sector_total_count",
        "leaders": "leader_close",
        "laggards": "laggard_close",
        "policy_series": "policy_event",
        "policy_event_series": "policy_event",
        "hotword_series": "hotword",
        "event_direction_series": "event_direction",
        "event_intensity_series": "event_intensity",
        "event_confidence_series": "event_confidence",
        "event_age_minutes": "event_age_minutes",
        "event_impact_series": "event_impact",
        "event_sector_hits": "event_sector_hits",
        "event_symbol_hits": "event_symbol_hits",
        "regulatory_event_series": "regulatory_event",
        "shareholder_event_series": "shareholder_event",
        "sentiment_series": "sentiment",
        "board_break_series": "board_break",
        "gap_series": "market_gap",
        "reversal_series": "market_reversal",
    }
    if parameter in mapping:
        return mapping[parameter]
    if parameter == "series":
        if factor_name == "limit_up_continuation_rate":
            return "sentiment"
        if factor_name == "board_break_rate":
            return "board_break"
        return None
    if parameter == "gaps":
        return "market_gap"
    if parameter == "flows":
        return (
            "sector_flow"
            if factor_name.startswith("sector_")
            else "stock_flow"
        )
    if parameter == "sector_flows":
        return "sector_flow"
    if parameter in _MINUTE_INPUTS:
        return None
    if parameter.endswith("_30m"):
        return None
    return None


def _direct_evaluator(
    function: Any,
    factor_name: str,
    *,
    numba: Any,
) -> tuple[Any, tuple[str, ...]] | None:
    positional = _function_positional_parameters(function)
    keyword = _function_keyword_parameters(function)
    required = [
        _direct_input_name(factor_name, parameter.name)
        for parameter in positional
    ]
    if any(name is None for name in required):
        return None
    source_inputs = [
        f"a{index}" for index in range(len(positional))
    ]
    sliced = [
        f"a{index}[max(0, end - 127):end + 1]"
        for index in range(len(positional))
    ]
    for parameter in keyword:
        if parameter.default is inspect.Parameter.empty:
            return None
        sliced.append(repr(parameter.default))
    signature = ", ".join(source_inputs)
    argument_text = ", ".join(sliced)
    source = f"""
def evaluate({signature}):
    rows = a0.shape[0]
    output = np.full(rows, np.nan, dtype=np.float64)
    for end in range(rows):
        output[end] = function({argument_text})
    return output
"""
    namespace = {"np": np, "function": function}
    exec(source, namespace)
    evaluator = numba.njit(namespace["evaluate"], cache=False)
    return evaluator, tuple(str(name) for name in required)


def _compile_direct_matrix(
    functions: dict[str, Any],
    inputs: dict[str, tuple[str, ...]],
    *,
    numba: Any,
) -> Any:
    names = sorted(functions)
    arguments = ", ".join(
        f"a{index}" for index in range(len(_BULK_COLUMNS))
    )
    column_index = {
        column: index for index, column in enumerate(_BULK_COLUMNS)
    }
    lines = [
        f"def evaluate({arguments}):",
        "    rows = a0.shape[0]",
        f"    output = np.empty((rows, {len(names)}), dtype=np.float64)",
        "    for end in range(rows):",
        "        begin = max(0, end - 127)",
        "        stop = end + 1",
    ]
    for index in range(len(_BULK_COLUMNS)):
        lines.append(
            f"        slice_{index} = a{index}[begin:stop]"
        )
    for factor_index, name in enumerate(names):
        function = functions[name]
        positional = _function_positional_parameters(function)
        keyword = _function_keyword_parameters(function)
        call_args = []
        for _ in positional:
            column = inputs[name][len(call_args)]
            call_args.append(f"slice_{column_index[column]}")
        call_args.extend(
            repr(parameter.default) for parameter in keyword
        )
        lines.append(
            f"        output[end, {factor_index}] = "
            f"f{factor_index}({', '.join(call_args)})"
        )
    lines.append("    return output")
    namespace: dict[str, Any] = {
        "np": np,
        **{
            f"f{index}": functions[name]
            for index, name in enumerate(names)
        },
    }
    exec("\n".join(lines), namespace)
    return numba.njit(
        namespace["evaluate"],
        cache=False,
    )


def _copy_common_aliases(values: dict[str, np.ndarray]) -> None:
    for alias, canonical in _LEGACY_ALIASES.items():
        if canonical in values:
            values[alias] = values[canonical].copy()


def _available_registry_inputs(
    frame: pd.DataFrame,
    *,
    fundamentals: dict[str, dict[str, dict[str, float]]],
) -> set[str]:
    available = {
        "prices_1d",
        "opens_1d",
        "lows_1d",
        "turnover_1d",
        "benchmark_prices",
    }
    for input_name, column in _SIDE_COLUMNS.items():
        if column in frame and np.isfinite(
            pd.to_numeric(frame[column], errors="coerce")
            .to_numpy(dtype=float)
        ).any():
            available.add(input_name)
    if "board_break" in frame and np.isfinite(
        pd.to_numeric(frame["board_break"], errors="coerce")
        .to_numpy(dtype=float)
    ).any():
        available.add("sentiment_proxy")
    if "sentiment" in frame and np.isfinite(
        pd.to_numeric(frame["sentiment"], errors="coerce")
        .to_numpy(dtype=float)
    ).any():
        available.add("sentiment_proxy")
    if any(fundamentals.values()):
        available.update(
            {
                "financial_income",
                "financial_balancesheet",
                "financial_cashflow",
            }
        )
    return available


def _dependencies_available(
    name: str,
    available: set[str],
    *,
    seen: frozenset[str] = frozenset(),
) -> bool:
    if name in seen:
        return False
    event_inputs = _EVENT_INPUTS.get(name)
    if event_inputs is not None:
        return all(item in available for item in event_inputs)
    definition = FACTOR_DEFINITIONS.get(name)
    if definition is None:
        return name in available
    for dependency in definition.dependencies:
        if dependency in FACTOR_DEFINITIONS:
            if not _dependencies_available(
                dependency,
                available,
                seen=seen | {name},
            ):
                return False
        elif dependency not in available:
            return False
    return True


def _compile_composite(
    *,
    scope: dict[str, Any],
    numba: Any,
) -> Any:
    function = scope["_apply_public_institutional_composites"]
    tree = _source_tree(function, _CompositeLowering())
    function_node = next(
        item for item in tree.body if isinstance(item, ast.FunctionDef)
    )
    function_node.name = "_apply_public_institutional_composites_bulk"
    function_node.args.args.append(
        ast.arg(arg="_zero", annotation=None)
    )
    function_node.args.kwonlyargs = []
    function_node.args.kw_defaults = []
    ast.fix_missing_locations(tree)
    namespace = dict(scope)
    exec(compile(tree, function.__code__.co_filename, "exec"), namespace)
    return numba.njit(
        namespace["_apply_public_institutional_composites_bulk"],
        cache=False,
    )


def _vector_max2(left: Any, right: Any) -> Any:
    return np.maximum(left, right)


def _vector_max3(left: Any, middle: Any, right: Any) -> Any:
    return np.maximum(np.maximum(left, middle), right)


def _vector_max4(
    first: Any, second: Any, third: Any, fourth: Any
) -> Any:
    return np.maximum(
        np.maximum(np.maximum(first, second), third),
        fourth,
    )


def _vector_min2(left: Any, right: Any) -> Any:
    return np.minimum(left, right)


def _vector_min3(left: Any, middle: Any, right: Any) -> Any:
    return np.minimum(np.minimum(left, middle), right)


def _vector_min4(
    first: Any, second: Any, third: Any, fourth: Any
) -> Any:
    return np.minimum(
        np.minimum(np.minimum(first, second), third),
        fourth,
    )


def _vector_clamp(value: Any, lower: float, upper: float) -> Any:
    return np.maximum(lower, np.minimum(upper, value))


def _compat_safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else np.nan


def _compat_mean(values: np.ndarray) -> float:
    if values.shape[0] == 0:
        return 0.0
    total = 0.0
    for value in values:
        total += value
    return total / values.shape[0]


def _compat_drawdown_recovery_convexity_20(
    prices: np.ndarray,
) -> float:
    size = prices.shape[0]
    if size < 21:
        return 0.0
    start = size - 21
    trough_index = start
    for index in range(start + 1, size):
        if prices[index] < prices[trough_index]:
            trough_index = index
    low = prices[start]
    for index in range(start + 1, size):
        if prices[index] < low:
            low = prices[index]
    if low <= 0.0:
        return 0.0
    if trough_index >= size - 4:
        return 0.0
    pre_peak = prices[start]
    for index in range(start + 1, trough_index + 1):
        if prices[index] > pre_peak:
            pre_peak = prices[index]
    trough = prices[trough_index]
    drawdown = _compat_safe_div(pre_peak - trough, pre_peak)
    if drawdown <= 0.015:
        return 0.0
    post_size = size - trough_index
    split = max(2, post_size // 2)
    progress_left = 0.0
    progress_right = 0.0
    for index in range(trough_index + 1, trough_index + split):
        progress_left += _compat_safe_div(
            prices[index] - trough,
            pre_peak - trough,
        )
    progress_left /= split - 1
    right_count = post_size - split
    for index in range(trough_index + split, size):
        progress_right += _compat_safe_div(
            prices[index] - trough,
            pre_peak - trough,
        )
    progress_right /= right_count
    convexity = max(progress_right - progress_left, 0.0)
    hold_count = 0
    hold_begin = max(trough_index, size - 5)
    for index in range(hold_begin, size):
        if prices[index] >= trough * 1.015:
            hold_count += 1
    hold_score = _compat_safe_div(
        hold_count,
        min(5, post_size),
    )
    max_one_bar = _compat_safe_div(
        prices[trough_index + 1] - prices[trough_index],
        prices[trough_index],
    )
    for index in range(trough_index + 1, size - 1):
        candidate = _compat_safe_div(
            prices[index + 1] - prices[index],
            prices[index],
        )
        if candidate > max_one_bar:
            max_one_bar = candidate
    one_bar_penalty = max(0.0, min(max_one_bar / 0.09, 1.0))
    recovery = _compat_safe_div(
        prices[size - 1] - trough,
        pre_peak - trough,
    )
    score = min(recovery, 1.25) / 1.25
    score *= (
        0.45 * min(convexity / 0.32, 1.0)
        + 0.35 * hold_score
        + 0.20 * min(drawdown / 0.18, 1.0)
    )
    score *= 1.0 - 0.35 * one_bar_penalty
    return max(min(score, 1.0), 0.0)


def _compat_breakout_failure_repair_15(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    turnover: np.ndarray,
) -> float:
    size = min(
        opens.shape[0],
        highs.shape[0],
        lows.shape[0],
        closes.shape[0],
        turnover.shape[0],
    )
    if size < 15:
        return 0.0
    start = size - 15
    best_score = 0.0
    for relative_index in range(9, 13):
        pivot_high = highs[start + relative_index - 9]
        for probe in range(relative_index - 8, relative_index):
            if highs[start + probe] > pivot_high:
                pivot_high = highs[start + probe]
        if (
            pivot_high <= 0.0
            or highs[start + relative_index] <= pivot_high * 1.004
            or closes[start + relative_index] >= pivot_high
        ):
            continue
        breakout_extension = min(
            max(
                _compat_safe_div(
                    highs[start + relative_index] - pivot_high,
                    pivot_high,
                ),
                0.0,
            )
            / 0.05,
            1.0,
        )
        failure_close_back = min(
            max(
                _compat_safe_div(
                    pivot_high - closes[start + relative_index],
                    pivot_high,
                ),
                0.0,
            )
            / 0.04,
            1.0,
        )
        repair_begin = relative_index + 1
        repair_end = min(relative_index + 5, 15)
        if repair_end - repair_begin < 2:
            continue
        reclaim_index = -1
        for probe in range(repair_begin, repair_end):
            if closes[start + probe] >= pivot_high:
                reclaim_index = probe
                break
        if reclaim_index < 0:
            continue
        reclaim_strength = min(
            max(
                _compat_safe_div(
                    closes[start + reclaim_index] - pivot_high,
                    pivot_high,
                ),
                0.0,
            )
            / 0.04,
            1.0,
        )
        follow_end = min(reclaim_index + 3, 15)
        hold_floor = lows[start + reclaim_index]
        for probe in range(reclaim_index + 1, follow_end):
            if lows[start + probe] < hold_floor:
                hold_floor = lows[start + probe]
        hold_score = max(
            1.0
            - min(
                max(
                    _compat_safe_div(
                        pivot_high - hold_floor,
                        pivot_high,
                    ),
                    0.0,
                )
                / 0.03,
                1.0,
            ),
            0.0,
        )
        close_cluster = 0.0
        for probe in range(reclaim_index, follow_end):
            close_cluster += max(
                _compat_safe_div(
                    closes[start + probe] - pivot_high,
                    pivot_high * 0.03,
                ),
                0.0,
            )
        close_cluster /= follow_end - reclaim_index
        pre_begin = max(relative_index - 4, 0)
        pre_turnover = 0.0
        for probe in range(pre_begin, relative_index + 1):
            pre_turnover += max(turnover[start + probe], 0.0)
        pre_turnover /= relative_index + 1 - pre_begin
        pre_turnover += 1.0
        repair_turnover = 0.0
        for probe in range(repair_begin, repair_end):
            repair_turnover += max(turnover[start + probe], 0.0)
        repair_turnover /= repair_end - repair_begin
        post_turnover = repair_turnover
        turnover_cool = max(
            1.0
            - min(
                _compat_safe_div(post_turnover, pre_turnover),
                1.0,
            ),
            0.0,
        )
        recency = max(
            1.0
            - _compat_safe_div(
                14 - reclaim_index,
                15.0,
            ),
            0.2,
        )
        score = (
            breakout_extension
            * (0.25 + 0.75 * (1.0 - failure_close_back))
            * (
                0.40 * reclaim_strength
                + 0.35 * hold_score
                + 0.15 * min(close_cluster, 1.0)
                + 0.10 * turnover_cool
            )
            * recency
        )
        if score > best_score:
            best_score = score
    return max(min(best_score, 1.25), 0.0)


@dataclass
class _BulkProgram:
    direct: dict[str, tuple[None, tuple[str, ...]]]
    direct_names: tuple[str, ...]
    direct_bulk: Any
    core_names: tuple[str, ...]
    core_bulk: Any
    common_names: dict[str, tuple[str, int]]
    common_bulk: dict[str, Any]
    composite: Any
    crowding: Any


@lru_cache(maxsize=1)
def _program() -> _BulkProgram:
    import numba
    from numba.extending import register_jitable

    register_jitable(_compat_safe_div)
    register_jitable(_compat_mean)

    def first_extreme_index(
        values: np.ndarray, limit: int, maximum: bool
    ) -> int:
        best = 0
        for index in range(1, limit):
            better = (
                values[index] > values[best]
                if maximum
                else values[index] < values[best]
            )
            if better:
                best = index
        return best

    first = numba.njit(first_extreme_index, cache=False)
    factor_scope = _clone_module(
        factors, {"_first_extreme_index": first}
    )
    event_scope = _clone_module(
        event_factors, {"_first_extreme_index": first}
    )
    factor_scope["drawdown_recovery_convexity_20"] = numba.njit(
        _compat_drawdown_recovery_convexity_20,
        cache=False,
    )
    factor_scope["breakout_failure_repair_15"] = numba.njit(
        _compat_breakout_failure_repair_15,
        cache=False,
    )
    _compile_helpers(
        module=factors,
        scope=factor_scope,
        numba=numba,
    )

    box_function = factor_scope["box_ceiling_residence_20"]
    false_breakdown_function = factor_scope[
        "false_breakdown_repair_10"
    ]

    @numba.njit(cache=False)
    def box_ceiling_wrapper(prices: np.ndarray) -> float:
        return box_function(prices, 0.2)

    @numba.njit(cache=False)
    def false_breakdown_wrapper(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> float:
        return false_breakdown_function(highs, lows, closes, 0.01)

    factor_scope["box_ceiling_residence_20"] = box_ceiling_wrapper
    factor_scope["false_breakdown_repair_10"] = false_breakdown_wrapper
    _compile_helpers(
        module=event_factors,
        scope=event_scope,
        numba=numba,
    )

    common_scope = _clone_module(
        factor_common,
        {"_first_extreme_index": first},
    )
    math_scope = _clone_module(factor_math, {})
    _compile_helpers(
        module=factor_math,
        scope=math_scope,
        numba=numba,
    )
    for name, value in math_scope.items():
        if isinstance(
            value, numba.core.registry.CPUDispatcher
        ):
            common_scope[name] = value
    for name, value in factor_scope.items():
        if isinstance(
            value, numba.core.registry.CPUDispatcher
        ):
            common_scope[name] = value
    _compile_helpers(
        module=factor_common,
        scope=common_scope,
        numba=numba,
    )

    core_scope = _clone_module(numba_kernels, {})

    @numba.njit(cache=False)
    def core_mean(values: np.ndarray) -> float:
        size = len(values)
        if size == 0:
            return 0.0
        total = 0.0
        for value in values:
            total += value
        return total / size

    core_scope["_mean_py"] = core_mean
    _compile_helpers(
        module=numba_kernels,
        scope=core_scope,
        numba=numba,
    )
    core_function = numba_kernels._core_price_turnover_bundle_py
    core_tree = _source_tree(
        core_function,
        _CoreReturnLowering(),
    )
    core_node = next(
        item
        for item in core_tree.body
        if isinstance(item, ast.FunctionDef)
    )
    core_node.name = "_core_price_turnover_bundle_bulk"
    ast.fix_missing_locations(core_tree)
    core_namespace = dict(core_scope)
    exec(
        compile(
            core_tree,
            core_function.__code__.co_filename,
            "exec",
        ),
        core_namespace,
    )
    core_bulk = numba.njit(
        core_namespace["_core_price_turnover_bundle_bulk"],
        cache=False,
    )

    common_names: dict[str, tuple[str, int]] = {}
    common_bulk: dict[str, Any] = {}
    for function_name, _, block, _ in _COMMON_SPECS:
        function = factor_common.__dict__[function_name]
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        return_node = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Dict)
        )
        keys = [
            str(key.value) for key in return_node.value.keys
        ]
        common_bulk[block] = _build_common_bulk(
            function,
            count=len(keys),
            block=block,
            scope=common_scope,
            numba=numba,
        )
        for index, name in enumerate(keys):
            common_names[name] = (block, index)

    signal_scope = _clone_module(
        signal_engine,
        {
            "_first_extreme_index": first,
            "_vmax2": _vector_max2,
            "_vmax3": _vector_max3,
            "_vmax4": _vector_max4,
            "_vmin2": _vector_min2,
            "_vmin3": _vector_min3,
            "_vmin4": _vector_min4,
            "_vclamp": _vector_clamp,
        },
    )
    for name, function in {
        "_vmax2": _vector_max2,
        "_vmax3": _vector_max3,
        "_vmax4": _vector_max4,
        "_vmin2": _vector_min2,
        "_vmin3": _vector_min3,
        "_vmin4": _vector_min4,
        "_vclamp": _vector_clamp,
    }.items():
        signal_scope[name] = numba.njit(function, cache=False)
    for name, value in factor_scope.items():
        if isinstance(
            value, numba.core.registry.CPUDispatcher
        ):
            signal_scope[name] = value
    for name, value in event_scope.items():
        if isinstance(
            value, numba.core.registry.CPUDispatcher
        ):
            signal_scope[name] = value

    referenced = {
        name
        for name in FACTOR_DEFINITIONS
        if isinstance(
            signal_scope.get(name) or factor_scope.get(name),
            numba.core.registry.CPUDispatcher,
        )
    }
    referenced.update(_CORE_PRICE_TURNOVER_NAMES)
    referenced.update(
        {
            "liquidity_impact_relief_10",
            "recrowding_reacceleration_quality_12",
        }
    )
    for function_name in (
        "_compute_market_structure_factors",
        "_compute_flow_sentiment_event_factors",
    ):
        tree = ast.parse(
            textwrap.dedent(
                inspect.getsource(signal_scope[function_name])
            )
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id
                in FACTOR_DEFINITIONS
            ):
                referenced.add(node.func.id)
    for function_name, _, _, _ in _COMMON_SPECS:
        tree = ast.parse(
            textwrap.dedent(
                inspect.getsource(
                    factor_common.__dict__[function_name]
                )
            )
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in FACTOR_DEFINITIONS
            ):
                referenced.add(node.func.id)

    direct: dict[str, tuple[None, tuple[str, ...]]] = {}
    direct_functions: dict[str, Any] = {}
    direct_inputs: dict[str, tuple[str, ...]] = {}
    for name in sorted(referenced):
        function = signal_scope.get(name) or factor_scope.get(name)
        if not isinstance(
            function, numba.core.registry.CPUDispatcher
        ):
            continue
        positional = _function_positional_parameters(function)
        columns = tuple(
            _direct_input_name(name, parameter.name)
            for parameter in positional
        )
        if any(column is None for column in columns):
            continue
        direct_functions[name] = function
        direct_inputs[name] = tuple(
            str(column) for column in columns
        )
        direct[name] = (None, direct_inputs[name])
    direct_names = tuple(sorted(direct_functions))
    direct_bulk = _compile_direct_matrix(
        direct_functions,
        direct_inputs,
        numba=numba,
    )

    composite = _compile_composite(
        scope=signal_scope,
        numba=numba,
    )

    crowding_function = signal_scope.get("crowding_risk_score")
    if not isinstance(
        crowding_function,
        numba.core.registry.CPUDispatcher,
    ):
        crowding_function = event_scope.get("crowding_risk_score")
    if not isinstance(
        crowding_function,
        numba.core.registry.CPUDispatcher,
    ):
        crowding_function = numba.njit(
            factors.crowding_risk_score,
            cache=False,
        )

    @numba.njit(cache=False)
    def crowding_factor(
        stock_signal: np.ndarray,
        turnover_signal: np.ndarray,
        sentiment_signal: np.ndarray,
    ) -> np.ndarray:
        rows = stock_signal.shape[0]
        output = np.empty(rows, dtype=np.float64)
        for index in range(rows):
            output[index] = crowding_function(
                min(max(stock_signal[index], 0.0), 1.0),
                min(max(turnover_signal[index], 0.0) / 3.0, 1.0),
                min(
                    max(sentiment_signal[index] - 0.5, 0.0) * 2.0,
                    1.0,
                ),
            )
        return output

    return _BulkProgram(
        direct=direct,
        direct_names=direct_names,
        direct_bulk=direct_bulk,
        core_names=tuple(numba_kernels._CORE_PRICE_TURNOVER_KEYS),
        core_bulk=core_bulk,
        common_names=common_names,
        common_bulk=common_bulk,
        composite=composite,
        crowding=crowding_factor,
    )


def _numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(
        dtype=np.float64
    )


def compute_registry_panel_bulk(
    prepared: pd.DataFrame,
    *,
    fundamental_by_symbol_date: dict[
        str, dict[str, dict[str, float]]
    ]
    | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute registry factors from whole-history compiled kernels."""

    if not numba_available():
        raise RuntimeError("r441_bulk_requires_numba")
    program = _program()
    frame = prepared.reset_index(drop=True)
    required = {
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "amount",
        "benchmark_close",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "registry panel missing columns: " + ", ".join(missing)
        )
    fundamentals = fundamental_by_symbol_date or {}
    available_inputs = _available_registry_inputs(
        frame,
        fundamentals=fundamentals,
    )
    columns = sorted(FACTOR_DEFINITIONS)
    output = {
        name: np.full(len(frame), np.nan, dtype=np.float32)
        for name in columns
    }
    arrays = {
        name: _numeric_array(frame, name)
        for name in {
            "open",
            "high",
            "low",
            "close",
            "amount",
            "benchmark_close",
            *_SIDE_COLUMNS.values(),
        }
        if name in frame
    }
    date_values = frame["trade_date"].astype(str).to_numpy()
    groups = frame.groupby("ts_code", sort=False).indices
    profile: dict[str, float] = {
        "direct": 0.0,
        "core": 0.0,
        "common": 0.0,
        "composite": 0.0,
    }

    for raw_symbol, raw_positions in groups.items():
        symbol = str(raw_symbol)
        positions = np.asarray(raw_positions, dtype=np.int64)
        symbol_length = len(positions)

        def symbol_array(column: str) -> np.ndarray:
            values = arrays.get(column)
            if values is None:
                return np.empty(0, dtype=np.float64)
            return np.asarray(values[positions], dtype=np.float64)

        base = {
            "open": symbol_array("open"),
            "high": symbol_array("high"),
            "low": symbol_array("low"),
            "close": symbol_array("close"),
            "amount": symbol_array("amount"),
            "benchmark_close": symbol_array("benchmark_close"),
        }
        for column in _SIDE_COLUMNS.values():
            base[column] = symbol_array(column)
        close_returns = np.diff(base["close"]) / base["close"][:-1]
        close_returns = np.where(
            base["close"][:-1] != 0.0,
            close_returns,
            0.0,
        )
        turnover_returns = np.diff(base["amount"]) / base["amount"][:-1]
        turnover_returns = np.where(
            base["amount"][:-1] != 0.0,
            turnover_returns,
            0.0,
        )
        benchmark_returns = (
            np.diff(base["benchmark_close"])
            / base["benchmark_close"][:-1]
        )
        benchmark_returns = np.where(
            base["benchmark_close"][:-1] != 0.0,
            benchmark_returns,
            0.0,
        )

        values: dict[str, np.ndarray] = {}
        present_keys: set[str] = set()
        started = time.perf_counter()
        direct_matrix = program.direct_bulk(
            *[
                base.get(column, np.empty(0))
                for column in _BULK_COLUMNS
            ]
        )
        for index, name in enumerate(program.direct_names):
            result = direct_matrix[:, index]
            result = np.asarray(result, dtype=np.float64)
            if result.shape[0] != symbol_length:
                continue
            values[name] = result
            if np.isfinite(result).any():
                present_keys.add(name)
        profile["direct"] += time.perf_counter() - started

        started = time.perf_counter()
        core_values = _core_matrix(
            base["close"],
            base["amount"],
            bulk=program.core_bulk,
            count=len(program.core_names),
            numba=__import__("numba"),
        )
        for index, name in enumerate(program.core_names):
            values[name] = core_values[:, index]
            if np.isfinite(values[name]).any():
                present_keys.add(name)
        profile["core"] += time.perf_counter() - started

        started = time.perf_counter()
        adapter_common: dict[str, np.ndarray] = {}
        common_inputs = {
            "price": (
                base["open"],
                base["close"],
                base["high"],
                base["low"],
                base["amount"],
            ),
            "risk": (
                base["close"],
                base["high"],
                base["low"],
                close_returns,
            ),
            "liquidity": (
                base["close"],
                base["amount"],
                close_returns,
                turnover_returns,
            ),
            "relative": (
                base["close"],
                base["benchmark_close"],
                close_returns,
                benchmark_returns,
            ),
        }
        end_counts = {
            block: end_count
            for _, _, block, end_count in _COMMON_SPECS
        }
        for block, arrays_for_block in common_inputs.items():
            matrix = _common_matrix(
                block,
                arrays_for_block,
                bulk=program.common_bulk[block],
                count=max(
                    index + 1
                    for name, (candidate, index)
                    in program.common_names.items()
                    if candidate == block
                ),
                end_count=end_counts[block],
                numba=__import__("numba"),
            )
            for name, (candidate, index) in program.common_names.items():
                if candidate == block:
                    adapter_common[name] = matrix[:, index]

        for name in program.common_names:
            if (
                name in _PRECOMPOSITE_COMMON_NAMES
                and name in adapter_common
            ):
                values[name] = adapter_common[name]
            if name in values and np.isfinite(values[name]).any():
                present_keys.add(name)
        profile["common"] += time.perf_counter() - started

        per_day = fundamentals.get(symbol, {})
        financial_names = {
            name
            for day_values in per_day.values()
            for name in day_values
            if name in FACTOR_DEFINITIONS
        }
        financial_arrays: dict[str, np.ndarray] = {}
        for name in financial_names:
            payload = np.zeros(symbol_length, dtype=np.float64)
            observed = np.zeros(symbol_length, dtype=bool)
            for offset, position in enumerate(positions):
                value = per_day.get(str(date_values[position]), {}).get(
                    name
                )
                if value is None:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(number):
                    payload[offset] = number
                    observed[offset] = True
            if observed.any():
                financial_arrays[name] = payload

        crowding_values = program.crowding(
            values["stock_net_inflow_5_20"],
            values["abnormal_turnover_z20"],
            values["limit_up_continuation_rate"],
        )
        values["crowding_risk_score"] = crowding_values
        if np.isfinite(crowding_values).any():
            present_keys.add("crowding_risk_score")

        from numba import typed
        from numba import types as nb_types

        started = time.perf_counter()
        value_type = nb_types.float64[:]
        composite_values = typed.Dict.empty(
            key_type=nb_types.unicode_type,
            value_type=value_type,
        )
        zero = np.zeros(symbol_length, dtype=np.float64)
        for name, source in values.items():
            if name not in FACTOR_DEFINITIONS:
                continue
            composite_values[name] = np.ascontiguousarray(
                np.asarray(source, dtype=np.float64)
            )
        program.composite(composite_values, zero)
        for name, payload in financial_arrays.items():
            values[name] = np.where(
                np.asarray(
                    [
                        name
                        in per_day.get(
                            str(date_values[position]),
                            {},
                        )
                        for position in positions
                    ],
                    dtype=bool,
                ),
                payload,
                np.nan,
            )
            composite_values[name] = np.ascontiguousarray(payload)
            present_keys.add(name)
        program.composite(composite_values, zero)
        for name in FACTOR_DEFINITIONS:
            if name in composite_values:
                values[name] = np.asarray(
                    composite_values[name], dtype=np.float64
                )
                if np.isfinite(values[name]).any():
                    present_keys.add(name)

        values.update(adapter_common)
        _copy_common_aliases(values)
        for name in _LEGACY_ALIASES:
            if name in values and np.isfinite(values[name]).any():
                present_keys.add(name)
        profile["composite"] += time.perf_counter() - started

        def interaction(left: str, right: str) -> np.ndarray:
            lhs = values.get(left)
            rhs = values.get(right)
            if lhs is None or rhs is None:
                return np.zeros(symbol_length, dtype=np.float64)
            lhs = np.asarray(lhs, dtype=np.float64)
            rhs = np.asarray(rhs, dtype=np.float64)
            valid = np.isfinite(lhs) & np.isfinite(rhs)
            safe_lhs = np.where(valid, lhs, 0.0)
            safe_rhs = np.where(valid, rhs, 0.0)
            return np.maximum(
                -1.5,
                np.minimum(
                    1.5,
                    safe_lhs * np.maximum(safe_rhs, 0.0),
                ),
            )

        values["clean_acceptance_liquidity_bridge_20"] = interaction(
            "institutional_trend_acceptance_clean_20",
            "liquidity_sponsorship_20",
        )
        values["clean_acceptance_outflow_dampening_20"] = interaction(
            "institutional_trend_acceptance_clean_20",
            "serial_outflow_damage_dampening_12",
        )
        values["support_lift_flow_resilience_20"] = interaction(
            "support_lift_strength_20",
            "flow_shock_half_life_resilience_20",
        )

        for name, result in values.items():
            if name.startswith("_composite_") or name not in output:
                continue
            if not _dependencies_available(name, available_inputs):
                continue
            finite = np.isfinite(result)
            if not finite.any():
                continue
            output[name][positions] = np.where(
                finite, result, np.nan
            ).astype(np.float32)

    generated = pd.DataFrame(output, index=frame.index)
    names = [
        name
        for name in columns
        if np.isfinite(generated[name].to_numpy(dtype=float)).any()
    ]
    if os.environ.get("QUANT_R441_PROFILE"):
        print(
            "r441_bulk_profile "
            + " ".join(
                f"{key}={value:.3f}"
                for key, value in profile.items()
            ),
            file=sys.stderr,
        )
    return generated[names].copy(), names
