"""Exact registry-factor snapshots over the frozen r441 oracle.

The catalog registry contains 241 named factors.  The older built-in family
modules implement only a subset; this adapter fans the complete frozen oracle
into one snapshot while retaining the registry's canonical names.

The three interaction factors were registered without a frozen executable in
the historical package.  They are derived here from their declared component
factors and use a stable ``interaction-v1`` contract.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from factors_catalog.factor_registry import FACTOR_DEFINITIONS
from factors_catalog.r441_oracle import build_main_wave_factor_snapshot
from factors_catalog.r441_oracle.factor_common import compute_common_factor_values
from factors_catalog.r441_oracle.factor_models import FactorInputs
from factors_catalog.r441_oracle.factors import recrowding_reacceleration_quality_12

INTERACTION_VERSION = "interaction-v1"

EVENT_FACTOR_INPUTS: dict[str, tuple[str, ...]] = {
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


@dataclass(frozen=True)
class RegistryFactorInputs:
    prices: Sequence[float]
    turnover: Sequence[float]
    opens: Sequence[float]
    highs: Sequence[float]
    lows: Sequence[float]
    benchmark_prices: Sequence[float]
    stock_flows: Sequence[float] = ()
    sector_flows: Sequence[float] = ()
    sector_closes: Sequence[float] = ()
    market_closes: Sequence[float] = ()
    sector_up_counts: Sequence[float] = ()
    sector_total_counts: Sequence[float] = ()
    leader_prices: Sequence[float] = ()
    laggard_prices: Sequence[float] = ()
    sentiment_series: Sequence[float] = ()
    board_break_series: Sequence[float] = ()
    gap_series: Sequence[float] = ()
    reversal_series: Sequence[float] = ()
    event_direction_series: Sequence[float] = ()
    event_intensity_series: Sequence[float] = ()
    event_confidence_series: Sequence[float] = ()
    event_age_minutes: Sequence[float] = ()
    event_impact_series: Sequence[float] = ()
    event_sector_hits: Sequence[float] = ()
    event_symbol_hits: Sequence[float] = ()
    regulatory_event_series: Sequence[float] = ()
    shareholder_event_series: Sequence[float] = ()
    policy_event_series: Sequence[float] = ()
    hotword_series: Sequence[float] = ()
    fundamental_factors: dict[str, float] | None = None
    enable_numba_kernels: bool = True


def _as_list(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_interaction(left: Any, right: Any) -> float:
    lhs = _finite(left) or 0.0
    rhs = _finite(right) or 0.0
    return max(-1.5, min(1.5, lhs * max(rhs, 0.0)))


def compute_registry_factor_snapshot(
    inputs: RegistryFactorInputs,
) -> dict[str, float]:
    """Return every registry factor that can be computed from this window.

    Missing side inputs remain absent.  In particular, event and intraday
    factors are not synthesized when their declared series are empty.
    """

    prices = _as_list(inputs.prices)
    turnover = _as_list(inputs.turnover)
    kwargs: dict[str, Any] = {
        "opens_1d": _as_list(inputs.opens),
        "highs_1d": _as_list(inputs.highs),
        "lows_1d": _as_list(inputs.lows),
        "benchmark_prices": _as_list(inputs.benchmark_prices),
        "allow_flow_proxy_fallback": False,
        "daily_only_mode": False,
        "official_clean_mode": False,
        "enable_numba_kernels": bool(inputs.enable_numba_kernels),
    }
    optional = {
        "stock_flows": inputs.stock_flows,
        "sector_flows": inputs.sector_flows,
        "sector_closes": inputs.sector_closes,
        "market_closes": inputs.market_closes,
        "sector_up_counts": inputs.sector_up_counts,
        "sector_total_counts": inputs.sector_total_counts,
        "leader_prices": inputs.leader_prices,
        "laggard_prices": inputs.laggard_prices,
        "sentiment_series": inputs.sentiment_series,
        "board_break_series": inputs.board_break_series,
        "gap_series": inputs.gap_series,
        "reversal_series": inputs.reversal_series,
        "event_direction_series": inputs.event_direction_series,
        "event_intensity_series": inputs.event_intensity_series,
        "event_confidence_series": inputs.event_confidence_series,
        "event_age_minutes": inputs.event_age_minutes,
        "event_impact_series": inputs.event_impact_series,
        "event_sector_hits": inputs.event_sector_hits,
        "event_symbol_hits": inputs.event_symbol_hits,
        "regulatory_event_series": inputs.regulatory_event_series,
        "shareholder_event_series": inputs.shareholder_event_series,
        "policy_event_series": inputs.policy_event_series,
        "hotword_series": inputs.hotword_series,
    }
    for name, values in optional.items():
        if values:
            kwargs[name] = _as_list(values)

    factors = build_main_wave_factor_snapshot(
        prices,
        turnover,
        factor_overrides=dict(inputs.fundamental_factors or {}),
        factor_overrides_weight=1.0,
        **kwargs,
    )

    # ``_compute_factors`` intentionally copies only a live-signal subset from
    # the common oracle.  The benchmark registry needs the complete common set.
    common = compute_common_factor_values(
        FactorInputs(
            opens=_as_list(inputs.opens),
            highs=_as_list(inputs.highs),
            lows=_as_list(inputs.lows),
            closes=prices,
            turnover=turnover,
            benchmark_closes=_as_list(inputs.benchmark_prices),
            close_returns_end_index=max(len(prices) - 1, 0),
            turnover_returns_end_index=max(len(turnover) - 1, 0),
            benchmark_returns_end_index=max(len(inputs.benchmark_prices) - 1, 0),
            end_index=len(prices),
            benchmark_end_index=len(inputs.benchmark_prices),
        )
    )
    factors.update(common)

    if inputs.stock_flows:
        factors["recrowding_reacceleration_quality_12"] = (
            recrowding_reacceleration_quality_12(
                prices,
                turnover,
                _as_list(inputs.stock_flows),
            )
        )

    factors["clean_acceptance_liquidity_bridge_20"] = _positive_interaction(
        factors.get("institutional_trend_acceptance_clean_20"),
        factors.get("liquidity_sponsorship_20"),
    )
    factors["clean_acceptance_outflow_dampening_20"] = _positive_interaction(
        factors.get("institutional_trend_acceptance_clean_20"),
        factors.get("serial_outflow_damage_dampening_12"),
    )
    factors["support_lift_flow_resilience_20"] = _positive_interaction(
        factors.get("support_lift_strength_20"),
        factors.get("flow_shock_half_life_resilience_20"),
    )

    available_inputs = {
        "prices_1d",
        "opens_1d",
        "lows_1d",
        "turnover_1d",
        "benchmark_prices",
    }
    dependency_columns = {
        "stock_flows": inputs.stock_flows,
        "sector_flows": inputs.sector_flows,
        "sector_closes": inputs.sector_closes,
        "market_closes": inputs.market_closes,
        "sector_up_counts": inputs.sector_up_counts,
        "sector_total_counts": inputs.sector_total_counts,
        "leader_prices": inputs.leader_prices,
        "laggard_prices": inputs.laggard_prices,
        "sentiment_proxy": (
            inputs.sentiment_series or inputs.board_break_series
        ),
    }
    for input_name in {
        dependency
        for dependencies in EVENT_FACTOR_INPUTS.values()
        for dependency in dependencies
    }:
        if getattr(inputs, input_name):
            available_inputs.add(input_name)
    for name, values in dependency_columns.items():
        if values:
            available_inputs.add(name)
    if inputs.fundamental_factors:
        available_inputs.update(
            {"financial_income", "financial_balancesheet", "financial_cashflow"}
        )

    def dependencies_available(name: str, seen: frozenset[str] = frozenset()) -> bool:
        if name in seen:
            return False
        event_inputs = EVENT_FACTOR_INPUTS.get(name)
        if event_inputs is not None:
            return all(
                dependency in available_inputs
                for dependency in event_inputs
            )
        definition = FACTOR_DEFINITIONS.get(name)
        if definition is None:
            return name in available_inputs
        for dependency in definition.dependencies:
            if dependency in FACTOR_DEFINITIONS:
                if not dependencies_available(
                    dependency, seen | {name}
                ):
                    return False
            elif dependency not in available_inputs:
                return False
        return True

    output: dict[str, float] = {}
    for name in FACTOR_DEFINITIONS:
        if not dependencies_available(name):
            continue
        value = _finite(factors.get(name))
        if value is not None:
            output[name] = value
    return output


def implemented_registry_names(inputs: RegistryFactorInputs) -> set[str]:
    """Names emitted for a representative window, used for boundary tests."""

    return set(compute_registry_factor_snapshot(inputs))
