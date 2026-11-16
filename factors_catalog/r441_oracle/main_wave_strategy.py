# ruff: noqa
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import Any

from factors_catalog.r441_flags import as_bool, as_float, normalized_state


@dataclass(frozen=True)
class MainWaveProfile:
    name: str
    entry_threshold: float
    breakout_threshold: float
    liquidity_threshold: float
    hard_stop_loss: float
    time_exit_days: int
    max_positions: int


SetupType = Literal["repair_reclaim", "trend_continuation", "none"]
ExitMode = Literal["realization", "trend_runner"]


@dataclass(frozen=True)
class MainWaveSetupDecision:
    setup_type: SetupType
    enter: bool
    trend_strength: float
    repair_strength: float
    flow_strength: float
    quality_strength: float
    support_strength: float
    risk_penalty: float
    score: float


@dataclass(frozen=True)
class MainWaveExitDecision:
    should_exit: bool
    exit_mode: ExitMode
    runner_eligible: bool
    base_time_exit_days: int
    max_hold_days: int
    deterioration_limit: float
    notes: tuple[str, ...] = ()


BASE_BALANCED = MainWaveProfile(
    name="base_balanced",
    entry_threshold=0.08,
    breakout_threshold=0.0,
    liquidity_threshold=0.05,
    hard_stop_loss=0.06,
    time_exit_days=20,
    max_positions=6,
)

ACCELERATION_PROBE = MainWaveProfile(
    name="acceleration_probe",
    entry_threshold=0.20,
    breakout_threshold=0.025,
    liquidity_threshold=0.08,
    hard_stop_loss=0.05,
    time_exit_days=15,
    max_positions=8,
)

TREND_HYPER = MainWaveProfile(
    name="trend_hyper",
    entry_threshold=0.08,
    breakout_threshold=0.01,
    liquidity_threshold=-0.02,
    hard_stop_loss=0.08,
    time_exit_days=45,
    max_positions=4,
)

TREND_TURBO = MainWaveProfile(
    name="trend_turbo",
    entry_threshold=0.03,
    breakout_threshold=0.005,
    liquidity_threshold=-0.08,
    hard_stop_loss=0.09,
    time_exit_days=60,
    max_positions=2,
)

DEFENSE_RECOVERY = MainWaveProfile(
    name="defense_recovery",
    entry_threshold=0.10,
    breakout_threshold=0.015,
    liquidity_threshold=0.03,
    hard_stop_loss=0.04,
    time_exit_days=12,
    max_positions=4,
)

PARAMETER_PROFILES: dict[str, MainWaveProfile] = {
    profile.name: profile
    for profile in (BASE_BALANCED, ACCELERATION_PROBE, TREND_HYPER, TREND_TURBO, DEFENSE_RECOVERY)
}

OBJECTIVE_EXECUTION_FILL_RATIO_FLOOR = 0.92
OBJECTIVE_EXECUTION_REJECT_RATIO_CEILING = 0.08

MAIN_WAVE_TREND_STRUCTURE_FACTORS: tuple[str, ...] = (
    "trend_quality_20",
    "relative_strength_60",
    "momentum_efficiency_20",
    "recovery_efficiency_20",
    "low_volume_box_support_20",
    "false_breakdown_repair_10",
    "breakout_retest_absorption_20",
    "resistance_memory_decay_20",
)

MAIN_WAVE_FLOW_SPONSORSHIP_FACTORS: tuple[str, ...] = (
    "outflow_retest_absorption_15",
    "negative_flow_impact_beta_10",
    "stock_flow_residual_vs_sector_10",
    "stock_defensive_residual_on_sector_selloff_15",
)

MAIN_WAVE_QUALITY_FILTER_FACTORS: tuple[str, ...] = (
    "piotroski_f_score_proxy_20",
    "operating_accruals_assets_20",
    "distress_resilience_public_20",
)


def canonical_main_wave_factor_layers() -> dict[str, tuple[str, ...]]:
    return {
        "trend_structure": MAIN_WAVE_TREND_STRUCTURE_FACTORS,
        "flow_sponsorship": MAIN_WAVE_FLOW_SPONSORSHIP_FACTORS,
        "quality_filter": MAIN_WAVE_QUALITY_FILTER_FACTORS,
    }


def canonical_main_wave_specs_by_regime() -> dict[str, tuple[dict[str, float | int | str], ...]]:
    return {
        "default": (
            {"factor": "trend_quality_20", "weight": 1.15, "direction": 1},
            {"factor": "relative_strength_60", "weight": 1.10, "direction": 1},
            {"factor": "momentum_efficiency_20", "weight": 0.95, "direction": 1},
            {"factor": "recovery_efficiency_20", "weight": 1.00, "direction": 1},
            {"factor": "low_volume_box_support_20", "weight": 1.20, "direction": 1},
            {"factor": "false_breakdown_repair_10", "weight": 1.18, "direction": 1},
            {"factor": "breakout_retest_absorption_20", "weight": 1.02, "direction": 1},
            {"factor": "resistance_memory_decay_20", "weight": 0.92, "direction": 1},
            {"factor": "outflow_retest_absorption_15", "weight": 1.05, "direction": 1},
            {"factor": "negative_flow_impact_beta_10", "weight": 0.98, "direction": 1},
            {"factor": "stock_flow_residual_vs_sector_10", "weight": 0.92, "direction": 1},
            {"factor": "stock_defensive_residual_on_sector_selloff_15", "weight": 0.88, "direction": 1},
            {"factor": "piotroski_f_score_proxy_20", "weight": 0.80, "direction": 1},
            {"factor": "operating_accruals_assets_20", "weight": 0.76, "direction": 1},
            {"factor": "distress_resilience_public_20", "weight": 0.82, "direction": 1},
        ),
        "trend": (
            {"factor": "trend_quality_20", "weight": 1.22, "direction": 1},
            {"factor": "relative_strength_60", "weight": 1.18, "direction": 1},
            {"factor": "momentum_efficiency_20", "weight": 1.08, "direction": 1},
            {"factor": "recovery_efficiency_20", "weight": 0.92, "direction": 1},
            {"factor": "low_volume_box_support_20", "weight": 1.14, "direction": 1},
            {"factor": "false_breakdown_repair_10", "weight": 1.06, "direction": 1},
            {"factor": "breakout_retest_absorption_20", "weight": 1.05, "direction": 1},
            {"factor": "resistance_memory_decay_20", "weight": 1.04, "direction": 1},
            {"factor": "outflow_retest_absorption_15", "weight": 0.96, "direction": 1},
            {"factor": "negative_flow_impact_beta_10", "weight": 0.90, "direction": 1},
            {"factor": "stock_flow_residual_vs_sector_10", "weight": 0.88, "direction": 1},
            {"factor": "stock_defensive_residual_on_sector_selloff_15", "weight": 0.72, "direction": 1},
            {"factor": "piotroski_f_score_proxy_20", "weight": 0.62, "direction": 1},
            {"factor": "operating_accruals_assets_20", "weight": 0.60, "direction": 1},
            {"factor": "distress_resilience_public_20", "weight": 0.64, "direction": 1},
        ),
        "cautious": (
            {"factor": "trend_quality_20", "weight": 1.00, "direction": 1},
            {"factor": "relative_strength_60", "weight": 0.95, "direction": 1},
            {"factor": "momentum_efficiency_20", "weight": 0.80, "direction": 1},
            {"factor": "recovery_efficiency_20", "weight": 1.10, "direction": 1},
            {"factor": "low_volume_box_support_20", "weight": 1.24, "direction": 1},
            {"factor": "false_breakdown_repair_10", "weight": 1.24, "direction": 1},
            {"factor": "breakout_retest_absorption_20", "weight": 1.10, "direction": 1},
            {"factor": "resistance_memory_decay_20", "weight": 0.84, "direction": 1},
            {"factor": "outflow_retest_absorption_15", "weight": 1.12, "direction": 1},
            {"factor": "negative_flow_impact_beta_10", "weight": 1.08, "direction": 1},
            {"factor": "stock_flow_residual_vs_sector_10", "weight": 1.02, "direction": 1},
            {"factor": "stock_defensive_residual_on_sector_selloff_15", "weight": 1.00, "direction": 1},
            {"factor": "piotroski_f_score_proxy_20", "weight": 0.90, "direction": 1},
            {"factor": "operating_accruals_assets_20", "weight": 0.86, "direction": 1},
            {"factor": "distress_resilience_public_20", "weight": 0.94, "direction": 1},
        ),
        "risk_off": (
            {"factor": "trend_quality_20", "weight": 0.72, "direction": 1},
            {"factor": "relative_strength_60", "weight": 0.78, "direction": 1},
            {"factor": "momentum_efficiency_20", "weight": 0.58, "direction": 1},
            {"factor": "recovery_efficiency_20", "weight": 1.08, "direction": 1},
            {"factor": "low_volume_box_support_20", "weight": 1.18, "direction": 1},
            {"factor": "false_breakdown_repair_10", "weight": 1.10, "direction": 1},
            {"factor": "breakout_retest_absorption_20", "weight": 0.92, "direction": 1},
            {"factor": "resistance_memory_decay_20", "weight": 0.70, "direction": 1},
            {"factor": "outflow_retest_absorption_15", "weight": 1.18, "direction": 1},
            {"factor": "negative_flow_impact_beta_10", "weight": 1.14, "direction": 1},
            {"factor": "stock_flow_residual_vs_sector_10", "weight": 1.02, "direction": 1},
            {"factor": "stock_defensive_residual_on_sector_selloff_15", "weight": 1.16, "direction": 1},
            {"factor": "piotroski_f_score_proxy_20", "weight": 0.98, "direction": 1},
            {"factor": "operating_accruals_assets_20", "weight": 0.92, "direction": 1},
            {"factor": "distress_resilience_public_20", "weight": 1.02, "direction": 1},
        ),
    }


def resolve_profile_selection_inputs(
    *,
    state: str | None = None,
    execution_quality_ok: bool | None = None,
    objective_state: dict[str, Any] | None = None,
    archive_replay: dict[str, Any] | None = None,
    release_governance: dict[str, Any] | None = None,
    risk_governance: dict[str, Any] | None = None,
    observability_governance: dict[str, Any] | None = None,
) -> dict[str, object]:
    objective = dict(objective_state or {})
    archive = dict(archive_replay or {})
    release = dict(release_governance or {})
    risk = dict(risk_governance or {})
    observability = dict(observability_governance or {})

    explicit_state = str(state or "").strip()
    resolved_state = normalized_state(explicit_state or objective.get("state"))
    kill_switch = as_bool(dict(risk.get("kill_switch") or {}).get("enabled"), default=False)
    release_blockers = [str(item) for item in list(release.get("blocking_reasons") or []) if str(item).strip()]
    observability_degraded = as_bool(objective.get("observability_degraded"), default=False)
    runtime_persist_error = str(objective.get("runtime_state_persist_error") or "").strip()

    if kill_switch:
        resolved_state = "risk_hold"
    elif release_blockers and resolved_state == "running":
        resolved_state = "risk_hold"
    elif (observability_degraded or runtime_persist_error) and resolved_state == "stable":
        resolved_state = "running"

    resolved_execution_quality = execution_quality_ok
    if resolved_execution_quality is None:
        signals: list[bool] = []
        has_explicit_quality_evidence = False
        latest_run = dict(archive.get("latest_archive_replay_run") or archive.get("latest_run") or {})
        simulation_summary = dict(archive.get("simulation_summary") or {})
        execution_summary = dict(latest_run.get("execution_summary") or archive.get("execution_summary") or {})

        if "execution_quality_pass" in execution_summary:
            has_explicit_quality_evidence = True
            signals.append(as_bool(execution_summary.get("execution_quality_pass"), default=False))
        elif "execution_quality_pass" in simulation_summary:
            has_explicit_quality_evidence = True
            signals.append(as_bool(simulation_summary.get("execution_quality_pass"), default=False))

        if "fill_ratio" in objective or "rejected_ratio" in objective:
            has_explicit_quality_evidence = True
            fill_ratio = as_float(objective.get("fill_ratio"), default=0.0)
            rejected_ratio = as_float(objective.get("rejected_ratio"), default=1.0)
            signals.append(
                fill_ratio >= float(OBJECTIVE_EXECUTION_FILL_RATIO_FLOOR)
                and rejected_ratio <= float(OBJECTIVE_EXECUTION_REJECT_RATIO_CEILING)
            )

        if "traceability_ok" in latest_run:
            has_explicit_quality_evidence = True
            signals.append(as_bool(latest_run.get("traceability_ok"), default=False))
        elif "traceability_ok" in simulation_summary:
            has_explicit_quality_evidence = True
            signals.append(as_bool(simulation_summary.get("traceability_ok"), default=False))

        if release and ("ready_for_preopen_activation" in release or release_blockers):
            has_explicit_quality_evidence = True
            signals.append(
                as_bool(release.get("ready_for_preopen_activation"), default=False)
                and not release_blockers
            )
        if observability and ("routing_ready" in observability or "routing_health_green" in observability):
            has_explicit_quality_evidence = True
            signals.append(
                as_bool(observability.get("routing_ready"), default=False)
                and as_bool(observability.get("routing_health_green"), default=False)
            )
        if kill_switch:
            has_explicit_quality_evidence = True
            signals.append(False)

        resolved_execution_quality = all(signals) if has_explicit_quality_evidence and signals else False

    return {
        "state": resolved_state,
        "execution_quality_ok": bool(resolved_execution_quality),
        "kill_switch_enabled": kill_switch,
        "release_blockers": release_blockers,
        "has_execution_quality_evidence": bool(has_explicit_quality_evidence),
    }


def choose_profile(
    state: str,
    execution_quality_ok: bool,
    *,
    regime: str = "",
    has_execution_quality_evidence: bool = False,
) -> MainWaveProfile:
    normalized_state = str(state or "").strip().lower()
    normalized_regime = str(regime or "").strip().lower()
    if normalized_state == "risk_hold" or normalized_regime == "risk_off":
        return DEFENSE_RECOVERY
    if normalized_state in {"reached", "stable"}:
        return BASE_BALANCED
    if not has_execution_quality_evidence:
        return BASE_BALANCED if normalized_regime != "risk_off" else DEFENSE_RECOVERY
    if normalized_regime == "cautious":
        return ACCELERATION_PROBE if execution_quality_ok else BASE_BALANCED
    if execution_quality_ok:
        return TREND_TURBO if normalized_regime == "trend" else TREND_HYPER
    if normalized_regime == "trend":
        return ACCELERATION_PROBE
    return BASE_BALANCED


def _factor_value(factors: dict[str, float] | None, name: str, default: float = 0.0) -> float:
    if not isinstance(factors, dict):
        return float(default)
    try:
        return float(factors.get(name, default) or default)
    except (TypeError, ValueError):
        return float(default)


def _bounded_strength(value: float, *, floor: float = 0.0, cap: float = 1.0) -> float:
    clipped = max(min(float(value), cap), -cap)
    return max(clipped - float(floor), 0.0)


def classify_setup(
    factors: dict[str, float] | None,
    profile: MainWaveProfile,
    *,
    regime: str = "",
    factor_threshold_bias: float = 0.0,
    regime_overlay_score: float = 0.0,
    regime_overlay_floor: float = 0.0,
    regime_support_score: float = 0.0,
    regime_support_floor: float = 0.0,
) -> MainWaveSetupDecision:
    normalized_regime = str(regime or "").strip().lower()
    rel_strength = _factor_value(factors, "relative_strength_60")
    trend_quality = _factor_value(factors, "trend_quality_20")
    momentum_efficiency = _factor_value(factors, "momentum_efficiency_20")
    recovery_efficiency = _factor_value(factors, "recovery_efficiency_20")
    low_volume_box_support = _factor_value(factors, "low_volume_box_support_20")
    false_breakdown_repair = _factor_value(factors, "false_breakdown_repair_10")
    breakout_retest_absorption = _factor_value(factors, "breakout_retest_absorption_20")
    resistance_memory_decay = _factor_value(factors, "resistance_memory_decay_20")
    outflow_retest_absorption = _factor_value(factors, "outflow_retest_absorption_15")
    negative_flow_impact_beta = _factor_value(factors, "negative_flow_impact_beta_10")
    stock_flow_residual_vs_sector = _factor_value(factors, "stock_flow_residual_vs_sector_10")
    defensive_residual = _factor_value(factors, "stock_defensive_residual_on_sector_selloff_15")
    piotroski = _factor_value(factors, "piotroski_f_score_proxy_20")
    accruals = _factor_value(factors, "operating_accruals_assets_20")
    distress = _factor_value(factors, "distress_resilience_public_20")
    turnover_stability = _factor_value(factors, "turnover_stability_20")
    liquidity = _factor_value(factors, "liquidity_sponsorship_20")
    breakout_quality = _factor_value(factors, "breakout_quality_10")
    trend_acceleration = _factor_value(factors, "trend_acceleration_20")
    downside_volatility = _factor_value(factors, "downside_volatility_20")
    crowding_risk = _factor_value(factors, "crowding_risk_score")
    distribution_warning = _factor_value(factors, "distribution_warning_5")
    exhaustion_risk = _factor_value(factors, "exhaustion_risk_10")
    institutional_alpha_core = _factor_value(factors, "institutional_alpha_core_20")
    tail_adjusted_residual_alpha = _factor_value(factors, "tail_adjusted_residual_alpha_20")
    orthogonalized_residual_alpha = _factor_value(factors, "orthogonalized_residual_alpha_20")
    weak_market_reaccumulation_alpha = _factor_value(factors, "weak_market_reaccumulation_alpha_20")
    institutional_orthogonal_reaccumulation = _factor_value(factors, "institutional_orthogonal_reaccumulation_20")
    defensive_alpha_core = _factor_value(factors, "defensive_alpha_core_20")
    quality_residual_sponsorship_spread = _factor_value(factors, "quality_residual_sponsorship_spread_20")
    flow_absorption_half_life = _factor_value(factors, "flow_absorption_half_life_15")
    supply_absorption_repair = _factor_value(factors, "supply_absorption_repair_20")

    trend_strength = (
        0.22 * _bounded_strength(rel_strength, floor=-0.03, cap=0.9)
        + 0.18 * _bounded_strength(trend_quality, floor=-0.02, cap=1.2)
        + 0.16 * _bounded_strength(momentum_efficiency, floor=-0.03, cap=0.9)
        + 0.11 * _bounded_strength(recovery_efficiency, floor=0.0, cap=0.9)
        + 0.11 * _bounded_strength(low_volume_box_support, floor=0.04, cap=0.9)
        + 0.10 * _bounded_strength(breakout_retest_absorption, floor=0.03, cap=0.9)
        + 0.12 * _bounded_strength(resistance_memory_decay, floor=0.02, cap=0.9)
        + 0.08 * _bounded_strength(institutional_alpha_core, floor=0.20, cap=1.2)
        + 0.06 * _bounded_strength(tail_adjusted_residual_alpha, floor=0.18, cap=1.2)
    )
    repair_strength = (
        0.24 * _bounded_strength(recovery_efficiency, floor=0.0, cap=0.9)
        + 0.22 * _bounded_strength(low_volume_box_support, floor=0.04, cap=0.9)
        + 0.22 * _bounded_strength(false_breakdown_repair, floor=0.03, cap=0.9)
        + 0.18 * _bounded_strength(breakout_retest_absorption, floor=0.03, cap=0.9)
        + 0.14 * _bounded_strength(_factor_value(factors, "support_test_reclaim_15"), floor=0.05, cap=0.9)
        + 0.08 * _bounded_strength(weak_market_reaccumulation_alpha, floor=0.18, cap=1.2)
        + 0.08 * _bounded_strength(institutional_orthogonal_reaccumulation, floor=0.18, cap=1.2)
        + 0.07 * _bounded_strength(supply_absorption_repair, floor=0.16, cap=1.2)
        + 0.06 * _bounded_strength(flow_absorption_half_life, floor=0.14, cap=1.2)
    )
    flow_strength = (
        0.28 * _bounded_strength(outflow_retest_absorption, floor=0.04, cap=0.9)
        + 0.24 * _bounded_strength(negative_flow_impact_beta, floor=0.06, cap=0.9)
        + 0.22 * _bounded_strength(stock_flow_residual_vs_sector, floor=0.0, cap=1.0)
        + 0.20 * _bounded_strength(defensive_residual, floor=0.0, cap=1.0)
        + 0.06 * _bounded_strength(turnover_stability, floor=0.55, cap=1.2)
        + 0.12 * _bounded_strength(institutional_orthogonal_reaccumulation, floor=0.18, cap=1.2)
        + 0.12 * _bounded_strength(orthogonalized_residual_alpha, floor=0.18, cap=1.2)
        + 0.10 * _bounded_strength(tail_adjusted_residual_alpha, floor=0.18, cap=1.2)
        + 0.08 * _bounded_strength(flow_absorption_half_life, floor=0.14, cap=1.2)
    )
    quality_strength = (
        0.40 * _bounded_strength(piotroski, floor=0.0, cap=1.5)
        + 0.28 * _bounded_strength(accruals, floor=0.0, cap=1.5)
        + 0.32 * _bounded_strength(distress, floor=0.0, cap=1.5)
        + 0.10 * _bounded_strength(quality_residual_sponsorship_spread, floor=0.16, cap=1.2)
        + 0.08 * _bounded_strength(defensive_alpha_core, floor=0.18, cap=1.2)
    )
    support_strength = (
        0.45 * _bounded_strength(liquidity, floor=float(profile.liquidity_threshold), cap=1.0)
        + 0.35 * _bounded_strength(turnover_stability, floor=0.55, cap=1.2)
        + 0.20 * _bounded_strength(regime_support_score - regime_support_floor, floor=-0.04, cap=0.8)
    )
    risk_penalty = (
        max(crowding_risk - 0.62, 0.0) * 0.20
        + max(downside_volatility - 0.034, 0.0) * 4.0
        + max(distribution_warning - 0.20, 0.0) * 0.35
        + max(exhaustion_risk - 0.62, 0.0) * 0.32
    )
    composite_alpha_strength = (
        0.20 * _bounded_strength(institutional_alpha_core, floor=0.20, cap=1.2)
        + 0.17 * _bounded_strength(tail_adjusted_residual_alpha, floor=0.18, cap=1.2)
        + 0.16 * _bounded_strength(institutional_orthogonal_reaccumulation, floor=0.18, cap=1.2)
        + 0.15 * _bounded_strength(orthogonalized_residual_alpha, floor=0.18, cap=1.2)
        + 0.12 * _bounded_strength(weak_market_reaccumulation_alpha, floor=0.18, cap=1.2)
        + 0.10 * _bounded_strength(defensive_alpha_core, floor=0.18, cap=1.2)
        + 0.08 * _bounded_strength(quality_residual_sponsorship_spread, floor=0.16, cap=1.2)
        + 0.07 * _bounded_strength(flow_absorption_half_life, floor=0.14, cap=1.2)
        + 0.05 * _bounded_strength(supply_absorption_repair, floor=0.16, cap=1.2)
    )
    composite_alpha_bonus = max(composite_alpha_strength - risk_penalty * 1.5, 0.0)
    overlay_gap = float(regime_overlay_score) - float(regime_overlay_floor)
    support_gap = float(regime_support_score) - float(regime_support_floor)
    overlay_bonus = max(overlay_gap, -0.05) * 0.10 + max(support_gap, -0.05) * 0.12

    repair_score = (
        0.34 * repair_strength
        + 0.24 * flow_strength
        + 0.14 * quality_strength
        + 0.16 * support_strength
        + 0.06 * max(rel_strength, 0.0)
        + 0.06 * composite_alpha_bonus
        - risk_penalty
        + overlay_bonus
    )
    continuation_score = (
        0.34 * trend_strength
        + 0.18 * flow_strength
        + 0.10 * quality_strength
        + 0.14 * support_strength
        + 0.08 * max(trend_acceleration, 0.0)
        + 0.06 * max(breakout_quality, 0.0)
        + 0.07 * composite_alpha_bonus
        - risk_penalty
        + overlay_bonus
    )

    repair_floor = 0.20 + max(float(factor_threshold_bias), 0.0)
    continuation_floor = max(float(profile.entry_threshold), 0.10) + max(float(factor_threshold_bias), 0.0)
    if normalized_regime == "risk_off":
        return MainWaveSetupDecision(
            setup_type="none",
            enter=False,
            trend_strength=round(trend_strength, 6),
            repair_strength=round(repair_strength, 6),
            flow_strength=round(flow_strength, 6),
            quality_strength=round(quality_strength, 6),
            support_strength=round(support_strength, 6),
            risk_penalty=round(risk_penalty, 6),
            score=round(max(repair_score, continuation_score), 6),
        )
    prefer_repair = repair_score >= repair_floor and (
        repair_strength >= max(trend_strength * 0.85, 0.08)
        or false_breakdown_repair >= 0.05
        or breakout_retest_absorption >= 0.06
    )
    prefer_continuation = continuation_score >= continuation_floor and (
        trend_strength >= 0.10
        and support_strength >= 0.04
        and breakout_quality <= 0.10
    )
    if not prefer_continuation and continuation_score >= continuation_floor:
        prefer_continuation = (
            composite_alpha_strength >= 0.42
            and trend_strength >= 0.08
            and flow_strength >= 0.12
            and support_strength >= 0.04
            and risk_penalty <= 0.08
            and breakout_quality <= 0.10
        )
    if normalized_regime == "cautious":
        prefer_continuation = prefer_continuation and continuation_score >= repair_score + 0.03
    if prefer_repair and repair_score >= continuation_score - 0.02:
        return MainWaveSetupDecision(
            setup_type="repair_reclaim",
            enter=True,
            trend_strength=round(trend_strength, 6),
            repair_strength=round(repair_strength, 6),
            flow_strength=round(flow_strength, 6),
            quality_strength=round(quality_strength, 6),
            support_strength=round(support_strength, 6),
            risk_penalty=round(risk_penalty, 6),
            score=round(repair_score, 6),
        )
    if prefer_continuation:
        return MainWaveSetupDecision(
            setup_type="trend_continuation",
            enter=True,
            trend_strength=round(trend_strength, 6),
            repair_strength=round(repair_strength, 6),
            flow_strength=round(flow_strength, 6),
            quality_strength=round(quality_strength, 6),
            support_strength=round(support_strength, 6),
            risk_penalty=round(risk_penalty, 6),
            score=round(continuation_score, 6),
        )
    fallback_score = max(repair_score, continuation_score)
    return MainWaveSetupDecision(
        setup_type="none",
        enter=False,
        trend_strength=round(trend_strength, 6),
        repair_strength=round(repair_strength, 6),
        flow_strength=round(flow_strength, 6),
        quality_strength=round(quality_strength, 6),
        support_strength=round(support_strength, 6),
        risk_penalty=round(risk_penalty, 6),
        score=round(fallback_score, 6),
    )


def evaluate_exit(
    *,
    pnl_pct: float,
    deterioration_score: float,
    held_days: int,
    profile: MainWaveProfile,
    structure_break: bool = False,
    crowding_risk: float = 0.0,
    setup_type: str = "",
    trend_strength: float = 0.0,
    support_strength: float = 0.0,
    flow_strength: float = 0.0,
    deterioration_threshold: float = 0.07,
) -> MainWaveExitDecision:
    notes: list[str] = []
    base_time_exit_days = max(int(profile.time_exit_days), 1)
    runner_bias = 0.0
    if str(setup_type).strip().lower() in {"trend_continuation", "repair_reclaim"}:
        runner_bias += 0.5 * max(float(trend_strength), 0.0)
        runner_bias += 0.3 * max(float(support_strength), 0.0)
        runner_bias += 0.2 * max(float(flow_strength), 0.0)
    runner_eligible = (
        pnl_pct >= 0.03
        and deterioration_score <= 0.035
        and crowding_risk <= 0.68
        and structure_break is False
        and runner_bias >= 0.09
    )
    deterioration_limit = min(float(deterioration_threshold), 0.065)
    max_hold_days = base_time_exit_days
    if runner_eligible:
        runner_extension = 8
        if pnl_pct >= 0.06:
            runner_extension += 8
        if pnl_pct >= 0.10:
            runner_extension += 10
        if trend_strength >= 0.18:
            runner_extension += 8
        if support_strength >= 0.10 and flow_strength >= 0.08:
            runner_extension += 6
        max_hold_days = base_time_exit_days + runner_extension
        deterioration_limit = min(deterioration_limit + 0.01, 0.075)
        notes.append("runner_extension_active")
    else:
        if pnl_pct >= 0.02:
            max_hold_days += 4
        if pnl_pct >= 0.05:
            max_hold_days += 6

    if pnl_pct <= -profile.hard_stop_loss:
        return MainWaveExitDecision(True, "realization", runner_eligible, base_time_exit_days, max_hold_days, deterioration_limit, tuple(["hard_stop"]))
    if structure_break:
        return MainWaveExitDecision(True, "realization", runner_eligible, base_time_exit_days, max_hold_days, deterioration_limit, tuple(["structure_break"]))
    if crowding_risk >= 0.82:
        return MainWaveExitDecision(True, "realization", runner_eligible, base_time_exit_days, max_hold_days, deterioration_limit, tuple(["crowding_risk"]))
    if deterioration_score >= deterioration_limit:
        return MainWaveExitDecision(True, "realization", runner_eligible, base_time_exit_days, max_hold_days, deterioration_limit, tuple([*notes, "deterioration"]))
    if pnl_pct < 0.0 and deterioration_score >= 0.05:
        return MainWaveExitDecision(True, "realization", runner_eligible, base_time_exit_days, max_hold_days, deterioration_limit, tuple([*notes, "negative_pnl_deterioration"]))
    if pnl_pct <= 0.0:
        time_stop = held_days >= max(base_time_exit_days - 1, 1)
        return MainWaveExitDecision(bool(time_stop), "realization", runner_eligible, base_time_exit_days, max_hold_days, deterioration_limit, tuple(notes))
    should_realize = (not runner_eligible) and held_days >= max_hold_days
    should_runner_exit = runner_eligible and held_days >= max_hold_days
    if should_realize:
        return MainWaveExitDecision(True, "realization", runner_eligible, base_time_exit_days, max_hold_days, deterioration_limit, tuple([*notes, "time_realization"]))
    if should_runner_exit:
        return MainWaveExitDecision(True, "trend_runner", runner_eligible, base_time_exit_days, max_hold_days, deterioration_limit, tuple([*notes, "runner_time_limit"]))
    return MainWaveExitDecision(False, "trend_runner" if runner_eligible else "realization", runner_eligible, base_time_exit_days, max_hold_days, deterioration_limit, tuple(notes))


def should_enter(
    trend_acceleration: float,
    breakout_quality: float,
    liquidity_sponsorship: float,
    profile: MainWaveProfile,
    *,
    factor_threshold_bias: float = 0.0,
    official_clean_mode: bool = False,
    regime: str = "",
    regime_overlay_score: float = 0.0,
    regime_overlay_floor: float = 0.0,
    regime_support_score: float = 0.0,
    regime_support_floor: float = 0.0,
) -> bool:
    # This helper only sees three observables, so keep it as a bounded legacy
    # heuristic instead of pretending it is equivalent to the full live entry
    # contract.
    normalized_regime = str(regime or "").strip().lower()
    trend_floor = float(profile.entry_threshold) + max(float(factor_threshold_bias), 0.0)
    breakout_floor = float(profile.breakout_threshold)
    liquidity_floor = max(float(profile.liquidity_threshold), -0.12)
    breakout_extension_cap = 0.07
    if normalized_regime == "cautious":
        breakout_floor = max(float(profile.breakout_threshold), 0.01)
        liquidity_floor = max(liquidity_floor, 0.05)
        trend_floor = max(trend_floor, float(profile.entry_threshold) + 0.01)
        breakout_extension_cap = 0.06
    trend_confirm = float(trend_acceleration) >= trend_floor
    breakout_confirm = float(breakout_quality) >= breakout_floor and float(breakout_quality) <= breakout_extension_cap
    liquidity_confirm = float(liquidity_sponsorship) >= liquidity_floor
    if normalized_regime == "risk_off":
        return False
    regime_confirm = True
    if normalized_regime:
        effective_overlay_floor = float(regime_overlay_floor)
        overlay_gap = float(regime_overlay_score) - effective_overlay_floor
        support_gap = float(regime_support_score) - float(regime_support_floor)
        if normalized_regime == "cautious":
            regime_confirm = overlay_gap >= 0.0 and support_gap >= 0.03
        else:
            regime_confirm = overlay_gap >= 0.0 and support_gap >= 0.0
    if bool(official_clean_mode):
        return bool(trend_confirm and liquidity_confirm and regime_confirm)
    return bool(trend_confirm and liquidity_confirm and breakout_confirm and regime_confirm)


def should_exit(
    pnl_pct: float,
    deterioration_score: float,
    held_days: int,
    profile: MainWaveProfile,
    *,
    structure_break: bool = False,
    crowding_risk: float = 0.0,
    deterioration_threshold: float = 0.07,
    setup_type: str = "",
    trend_strength: float = 0.0,
    support_strength: float = 0.0,
    flow_strength: float = 0.0,
) -> bool:
    decision = evaluate_exit(
        pnl_pct=float(pnl_pct),
        deterioration_score=float(deterioration_score),
        held_days=max(int(held_days), 0),
        profile=profile,
        structure_break=bool(structure_break),
        crowding_risk=float(crowding_risk),
        deterioration_threshold=float(deterioration_threshold),
        setup_type=str(setup_type or ""),
        trend_strength=float(trend_strength),
        support_strength=float(support_strength),
        flow_strength=float(flow_strength),
    )
    return bool(decision.should_exit)
