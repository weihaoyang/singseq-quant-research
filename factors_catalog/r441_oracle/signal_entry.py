# ruff: noqa
from __future__ import annotations

from factors_catalog.r441_oracle.main_wave_strategy import MainWaveProfile


def _thresholds_for_profile(profile: MainWaveProfile) -> dict[str, float]:
    if profile.name == "trend_turbo":
        return {
            "relative_strength_floor": -0.10,
            "pullback_floor": 0.25,
            "flow_persistence_floor": 0.28,
            "flow_divergence_cap": 0.72,
            "rotation_rs_floor": -0.15,
            "rotation_flow_floor": -0.12,
            "momentum_efficiency_floor": 0.05,
            "trend_stability_floor": 0.14,
            "breakout_pressure_floor": -0.01,
            "turnover_impulse_floor": -0.08,
            "exhaustion_cap": 0.78,
            "vol_compression_floor": 0.0,
            "turnover_stability_floor": 0.60,
            "liquidity_floor": 0.02,
            "downside_volatility_cap": 0.036,
            "crowding_cap": 0.66,
            "breakout_extension_cap": 0.075,
        }
    if profile.name == "trend_hyper":
        return {
            "relative_strength_floor": -0.04,
            "pullback_floor": 0.30,
            "flow_persistence_floor": 0.36,
            "flow_divergence_cap": 0.58,
            "rotation_rs_floor": -0.05,
            "rotation_flow_floor": -0.06,
            "momentum_efficiency_floor": 0.07,
            "trend_stability_floor": 0.23,
            "breakout_pressure_floor": 0.0,
            "turnover_impulse_floor": -0.04,
            "exhaustion_cap": 0.72,
            "vol_compression_floor": 0.0,
            "turnover_stability_floor": 0.64,
            "liquidity_floor": 0.04,
            "downside_volatility_cap": 0.033,
            "crowding_cap": 0.60,
            "breakout_extension_cap": 0.060,
        }
    return {
        "relative_strength_floor": 0.00,
        "pullback_floor": 0.45,
        "flow_persistence_floor": 0.45,
        "flow_divergence_cap": 0.42,
        "rotation_rs_floor": 0.0,
        "rotation_flow_floor": -0.02,
        "momentum_efficiency_floor": 0.10,
        "trend_stability_floor": 0.24,
        "breakout_pressure_floor": 0.0,
        "turnover_impulse_floor": 0.0,
        "exhaustion_cap": 0.56,
        "vol_compression_floor": 0.0,
        "turnover_stability_floor": 0.68,
        "liquidity_floor": 0.06,
        "downside_volatility_cap": 0.030,
        "crowding_cap": 0.55,
        "breakout_extension_cap": 0.050,
    }


def _apply_bias(thresholds: dict[str, float], factor_threshold_bias: float) -> dict[str, float]:
    adjusted = dict(thresholds)
    bias = float(factor_threshold_bias)
    adjusted["relative_strength_floor"] -= bias
    adjusted["pullback_floor"] -= bias
    adjusted["flow_persistence_floor"] -= bias
    adjusted["flow_divergence_cap"] = max(0.1, min(adjusted["flow_divergence_cap"] + bias, 1.5))
    adjusted["rotation_rs_floor"] -= bias
    adjusted["rotation_flow_floor"] -= bias
    adjusted["momentum_efficiency_floor"] -= bias
    adjusted["trend_stability_floor"] -= bias
    adjusted["breakout_pressure_floor"] -= bias
    adjusted["turnover_impulse_floor"] -= bias
    adjusted["exhaustion_cap"] = max(0.2, min(adjusted["exhaustion_cap"] + bias, 1.0))
    return adjusted


def _risk_off_defensive_confirm(factors: dict[str, float], thresholds: dict[str, float]) -> bool:
    defensive_core = (
        factors.get("relative_strength_60", 0.0) >= max(-0.05, thresholds["relative_strength_floor"] - 0.02)
        and factors.get("turnover_stability_20", 0.0) >= 0.58
        and factors.get("flow_correlation_10", 0.0) <= 1.02
        and factors.get("volatility_compression_20", 0.0) >= -0.05
        and factors.get("exhaustion_risk_10", 0.0) <= min(0.68, thresholds["exhaustion_cap"] + 0.14)
        and factors.get("flow_price_divergence_10", 0.0) <= min(0.60, thresholds["flow_divergence_cap"] + 0.20)
        and factors.get("downside_volatility_20", 0.0) <= 0.05
    )
    catalyst_count = sum(
        1
        for passed in (
            factors.get("panic_reversal_score", 0.0) >= 0.001,
            factors.get("policy_alignment_score_10d", 0.0) >= 0.0,
            factors.get("event_surprise_score_5", 0.0) >= 0.0,
            factors.get("shareholder_alignment_60d", 0.0) >= 0.0,
            factors.get("market_breadth_support_10", 0.0) >= 0.03,
            factors.get("sentiment_tailwind_10", 0.0) >= 0.0,
        )
        if passed
    )
    return defensive_core and catalyst_count >= 1


def _trend_exhaustion_cap(profile: MainWaveProfile, thresholds: dict[str, float]) -> float:
    cap = float(thresholds["exhaustion_cap"])
    if profile.name == "trend_hyper":
        return min(max(cap + 0.22, 0.2), 0.9)
    if profile.name == "trend_turbo":
        return min(max(cap + 0.12, 0.2), 0.85)
    return cap


def _trend_quality_confirm(
    factors: dict[str, float],
    *,
    profile: MainWaveProfile,
    thresholds: dict[str, float],
    regime: str,
) -> bool:
    normalized_regime = str(regime or "").strip().lower()
    relative_strength_floor = max(0.02, float(thresholds["relative_strength_floor"]))
    liquidity_floor = max(float(thresholds["liquidity_floor"]), float(profile.liquidity_threshold), 0.0)
    turnover_stability_floor = float(thresholds["turnover_stability_floor"])
    downside_volatility_cap = float(thresholds["downside_volatility_cap"])
    trend_stability_floor = float(thresholds["trend_stability_floor"])
    crowding_cap = float(thresholds["crowding_cap"])
    if normalized_regime == "cautious":
        relative_strength_floor = max(relative_strength_floor, 0.00)
        liquidity_floor = max(liquidity_floor, 0.0)
        turnover_stability_floor = max(turnover_stability_floor, 0.60)
        downside_volatility_cap = max(downside_volatility_cap, 0.038)
        crowding_cap = max(crowding_cap, 0.66)
    liquidity_confirm = factors.get("liquidity_sponsorship_20", 0.0) >= liquidity_floor
    relative_strength_confirm = factors.get("relative_strength_60", 0.0) >= relative_strength_floor
    stability_confirm = factors.get("trend_stability_15", 0.0) >= trend_stability_floor
    turnover_confirm = (
        "turnover_stability_20" not in factors
        or factors.get("turnover_stability_20", 0.0) >= turnover_stability_floor
    )
    downside_confirm = factors.get("downside_volatility_20", 0.0) <= downside_volatility_cap
    crowding_confirm = factors.get("crowding_risk_score", 0.0) <= crowding_cap
    return (
        liquidity_confirm
        and stability_confirm
        and turnover_confirm
        and downside_confirm
        and crowding_confirm
        and (
            relative_strength_confirm
            or factors.get("trend_quality_20", 0.0) >= max(trend_stability_floor * 0.75, 0.0)
            or factors.get("momentum_efficiency_20", 0.0) >= thresholds["momentum_efficiency_floor"]
        )
    )


def _daily_trend_stack_score(factors: dict[str, float]) -> float:
    return (
        max(float(factors.get("relative_strength_60", 0.0) or 0.0), 0.0) * 1.2
        + max(float(factors.get("trend_quality_20", 0.0) or 0.0), 0.0) * 1.1
        + max(float(factors.get("momentum_efficiency_20", 0.0) or 0.0), 0.0) * 0.9
        + max(float(factors.get("drawdown_recovery_20", 0.0) or 0.0), 0.0) * 0.6
        + max(float(factors.get("recovery_efficiency_20", 0.0) or 0.0), 0.0) * 0.8
        + max(float(factors.get("support_lift_strength_20", 0.0) or 0.0), 0.0) * 0.5
        + max(float(factors.get("box_ceiling_residence_20", 0.0) or 0.0), 0.0) * 0.5
        + max(float(factors.get("resistance_memory_decay_20", 0.0) or 0.0) - 0.05, 0.0) * 0.8
        + max(float(factors.get("breakout_retention_15", 0.0) or 0.0) - 0.04, 0.0) * 0.6
        + max(float(factors.get("low_volume_box_support_20", 0.0) or 0.0) - 0.06, 0.0) * 1.0
        + max(float(factors.get("support_test_reclaim_15", 0.0) or 0.0) - 0.08, 0.0) * 0.8
        + max(float(factors.get("false_breakdown_repair_10", 0.0) or 0.0) - 0.04, 0.0) * 1.0
        + max(float(factors.get("price_momentum_20", 0.0) or 0.0), 0.0) * 0.4
        + max(float(factors.get("price_momentum_60", 0.0) or 0.0), 0.0) * 0.3
    )


def _daily_confirmation_stack_score(factors: dict[str, float]) -> float:
    turnover_stability = float(factors.get("turnover_stability_20", 0.0) or 0.0)
    return (
        max(float(factors.get("liquidity_sponsorship_20", 0.0) or 0.0), 0.0) * 1.0
        + max(turnover_stability - 0.60, 0.0) * 1.6
        + max(float(factors.get("stock_net_inflow_5_20", 0.0) or 0.0), 0.0) * 0.08
        + max(float(factors.get("volume_price_confirmation_10", 0.0) or 0.0), 0.0) * 0.8
        + max(float(factors.get("selloff_absorption_flow_10", 0.0) or 0.0), 0.0) * 0.6
        + max(float(factors.get("negative_flow_impact_beta_10", 0.0) or 0.0) - 0.08, 0.0) * 0.9
        + max(float(factors.get("downside_impact_improvement_5_15", 0.0) or 0.0), 0.0) * 0.8
        + max(float(factors.get("outflow_decay_price_repair_3_10", 0.0) or 0.0), 0.0) * 0.9
        + max(float(factors.get("outflow_reversal_efficiency_10", 0.0) or 0.0) - 0.30, 0.0) * 0.9
        + max(float(factors.get("turnover_adjusted_outflow_damage_10", 0.0) or 0.0) - 0.18, 0.0) * 0.8
        + max(float(factors.get("severe_outflow_nonbreak_rate_15", 0.0) or 0.0) - 0.28, 0.0) * 0.8
        + max(float(factors.get("shrink_volume_nonbreak_5_20", 0.0) or 0.0) - 0.04, 0.0) * 0.8
        + max(float(factors.get("pullback_energy_decay_20", 0.0) or 0.0), 0.0) * 0.6
        + max(float(factors.get("lower_shadow_absorption_5", 0.0) or 0.0) - 0.03, 0.0) * 0.4
        + max(float(factors.get("pullback_absorption_10", 0.0) or 0.0) - 0.35, 0.0) * 0.5
    )


def _daily_only_structural_confirm(
    factors: dict[str, float],
    *,
    regime: str,
    regime_support_score: float = 0.0,
) -> bool:
    normalized_regime = str(regime or "").strip().lower()
    trend_stack = _daily_trend_stack_score(factors)
    confirmation_stack = _daily_confirmation_stack_score(factors)
    risk_drag = _daily_risk_penalty(factors)
    liquidity = float(factors.get("liquidity_sponsorship_20", 0.0) or 0.0)
    turnover_stability = float(factors.get("turnover_stability_20", 0.0) or 0.0)
    rel_strength = float(factors.get("relative_strength_60", 0.0) or 0.0)
    trend_quality = float(factors.get("trend_quality_20", 0.0) or 0.0)
    momentum_efficiency = float(factors.get("momentum_efficiency_20", 0.0) or 0.0)
    support_lift_strength = float(factors.get("support_lift_strength_20", 0.0) or 0.0)
    downside_volatility = float(factors.get("downside_volatility_20", 0.0) or 0.0)
    if normalized_regime == "risk_off":
        support_backed_defensive = (
            float(regime_support_score) >= 0.28
            and rel_strength >= 0.04
            and turnover_stability >= 0.66
            and downside_volatility <= 0.022
            and trend_stack >= 0.22
            and confirmation_stack >= 0.22
            and trend_stack + confirmation_stack - risk_drag >= 0.58
        )
        return (
            (
                rel_strength >= -0.01
                and liquidity >= -0.08
                and turnover_stability >= 0.62
                and downside_volatility <= 0.038
                and trend_stack >= 0.16
                and confirmation_stack >= 0.12
                and trend_stack + confirmation_stack - risk_drag >= 0.40
            )
            or support_backed_defensive
        )
    if normalized_regime == "cautious":
        structural_recovery_confirm = (
            turnover_stability >= 0.60
            and (
                float(factors.get("negative_flow_impact_beta_10", 0.0) or 0.0) >= 0.12
                or float(factors.get("downside_impact_improvement_5_15", 0.0) or 0.0) >= 0.04
                or float(factors.get("outflow_decay_price_repair_3_10", 0.0) or 0.0) >= 0.06
                or float(factors.get("outflow_reversal_efficiency_10", 0.0) or 0.0) >= 0.38
                or float(factors.get("turnover_adjusted_outflow_damage_10", 0.0) or 0.0) >= 0.28
                or float(factors.get("severe_outflow_nonbreak_rate_15", 0.0) or 0.0) >= 0.40
                or float(factors.get("shrink_volume_nonbreak_5_20", 0.0) or 0.0) >= 0.08
            )
            and downside_volatility <= 0.040
            and trend_stack >= 0.16
            and confirmation_stack >= 0.78
            and trend_stack + confirmation_stack - risk_drag >= 0.86
            and (
                rel_strength >= -0.03
                or float(factors.get("drawdown_recovery_20", 0.0) or 0.0) >= 0.10
                or float(regime_support_score) >= 0.12
            )
        )
        return (
            (
                rel_strength >= -0.02
                and liquidity >= -0.04
                and turnover_stability >= 0.60
                and downside_volatility <= 0.040
                and trend_stack >= 0.16
                and confirmation_stack >= 0.78
                and trend_stack + confirmation_stack - risk_drag >= 0.90
            )
            or structural_recovery_confirm
        )
    return (
        (
            rel_strength >= 0.02
            and trend_quality >= 0.0
            and momentum_efficiency >= 0.0
            and liquidity >= 0.03
            and turnover_stability >= 0.62
            and downside_volatility <= 0.036
            and trend_stack >= 0.24
            and confirmation_stack >= 0.36
            and trend_stack + confirmation_stack - risk_drag >= 0.66
        )
        or (
            turnover_stability >= 0.62
            and (
                float(factors.get("negative_flow_impact_beta_10", 0.0) or 0.0) >= 0.12
                or float(factors.get("downside_impact_improvement_5_15", 0.0) or 0.0) >= 0.04
                or float(factors.get("outflow_decay_price_repair_3_10", 0.0) or 0.0) >= 0.06
                or float(factors.get("outflow_reversal_efficiency_10", 0.0) or 0.0) >= 0.38
                or float(factors.get("turnover_adjusted_outflow_damage_10", 0.0) or 0.0) >= 0.28
                or float(factors.get("severe_outflow_nonbreak_rate_15", 0.0) or 0.0) >= 0.40
                or float(factors.get("shrink_volume_nonbreak_5_20", 0.0) or 0.0) >= 0.08
                or float(factors.get("pullback_energy_decay_20", 0.0) or 0.0) >= 0.05
            )
            and downside_volatility <= 0.036
            and trend_stack >= 0.16
            and confirmation_stack >= 0.72
            and trend_stack + confirmation_stack - risk_drag >= 0.82
            and (
                rel_strength >= 0.0
                or trend_quality >= 0.04
                or momentum_efficiency >= 0.03
                or (
                    float(factors.get("drawdown_recovery_20", 0.0) or 0.0) >= 0.12
                    and support_lift_strength >= 0.05
                )
            )
        )
    )


def _daily_risk_penalty(factors: dict[str, float]) -> float:
    return (
        max(float(factors.get("crowding_risk_score", 0.0) or 0.0) - 0.58, 0.0) * 0.8
        + max(float(factors.get("exhaustion_risk_10", 0.0) or 0.0) - 0.52, 0.0) * 0.7
        + max(float(factors.get("distribution_warning_5", 0.0) or 0.0) - 0.14, 0.0) * 0.7
        + max(float(factors.get("downside_volatility_20", 0.0) or 0.0) - 0.031, 0.0) * 12.0
    )


def _risk_off_quality_confirm(factors: dict[str, float], thresholds: dict[str, float]) -> bool:
    return (
        factors.get("relative_strength_60", 0.0) >= max(-0.06, thresholds["relative_strength_floor"] - 0.03)
        and factors.get("turnover_stability_20", 0.0) >= 0.54
        and factors.get("liquidity_sponsorship_20", 0.0) >= -0.25
        and factors.get("crowding_risk_score", 0.0) <= 0.36
        and factors.get("downside_volatility_20", 0.0) <= 0.05
    )


def _breakout_quality_floor(profile: MainWaveProfile, regime: str) -> float:
    normalized_regime = str(regime or "").strip().lower()
    if normalized_regime == "cautious":
        return max(float(profile.breakout_threshold), 0.0)
    return float(profile.breakout_threshold)


def _breakout_extension_cap(profile: MainWaveProfile, thresholds: dict[str, float], regime: str) -> float:
    normalized_regime = str(regime or "").strip().lower()
    cap = float(thresholds["breakout_extension_cap"])
    if normalized_regime == "cautious":
        return cap
    if profile.name == "acceleration_probe":
        return min(cap + 0.01, 0.09)
    return cap


def _regime_overlay_floor(profile: MainWaveProfile, regime: str, overlay_floor: float) -> float:
    normalized_regime = str(regime or "").strip().lower()
    return float(overlay_floor)


def _regime_confirm(
    *,
    profile: MainWaveProfile,
    regime: str,
    regime_overlay_score: float,
    regime_overlay_floor: float,
    regime_support_score: float,
    regime_support_floor: float,
) -> bool:
    normalized_regime = str(regime or "").strip().lower()
    if not normalized_regime:
        return True
    effective_overlay_floor = _regime_overlay_floor(profile, normalized_regime, regime_overlay_floor)
    overlay_gap = float(regime_overlay_score) - effective_overlay_floor
    support_gap = float(regime_support_score) - float(regime_support_floor)
    if normalized_regime == "risk_off":
        return (
            (overlay_gap >= 0.0 and support_gap >= -0.04)
            or (support_gap >= 0.0 and overlay_gap >= -0.03)
            or (
                float(regime_overlay_score) >= 0.03
                and float(regime_support_score) >= max(float(regime_support_floor) - 0.08, 0.0)
            )
        )
    if normalized_regime == "cautious":
        return (
            (overlay_gap >= 0.0 and support_gap >= -0.03)
            or (support_gap >= 0.0 and overlay_gap >= -0.03)
            or (overlay_gap >= -0.02 and support_gap >= -0.06)
        )
    return overlay_gap >= 0.0 or (overlay_gap >= -0.03 and support_gap >= 0.04)


def _hard_entry_conflict(
    *,
    factors: dict[str, float],
    thresholds: dict[str, float],
    regime: str,
) -> tuple[bool, list[str]]:
    normalized_regime = str(regime or "").strip().lower()
    exhaustion = float(factors.get("exhaustion_risk_10", 0.0))
    downside_volatility = float(factors.get("downside_volatility_20", 0.0))
    distribution_warning = float(factors.get("distribution_warning_5", 0.0))
    crowding_risk = float(factors.get("crowding_risk_score", 0.0))
    stock_net_inflow = float(factors.get("stock_net_inflow_5_20", 0.0))
    flow_correlation = float(factors.get("flow_correlation_10", 0.0))
    sector_flow_accel = float(factors.get("sector_flow_accel_5", 0.0))
    reasons: list[str] = []
    if normalized_regime == "risk_off":
        if downside_volatility >= 0.05:
            reasons.append("hard_downside_volatility_conflict")
        if exhaustion >= min(0.82, thresholds["exhaustion_cap"] + 0.24):
            reasons.append("hard_exhaustion_conflict")
        if crowding_risk >= 0.78 and distribution_warning >= 0.24:
            reasons.append("hard_crowding_distribution_conflict")
    else:
        if downside_volatility >= thresholds["downside_volatility_cap"] + 0.008:
            reasons.append("hard_downside_volatility_conflict")
        if exhaustion >= min(0.88, thresholds["exhaustion_cap"] + 0.20):
            reasons.append("hard_exhaustion_conflict")
        if distribution_warning >= (0.48 if normalized_regime == "cautious" else 0.30):
            reasons.append("hard_distribution_conflict")
        if crowding_risk >= max(0.62, thresholds["crowding_cap"] + 0.05) and (
            distribution_warning >= 0.20
            or downside_volatility >= thresholds["downside_volatility_cap"]
            or exhaustion >= min(0.78, thresholds["exhaustion_cap"] + 0.12)
        ):
            reasons.append("hard_crowding_conflict")
        if (
            normalized_regime not in {"risk_off", "cautious"}
            and stock_net_inflow < -0.10
            and flow_correlation >= 0.985
            and sector_flow_accel <= 0.02
        ):
            reasons.append("hard_flow_conflict_blocked")
    return bool(reasons), reasons


def entry_confirmations(
    factors: dict[str, float],
    profile: MainWaveProfile,
    *,
    factor_threshold_bias: float = 0.0,
    official_clean_mode: bool = False,
    regime: str = "",
    regime_overlay_score: float = 0.0,
    regime_overlay_floor: float = 0.0,
    regime_support_score: float = 0.0,
    regime_support_floor: float = 0.0,
    collect_reasons: bool = True,
    daily_only_mode: bool = False,
) -> tuple[bool, list[str]]:
    normalized_regime = str(regime or "").strip().lower()
    effective_regime = normalized_regime or "trend"
    if bool(daily_only_mode):
        reasons: list[str] = []
        trend_stack = _daily_trend_stack_score(factors)
        confirmation_stack = _daily_confirmation_stack_score(factors)
        risk_drag = _daily_risk_penalty(factors)
        regime_confirm = _regime_confirm(
            profile=profile,
            regime=normalized_regime,
            regime_overlay_score=float(regime_overlay_score),
            regime_overlay_floor=float(regime_overlay_floor),
            regime_support_score=float(regime_support_score),
            regime_support_floor=float(regime_support_floor),
        )
        hard_conflict, hard_conflict_reasons = _hard_entry_conflict(
            factors=factors,
            thresholds=_apply_bias(_thresholds_for_profile(profile), factor_threshold_bias),
            regime=effective_regime,
        )
        structural_confirm = _daily_only_structural_confirm(
            factors,
            regime=effective_regime,
            regime_support_score=float(regime_support_score),
        )
        if normalized_regime == "risk_off":
            risk_filter_confirm = (
                float(factors.get("crowding_risk_score", 0.0) or 0.0) <= 0.45
                and float(factors.get("distribution_warning_5", 0.0) or 0.0) <= 0.28
                and float(factors.get("exhaustion_risk_10", 0.0) or 0.0) <= 0.70
            )
        elif normalized_regime == "cautious":
            risk_filter_confirm = (
                float(factors.get("crowding_risk_score", 0.0) or 0.0) <= 0.58
                and float(factors.get("distribution_warning_5", 0.0) or 0.0) <= 0.34
                and float(factors.get("exhaustion_risk_10", 0.0) or 0.0) <= 0.76
            )
        else:
            risk_filter_confirm = (
                float(factors.get("crowding_risk_score", 0.0) or 0.0) <= 0.62
                and float(factors.get("distribution_warning_5", 0.0) or 0.0) <= 0.30
                and float(factors.get("exhaustion_risk_10", 0.0) or 0.0) <= 0.78
            )
        quality_confirm = structural_confirm
        if collect_reasons:
            if structural_confirm:
                reasons.append("daily_only_structural_confirmed")
            if risk_filter_confirm:
                reasons.append("risk_filter_confirmed")
            if quality_confirm:
                reasons.append("quality_confirmed")
            if regime_confirm and str(regime or "").strip():
                reasons.append("regime_overlay_confirmed")
                reasons.append(f"regime_support={float(regime_support_score):.3f}")
            reasons.extend(hard_conflict_reasons)
            reasons.append(f"daily_trend_stack={trend_stack:.3f}")
            reasons.append(f"daily_confirmation_stack={confirmation_stack:.3f}")
            reasons.append(f"daily_risk_drag={risk_drag:.3f}")
        return structural_confirm and risk_filter_confirm and quality_confirm and regime_confirm and not hard_conflict, reasons
    thresholds = _apply_bias(_thresholds_for_profile(profile), factor_threshold_bias)
    # Keep entry vetoes aligned with the live score: score already penalizes
    # stretched/chasing names, so confirmations should focus on structural
    # validity instead of re-requiring the same momentum-style expansion traits.
    trend_confirm = (
        factors["volatility_compression_20"] >= thresholds["vol_compression_floor"]
    )
    breakout_confirm = (
        factors["breakout_quality_10"] >= _breakout_quality_floor(profile, normalized_regime)
    )
    breakout_not_stretched = (
        factors["breakout_quality_10"] <= _breakout_extension_cap(profile, thresholds, effective_regime)
    )
    flow_confirm = (
        factors["flow_persistence_10"] >= thresholds["flow_persistence_floor"]
        or (
            factors["flow_price_divergence_10"] <= thresholds["flow_divergence_cap"]
            and factors.get("stock_net_inflow_5_20", 0.0) >= -0.04
        )
    )
    rotation_confirm = bool(official_clean_mode) or (
        factors["sector_rs_20"] >= thresholds["rotation_rs_floor"]
        and factors["sector_flow_accel_5"] >= thresholds["rotation_flow_floor"]
    )
    risk_filter_confirm = factors["exhaustion_risk_10"] <= _trend_exhaustion_cap(profile, thresholds)
    quality_confirm = _trend_quality_confirm(
        factors,
        profile=profile,
        thresholds=thresholds,
        regime=effective_regime,
    )
    if effective_regime not in {"risk_off", "cautious"}:
        crowding_risk = factors.get("crowding_risk_score", 0.0)
        severe_crowding_conflict = (
            crowding_risk >= thresholds["crowding_cap"]
            and (
                factors.get("distribution_warning_5", 0.0) >= 0.18
                or factors.get("downside_volatility_20", 0.0) >= thresholds["downside_volatility_cap"] - 0.002
                or factors.get("exhaustion_risk_10", 0.0) >= min(0.76, thresholds["exhaustion_cap"] + 0.10)
            )
        )
        risk_filter_confirm = (
            risk_filter_confirm
            and factors.get("downside_volatility_20", 0.0) <= thresholds["downside_volatility_cap"]
            and not severe_crowding_conflict
        )
    if normalized_regime == "cautious":
        rotation_confirm = bool(official_clean_mode) or (
            factors.get("sector_rs_20", 0.0) >= thresholds["rotation_rs_floor"]
            or factors.get("sector_flow_accel_5", 0.0) >= thresholds["rotation_flow_floor"]
        )
        breakout_confirm = breakout_confirm and breakout_not_stretched
        risk_filter_confirm = (
            risk_filter_confirm
            and not (
                factors.get("sector_flow_accel_5", 0.0) >= 1.30
                and factors.get("exhaustion_risk_10", 0.0) >= 0.58
            )
        )
    if normalized_regime == "risk_off":
        support_backed_defensive_flow = (
            float(regime_support_score) >= max(float(regime_support_floor) + 0.08, 0.26)
            and factors.get("relative_strength_60", 0.0) >= -0.02
            and factors.get("downside_volatility_20", 0.0) <= 0.03
        )
        breakout_confirm = factors.get("breakout_quality_10", 0.0) >= max(0.0, _breakout_quality_floor(profile, normalized_regime))
        flow_confirm = (
            (
                factors.get("flow_persistence_10", 0.0) >= max(0.05, thresholds["flow_persistence_floor"] - 0.30)
                and factors.get("flow_price_divergence_10", 0.0) <= min(0.65, thresholds["flow_divergence_cap"] + 0.25)
                and factors.get("stock_net_inflow_5_20", 0.0) >= -1.0
            )
            or support_backed_defensive_flow
        )
        if not official_clean_mode:
            rotation_confirm = (
                factors.get("sector_rs_20", 0.0) >= -0.06
                and factors.get("sector_flow_accel_5", 0.0) >= -0.28
            )
        risk_filter_confirm = risk_filter_confirm and _risk_off_defensive_confirm(factors, thresholds)
        quality_confirm = _risk_off_quality_confirm(factors, thresholds)
    elif normalized_regime in {"trend", ""}:
        breakout_confirm = breakout_confirm and breakout_not_stretched
        flow_confirm = flow_confirm and (
            "turnover_stability_20" not in factors
            or factors.get("turnover_stability_20", 0.0) >= thresholds["turnover_stability_floor"]
        )
    regime_confirm = _regime_confirm(
        profile=profile,
        regime=normalized_regime,
        regime_overlay_score=float(regime_overlay_score),
        regime_overlay_floor=float(regime_overlay_floor),
        regime_support_score=float(regime_support_score),
        regime_support_floor=float(regime_support_floor),
    )
    hard_conflict, hard_conflict_reasons = _hard_entry_conflict(
        factors=factors,
        thresholds=thresholds,
        regime=effective_regime,
    )
    if normalized_regime == "risk_off":
        min_structural_confirms = 1
    elif normalized_regime == "cautious":
        min_structural_confirms = 2
    else:
        min_structural_confirms = 3
    structural_confirm_count = sum(
        1
        for passed in (
            trend_confirm,
            breakout_confirm,
            flow_confirm,
            rotation_confirm,
        )
        if passed
    )
    daily_trend_stack = _daily_trend_stack_score(factors)
    daily_confirmation_stack = _daily_confirmation_stack_score(factors)
    daily_risk_drag = _daily_risk_penalty(factors)
    if normalized_regime == "risk_off":
        structural_confirm = structural_confirm_count >= min_structural_confirms and (
            daily_trend_stack + daily_confirmation_stack - daily_risk_drag
        ) >= 0.10
    elif normalized_regime == "cautious":
        structural_confirm = structural_confirm_count >= min_structural_confirms and (
            daily_trend_stack + daily_confirmation_stack - daily_risk_drag
        ) >= 0.26
    else:
        structural_confirm = structural_confirm_count >= min_structural_confirms and (
            daily_trend_stack >= 0.14
            and daily_confirmation_stack >= 0.08
            and daily_trend_stack + daily_confirmation_stack - daily_risk_drag >= 0.44
        )
    reasons: list[str] = []
    if collect_reasons:
        if trend_confirm:
            reasons.append("trend_confirmed")
        if breakout_confirm:
            reasons.append("breakout_confirmed")
        if breakout_not_stretched and breakout_confirm:
            reasons.append("breakout_not_stretched")
        if flow_confirm:
            reasons.append("flow_confirmed")
        if official_clean_mode:
            reasons.append("rotation_clean_mode_skipped")
        elif rotation_confirm:
            reasons.append("rotation_confirmed")
        if risk_filter_confirm:
            reasons.append("risk_filter_confirmed")
        if quality_confirm:
            reasons.append("quality_confirmed")
        if normalized_regime == "risk_off" and risk_filter_confirm:
            reasons.append("risk_off_defensive_confirmed")
        if regime_confirm and str(regime or "").strip():
            reasons.append("regime_overlay_confirmed")
            reasons.append(f"regime_support={float(regime_support_score):.3f}")
        reasons.extend(hard_conflict_reasons)
        if structural_confirm:
            reasons.append(f"structural_confirm_quorum={structural_confirm_count}")
        reasons.append(f"daily_trend_stack={daily_trend_stack:.3f}")
        reasons.append(f"daily_confirmation_stack={daily_confirmation_stack:.3f}")
        reasons.append(f"daily_risk_drag={daily_risk_drag:.3f}")
    return structural_confirm and risk_filter_confirm and quality_confirm and regime_confirm and not hard_conflict, reasons
