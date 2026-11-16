# ruff: noqa
from __future__ import annotations

from math import tanh
from typing import Any

from factors_catalog.r441_oracle.regime_factor_policy import calc_regime_decision_adjustments


def _bounded(value: float, scale: float, *, center: float = 0.0) -> float:
    safe_scale = max(float(scale), 1e-9)
    return tanh((float(value) - float(center)) / safe_scale)


def _positive_signal(value: float | None) -> float:
    return max(float(value or 0.0), 0.0)


def _inverse_signal(value: float | None, *, ceiling: float = 1.0) -> float:
    bounded = max(min(float(value or 0.0), float(ceiling)), 0.0)
    return max(float(ceiling) - bounded, 0.0)


def _factor_value(factors: dict[str, float], name: str, default: float = 0.0) -> float:
    return float(factors.get(name, default) or default)


def calc_exit_guard_score(factors: dict[str, float]) -> float:
    return (
        max(float(factors.get("exit_deterioration_score_5", 0.0) or 0.0), 0.0) * 0.10
        + max(float(factors.get("distribution_warning_5", 0.0) or 0.0), 0.0) * 0.06
        + max(float(factors.get("exhaustion_risk_10", 0.0) or 0.0), 0.0) * 0.05
        + max(float(factors.get("crowding_risk_score", 0.0) or 0.0), 0.0) * 0.06
        + max(float(factors.get("regulatory_risk_penalty_20d", 0.0) or 0.0), 0.0) * 0.04
        + max(float(factors.get("crowding_event_conflict_5d", 0.0) or 0.0), 0.0) * 0.04
        + max(float(factors.get("downside_volatility_20", 0.0) or 0.0), 0.0) * 0.10
        + max(float(factors.get("amihud_illiquidity_20", 0.0) or 0.0), 0.0) * 0.03
        + max(-float(factors.get("opening_impulse_retention_30m", 0.0) or 0.0), 0.0) * 0.04
        + max(-float(factors.get("intraday_range_close_location_30m", 0.0) or 0.0), 0.0) * 0.04
        + max(-float(factors.get("stock_net_inflow_5_20", 0.0) or 0.0), 0.0) * 0.06
    )


def calc_daily_alpha_components(
    factors: dict[str, float],
    *,
    official_clean_mode: bool = False,
) -> tuple[float, float, float]:
    sector_rs = 0.0 if official_clean_mode else _factor_value(factors, "sector_rs_20")
    relative_strength = _bounded(_factor_value(factors, "relative_strength_60"), 0.15)
    breakout = _bounded(_factor_value(factors, "breakout_quality_10"), 0.05)
    breakout_pressure = _bounded(_factor_value(factors, "breakout_pressure_10"), 0.05)
    trend_acceleration = _bounded(_factor_value(factors, "trend_acceleration_20"), 0.03)
    momentum_20 = _bounded(_factor_value(factors, "price_momentum_20"), 0.12)
    momentum_60 = _bounded(_factor_value(factors, "price_momentum_60"), 0.20)
    momentum_efficiency = _bounded(_factor_value(factors, "momentum_efficiency_20"), 0.35)
    trend_stability = _bounded(_factor_value(factors, "trend_stability_15"), 0.35)
    trend_quality = _bounded(_factor_value(factors, "trend_quality_20"), 0.90)
    drawdown_recovery = _bounded(_factor_value(factors, "drawdown_recovery_20"), 0.35, center=0.25)
    recovery_efficiency = _bounded(_factor_value(factors, "recovery_efficiency_20"), 0.30, center=0.18)
    support_lift_strength = _bounded(_factor_value(factors, "support_lift_strength_20"), 0.90)
    box_ceiling_residence = _bounded(_factor_value(factors, "box_ceiling_residence_20"), 0.24, center=0.18)
    resistance_memory_decay = _bounded(_factor_value(factors, "resistance_memory_decay_20"), 0.24, center=0.06)
    breakout_retention = _bounded(_factor_value(factors, "breakout_retention_15"), 0.20, center=0.05)
    breakout_retest_absorption = _bounded(_factor_value(factors, "breakout_retest_absorption_20"), 0.24, center=0.08)
    low_volume_box_support = _bounded(_factor_value(factors, "low_volume_box_support_20"), 0.22, center=0.10)
    support_test_reclaim = _bounded(_factor_value(factors, "support_test_reclaim_15"), 0.28, center=0.12)
    false_breakdown_repair = _bounded(_factor_value(factors, "false_breakdown_repair_10"), 0.28, center=0.08)
    pullback_absorption = _bounded(_factor_value(factors, "pullback_absorption_10"), 0.25, center=0.45)
    liquidity_sponsorship = _bounded(_factor_value(factors, "liquidity_sponsorship_20"), 0.50)
    turnover_stability = _bounded(_factor_value(factors, "turnover_stability_20"), 0.12, center=0.72)
    flow_persistence = _bounded(_factor_value(factors, "flow_persistence_10"), 0.45)
    stock_inflow = _bounded(_factor_value(factors, "stock_net_inflow_5_20"), 1.00)
    selloff_absorption_flow = _bounded(_factor_value(factors, "selloff_absorption_flow_10"), 0.45)
    negative_flow_impact_beta = _bounded(_factor_value(factors, "negative_flow_impact_beta_10"), 0.22, center=0.18)
    downside_impact_improvement = _bounded(_factor_value(factors, "downside_impact_improvement_5_15"), 0.20, center=0.05)
    outflow_decay_price_repair = _bounded(_factor_value(factors, "outflow_decay_price_repair_3_10"), 0.32, center=0.05)
    outflow_repair_without_inflow = _bounded(_factor_value(factors, "outflow_repair_without_inflow_10"), 0.28, center=0.10)
    outflow_reversal_efficiency = _bounded(_factor_value(factors, "outflow_reversal_efficiency_10"), 0.24, center=0.30)
    turnover_adjusted_outflow_damage = _bounded(_factor_value(factors, "turnover_adjusted_outflow_damage_10"), 0.24, center=0.18)
    severe_outflow_nonbreak_rate = _bounded(_factor_value(factors, "severe_outflow_nonbreak_rate_15"), 0.24, center=0.28)
    volume_price_confirmation = _bounded(_factor_value(factors, "volume_price_confirmation_10"), 0.20)
    volatility_compression = _bounded(_factor_value(factors, "volatility_compression_20"), 0.25)
    shrink_volume_nonbreak = _bounded(_factor_value(factors, "shrink_volume_nonbreak_5_20"), 0.22, center=0.08)
    pullback_energy_decay = _bounded(_factor_value(factors, "pullback_energy_decay_20"), 0.22, center=0.06)
    lower_shadow_absorption = _bounded(_factor_value(factors, "lower_shadow_absorption_5"), 0.18, center=0.05)
    large_candle_followthrough = _bounded(_factor_value(factors, "large_candle_followthrough_5"), 0.40, center=0.20)
    sector_rs_score = _bounded(sector_rs, 0.12)
    flow_divergence = _bounded(_factor_value(factors, "flow_price_divergence_10"), 0.40)
    distribution_warning = _bounded(_factor_value(factors, "distribution_warning_5"), 0.30)
    exhaustion_risk = _bounded(_factor_value(factors, "exhaustion_risk_10"), 0.35)
    crowding_risk = _bounded(_factor_value(factors, "crowding_risk_score"), 0.35)
    realized_volatility = _bounded(_factor_value(factors, "realized_volatility_20"), 0.03)
    downside_volatility = _bounded(_factor_value(factors, "downside_volatility_20"), 0.03)
    illiquidity = _bounded(_factor_value(factors, "amihud_illiquidity_20"), 0.60)

    trend_core = (
        0.24 * relative_strength
        + 0.21 * trend_quality
        + 0.18 * momentum_efficiency
        + 0.07 * drawdown_recovery
        + 0.11 * recovery_efficiency
        + 0.07 * support_lift_strength
        + 0.06 * box_ceiling_residence
        + 0.09 * resistance_memory_decay
        + 0.07 * breakout_retention
        + 0.10 * breakout_retest_absorption
        + 0.11 * low_volume_box_support
        + 0.10 * support_test_reclaim
        + 0.11 * false_breakdown_repair
        + 0.07 * trend_stability
        + 0.06 * pullback_absorption
        + 0.05 * max(momentum_20, 0.0)
        + 0.04 * max(momentum_60, 0.0)
        + 0.03 * max(trend_acceleration, 0.0)
        + 0.01 * max(breakout_pressure, 0.0)
    )
    confirmation_quality = (
        0.22 * liquidity_sponsorship
        + 0.18 * turnover_stability
        + 0.16 * volume_price_confirmation
        + 0.01 * max(stock_inflow, 0.0)
        + 0.08 * max(selloff_absorption_flow, 0.0)
        + 0.11 * negative_flow_impact_beta
        + 0.10 * downside_impact_improvement
        + 0.13 * outflow_decay_price_repair
        + 0.11 * outflow_repair_without_inflow
        + 0.12 * outflow_reversal_efficiency
        + 0.10 * turnover_adjusted_outflow_damage
        + 0.10 * severe_outflow_nonbreak_rate
        + 0.10 * shrink_volume_nonbreak
        + 0.08 * pullback_energy_decay
        + 0.07 * lower_shadow_absorption
        + 0.06 * volatility_compression
        + 0.04 * large_candle_followthrough
        + 0.03 * sector_rs_score
    )
    risk_drag = (
        0.24 * max(exhaustion_risk, 0.0)
        + 0.22 * max(crowding_risk, 0.0)
        + 0.18 * max(downside_volatility, 0.0)
        + 0.12 * max(flow_divergence, 0.0)
        + 0.10 * max(distribution_warning, 0.0)
        + 0.08 * max(realized_volatility, 0.0)
        + 0.06 * max(illiquidity, 0.0)
    )
    repair_leg = (
        recovery_efficiency
        + low_volume_box_support
        + false_breakdown_repair
        + support_test_reclaim
        + breakout_retest_absorption
    ) / 5.0
    flow_leg = (
        turnover_stability
        + outflow_decay_price_repair
        + outflow_repair_without_inflow
        + negative_flow_impact_beta
        + severe_outflow_nonbreak_rate
    ) / 5.0
    trend_anchor = max(relative_strength, trend_quality, momentum_efficiency)
    defensive_confirmation_bias = max((0.58 * repair_leg + 0.42 * flow_leg) - trend_anchor, 0.0)
    defensive_drift_penalty = defensive_confirmation_bias * 0.22 + max(0.04 - trend_anchor, 0.0) * 0.16
    chase_penalty = max(breakout, 0.0) * max(0.35 - repair_leg, 0.0) * 0.30 + max(flow_persistence, 0.0) * max(0.30 - flow_leg, 0.0) * 0.22
    trend_core = max(trend_core - chase_penalty - defensive_drift_penalty, 0.0)
    confirmation_quality = max(confirmation_quality - chase_penalty * 0.5 - defensive_drift_penalty * 0.35, 0.0)
    risk_drag = max(risk_drag + defensive_drift_penalty * 0.45, 0.0)
    return trend_core, confirmation_quality, risk_drag


def calc_alpha_score(
    factors: dict[str, float],
    *,
    official_clean_mode: bool = False,
    daily_only_mode: bool = False,
) -> float:
    sector_rs = 0.0 if official_clean_mode else _factor_value(factors, "sector_rs_20")
    sector_flow_accel = 0.0 if official_clean_mode else _factor_value(factors, "sector_flow_accel_5")
    # Keep the score narrow: cross-sectional ranking should mostly reflect stock-
    # level alpha, while market state, event noise, and regime decisions live in
    # the entry/sizing layers.
    breakout = _bounded(_factor_value(factors, "breakout_quality_10"), 0.05)
    compression = _bounded(_factor_value(factors, "volatility_compression_20"), 0.25)
    turnover_stability = _bounded(_factor_value(factors, "turnover_stability_20"), 0.12, center=0.72)
    auction_gap = _bounded(_factor_value(factors, "auction_gap_strength_1d"), 0.04)
    trend_acceleration = _bounded(_factor_value(factors, "trend_acceleration_20"), 0.03)
    momentum_20 = _bounded(_factor_value(factors, "price_momentum_20"), 0.12)
    momentum_60 = _bounded(_factor_value(factors, "price_momentum_60"), 0.20)
    ma_gap_20 = _bounded(_factor_value(factors, "ma_gap_20"), 0.08)
    ma_gap_60 = _bounded(_factor_value(factors, "ma_gap_60"), 0.12)
    price_position = _bounded(_factor_value(factors, "price_position_20"), 0.20)
    pullback_absorption = _bounded(_factor_value(factors, "pullback_absorption_10"), 0.25, center=0.45)
    liquidity_sponsorship = _bounded(_factor_value(factors, "liquidity_sponsorship_20"), 0.50)
    relative_strength = _bounded(_factor_value(factors, "relative_strength_60"), 0.15)
    momentum_efficiency = _bounded(_factor_value(factors, "momentum_efficiency_20"), 0.35)
    trend_stability = _bounded(_factor_value(factors, "trend_stability_15"), 0.35)
    breakout_pressure = _bounded(_factor_value(factors, "breakout_pressure_10"), 0.05)
    close_drive = _bounded(_factor_value(factors, "close_drive_strength_30m"), 0.30)
    vwap_dislocation = _bounded(_factor_value(factors, "vwap_dislocation_30m"), 0.05)
    auction_turnover_imbalance = _bounded(_factor_value(factors, "auction_turnover_imbalance_30m"), 0.80)
    opening_impulse_retention = _bounded(_factor_value(factors, "opening_impulse_retention_30m"), 0.70)
    intraday_trend_consistency = _bounded(_factor_value(factors, "intraday_trend_consistency_30m"), 0.50)
    intraday_range_close = _bounded(_factor_value(factors, "intraday_range_close_location_30m"), 0.25)
    sector_rs_score = _bounded(sector_rs, 0.12)
    sector_flow_score = _bounded(sector_flow_accel, 0.60)
    stock_inflow = _bounded(_factor_value(factors, "stock_net_inflow_5_20"), 1.00)
    turnover_impulse = _bounded(_factor_value(factors, "turnover_impulse_5_20"), 0.60)
    flow_divergence = _bounded(_factor_value(factors, "flow_price_divergence_10"), 0.40)
    flow_correlation = _bounded(_factor_value(factors, "flow_correlation_10"), 0.45)
    distribution_warning = _bounded(_factor_value(factors, "distribution_warning_5"), 0.30)
    exhaustion_risk = _bounded(_factor_value(factors, "exhaustion_risk_10"), 0.35)
    crowding_risk = _bounded(_factor_value(factors, "crowding_risk_score"), 0.35)
    regulatory_risk = _bounded(_factor_value(factors, "regulatory_risk_penalty_20d"), 0.30)
    crowding_event_conflict = _bounded(_factor_value(factors, "crowding_event_conflict_5d"), 0.35)
    realized_volatility = _bounded(_factor_value(factors, "realized_volatility_20"), 0.03)
    downside_volatility = _bounded(_factor_value(factors, "downside_volatility_20"), 0.03)
    illiquidity = _bounded(_factor_value(factors, "amihud_illiquidity_20"), 0.60)
    flow_persistence = _bounded(_factor_value(factors, "flow_persistence_10"), 0.45)
    trend_quality = _bounded(_factor_value(factors, "trend_quality_20"), 0.90)
    drawdown_recovery = _bounded(_factor_value(factors, "drawdown_recovery_20"), 0.35, center=0.25)
    recovery_efficiency = _bounded(_factor_value(factors, "recovery_efficiency_20"), 0.30, center=0.18)
    support_lift_strength = _bounded(_factor_value(factors, "support_lift_strength_20"), 0.90)
    box_ceiling_residence = _bounded(_factor_value(factors, "box_ceiling_residence_20"), 0.24, center=0.18)
    volume_price_confirmation = _bounded(_factor_value(factors, "volume_price_confirmation_10"), 0.20)
    selloff_absorption_flow = _bounded(_factor_value(factors, "selloff_absorption_flow_10"), 0.45)
    large_candle_followthrough = _bounded(_factor_value(factors, "large_candle_followthrough_5"), 0.40, center=0.20)
    has_intraday_signal = (not daily_only_mode) and any(
        abs(float(factors.get(name, 0.0) or 0.0)) > 1e-9
        for name in (
            "close_drive_strength_30m",
            "vwap_dislocation_30m",
            "auction_turnover_imbalance_30m",
            "opening_impulse_retention_30m",
            "intraday_trend_consistency_30m",
            "intraday_range_close_location_30m",
        )
    )
    if not has_intraday_signal:
        trend_core, confirmation_quality, risk_drag = calc_daily_alpha_components(
            factors,
            official_clean_mode=official_clean_mode,
        )
        defensive_gap_penalty = max(confirmation_quality - trend_core, 0.0)
        daily_raw = (
            0.66 * trend_core
            + 0.24 * confirmation_quality
            - 0.50 * risk_drag
            + 0.02 * auction_gap
            - 0.18 * defensive_gap_penalty
        )
        scaled_daily = 0.5 + 0.5 * tanh(daily_raw / 0.22)
        return round(max(scaled_daily, 0.0), 6)
    stretched_trend_penalty = max(breakout, 0.0) * max(trend_acceleration, 0.0) * max(relative_strength, 0.0)
    liquidity_gap_penalty = max(0.0, 0.18 - _factor_value(factors, "liquidity_sponsorship_20")) / 0.18
    trend_fragility_penalty = max(trend_acceleration, 0.0) * max(-trend_stability, 0.0)
    breakout_chase_penalty = max(breakout, 0.0) * max(breakout_pressure, 0.0) * (
        0.55 + 0.45 * max(crowding_risk, 0.0)
    )
    downside_liquidity_penalty = max(downside_volatility, 0.0) * (0.60 + 0.40 * max(illiquidity, 0.0))
    raw = (
        + breakout * 0.02
        + trend_acceleration * 0.02
        + relative_strength * 0.17
        + liquidity_sponsorship * 0.18
        + stock_inflow * 0.07
        + momentum_efficiency * 0.09
        + trend_quality * 0.06
        + drawdown_recovery * 0.03
        + recovery_efficiency * 0.04
        + support_lift_strength * 0.03
        + box_ceiling_residence * 0.03
        + volume_price_confirmation * 0.04
        + selloff_absorption_flow * 0.03
        + large_candle_followthrough * 0.02
        + flow_correlation * 0.03
        + momentum_20 * 0.00
        + breakout_pressure * 0.00
        + close_drive * 0.04
        + opening_impulse_retention * 0.04
        + sector_rs_score * 0.07
        + intraday_trend_consistency * 0.03
        + sector_flow_score * 0.06
        + ma_gap_20 * 0.00
        + pullback_absorption * 0.04
        + turnover_impulse * 0.02
        + compression * 0.02
        + momentum_60 * 0.01
        + auction_gap * 0.02
        + price_position * 0.00
        + trend_stability * 0.04
        + intraday_range_close * 0.02
        + turnover_stability * 0.05
        - ma_gap_60 * 0.01
        - vwap_dislocation * 0.03
        - auction_turnover_imbalance * 0.02
        - flow_divergence * 0.06
        - distribution_warning * 0.07
        - exhaustion_risk * 0.10
        - crowding_risk * 0.11
        - regulatory_risk * 0.02
        - crowding_event_conflict * 0.05
        - realized_volatility * 0.06
        - downside_volatility * 0.09
        - illiquidity * 0.06
        - stretched_trend_penalty * 0.08
        - liquidity_gap_penalty * 0.08
        - trend_fragility_penalty * 0.06
        - breakout_chase_penalty * 0.08
        - downside_liquidity_penalty * 0.07
    )
    scaled = 0.5 + 0.5 * tanh(raw / 0.35)
    return round(max(scaled, 0.0), 6)


def calc_live_quality_components(factors: dict[str, float]) -> tuple[float, float, float]:
    flow_alignment = _inverse_signal(factors.get("flow_price_divergence_10"))
    core_trend_anchor = (
        0.55 * _positive_signal(factors.get("relative_strength_60"))
        + 0.45 * _positive_signal(factors.get("momentum_efficiency_20"))
        + 0.40 * _positive_signal(factors.get("breakout_quality_10"))
    )
    secondary_trend_raw = (
        0.50 * _positive_signal(factors.get("trend_acceleration_20"))
        + 0.35 * _positive_signal(factors.get("price_momentum_60"))
        + 0.30 * _positive_signal(factors.get("price_momentum_20"))
        + 0.25 * _positive_signal(factors.get("breakout_pressure_10"))
        + 0.20 * _positive_signal(factors.get("trend_stability_15"))
        + 0.18 * _positive_signal(factors.get("pullback_absorption_10"))
    )
    secondary_trend_scale = min(core_trend_anchor / 0.32, 1.0) if core_trend_anchor > 0.0 else 0.0
    secondary_trend_stack = secondary_trend_raw * secondary_trend_scale
    trend_stack = (
        0.26 * _positive_signal(factors.get("relative_strength_60"))
        + 0.22 * _positive_signal(factors.get("momentum_efficiency_20"))
        + 0.18 * _positive_signal(factors.get("breakout_quality_10"))
        + 0.14 * secondary_trend_stack
    )
    confirmation_stack = (
        0.18 * _positive_signal(factors.get("flow_persistence_10"))
        + 0.14 * flow_alignment
        + 0.14 * _positive_signal(factors.get("breakout_pressure_10"))
        + 0.10 * _positive_signal(factors.get("price_position_20"))
        + 0.10 * _positive_signal(factors.get("opening_impulse_retention_30m"))
        + 0.08 * _positive_signal(factors.get("intraday_trend_consistency_30m"))
        + 0.08 * _positive_signal(factors.get("close_drive_strength_30m"))
        + 0.10 * _positive_signal(factors.get("volatility_compression_20"))
        + 0.08 * _positive_signal(factors.get("liquidity_sponsorship_20"))
    )
    sponsorship_stack = (
        0.16 * _positive_signal(factors.get("sector_rs_20"))
        + 0.12 * _positive_signal(factors.get("stock_net_inflow_5_20"))
        + 0.10 * max(float(factors.get("turnover_stability_20", 0.0) or 0.0) - 0.70, 0.0)
        + 0.08 * _positive_signal(factors.get("market_breadth_support_10"))
        + 0.06 * _positive_signal(factors.get("sentiment_tailwind_10"))
        + 0.06 * _positive_signal(factors.get("event_consistency_20d"))
    )
    risk_stack = (
        0.32 * _positive_signal(factors.get("crowding_risk_score"))
        + 0.26 * _positive_signal(factors.get("exhaustion_risk_10"))
        + 0.12 * _positive_signal(factors.get("flow_price_divergence_10"))
        + 0.14 * _positive_signal(factors.get("gap_exhaustion_score"))
        + 0.10 * _positive_signal(factors.get("distribution_warning_5"))
        + 0.08 * _positive_signal(factors.get("crowding_event_conflict_5d"))
        + 0.08 * _positive_signal(factors.get("exit_deterioration_score_5"))
        + 0.06 * _positive_signal(factors.get("drawdown_from_high_20"))
        + 1.60 * _positive_signal(factors.get("downside_volatility_20"))
    )
    trend_quality = tanh(0.85 * trend_stack)
    confirmation_quality = tanh(0.70 * (confirmation_stack + sponsorship_stack))
    risk_drag = tanh(0.90 * risk_stack)
    return trend_quality, confirmation_quality, risk_drag


def calc_live_quality_adjustment(
    factors: dict[str, float],
    *,
    regime: str,
    position_scale: float,
    overlay_score: float,
    support_score: float,
) -> float:
    trend_quality, confirmation_quality, risk_drag = calc_live_quality_components(factors)
    turnover_stability = float(factors.get("turnover_stability_20", 0.0) or 0.0)
    liquidity_sponsorship = float(factors.get("liquidity_sponsorship_20", 0.0) or 0.0)
    downside_volatility = float(factors.get("downside_volatility_20", 0.0) or 0.0)
    quality_scale = min(
        max(
            0.35
            + 0.38 * max(float(support_score), 0.0)
            + 0.22 * max(float(overlay_score), 0.0)
            + 0.18 * max(float(position_scale) - 0.55, 0.0),
            0.25,
        ),
        1.10,
    )
    normalized_regime = str(regime or "").strip().lower()
    if normalized_regime == "risk_off":
        return 0.010 * confirmation_quality * min(quality_scale, 0.50) - 0.060 * risk_drag
    if normalized_regime == "cautious":
        cautious_fragility_penalty = (
            max(0.70 - turnover_stability, 0.0) * 0.18
            + max(0.08 - liquidity_sponsorship, 0.0) * 0.22
            + max(downside_volatility - 0.031, 0.0) * 5.5
        )
        return (
            0.028 * trend_quality * quality_scale
            + 0.020 * confirmation_quality * quality_scale
            - 0.050 * risk_drag
            - cautious_fragility_penalty
        )
    trend_fragility_penalty = (
        max(0.68 - turnover_stability, 0.0) * 0.12
        + max(0.06 - liquidity_sponsorship, 0.0) * 0.14
        + max(downside_volatility - 0.033, 0.0) * 3.5
    )
    return (
        0.040 * trend_quality * quality_scale
        + 0.026 * confirmation_quality * quality_scale
        - 0.045 * risk_drag
        - trend_fragility_penalty
    )


def calc_total_score(
    factors: dict[str, float],
    *,
    official_clean_mode: bool = False,
    daily_only_mode: bool = False,
    regime: str | None = None,
    regime_factor_policy_overrides: dict[str, list[dict[str, Any]]] | None = None,
    regime_decision_policy_overrides: dict[str, dict[str, Any]] | None = None,
    regime_factor_overlay_scale: float = 1.0,
    alpha_score: float | None = None,
) -> float:
    if alpha_score is None:
        alpha_score = calc_alpha_score(
            factors,
            official_clean_mode=official_clean_mode,
            daily_only_mode=daily_only_mode,
        )
    alpha_score = float(alpha_score)
    normalized_regime = str(regime or "").strip().lower()
    if not normalized_regime:
        return alpha_score
    decision = calc_regime_decision_adjustments(
        factors,
        normalized_regime,
        overrides=regime_factor_policy_overrides,
        decision_overrides=regime_decision_policy_overrides,
        overlay_scale=regime_factor_overlay_scale,
    )
    overlay_score = float(decision.get("effective_overlay_score", 0.0) or 0.0)
    support_score = float(decision.get("support_score", 0.0) or 0.0)
    position_scale = float(decision.get("position_scale", 1.0) or 1.0)
    regime_adjustment = (
        min(0.0, 0.08 * overlay_score)
        + min(0.0, 0.10 * (support_score - 0.5))
        + min(0.0, 0.05 * (position_scale - 1.0))
        + calc_live_quality_adjustment(
            factors,
            regime=normalized_regime,
            position_scale=position_scale,
            overlay_score=overlay_score,
            support_score=support_score,
        )
    )
    scaled = max(0.0, min(alpha_score + regime_adjustment, 1.0))
    return round(scaled, 6)


def calc_event_reweight(factors: dict[str, float]) -> float:
    value = (
        1.0
        - max(-factors["policy_alignment_score_10d"], 0.0) * 0.08
        - max(-factors["shareholder_alignment_60d"], 0.0) * 0.08
        - factors["regulatory_risk_penalty_20d"] * 0.20
        - factors["crowding_event_conflict_5d"] * 0.16
    )
    return max(0.70, min(1.0, value))


def apply_layer_d_guard(factors: dict[str, float], core_enter: bool) -> tuple[bool, bool]:
    triggered = factors["crowding_risk_score"] >= 0.85 and factors["board_break_rate"] >= 0.55
    return (False, True) if triggered else (core_enter, False)
