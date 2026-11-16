# ruff: noqa
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from factors_catalog.r441_oracle.numba_kernels import compute_core_price_turnover_factor_bundle
from factors_catalog.r441_oracle.factor_common import compute_common_factor_values
from factors_catalog.r441_oracle.factor_models import FactorInputs
from factors_catalog.factor_registry import canonical_factor_name
from factors_catalog.r441_oracle.factors import (
    advance_participation_uniformity_20,
    abnormal_turnover_z20,
    amihud_illiquidity_20,
    auction_gap_strength_1d,
    auction_turnover_imbalance_30m,
    board_break_rate,
    bullish_engulfing_hold_10,
    breakout_close_control_10,
    breakout_pressure_10,
    breakout_quality_10,
    close_drive_strength_30m,
    compression_release_quality_15,
    crowding_event_conflict_5d,
    crowding_risk_score,
    distribution_decay_after_inflow_spike_8,
    distribution_warning_5,
    downside_impact_decay_15,
    downside_impact_improvement_5_15,
    downside_volatility_20,
    event_consistency_20d,
    event_recency_decay_30m,
    event_sector_diffusion_5d,
    event_surprise_score_5,
    exhaustion_risk_10,
    exit_deterioration_score_5,
    flat_top_higher_low_coil_20,
    fractal_trend_persistence_20,
    failed_breakout_scarcity_20,
    failed_breakout_reversal_intensity_12,
    failed_breakout_recoil_repair_15,
    failed_reclaim_persistence_15,
    false_breakdown_repair_10,
    flow_persistence_10,
    flow_price_divergence_10,
    flow_shock_half_life_resilience_20,
    gap_fill_acceptance_8,
    gap_exhaustion_score,
    gap_failure_containment_10,
    impact_absorption_resilience_20,
    institutional_trend_acceptance_20,
    institutional_trend_acceptance_clean_20,
    institutional_trend_false_acceptance_risk_20,
    innovation_drift_absorption_20,
    intraday_range_close_location_30m,
    intraday_trend_consistency_30m,
    large_candle_followthrough_5,
    lower_shadow_absorption_5,
    leader_laggard_spread_10,
    limit_up_continuation_rate,
    liquidity_impact_relief_10,
    liquidity_sponsorship_20,
    ma_gap_20,
    ma_gap_60,
    market_breadth_support_10,
    momentum_efficiency_20,
    negative_flow_impact_beta_10,
    opening_impulse_retention_30m,
    outflow_repair_without_inflow_10,
    panic_followthrough_repair_8,
    panic_reversal_score,
    policy_alignment_score_10d,
    price_momentum_20,
    price_momentum_60,
    price_position_20,
    pullback_absorption_10,
    realized_volatility_20,
    regulatory_risk_penalty_20d,
    relative_drawdown_resilience_15,
    relative_strength_60,
    residual_momentum_20_60,
    outflow_decay_price_repair_3_10,
    breakout_failure_repair_15,
    delayed_undercut_reclaim_12,
    outflow_reversal_efficiency_10,
    outflow_retest_absorption_15,
    selloff_absorption_flow_10,
    severe_outflow_nonbreak_rate_15,
    sector_breadth_10,
    sector_flow_conversion_efficiency_20,
    sector_flow_breadth_confirm_5_10,
    sector_flow_accel_5,
    sector_flow_breadth_divergence_crowding_15,
    sector_negative_flow_breadth_resilience_10,
    rotation_early_diffusion_10_20,
    sector_net_inflow_5_20,
    sector_rs_20,
    shareholder_alignment_60d,
    stock_net_inflow_5_20,
    stock_defensive_residual_on_sector_selloff_15,
    stock_sector_breadth_lead_10,
    stock_flow_residual_vs_sector_10,
    stock_flow_residual_acceleration_5_20,
    stock_sector_breadth_capture_10,
    stock_sector_flow_resonance_10,
    stock_relative_sponsorship_under_sector_crowding_10,
    sentiment_tailwind_10,
    shrink_volume_nonbreak_5_20,
    support_test_reclaim_15,
    serial_outflow_damage_dampening_12,
    turnover_adjusted_outflow_damage_10,
    turnover_impulse_5_20,
    turnover_stability_20,
    trend_liquidity_persistence_20,
    trend_stability_15,
    trend_acceleration_20,
    upper_shadow_distribution_pressure_10,
    volatility_compression_20,
    vwap_dislocation_30m,
    flow_correlation_10,
    pullback_energy_decay_20,
    reaction_high_retake_quality_15,
    repair_base_tightness_12,
    repair_liquidity_retention_10,
    recrowding_reacceleration_quality_12,
    resistance_memory_decay_20,
    breakout_retention_15,
    breakout_retest_absorption_20,
    shock_relaxation_convexity_15,
    spectral_trend_coherence_20,
    steady_trend_advancement_20,
    upper_half_churn_trap_20,
    panic_followthrough_decay_12,
)
from factors_catalog.r441_oracle.main_wave_strategy import MainWaveProfile, classify_setup
from factors_catalog.r441_oracle.regime_factor_policy import calc_regime_decision_adjustments
from factors_catalog.r441_oracle.signal_entry import entry_confirmations
from factors_catalog.r441_oracle.signal_inputs import intraday_input_quality, normalize_inputs
from factors_catalog.r441_oracle.signal_scoring import apply_layer_d_guard, calc_event_reweight, calc_exit_guard_score, calc_total_score


@dataclass(frozen=True)
class MainWaveSignal:
    enter: bool
    score: float
    reasons: list[str]
    alpha_score: float = 0.0
    factors: dict[str, float] = field(default_factory=dict)
    exit_guard_score: float = 0.0
    regime_overlay_score: float = 0.0
    regime_support_score: float = 0.0
    position_scale: float = 1.0
    setup_type: str = ""
    trend_strength: float = 0.0
    flow_strength: float = 0.0
    support_strength: float = 0.0


_LIVE_FACTOR_ALIASES = {
    "price_momentum_20": ("momentum_20",),
    "price_momentum_60": ("momentum_60",),
    "ma_gap_20": ("price_vs_ma_20",),
    "ma_gap_60": ("price_vs_ma_60",),
    "volatility_compression_20": ("compression_ratio_20_60",),
    "realized_volatility_20": ("realized_vol_20",),
    "downside_volatility_20": ("downside_vol_20",),
    "abnormal_turnover_z20": ("turnover_zscore_20",),
}


_DAILY_ONLY_NEUTRALIZED_FACTORS = {
    "event_recency_decay_30m",
}


def _compute_factors(
    prices: list[float],
    turnover: list[float],
    inputs: dict[str, list[float]],
    *,
    enable_numba_kernels: bool = True,
) -> dict[str, float]:
    factors = _compute_market_structure_factors(prices=prices, inputs=inputs)
    factors.update(_compute_flow_sentiment_event_factors(prices=prices, turnover=turnover, inputs=inputs))
    factors.update(
        compute_core_price_turnover_factor_bundle(
            prices,
            turnover,
            enable_numba=bool(enable_numba_kernels),
        )
    )
    factors["liquidity_impact_relief_10"] = liquidity_impact_relief_10(prices, turnover)
    _apply_common_factor_values(factors=factors, prices=prices, turnover=turnover, inputs=inputs)
    _apply_crowding_policy_factors(factors=factors, inputs=inputs)
    _apply_public_institutional_composites(factors)
    _apply_factor_aliases(factors)
    return factors


def _neutralize_clean_mode_factors(factors: dict[str, float]) -> None:
    for name in (
        "sector_rs_20",
        "sector_flow_accel_5",
        "market_breadth_support_10",
        "policy_alignment_score_10d",
        "event_sector_diffusion_5d",
        "sector_breadth_10",
        "rotation_early_diffusion_10_20",
    ):
        factors[name] = 0.0


def _neutralize_daily_only_mode_factors(factors: dict[str, float]) -> None:
    # Daily-only mode should disable minute-specific event recency behavior, but
    # still keep valid daily alpha inputs such as turnover, flow, rotation, and
    # daily sentiment proxies. The daily research path depends on those signals.
    for name in _DAILY_ONLY_NEUTRALIZED_FACTORS:
        factors[name] = 0.0


def _safe_return(curr: float, prev: float) -> float:
    return (curr - prev) / prev if prev else 0.0


def _series_returns(values: list[float]) -> list[float]:
    return [_safe_return(values[index + 1], values[index]) for index in range(len(values) - 1)] if len(values) >= 2 else []


def _compute_market_structure_factors(*, prices: list[float], inputs: dict[str, list[float]]) -> dict[str, float]:
    return {
        "relative_strength_60": relative_strength_60(prices, inputs["benchmark_prices"]),
        "close_drive_strength_30m": close_drive_strength_30m(inputs["closes_30m"]),
        "vwap_dislocation_30m": vwap_dislocation_30m(inputs["closes_30m"], inputs["vwap_30m"]),
        "auction_gap_strength_1d": auction_gap_strength_1d(inputs["opens_1d"], prices),
        "auction_turnover_imbalance_30m": auction_turnover_imbalance_30m(inputs["turnover_30m"]),
        "opening_impulse_retention_30m": opening_impulse_retention_30m(inputs["opens_30m"], inputs["closes_30m"]),
        "intraday_trend_consistency_30m": intraday_trend_consistency_30m(inputs["closes_30m"]),
        "intraday_range_close_location_30m": intraday_range_close_location_30m(inputs["highs_30m"], inputs["lows_30m"], inputs["closes_30m"]),
        "sector_rs_20": sector_rs_20(inputs["sector_closes"], inputs["market_closes"]),
        "sector_breadth_10": sector_breadth_10(inputs["sector_up_counts"], inputs["sector_total_counts"]),
        "sector_flow_accel_5": sector_flow_accel_5(inputs["sector_flows"]),
        "sector_flow_breadth_confirm_5_10": sector_flow_breadth_confirm_5_10(
            inputs["sector_flows"],
            inputs["sector_up_counts"],
            inputs["sector_total_counts"],
        ),
        "rotation_early_diffusion_10_20": rotation_early_diffusion_10_20(
            inputs["sector_closes"],
            inputs["market_closes"],
            inputs["sector_up_counts"],
            inputs["sector_total_counts"],
        ),
        "sector_flow_breadth_divergence_crowding_15": sector_flow_breadth_divergence_crowding_15(
            inputs["sector_flows"],
            inputs["sector_up_counts"],
            inputs["sector_total_counts"],
        ),
        "sector_negative_flow_breadth_resilience_10": sector_negative_flow_breadth_resilience_10(
            inputs["sector_flows"],
            inputs["sector_up_counts"],
            inputs["sector_total_counts"],
        ),
        "leader_laggard_spread_10": leader_laggard_spread_10(inputs["leader_prices"], inputs["laggard_prices"]),
    }


def _compute_flow_sentiment_event_factors(
    *,
    prices: list[float],
    turnover: list[float],
    inputs: dict[str, list[float]],
) -> dict[str, float]:
    return {
        "stock_net_inflow_5_20": stock_net_inflow_5_20(inputs["stock_flows"]),
        "stock_sector_flow_resonance_10": stock_sector_flow_resonance_10(inputs["stock_flows"], inputs["sector_flows"]),
        "stock_flow_residual_vs_sector_10": stock_flow_residual_vs_sector_10(prices, inputs["stock_flows"], inputs["sector_flows"]),
        "stock_flow_residual_acceleration_5_20": stock_flow_residual_acceleration_5_20(prices, inputs["stock_flows"], inputs["sector_flows"]),
        "stock_defensive_residual_on_sector_selloff_15": stock_defensive_residual_on_sector_selloff_15(
            prices,
            inputs["sector_closes"],
            inputs["stock_flows"],
            inputs["sector_flows"],
        ),
        "relative_drawdown_resilience_15": relative_drawdown_resilience_15(
            prices,
            inputs["sector_closes"],
            inputs["market_closes"],
        ),
        "residual_momentum_20_60": residual_momentum_20_60(
            prices,
            inputs["sector_closes"],
            inputs["market_closes"],
        ),
        "stock_sector_breadth_capture_10": stock_sector_breadth_capture_10(
            prices,
            inputs["sector_closes"],
            inputs["stock_flows"],
            inputs["sector_flows"],
            inputs["sector_up_counts"],
            inputs["sector_total_counts"],
        ),
        "stock_sector_breadth_lead_10": stock_sector_breadth_lead_10(
            inputs["stock_flows"],
            inputs["sector_flows"],
            inputs["sector_up_counts"],
            inputs["sector_total_counts"],
        ),
        "stock_relative_sponsorship_under_sector_crowding_10": stock_relative_sponsorship_under_sector_crowding_10(
            inputs["stock_flows"],
            inputs["sector_flows"],
            inputs["sector_up_counts"],
            inputs["sector_total_counts"],
        ),
        "selloff_absorption_flow_10": selloff_absorption_flow_10(prices, inputs["stock_flows"]),
        "downside_impact_decay_15": downside_impact_decay_15(prices, inputs["stock_flows"]),
        "negative_flow_impact_beta_10": negative_flow_impact_beta_10(prices, inputs["stock_flows"]),
        "impact_absorption_resilience_20": impact_absorption_resilience_20(
            prices,
            inputs["lows_1d"],
            turnover,
            inputs["stock_flows"],
        ),
        "downside_impact_improvement_5_15": downside_impact_improvement_5_15(prices, inputs["stock_flows"]),
        "panic_followthrough_repair_8": panic_followthrough_repair_8(
            inputs["opens_1d"],
            inputs["highs_1d"],
            inputs["lows_1d"],
            prices,
            turnover,
        ),
        "panic_followthrough_decay_12": panic_followthrough_decay_12(
            inputs["opens_1d"],
            inputs["highs_1d"],
            inputs["lows_1d"],
            prices,
            turnover,
        ),
        "outflow_decay_price_repair_3_10": outflow_decay_price_repair_3_10(prices, inputs["stock_flows"]),
        "outflow_repair_without_inflow_10": outflow_repair_without_inflow_10(prices, inputs["stock_flows"]),
        "outflow_reversal_efficiency_10": outflow_reversal_efficiency_10(inputs["lows_1d"], prices, inputs["stock_flows"]),
        "outflow_retest_absorption_15": outflow_retest_absorption_15(
            inputs["lows_1d"],
            prices,
            turnover,
            inputs["stock_flows"],
        ),
        "shock_relaxation_convexity_15": shock_relaxation_convexity_15(
            prices,
            turnover,
            inputs["stock_flows"],
        ),
        "distribution_decay_after_inflow_spike_8": distribution_decay_after_inflow_spike_8(
            prices,
            turnover,
            inputs["stock_flows"],
        ),
        "serial_outflow_damage_dampening_12": serial_outflow_damage_dampening_12(
            prices,
            turnover,
            inputs["stock_flows"],
        ),
        "turnover_adjusted_outflow_damage_10": turnover_adjusted_outflow_damage_10(prices, turnover, inputs["stock_flows"]),
        "severe_outflow_nonbreak_rate_15": severe_outflow_nonbreak_rate_15(inputs["lows_1d"], prices, inputs["stock_flows"]),
        "sector_net_inflow_5_20": sector_net_inflow_5_20(inputs["sector_flows"]),
        "sector_flow_conversion_efficiency_20": sector_flow_conversion_efficiency_20(
            inputs["sector_flows"],
            inputs["sector_closes"],
            inputs["market_closes"],
        ),
        "flow_persistence_10": flow_persistence_10(inputs["stock_flows"]),
        "flow_price_divergence_10": flow_price_divergence_10(inputs["stock_flows"], prices),
        "flow_correlation_10": flow_correlation_10(prices, inputs["stock_flows"]),
        "flow_shock_half_life_resilience_20": flow_shock_half_life_resilience_20(
            prices,
            turnover,
            inputs["stock_flows"],
        ),
        "institutional_trend_acceptance_20": institutional_trend_acceptance_20(
            prices,
            turnover,
            inputs["stock_flows"],
        ),
        "institutional_trend_acceptance_clean_20": institutional_trend_acceptance_clean_20(
            prices,
            turnover,
            inputs["stock_flows"],
        ),
        "institutional_trend_false_acceptance_risk_20": institutional_trend_false_acceptance_risk_20(
            prices,
            turnover,
            inputs["stock_flows"],
        ),
        "limit_up_continuation_rate": limit_up_continuation_rate(inputs["sentiment_series"]),
        "board_break_rate": board_break_rate(inputs["board_break_series"]),
        "gap_exhaustion_score": gap_exhaustion_score(inputs["gap_series"]),
        "panic_reversal_score": panic_reversal_score(inputs["reversal_series"]),
        "market_breadth_support_10": market_breadth_support_10(inputs["sentiment_series"]),
        "sentiment_tailwind_10": sentiment_tailwind_10(inputs["sentiment_series"], inputs["board_break_series"]),
        "event_surprise_score_5": event_surprise_score_5(inputs["event_direction_series"], inputs["event_intensity_series"], inputs["event_confidence_series"]),
        "event_recency_decay_30m": event_recency_decay_30m(inputs["event_age_minutes"], inputs["event_impact_series"]),
        "event_consistency_20d": event_consistency_20d(inputs["event_impact_series"]),
        "event_sector_diffusion_5d": event_sector_diffusion_5d(inputs["event_symbol_hits"], inputs["event_sector_hits"]),
        "regulatory_risk_penalty_20d": regulatory_risk_penalty_20d(inputs["regulatory_event_series"]),
        "shareholder_alignment_60d": shareholder_alignment_60d(inputs["shareholder_event_series"]),
        "lower_shadow_absorption_5": lower_shadow_absorption_5(inputs["opens_1d"], inputs["highs_1d"], inputs["lows_1d"], prices),
        "reaction_high_retake_quality_15": reaction_high_retake_quality_15(
            inputs["highs_1d"],
            inputs["lows_1d"],
            prices,
            turnover,
        ),
        "gap_failure_containment_10": gap_failure_containment_10(
            inputs["opens_1d"],
            inputs["highs_1d"],
            inputs["lows_1d"],
            prices,
            turnover,
        ),
        "breakout_close_control_10": breakout_close_control_10(inputs["opens_1d"], inputs["highs_1d"], inputs["lows_1d"], prices),
        "gap_fill_acceptance_8": gap_fill_acceptance_8(
            inputs["opens_1d"],
            inputs["highs_1d"],
            inputs["lows_1d"],
            prices,
            turnover,
        ),
        "bullish_engulfing_hold_10": bullish_engulfing_hold_10(
            inputs["opens_1d"],
            inputs["highs_1d"],
            inputs["lows_1d"],
            prices,
        ),
        "upper_shadow_distribution_pressure_10": upper_shadow_distribution_pressure_10(
            inputs["opens_1d"],
            inputs["highs_1d"],
            inputs["lows_1d"],
            prices,
            turnover,
        ),
        "upper_half_churn_trap_20": upper_half_churn_trap_20(
            inputs["opens_1d"],
            inputs["highs_1d"],
            inputs["lows_1d"],
            prices,
            turnover,
        ),
        "compression_release_quality_15": compression_release_quality_15(
            inputs["highs_1d"],
            inputs["lows_1d"],
            prices,
            turnover,
        ),
        "failed_reclaim_persistence_15": failed_reclaim_persistence_15(
            inputs["highs_1d"],
            inputs["lows_1d"],
            prices,
            turnover,
        ),
        "support_test_reclaim_15": support_test_reclaim_15(inputs["highs_1d"], inputs["lows_1d"], prices),
        "false_breakdown_repair_10": false_breakdown_repair_10(inputs["highs_1d"], inputs["lows_1d"], prices),
        "delayed_undercut_reclaim_12": delayed_undercut_reclaim_12(
            inputs["highs_1d"],
            inputs["lows_1d"],
            prices,
            turnover,
        ),
        "failed_breakout_reversal_intensity_12": failed_breakout_reversal_intensity_12(
            inputs["opens_1d"],
            inputs["highs_1d"],
            inputs["lows_1d"],
            prices,
            turnover,
        ),
        "breakout_failure_repair_15": breakout_failure_repair_15(
            inputs["opens_1d"],
            inputs["highs_1d"],
            inputs["lows_1d"],
            prices,
            turnover,
        ),
        "failed_breakout_recoil_repair_15": failed_breakout_recoil_repair_15(
            inputs["highs_1d"],
            inputs["lows_1d"],
            prices,
            turnover,
        ),
        "flat_top_higher_low_coil_20": flat_top_higher_low_coil_20(
            inputs["highs_1d"],
            inputs["lows_1d"],
            prices,
        ),
        "repair_base_tightness_12": repair_base_tightness_12(
            inputs["highs_1d"],
            inputs["lows_1d"],
            prices,
            turnover,
        ),
        "repair_liquidity_retention_10": repair_liquidity_retention_10(
            inputs["highs_1d"],
            inputs["lows_1d"],
            prices,
            turnover,
        ),
        "resistance_memory_decay_20": resistance_memory_decay_20(inputs["highs_1d"], inputs["lows_1d"], prices, turnover),
        "breakout_retention_15": breakout_retention_15(inputs["highs_1d"], prices, turnover),
        "breakout_retest_absorption_20": breakout_retest_absorption_20(inputs["highs_1d"], inputs["lows_1d"], prices, turnover),
        "shrink_volume_nonbreak_5_20": shrink_volume_nonbreak_5_20(prices, turnover),
        "pullback_energy_decay_20": pullback_energy_decay_20(prices, turnover),
        "failed_breakout_scarcity_20": failed_breakout_scarcity_20(
            inputs["opens_1d"],
            inputs["highs_1d"],
            inputs["lows_1d"],
            prices,
            turnover,
        ),
        "advance_participation_uniformity_20": advance_participation_uniformity_20(prices),
    }


def _apply_common_factor_values(
    *,
    factors: dict[str, float],
    prices: list[float],
    turnover: list[float],
    inputs: dict[str, list[float]],
) -> None:
    benchmark_prices = inputs["benchmark_prices"]
    common_factors = compute_common_factor_values(
        FactorInputs(
            opens=inputs["opens_1d"],
            highs=inputs["highs_1d"],
            lows=inputs["lows_1d"],
            closes=prices,
            turnover=turnover,
            benchmark_closes=inputs["benchmark_prices"],
            close_returns=_series_returns(prices),
            close_returns_end_index=max(len(prices) - 1, 0),
            turnover_returns=_series_returns(turnover),
            turnover_returns_end_index=max(len(turnover) - 1, 0),
            benchmark_returns=_series_returns(benchmark_prices),
            benchmark_returns_end_index=max(len(benchmark_prices) - 1, 0),
            end_index=len(prices),
            benchmark_end_index=len(benchmark_prices),
        )
    )
    for name in (
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
    ):
        value = common_factors.get(name)
        if isinstance(value, (int, float)):
            factors[name] = float(value)


def _apply_crowding_policy_factors(*, factors: dict[str, float], inputs: dict[str, list[float]]) -> None:
    flow_crowding_signal = max(float(factors["stock_net_inflow_5_20"]), 0.0)
    turnover_crowding_signal = max(float(factors["abnormal_turnover_z20"]), 0.0)
    sentiment_crowding_signal = max(float(factors["limit_up_continuation_rate"]) - 0.5, 0.0) * 2.0
    factors["crowding_risk_score"] = crowding_risk_score(
        min(flow_crowding_signal, 1.0),
        min(turnover_crowding_signal / 3.0, 1.0),
        min(sentiment_crowding_signal, 1.0),
    )
    policy_series = inputs["policy_event_series"]
    factors["policy_alignment_score_10d"] = policy_alignment_score_10d(policy_series, [factors["sector_rs_20"]] * max(len(policy_series), 1))
    factors["crowding_event_conflict_5d"] = crowding_event_conflict_5d(
        [factors["crowding_risk_score"]] * len(inputs["hotword_series"]),
        inputs["hotword_series"],
    )


def _apply_factor_aliases(factors: dict[str, float]) -> None:
    for target_name, aliases in _LIVE_FACTOR_ALIASES.items():
        if target_name not in factors:
            continue
        for alias in aliases:
            factors.setdefault(alias, float(factors[target_name]))


def _seed_factor_overrides(
    factors: dict[str, float],
    overrides: dict[str, float] | None,
    *,
    only_missing: bool,
) -> None:
    if not overrides:
        return
    for key, value in overrides.items():
        canonical = canonical_factor_name(key)
        if not canonical or not isinstance(value, (int, float)):
            continue
        if only_missing and canonical in factors:
            continue
        factors[canonical] = float(value)
        if canonical in _LIVE_FACTOR_ALIASES:
            for alias in _LIVE_FACTOR_ALIASES[canonical]:
                if only_missing and alias in factors:
                    continue
                factors[alias] = float(value)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _apply_public_institutional_composites(factors: dict[str, float]) -> None:
    growth_quality = float(factors.get("growth_quality_underreaction_20", 0.0) or 0.0)
    cash_conversion = float(factors.get("cash_conversion_balance_sheet_momentum_20", 0.0) or 0.0)
    quality_balance_sheet_repair = float(factors.get("quality_balance_sheet_repair_20", 0.0) or 0.0)
    accrual_cashflow_quality = float(factors.get("accrual_cashflow_quality_20", 0.0) or 0.0)
    post_report_quality_drift = float(factors.get("post_report_quality_drift_20", 0.0) or 0.0)
    gross_profitability_assets = float(factors.get("gross_profitability_assets_20", 0.0) or 0.0)
    gross_margin_level = float(factors.get("gross_margin_level_20", 0.0) or 0.0)
    gross_margin_improvement = float(factors.get("gross_margin_improvement_20", 0.0) or 0.0)
    conservative_investment_discipline = float(factors.get("conservative_investment_discipline_20", 0.0) or 0.0)
    financing_discipline_safety = float(factors.get("financing_discipline_safety_20", 0.0) or 0.0)
    operating_profitability_assets = float(factors.get("operating_profitability_assets_20", 0.0) or 0.0)
    operating_profitability_improvement = float(factors.get("operating_profitability_improvement_20", 0.0) or 0.0)
    cash_profitability_assets = float(factors.get("cash_profitability_assets_20", 0.0) or 0.0)
    cash_profitability_improvement = float(factors.get("cash_profitability_improvement_20", 0.0) or 0.0)
    return_on_assets_level = float(factors.get("return_on_assets_level_20", 0.0) or 0.0)
    return_on_assets_improvement = float(factors.get("return_on_assets_improvement_20", 0.0) or 0.0)
    asset_turnover_level = float(factors.get("asset_turnover_level_20", 0.0) or 0.0)
    operating_cashflow_to_liabilities = float(factors.get("operating_cashflow_to_liabilities_20", 0.0) or 0.0)
    asset_turnover_improvement = float(factors.get("asset_turnover_improvement_20", 0.0) or 0.0)
    net_external_financing_burden = float(factors.get("net_external_financing_burden_20", 0.0) or 0.0)
    liquidity_safety_buffer = float(factors.get("liquidity_safety_buffer_20", 0.0) or 0.0)
    working_capital_to_assets = float(factors.get("working_capital_to_assets_20", 0.0) or 0.0)
    retained_earnings_to_assets = float(factors.get("retained_earnings_to_assets_20", 0.0) or 0.0)
    operating_accruals_assets = float(factors.get("operating_accruals_assets_20", 0.0) or 0.0)
    working_capital_accruals = float(factors.get("working_capital_accruals_20", 0.0) or 0.0)
    net_operating_assets_ratio = float(factors.get("net_operating_assets_ratio_20", 0.0) or 0.0)
    working_capital_discipline = float(factors.get("working_capital_discipline_20", 0.0) or 0.0)
    beneish_anti_manipulation = float(factors.get("beneish_anti_manipulation_20", 0.0) or 0.0)
    piotroski_f_score_proxy = float(factors.get("piotroski_f_score_proxy_20", 0.0) or 0.0)
    quality_strength_persistence = float(factors.get("quality_strength_persistence_20", 0.0) or 0.0)
    residual_sponsorship_resilience = (
        0.24 * _clamp(float(factors.get("stock_flow_residual_vs_sector_10", 0.0) or 0.0), -1.5, 1.5)
        + 0.18 * _clamp(float(factors.get("stock_flow_residual_acceleration_5_20", 0.0) or 0.0), -1.5, 1.5)
        + 0.18 * _clamp(float(factors.get("stock_defensive_residual_on_sector_selloff_15", 0.0) or 0.0), -1.5, 1.5)
        + 0.14 * _clamp(float(factors.get("relative_drawdown_resilience_15", 0.0) or 0.0), -1.5, 1.5)
        + 0.12 * _clamp(float(factors.get("stock_sector_breadth_capture_10", 0.0) or 0.0), -1.5, 1.5)
        + 0.14 * _clamp(float(factors.get("stock_relative_sponsorship_under_sector_crowding_10", 0.0) or 0.0), -1.5, 1.5)
    )
    panic_repair_sponsorship = (
        0.18 * _clamp(float(factors.get("false_breakdown_repair_10", 0.0) or 0.0), 0.0, 1.2)
        + 0.16 * _clamp(float(factors.get("lower_shadow_absorption_5", 0.0) or 0.0), 0.0, 1.2)
        + 0.14 * _clamp(float(factors.get("panic_followthrough_repair_8", 0.0) or 0.0), 0.0, 1.2)
        + 0.14 * _clamp(float(factors.get("panic_followthrough_decay_12", 0.0) or 0.0), 0.0, 1.2)
        + 0.14 * _clamp(float(factors.get("outflow_retest_absorption_15", 0.0) or 0.0), 0.0, 1.2)
        + 0.12 * _clamp(float(factors.get("repair_liquidity_retention_10", 0.0) or 0.0), 0.0, 1.2)
        + 0.12 * _clamp(float(factors.get("turnover_adjusted_outflow_damage_10", 0.0) or 0.0), 0.0, 1.2)
    )
    quality_sponsorship_underreaction = (
        0.38 * _clamp(max(quality_balance_sheet_repair, (growth_quality + cash_conversion) / 2.0), -1.5, 1.5)
        + 0.30 * _clamp(residual_sponsorship_resilience, -1.5, 1.5)
        + 0.20 * _clamp(panic_repair_sponsorship, 0.0, 1.2)
        + 0.12 * _clamp(float(factors.get("sector_negative_flow_breadth_resilience_10", 0.0) or 0.0), -1.5, 1.5)
    )
    qmj_defensive_sponsorship = (
        0.12 * _clamp(piotroski_f_score_proxy, -1.5, 1.5)
        + 0.26 * _clamp(quality_balance_sheet_repair, -1.5, 1.5)
        + 0.22 * _clamp(accrual_cashflow_quality, -1.5, 1.5)
        + 0.12 * _clamp(working_capital_discipline, -1.5, 1.5)
        + 0.08 * _clamp(operating_accruals_assets, -1.5, 1.5)
        + 0.10 * _clamp(beneish_anti_manipulation, -1.5, 1.5)
        + 0.18 * _clamp(post_report_quality_drift, -1.5, 1.5)
        + 0.18 * _clamp(residual_sponsorship_resilience, -1.5, 1.5)
        + 0.10 * _clamp(liquidity_safety_buffer, -1.5, 1.5)
        + 0.04 * _clamp(float(factors.get("negative_flow_impact_beta_10", 0.0) or 0.0), -1.5, 1.5)
        + 0.02 * _clamp(float(factors.get("turnover_stability_20", 0.0) or 0.0), -1.5, 1.5)
    )
    qmj_profitability_safety = (
        0.18 * _clamp(gross_profitability_assets, -1.5, 1.5)
        + 0.10 * _clamp(gross_margin_level, -1.5, 1.5)
        + 0.10 * _clamp(gross_margin_improvement, -1.5, 1.5)
        + 0.16 * _clamp(operating_profitability_assets, -1.5, 1.5)
        + 0.10 * _clamp(operating_profitability_improvement, -1.5, 1.5)
        + 0.14 * _clamp(cash_profitability_assets, -1.5, 1.5)
        + 0.10 * _clamp(cash_profitability_improvement, -1.5, 1.5)
        + 0.10 * _clamp(return_on_assets_level, -1.5, 1.5)
        + 0.10 * _clamp(return_on_assets_improvement, -1.5, 1.5)
        + 0.14 * _clamp(conservative_investment_discipline, -1.5, 1.5)
        + 0.10 * _clamp(asset_turnover_level, -1.5, 1.5)
        + 0.12 * _clamp(asset_turnover_improvement, -1.5, 1.5)
        + 0.10 * _clamp(financing_discipline_safety, -1.5, 1.5)
        + 0.08 * _clamp(operating_cashflow_to_liabilities, -1.5, 1.5)
        + 0.08 * _clamp(working_capital_to_assets, -1.5, 1.5)
        + 0.08 * _clamp(retained_earnings_to_assets, -1.5, 1.5)
        + 0.10 * _clamp(working_capital_discipline, -1.5, 1.5)
        + 0.08 * _clamp(net_operating_assets_ratio, -1.5, 1.5)
        + 0.10 * _clamp(beneish_anti_manipulation, -1.5, 1.5)
        + 0.08 * _clamp(net_external_financing_burden, -1.5, 1.5)
        + 0.08 * _clamp(liquidity_safety_buffer, -1.5, 1.5)
        + 0.08 * _clamp(accrual_cashflow_quality, -1.5, 1.5)
        + 0.12 * _clamp(qmj_defensive_sponsorship, -1.5, 1.5)
        + 0.12 * _clamp(quality_strength_persistence, -1.5, 1.5)
        + 0.06 * _clamp(residual_sponsorship_resilience, -1.5, 1.5)
        + 0.04 * _clamp(post_report_quality_drift, -1.5, 1.5)
    )
    distress_resilience_public = (
        0.24 * _clamp(liquidity_safety_buffer, -1.5, 1.5)
        + 0.18 * _clamp(net_external_financing_burden, -1.5, 1.5)
        + 0.18 * _clamp(operating_cashflow_to_liabilities, -1.5, 1.5)
        + 0.14 * _clamp(financing_discipline_safety, -1.5, 1.5)
        + 0.12 * _clamp(float(factors.get("net_cash_to_assets_20", 0.0) or 0.0), -1.5, 1.5)
        + 0.08 * _clamp(float(factors.get("current_ratio_improvement_20", 0.0) or 0.0), -1.5, 1.5)
        + 0.06 * _clamp(float(factors.get("cash_coverage_relief_yoy", 0.0) or 0.0), -1.0, 1.0)
    )
    franchise_power_quality = (
        0.18 * _clamp(gross_profitability_assets, -1.5, 1.5)
        + 0.16 * _clamp(operating_profitability_assets, -1.5, 1.5)
        + 0.14 * _clamp(cash_profitability_assets, -1.5, 1.5)
        + 0.12 * _clamp(return_on_assets_improvement, -1.5, 1.5)
        + 0.10 * _clamp(gross_margin_improvement, -1.5, 1.5)
        + 0.10 * _clamp(asset_turnover_improvement, -1.5, 1.5)
        + 0.10 * _clamp(retained_earnings_to_assets, -1.5, 1.5)
        + 0.10 * _clamp(quality_strength_persistence, -1.5, 1.5)
    )
    institutional_quality_compound = (
        0.22 * _clamp(franchise_power_quality, -1.5, 1.5)
        + 0.20 * _clamp(distress_resilience_public, -1.5, 1.5)
        + 0.18 * _clamp(quality_strength_persistence, -1.5, 1.5)
        + 0.16 * _clamp(qmj_profitability_safety, -1.5, 1.5)
        + 0.12 * _clamp(qmj_defensive_sponsorship, -1.5, 1.5)
        + 0.06 * _clamp(beneish_anti_manipulation, -1.5, 1.5)
        + 0.06 * _clamp(post_report_quality_drift, -1.5, 1.5)
    )
    quality_residual_sponsorship_spread = (
        0.42 * _clamp(institutional_quality_compound, -1.5, 1.5)
        + 0.32 * _clamp(residual_sponsorship_resilience, -1.5, 1.5)
        + 0.18 * _clamp(float(factors.get("alpha_20_beta_60", 0.0) or 0.0), -1.5, 1.5)
        + 0.08 * _clamp(float(factors.get("stock_sector_breadth_capture_10", 0.0) or 0.0), -1.5, 1.5)
    )
    crowding_adjusted_sponsorship_escape = (
        0.38 * _clamp(float(factors.get("stock_relative_sponsorship_under_sector_crowding_10", 0.0) or 0.0), -1.5, 1.5)
        + 0.28 * _clamp(float(factors.get("stock_sector_breadth_capture_10", 0.0) or 0.0), -1.5, 1.5)
        + 0.20 * _clamp(float(factors.get("stock_flow_residual_acceleration_5_20", 0.0) or 0.0), -1.5, 1.5)
        + 0.14 * _clamp(float(factors.get("alpha_20_beta_60", 0.0) or 0.0), -1.5, 1.5)
        - 0.24 * _clamp(float(factors.get("sector_flow_breadth_divergence_crowding_15", 0.0) or 0.0), 0.0, 1.5)
    )
    residual_trend_sponsorship = (
        0.26 * _clamp(float(factors.get("residual_momentum_20_60", 0.0) or 0.0), -1.5, 1.5)
        + 0.22 * _clamp(float(factors.get("alpha_20_beta_60", 0.0) or 0.0), -1.5, 1.5)
        + 0.18 * _clamp(float(factors.get("crowding_adjusted_sponsorship_escape_10", 0.0) or 0.0), -1.5, 1.5)
        + 0.16 * _clamp(float(factors.get("stock_flow_residual_acceleration_5_20", 0.0) or 0.0), -1.5, 1.5)
        + 0.10 * _clamp(float(factors.get("trend_quality_20", 0.0) or 0.0), -1.5, 1.5)
        + 0.08 * _clamp(float(factors.get("turnover_stability_20", 0.0) or 0.0), -1.5, 1.5)
    )
    compression_breakout_sponsorship = (
        0.20 * _clamp(float(factors.get("compression_persistence_20", 0.0) or 0.0), 0.0, 1.5)
        + 0.18 * _clamp(float(factors.get("entropy_compression_breakout_potential_20", 0.0) or 0.0), 0.0, 1.5)
        + 0.16 * _clamp(float(factors.get("steady_trend_advancement_20", 0.0) or 0.0), 0.0, 1.5)
        + 0.14 * _clamp(float(factors.get("breakout_retest_absorption_20", 0.0) or 0.0), 0.0, 1.5)
        + 0.12 * _clamp(float(factors.get("fractal_trend_persistence_20", 0.0) or 0.0), 0.0, 1.5)
        + 0.10 * _clamp(float(factors.get("spectral_trend_coherence_20", 0.0) or 0.0), 0.0, 1.5)
        + 0.10 * _clamp(float(factors.get("stock_sector_breadth_capture_10", 0.0) or 0.0), -1.5, 1.5)
    )
    supply_absorption_repair = (
        0.18 * _clamp(float(factors.get("panic_repair_sponsorship_15", 0.0) or 0.0), 0.0, 1.5)
        + 0.16 * _clamp(float(factors.get("outflow_retest_absorption_15", 0.0) or 0.0), 0.0, 1.5)
        + 0.14 * _clamp(float(factors.get("serial_outflow_damage_dampening_12", 0.0) or 0.0), -1.5, 1.5)
        + 0.14 * _clamp(float(factors.get("negative_flow_impact_beta_10", 0.0) or 0.0), -1.5, 1.5)
        + 0.12 * _clamp(float(factors.get("relative_drawdown_resilience_15", 0.0) or 0.0), -1.5, 1.5)
        + 0.10 * _clamp(float(factors.get("liquidity_impact_relief_10", 0.0) or 0.0), -1.5, 1.5)
        - 0.10 * _clamp(float(factors.get("upper_shadow_distribution_pressure_10", 0.0) or 0.0), 0.0, 1.5)
        - 0.06 * _clamp(float(factors.get("failed_reclaim_persistence_15", 0.0) or 0.0), 0.0, 1.5)
    )
    institutional_alpha_core = (
        0.30 * _clamp(residual_trend_sponsorship, -1.5, 1.5)
        + 0.24 * _clamp(compression_breakout_sponsorship, 0.0, 1.5)
        + 0.20 * _clamp(quality_residual_sponsorship_spread, -1.5, 1.5)
        + 0.16 * _clamp(supply_absorption_repair, -1.5, 1.5)
        + 0.10 * _clamp(float(factors.get("state_space_trend_snr_20", 0.0) or 0.0), 0.0, 1.5)
    )
    tail_adjusted_residual_alpha = (
        0.30 * _clamp(residual_trend_sponsorship, -1.5, 1.5)
        + 0.22 * _clamp(quality_residual_sponsorship_spread, -1.5, 1.5)
        + 0.16 * _clamp(float(factors.get("alpha_20_beta_60", 0.0) or 0.0), -1.5, 1.5)
        + 0.14 * _clamp(float(factors.get("state_space_trend_snr_20", 0.0) or 0.0), 0.0, 1.5)
        + 0.10 * _clamp(float(factors.get("innovation_drift_absorption_20", 0.0) or 0.0), 0.0, 1.5)
        + 0.08 * _clamp(float(factors.get("spectral_trend_coherence_20", 0.0) or 0.0), 0.0, 1.5)
        - 0.22 * _clamp(float(factors.get("downside_volatility_20", 0.0) or 0.0) * 18.0, 0.0, 1.5)
        - 0.16 * _clamp(float(factors.get("crowding_risk_score", 0.0) or 0.0), 0.0, 1.5)
        - 0.14 * _clamp(float(factors.get("exhaustion_risk_10", 0.0) or 0.0), 0.0, 1.5)
        - 0.10 * _clamp(float(factors.get("beta_60", 0.0) or 0.0), 0.0, 1.5)
        - 0.08 * _clamp(float(factors.get("upper_shadow_distribution_pressure_10", 0.0) or 0.0), 0.0, 1.5)
    )
    weak_market_resilience = (
        0.22 * _clamp(quality_residual_sponsorship_spread, -1.5, 1.5)
        + 0.18 * _clamp(float(factors.get("relative_drawdown_resilience_15", 0.0) or 0.0), -1.5, 1.5)
        + 0.16 * _clamp(float(factors.get("negative_flow_impact_beta_10", 0.0) or 0.0), -1.5, 1.5)
        + 0.14 * _clamp(float(factors.get("selloff_absorption_flow_10", 0.0) or 0.0), -1.5, 1.5)
        + 0.12 * _clamp(float(factors.get("turnover_stability_20", 0.0) or 0.0), -1.5, 1.5)
        + 0.10 * _clamp(float(factors.get("price_position_20", 0.0) or 0.0), -1.5, 1.5)
        + 0.08 * _clamp(float(factors.get("stock_defensive_residual_on_sector_selloff_15", 0.0) or 0.0), -1.5, 1.5)
        - 0.16 * _clamp(float(factors.get("upper_shadow_distribution_pressure_10", 0.0) or 0.0), 0.0, 1.5)
        - 0.12 * _clamp(float(factors.get("upper_half_churn_trap_20", 0.0) or 0.0), 0.0, 1.5)
        - 0.08 * _clamp(float(factors.get("crowding_risk_score", 0.0) or 0.0), 0.0, 1.5)
    )
    weak_market_reaccumulation_alpha = (
        0.18 * _clamp(quality_residual_sponsorship_spread, -1.5, 1.5)
        + 0.16 * _clamp(residual_sponsorship_resilience, -1.5, 1.5)
        + 0.14 * _clamp(float(factors.get("sector_negative_flow_breadth_resilience_10", 0.0) or 0.0), -1.5, 1.5)
        + 0.12 * _clamp(float(factors.get("stock_sector_breadth_lead_10", 0.0) or 0.0), -1.5, 1.5)
        + 0.10 * _clamp(float(factors.get("rotation_early_diffusion_10_20", 0.0) or 0.0), -1.5, 1.5)
        + 0.10 * _clamp(float(factors.get("outflow_reversal_efficiency_10", 0.0) or 0.0), 0.0, 1.5)
        + 0.08 * _clamp(float(factors.get("selloff_absorption_flow_10", 0.0) or 0.0), -1.5, 1.5)
        + 0.08 * _clamp(float(factors.get("support_test_reclaim_15", 0.0) or 0.0), 0.0, 1.5)
        + 0.06 * _clamp(float(factors.get("false_breakdown_repair_10", 0.0) or 0.0), 0.0, 1.5)
        + 0.06 * _clamp(float(factors.get("negative_flow_impact_beta_10", 0.0) or 0.0), -1.5, 1.5)
        - 0.18 * _clamp(float(factors.get("downside_volatility_20", 0.0) or 0.0) * 18.0, 0.0, 1.5)
        - 0.14 * _clamp(float(factors.get("upper_half_churn_trap_20", 0.0) or 0.0), 0.0, 1.5)
        - 0.12 * _clamp(float(factors.get("failed_reclaim_persistence_15", 0.0) or 0.0), 0.0, 1.5)
        - 0.10 * _clamp(float(factors.get("beta_60", 0.0) or 0.0), 0.0, 1.5)
        - 0.08 * _clamp(float(factors.get("upper_shadow_distribution_pressure_10", 0.0) or 0.0), 0.0, 1.5)
    )
    orthogonalized_residual_alpha = (
        0.26 * _clamp(quality_residual_sponsorship_spread, -1.5, 1.5)
        + 0.22 * _clamp(residual_trend_sponsorship, -1.5, 1.5)
        + 0.18 * _clamp(weak_market_reaccumulation_alpha, -1.5, 1.5)
        + 0.14 * _clamp(float(factors.get("stock_flow_residual_acceleration_5_20", 0.0) or 0.0), -1.5, 1.5)
        + 0.12 * _clamp(float(factors.get("alpha_20_beta_60", 0.0) or 0.0), -1.5, 1.5)
        + 0.08 * _clamp(float(factors.get("stock_sector_breadth_lead_10", 0.0) or 0.0), -1.5, 1.5)
        - 0.18 * _clamp(float(factors.get("price_momentum_20", 0.0) or 0.0), 0.0, 1.5)
        - 0.14 * _clamp(float(factors.get("price_momentum_60", 0.0) or 0.0), 0.0, 1.5)
        - 0.14 * _clamp(float(factors.get("beta_60", 0.0) or 0.0), 0.0, 1.5)
        - 0.12 * _clamp(float(factors.get("crowding_risk_score", 0.0) or 0.0), 0.0, 1.5)
        - 0.10 * _clamp(float(factors.get("downside_volatility_20", 0.0) or 0.0) * 18.0, 0.0, 1.5)
    )
    flow_absorption_half_life = (
        0.22 * _clamp(float(factors.get("serial_outflow_damage_dampening_12", 0.0) or 0.0), -1.5, 1.5)
        + 0.18 * _clamp(float(factors.get("outflow_reversal_efficiency_10", 0.0) or 0.0), 0.0, 1.5)
        + 0.16 * _clamp(float(factors.get("outflow_retest_absorption_15", 0.0) or 0.0), 0.0, 1.5)
        + 0.14 * _clamp(float(factors.get("negative_flow_impact_beta_10", 0.0) or 0.0), -1.5, 1.5)
        + 0.12 * _clamp(float(factors.get("liquidity_impact_relief_10", 0.0) or 0.0), -1.5, 1.5)
        + 0.10 * _clamp(float(factors.get("selloff_absorption_flow_10", 0.0) or 0.0), -1.5, 1.5)
        + 0.08 * _clamp(float(factors.get("support_test_reclaim_15", 0.0) or 0.0), 0.0, 1.5)
        - 0.18 * _clamp(float(factors.get("failed_reclaim_persistence_15", 0.0) or 0.0), 0.0, 1.5)
        - 0.14 * _clamp(float(factors.get("upper_shadow_distribution_pressure_10", 0.0) or 0.0), 0.0, 1.5)
        - 0.12 * _clamp(float(factors.get("upper_half_churn_trap_20", 0.0) or 0.0), 0.0, 1.5)
        - 0.10 * _clamp(float(factors.get("downside_volatility_20", 0.0) or 0.0) * 18.0, 0.0, 1.5)
    )
    institutional_orthogonal_reaccumulation = (
        0.34 * _clamp(orthogonalized_residual_alpha, -1.5, 1.5)
        + 0.28 * _clamp(flow_absorption_half_life, -1.5, 1.5)
        + 0.18 * _clamp(supply_absorption_repair, -1.5, 1.5)
        + 0.12 * _clamp(float(factors.get("state_space_trend_snr_20", 0.0) or 0.0), 0.0, 1.5)
        + 0.08 * _clamp(float(factors.get("rotation_early_diffusion_10_20", 0.0) or 0.0), -1.5, 1.5)
        - 0.16 * _clamp(float(factors.get("failed_breakout_reversal_intensity_12", 0.0) or 0.0), 0.0, 1.5)
        - 0.12 * _clamp(float(factors.get("exhaustion_risk_10", 0.0) or 0.0), 0.0, 1.5)
    )
    oscillation_repair_quality = (
        0.18 * _clamp(float(factors.get("low_volume_box_support_20", 0.0) or 0.0), 0.0, 1.5)
        + 0.16 * _clamp(float(factors.get("false_breakdown_repair_10", 0.0) or 0.0), 0.0, 1.5)
        + 0.14 * _clamp(float(factors.get("outflow_retest_absorption_15", 0.0) or 0.0), 0.0, 1.5)
        + 0.14 * _clamp(float(factors.get("support_test_reclaim_15", 0.0) or 0.0), 0.0, 1.5)
        + 0.12 * _clamp(float(factors.get("shrink_volume_nonbreak_5_20", 0.0) or 0.0), 0.0, 1.5)
        + 0.10 * _clamp(float(factors.get("pullback_energy_decay_20", 0.0) or 0.0), -1.5, 1.5)
        + 0.10 * _clamp(float(factors.get("price_position_20", 0.0) or 0.0), -1.5, 1.5)
        + 0.08 * _clamp(float(factors.get("turnover_stability_20", 0.0) or 0.0), -1.5, 1.5)
        - 0.10 * _clamp(float(factors.get("failed_reclaim_persistence_15", 0.0) or 0.0), 0.0, 1.5)
        - 0.08 * _clamp(float(factors.get("failed_breakout_reversal_intensity_12", 0.0) or 0.0), 0.0, 1.5)
    )
    defensive_alpha_core = (
        0.30 * _clamp(weak_market_resilience, -1.5, 1.5)
        + 0.24 * _clamp(oscillation_repair_quality, -1.5, 1.5)
        + 0.18 * _clamp(supply_absorption_repair, -1.5, 1.5)
        + 0.16 * _clamp(quality_residual_sponsorship_spread, -1.5, 1.5)
        + 0.12 * _clamp(distress_resilience_public, -1.5, 1.5)
    )
    calm_residual_alpha = (
        0.30 * _clamp(quality_residual_sponsorship_spread, -1.5, 1.5)
        + 0.22 * _clamp(float(factors.get("turnover_stability_20", 0.0) or 0.0), -1.5, 1.5)
        + 0.16 * _clamp(float(factors.get("liquidity_sponsorship_20", 0.0) or 0.0), -1.5, 1.5)
        + 0.14 * _clamp(flow_absorption_half_life, -1.5, 1.5)
        + 0.10 * _clamp(float(factors.get("price_position_20", 0.0) or 0.0), -1.5, 1.5)
        + 0.08 * _clamp(float(factors.get("stock_defensive_residual_on_sector_selloff_15", 0.0) or 0.0), -1.5, 1.5)
        - 0.18 * _clamp(float(factors.get("realized_volatility_20", 0.0) or 0.0) * 18.0, 0.0, 1.5)
        - 0.16 * _clamp(float(factors.get("downside_volatility_20", 0.0) or 0.0) * 18.0, 0.0, 1.5)
        - 0.10 * _clamp(float(factors.get("upper_shadow_distribution_pressure_10", 0.0) or 0.0), 0.0, 1.5)
    )
    factors["residual_sponsorship_resilience_10"] = residual_sponsorship_resilience
    factors["panic_repair_sponsorship_15"] = panic_repair_sponsorship
    factors["quality_sponsorship_underreaction_20"] = quality_sponsorship_underreaction
    factors["qmj_defensive_sponsorship_20"] = qmj_defensive_sponsorship
    factors["qmj_profitability_safety_20"] = qmj_profitability_safety
    factors["piotroski_f_score_proxy_20"] = piotroski_f_score_proxy
    factors["quality_strength_persistence_20"] = quality_strength_persistence
    factors["distress_resilience_public_20"] = distress_resilience_public
    factors["franchise_power_quality_20"] = franchise_power_quality
    factors["institutional_quality_compound_20"] = institutional_quality_compound
    factors["quality_residual_sponsorship_spread_20"] = quality_residual_sponsorship_spread
    factors["crowding_adjusted_sponsorship_escape_10"] = crowding_adjusted_sponsorship_escape
    factors["residual_trend_sponsorship_20"] = residual_trend_sponsorship
    factors["compression_breakout_sponsorship_20"] = compression_breakout_sponsorship
    factors["supply_absorption_repair_20"] = supply_absorption_repair
    factors["institutional_alpha_core_20"] = institutional_alpha_core
    factors["tail_adjusted_residual_alpha_20"] = tail_adjusted_residual_alpha
    factors["weak_market_resilience_20"] = weak_market_resilience
    factors["weak_market_reaccumulation_alpha_20"] = weak_market_reaccumulation_alpha
    factors["orthogonalized_residual_alpha_20"] = orthogonalized_residual_alpha
    factors["flow_absorption_half_life_15"] = flow_absorption_half_life
    factors["institutional_orthogonal_reaccumulation_20"] = institutional_orthogonal_reaccumulation
    factors["oscillation_repair_quality_20"] = oscillation_repair_quality
    factors["defensive_alpha_core_20"] = defensive_alpha_core
    factors["calm_residual_alpha_20"] = calm_residual_alpha


def build_main_wave_factor_snapshot(
    prices: list[float],
    turnover: list[float],
    *,
    factor_overrides: dict[str, float] | None = None,
    factor_overrides_weight: float = 1.0,
    **kwargs: object,
) -> dict[str, float]:
    inputs = _main_wave_signal_inputs(prices=prices, turnover=turnover, kwargs=kwargs)
    factors = _compute_factors(
        prices,
        turnover,
        inputs,
        enable_numba_kernels=bool(kwargs.get("enable_numba_kernels", True)),
    )
    if bool(kwargs.get("official_clean_mode", False)):
        _neutralize_clean_mode_factors(factors)
    _seed_factor_overrides(
        factors,
        factor_overrides,
        only_missing=max(0.0, min(float(factor_overrides_weight), 1.0)) < 1.0,
    )
    _apply_factor_overrides(
        factors=factors,
        overrides=factor_overrides,
        weight=max(0.0, min(float(factor_overrides_weight), 1.0)),
    )
    if bool(kwargs.get("daily_only_mode", False)):
        _neutralize_daily_only_mode_factors(factors)
    _apply_public_institutional_composites(factors)
    _apply_factor_aliases(factors)
    return factors


def evaluate_main_wave_signal(prices: list[float], turnover: list[float], profile: MainWaveProfile, **kwargs: object) -> MainWaveSignal:
    options = _main_wave_signal_options(kwargs)
    inputs = _main_wave_signal_inputs(prices=prices, turnover=turnover, kwargs=kwargs)
    factors = _compute_factors(
        prices,
        turnover,
        inputs,
        enable_numba_kernels=bool(kwargs.get("enable_numba_kernels", True)),
    )
    if bool(options["official_clean_mode"]):
        _neutralize_clean_mode_factors(factors)
    raw_overrides = kwargs.get("factor_overrides")
    _seed_factor_overrides(
        factors,
        raw_overrides if isinstance(raw_overrides, dict) else None,
        only_missing=float(options["factor_overrides_weight"]) < 1.0,
    )
    _apply_factor_overrides(
        factors=factors,
        overrides=raw_overrides,
        weight=float(options["factor_overrides_weight"]),
    )
    if bool(options["daily_only_mode"]):
        _neutralize_daily_only_mode_factors(factors)
    _apply_public_institutional_composites(factors)
    decision = calc_regime_decision_adjustments(
        factors,
        str(options["regime"]),
        overrides=options["regime_factor_policy_overrides"],
        decision_overrides=options["regime_decision_policy_overrides"],
        overlay_scale=float(options["regime_factor_overlay_scale"]),
    )
    final_enter, guarded, event_reweight, reasons = _main_wave_signal_decision(
        factors,
        profile,
        options=options,
        decision=decision,
    )
    if _intraday_entry_blocked(options=options, kwargs=kwargs):
        final_enter = False
        if bool(options["include_reasons"]):
            reasons.append("intraday_context_unavailable")
    return _main_wave_signal_payload(
        enter=final_enter,
        factors=factors,
        profile=profile,
        reasons=reasons,
        event_reweight=event_reweight,
        decision=decision,
        options=options,
    )


def evaluate_main_wave_signal_from_factors(
    factor_values: dict[str, float] | None,
    profile: MainWaveProfile,
    **kwargs: object,
) -> MainWaveSignal:
    options = _main_wave_signal_options(kwargs)
    factors = {
        str(name): float(value)
        for name, value in dict(factor_values or {}).items()
        if isinstance(value, (int, float))
    }
    if bool(options["official_clean_mode"]):
        _neutralize_clean_mode_factors(factors)
    raw_overrides = kwargs.get("factor_overrides")
    _seed_factor_overrides(
        factors,
        raw_overrides if isinstance(raw_overrides, dict) else None,
        only_missing=float(options["factor_overrides_weight"]) < 1.0,
    )
    _apply_factor_overrides(
        factors=factors,
        overrides=raw_overrides,
        weight=float(options["factor_overrides_weight"]),
    )
    if bool(options["daily_only_mode"]):
        _neutralize_daily_only_mode_factors(factors)
    _apply_public_institutional_composites(factors)
    _apply_factor_aliases(factors)
    decision = calc_regime_decision_adjustments(
        factors,
        str(options["regime"]),
        overrides=options["regime_factor_policy_overrides"],
        decision_overrides=options["regime_decision_policy_overrides"],
        overlay_scale=float(options["regime_factor_overlay_scale"]),
    )
    final_enter, guarded, event_reweight, reasons = _main_wave_signal_decision(
        factors,
        profile,
        options=options,
        decision=decision,
    )
    return _main_wave_signal_payload(
        enter=final_enter,
        factors=factors,
        profile=profile,
        reasons=reasons,
        event_reweight=event_reweight,
        decision=decision,
        options=options,
    )


def _main_wave_signal_options(kwargs: dict[str, object]) -> dict[str, object]:
    return {
        "factor_threshold_bias": float(kwargs.get("factor_threshold_bias", 0.0)),
        "allow_flow_proxy_fallback": bool(kwargs.get("allow_flow_proxy_fallback", True)),
        "daily_only_mode": bool(kwargs.get("daily_only_mode", False)),
        "include_details": bool(kwargs.get("include_details", True)),
        "include_reasons": bool(kwargs.get("include_reasons", kwargs.get("include_details", True))),
        "official_clean_mode": bool(kwargs.get("official_clean_mode", False)),
        "regime": str(kwargs.get("regime", "") or "").strip().lower(),
        "regime_factor_policy_overrides": (
            kwargs.get("regime_factor_policy_overrides") if isinstance(kwargs.get("regime_factor_policy_overrides"), dict) else None
        ),
        "regime_decision_policy_overrides": (
            kwargs.get("regime_decision_policy_overrides") if isinstance(kwargs.get("regime_decision_policy_overrides"), dict) else None
        ),
        "regime_factor_overlay_scale": float(kwargs.get("regime_factor_overlay_scale", 1.0)),
        "factor_overrides_weight": max(0.0, min(float(kwargs.get("factor_overrides_weight", 1.0)), 1.0)),
    }


def _main_wave_signal_inputs(
    *,
    prices: list[float],
    turnover: list[float],
    kwargs: dict[str, object],
) -> dict[str, list[float]]:
    daily_only_mode = bool(kwargs.get("daily_only_mode", False))
    return normalize_inputs(
        prices,
        turnover,
        kwargs.get("benchmark_prices"), kwargs.get("opens_1d"), kwargs.get("highs_1d"), kwargs.get("lows_1d"), None if daily_only_mode else kwargs.get("closes_30m"), None if daily_only_mode else kwargs.get("opens_30m"),
        None if daily_only_mode else kwargs.get("highs_30m"), None if daily_only_mode else kwargs.get("lows_30m"), None if daily_only_mode else kwargs.get("turnover_30m"), None if daily_only_mode else kwargs.get("vwap_30m"),
        kwargs.get("stock_flows"), kwargs.get("sector_flows"), kwargs.get("sector_closes"), kwargs.get("market_closes"),
        kwargs.get("sector_up_counts"), kwargs.get("sector_total_counts"), kwargs.get("leader_prices"), kwargs.get("laggard_prices"),
        kwargs.get("sentiment_series"), kwargs.get("board_break_series"), kwargs.get("gap_series"), kwargs.get("reversal_series"),
        kwargs.get("event_direction_series"), kwargs.get("event_intensity_series"), kwargs.get("event_confidence_series"),
        kwargs.get("event_age_minutes"), kwargs.get("event_impact_series"), kwargs.get("event_sector_hits"), kwargs.get("event_symbol_hits"),
        kwargs.get("regulatory_event_series"), kwargs.get("shareholder_event_series"), kwargs.get("policy_event_series"), kwargs.get("hotword_series"),
        official_clean_mode=bool(kwargs.get("official_clean_mode", False)),
        allow_flow_proxy_fallback=bool(kwargs.get("allow_flow_proxy_fallback", True)),
    )


def _apply_factor_overrides(*, factors: dict[str, float], overrides: object, weight: float) -> None:
    if not isinstance(overrides, dict):
        return
    for key, value in overrides.items():
        if key in factors and isinstance(value, (int, float)):
            base_value = float(factors[key])
            override_value = float(value)
            factors[key] = (1.0 - weight) * base_value + weight * override_value


def _main_wave_signal_decision(
    factors: dict[str, float],
    profile: MainWaveProfile,
    *,
    options: dict[str, object],
    decision: dict[str, float],
) -> tuple[bool, bool, float, list[str]]:
    confirm, reasons = entry_confirmations(
        factors,
        profile,
        factor_threshold_bias=float(options["factor_threshold_bias"]) + float(decision["threshold_bias_shift"]),
        official_clean_mode=bool(options["official_clean_mode"]),
        regime=str(options["regime"]),
        regime_overlay_score=float(decision["effective_overlay_score"]),
        regime_overlay_floor=float(decision["overlay_floor"]),
        regime_support_score=float(decision["support_score"]),
        regime_support_floor=float(decision["support_floor"]),
        collect_reasons=bool(options["include_reasons"]),
        daily_only_mode=bool(options["daily_only_mode"]),
    )
    if not bool(decision.get("entry_allowed", True)):
        confirm = False
        if bool(options["include_reasons"]):
            reasons.append("regime_entry_blocked")
    final_enter, guarded = apply_layer_d_guard(factors, confirm)
    event_reweight = calc_event_reweight(factors)
    _append_main_wave_signal_reasons(reasons=reasons, guarded=guarded, event_reweight=event_reweight, options=options, decision=decision)
    return final_enter, guarded, event_reweight, reasons


def _intraday_entry_blocked(*, options: dict[str, object], kwargs: dict[str, object]) -> bool:
    """Do not let missing minute data become score-layer zeroes in live mode."""
    if bool(options["daily_only_mode"]):
        return False
    quality = intraday_input_quality(
        closes_30m=kwargs.get("closes_30m") if isinstance(kwargs.get("closes_30m"), list) else None,
        opens_30m=kwargs.get("opens_30m") if isinstance(kwargs.get("opens_30m"), list) else None,
        highs_30m=kwargs.get("highs_30m") if isinstance(kwargs.get("highs_30m"), list) else None,
        lows_30m=kwargs.get("lows_30m") if isinstance(kwargs.get("lows_30m"), list) else None,
        turnover_30m=kwargs.get("turnover_30m") if isinstance(kwargs.get("turnover_30m"), list) else None,
        vwap_30m=kwargs.get("vwap_30m") if isinstance(kwargs.get("vwap_30m"), list) else None,
    )
    return quality["intraday_source"] == "unavailable"


def _append_main_wave_signal_reasons(
    *,
    reasons: list[str],
    guarded: bool,
    event_reweight: float,
    options: dict[str, object],
    decision: dict[str, float],
) -> None:
    if not bool(options["include_reasons"]):
        return
    if guarded:
        reasons.append("crowding_conflict_protection")
    regime = str(options["regime"])
    if regime:
        reasons.append(f"regime={regime}")
        reasons.append(f"regime_overlay={float(decision['effective_overlay_score']):.4f}")
        reasons.append(f"regime_support={float(decision['support_score']):.4f}")
    reasons.append(f"event_reweight={event_reweight:.3f}")


def _main_wave_signal_payload(
    *,
    enter: bool,
    factors: dict[str, float],
    profile: MainWaveProfile,
    reasons: list[str],
    event_reweight: float,
    decision: dict[str, float],
    options: dict[str, object],
) -> MainWaveSignal:
    alpha_score = calc_total_score(
        factors,
        official_clean_mode=bool(options["official_clean_mode"]),
        daily_only_mode=bool(options["daily_only_mode"]),
        regime="",
        regime_factor_policy_overrides=options["regime_factor_policy_overrides"],
        regime_decision_policy_overrides=options["regime_decision_policy_overrides"],
        regime_factor_overlay_scale=float(options["regime_factor_overlay_scale"]),
    )
    total_score = (
        calc_total_score(
            factors,
            official_clean_mode=bool(options["official_clean_mode"]),
            daily_only_mode=bool(options["daily_only_mode"]),
            regime=str(options["regime"]),
            regime_factor_policy_overrides=options["regime_factor_policy_overrides"],
            regime_decision_policy_overrides=options["regime_decision_policy_overrides"],
            regime_factor_overlay_scale=float(options["regime_factor_overlay_scale"]),
            alpha_score=float(alpha_score),
        )
        * event_reweight
    )
    setup_decision = classify_setup(
        factors,
        profile,
        regime=str(options["regime"]),
        factor_threshold_bias=float(options["factor_threshold_bias"]) + float(decision["threshold_bias_shift"]),
        regime_overlay_score=float(decision["effective_overlay_score"]),
        regime_overlay_floor=float(decision["overlay_floor"]),
        regime_support_score=float(decision["support_score"]),
        regime_support_floor=float(decision["support_floor"]),
    )
    return MainWaveSignal(
        enter=enter,
        score=round(total_score, 6),
        alpha_score=round(alpha_score, 6),
        reasons=reasons if bool(options["include_reasons"]) else [],
        factors={name: round(value, 6) for name, value in factors.items()} if bool(options["include_details"]) else {},
        exit_guard_score=round(calc_exit_guard_score(factors), 6),
        regime_overlay_score=round(float(decision["effective_overlay_score"]), 6),
        regime_support_score=round(float(decision["support_score"]), 6),
        position_scale=round(float(decision["position_scale"]), 6),
        setup_type=str(setup_decision.setup_type),
        trend_strength=float(setup_decision.trend_strength),
        flow_strength=float(setup_decision.flow_strength),
        support_strength=float(setup_decision.support_strength),
    )

