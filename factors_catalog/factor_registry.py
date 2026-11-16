# ruff: noqa
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    family: str
    higher_is_better: bool
    dependencies: tuple[str, ...]
    description: str


_FACTOR_DEFINITIONS = (
    FactorDefinition("momentum_5", "momentum", True, ("prices_1d",), "Five-day raw price momentum."),
    FactorDefinition("breakout_distance_55", "trend", True, ("prices_1d",), "Distance from the prior 55-session high."),
    FactorDefinition("drawdown_from_high_20", "risk", False, ("prices_1d",), "Current drawdown versus the rolling 20-session high."),
    FactorDefinition("range_ratio_20", "volatility", False, ("prices_1d",), "Average 20-session intraday range normalized by close."),
    FactorDefinition("compression_ratio_20_60", "volatility", True, ("prices_1d",), "Recent volatility compression versus the prior longer window."),
    FactorDefinition("realized_vol_20", "volatility", False, ("prices_1d",), "Realized daily volatility over 20 sessions."),
    FactorDefinition("downside_vol_20", "volatility", False, ("prices_1d",), "Downside-only realized volatility over 20 sessions."),
    FactorDefinition("vol_adjusted_momentum_20", "momentum", True, ("prices_1d",), "Twenty-day momentum normalized by realized volatility."),
    FactorDefinition("sharpe_like_20", "momentum", True, ("prices_1d",), "Mean daily return divided by return volatility over 20 sessions."),
    FactorDefinition("drawdown_resilience_20", "risk", True, ("prices_1d",), "Negative rolling max drawdown over 20 sessions; higher means shallower drawdowns."),
    FactorDefinition("flow_proxy_strength_10", "flow", True, ("prices_1d", "turnover_1d"), "Turnover-weighted return pressure over the last 10 sessions."),
    FactorDefinition("turnover_return_corr_20", "flow", True, ("prices_1d", "turnover_1d"), "Correlation between daily returns and turnover changes over 20 sessions."),
    FactorDefinition("relative_strength_20", "momentum", True, ("prices_1d", "benchmark_prices"), "Twenty-day return relative to market proxy."),
    FactorDefinition("beta_60", "risk", False, ("prices_1d", "benchmark_prices"), "Sixty-day beta of returns versus market proxy."),
    FactorDefinition("alpha_20_beta_60", "momentum", True, ("prices_1d", "benchmark_prices"), "Twenty-day relative return after removing 60-day beta exposure."),
    FactorDefinition("trend_acceleration_20", "trend", True, ("prices_1d",), "Late-vs-early slope acceleration over 20 sessions."),
    FactorDefinition("breakout_quality_10", "trend", True, ("prices_1d",), "Distance above the recent breakout level."),
    FactorDefinition("price_momentum_20", "momentum", True, ("prices_1d",), "Twenty-day raw price momentum."),
    FactorDefinition("price_momentum_60", "momentum", True, ("prices_1d",), "Sixty-day raw price momentum."),
    FactorDefinition("ma_gap_20", "trend", True, ("prices_1d",), "Distance between close and 20-day moving average."),
    FactorDefinition("ma_gap_60", "trend", True, ("prices_1d",), "Distance between close and 60-day moving average."),
    FactorDefinition("ma_spread_20_60", "trend", True, ("prices_1d",), "Twenty-day moving average spread over the sixty-day base."),
    FactorDefinition("trend_streak_10", "trend", True, ("prices_1d",), "Net positive-minus-negative day streak over 10 sessions."),
    FactorDefinition("price_position_20", "trend", True, ("prices_1d",), "Close location inside the 20-day range."),
    FactorDefinition("range_position_20", "trend", True, ("prices_1d",), "Close location inside the 20-day high-low range."),
    FactorDefinition("breakout_distance_20", "trend", True, ("prices_1d",), "Distance from the prior 20-session closing high."),
    FactorDefinition("rebound_from_low_20", "trend", True, ("prices_1d",), "Recovery distance from the 20-session low."),
    FactorDefinition("pullback_absorption_10", "trend", True, ("prices_1d",), "Ability to hold gains after a local high."),
    FactorDefinition("drawdown_20", "risk", False, ("prices_1d",), "Drawdown from the rolling 20-session closing peak."),
    FactorDefinition("channel_width_20", "volatility", False, ("prices_1d",), "Twenty-session price channel width normalized by close."),
    FactorDefinition("liquidity_sponsorship_20", "liquidity", True, ("turnover_1d",), "Recent turnover sponsorship vs baseline."),
    FactorDefinition("exit_deterioration_score_5", "risk", False, ("prices_1d",), "Short-term downside deterioration pressure."),
    FactorDefinition("volatility_20", "volatility", False, ("prices_1d",), "Realized daily volatility over 20 sessions."),
    FactorDefinition("volatility_ratio_20_60", "volatility", False, ("prices_1d",), "Short-window volatility premium versus the 60-session baseline."),
    FactorDefinition("volatility_compression_20", "volatility", True, ("prices_1d",), "Compression of recent volatility vs prior window."),
    FactorDefinition("realized_volatility_20", "volatility", False, ("prices_1d",), "Realized daily volatility over 20 sessions."),
    FactorDefinition("downside_volatility_20", "volatility", False, ("prices_1d",), "Downside-only realized volatility over 20 sessions."),
    FactorDefinition("relative_strength_60", "momentum", True, ("prices_1d", "benchmark_prices"), "Sixty-day return relative to market proxy."),
    FactorDefinition("momentum_efficiency_20", "momentum", True, ("prices_1d",), "Net move divided by travelled path length."),
    FactorDefinition("trend_quality_20", "trend", True, ("prices_1d",), "Twenty-day advance quality after adjusting for path efficiency and volatility."),
    FactorDefinition("trend_stability_15", "trend", True, ("prices_1d",), "Directional consistency of daily returns."),
    FactorDefinition("drawdown_recovery_20", "trend", True, ("prices_1d",), "Recovery ratio from the latest 20-day pullback toward the prior peak."),
    FactorDefinition("recovery_efficiency_20", "trend", True, ("prices_1d",), "Recovery quality after pullback, scaled by path efficiency and speed."),
    FactorDefinition("support_lift_strength_20", "trend", True, ("prices_1d",), "Progressive lift in rolling support levels relative to realized range."),
    FactorDefinition("box_ceiling_residence_20", "trend", True, ("prices_1d",), "Upper-range residence inside a compressed 20-day box."),
    FactorDefinition("resistance_memory_decay_20", "trend", True, ("prices_1d", "turnover_1d"), "Repeated upper-bound tests with decaying rejection intensity and contained distribution."),
    FactorDefinition("breakout_retention_15", "trend", True, ("prices_1d", "turnover_1d"), "Acceptance quality after recent breakouts above the prior trading range."),
    FactorDefinition("breakout_retest_absorption_20", "trend", True, ("prices_1d", "turnover_1d"), "First breakout retest hold quality with contained supply and rebound support."),
    FactorDefinition("steady_trend_advancement_20", "trend", True, ("prices_1d", "turnover_1d"), "Steady upper-band trend advancement with broad participation and restrained turnover chase instead of one-bar impulse dependence."),
    FactorDefinition("fractal_trend_persistence_20", "trend", True, ("prices_1d",), "Fractal-persistence proxy estimating whether directional structure survives across multiple aggregation scales instead of collapsing into short-horizon noise."),
    FactorDefinition("spectral_trend_coherence_20", "trend", True, ("prices_1d",), "Low-frequency spectral dominance proxy rewarding trends whose return energy is concentrated in smooth slow modes rather than choppy oscillation."),
    FactorDefinition("state_space_trend_snr_20", "trend", True, ("prices_1d",), "State-space filtered trend signal-to-noise proxy rewarding advances whose latent drift survives observation noise."),
    FactorDefinition("path_efficiency_tail_resilience_20", "trend", True, ("prices_1d", "turnover_1d"), "Path-efficiency and left-tail-resilience proxy rewarding steady advances with broad participation, shallow drawdowns, muted CVaR-style downside tails, and controlled turnover chase."),
    FactorDefinition("trend_liquidity_persistence_20", "liquidity", True, ("prices_1d", "turnover_1d"), "Persistent upside participation proxy rewarding trends where turnover sponsorship remains durable, aligned with positive returns, and does not migrate into downside churn."),
    FactorDefinition("flow_shock_half_life_resilience_20", "flow", True, ("prices_1d", "turnover_1d", "stock_flows"), "Flow-shock half-life proxy rewarding names whose price damage after negative-flow shocks decays quickly with retained turnover capacity and repair follow-through."),
    FactorDefinition("institutional_trend_acceptance_20", "trend", True, ("prices_1d", "turnover_1d", "stock_flows"), "Composite institutional trend-acceptance proxy combining state-space trend SNR, path resilience, liquidity persistence, and fast absorption of flow shocks."),
    FactorDefinition("institutional_trend_acceptance_clean_20", "trend", True, ("prices_1d", "turnover_1d", "stock_flows"), "Clean institutional trend-acceptance proxy rewarding trend SNR only when path resilience, liquidity persistence, shock absorption, and balanced momentum agree without chase-like overextension."),
    FactorDefinition("institutional_trend_false_acceptance_risk_20", "risk", False, ("prices_1d", "turnover_1d", "stock_flows"), "False institutional acceptance risk proxy flagging high-SNR trends that lack path/liquidity/shock confirmation or show turnover-chase and overextension."),
    FactorDefinition("weak_trend_reaccumulation_spring_20", "trend", True, ("prices_1d", "turnover_1d"), "Weak-trend reaccumulation spring proxy rewarding compressed low-volume bases with lifting lows, upper-range reclaim, volatility contraction, and controlled downside tails before broad trend confirmation."),
    FactorDefinition("innovation_drift_absorption_20", "trend", True, ("prices_1d", "turnover_1d"), "State-space innovation absorption proxy rewarding positive shocks that keep drifting while residual noise and turnover chase stay controlled."),
    FactorDefinition("entropy_compression_breakout_potential_20", "trend", True, ("prices_1d", "turnover_1d"), "Entropy-compression breakout potential proxy rewarding quiet ordered bases near the upper range before expansion."),
    FactorDefinition("volatility_of_volatility_contraction_20", "volatility", True, ("prices_1d",), "Second-order volatility compression proxy rewarding names where vol-of-vol contracts into a calm but still advancing structure."),
    FactorDefinition("asymmetric_autocorr_repair_20", "trend", True, ("prices_1d",), "Asymmetric return-memory proxy rewarding upside continuation after up days and constructive repair after down days without one-bar chase."),
    FactorDefinition("drawdown_recovery_convexity_20", "trend", True, ("prices_1d",), "Recovery-convexity proxy rewarding multi-day acceleration out of a local trough rather than a single fragile bounce."),
    FactorDefinition("tail_repair_skew_balance_20", "risk", True, ("prices_1d",), "Tail repair skew proxy rewarding stronger upside tails and frequent positive follow-through after down days."),
    FactorDefinition("low_volume_box_support_20", "trend", True, ("prices_1d", "turnover_1d"), "Compressed box support with low-volume hold near the upper-middle structure."),
    FactorDefinition("compression_release_quality_15", "trend", True, ("prices_1d", "turnover_1d"), "Compression-to-release quality with turnover confirmation after a quiet base."),
    FactorDefinition("compression_persistence_20", "trend", True, ("prices_1d", "turnover_1d"), "Persistent quiet compression that keeps price in the upper structure with calm turnover."),
    FactorDefinition("flat_top_higher_low_coil_20", "trend", True, ("prices_1d",), "Coiled flat-top structure with rising lows and contracting range under a stable ceiling."),
    FactorDefinition("upper_shadow_distribution_pressure_10", "risk", False, ("opens_1d", "prices_1d", "turnover_1d"), "Distribution pressure from repeated upper shadows, weak closes, and elevated turnover near upper structure."),
    FactorDefinition("breakout_close_control_10", "trend", True, ("opens_1d", "prices_1d"), "Breakout bars that retain close control and avoid upper-shadow supply."),
    FactorDefinition("gap_fill_acceptance_8", "trend", True, ("opens_1d", "prices_1d", "turnover_1d"), "Whether negative gaps get constructively filled and accepted over the next few sessions instead of failing back down."),
    FactorDefinition("bullish_engulfing_hold_10", "trend", True, ("opens_1d", "prices_1d"), "Bullish engulfing reversal quality conditioned on follow-through and holding above the engulfing midpoint."),
    FactorDefinition("breakout_digestion_hold_10", "trend", True, ("prices_1d", "turnover_1d"), "Post-breakout digestion that holds above the base while turnover cools constructively."),
    FactorDefinition("failed_reclaim_persistence_15", "risk", False, ("prices_1d", "turnover_1d"), "Repeated support reclaims that fail to hold by the close, signaling weak repair quality."),
    FactorDefinition("support_test_reclaim_15", "trend", True, ("prices_1d",), "Repeated support tests that reclaim and close back above structure."),
    FactorDefinition("false_breakdown_repair_10", "trend", True, ("prices_1d",), "Failed-breakdown reclaim quality after a local support washout."),
    FactorDefinition("delayed_undercut_reclaim_12", "trend", True, ("prices_1d", "turnover_1d"), "Undercut events that reclaim within a few sessions and then hold with cooling turnover."),
    FactorDefinition("failed_breakout_reversal_intensity_12", "risk", False, ("opens_1d", "prices_1d", "turnover_1d"), "How violently recent breakout attempts reverse back below the prior ceiling with follow-through failure."),
    FactorDefinition("breakout_failure_repair_15", "trend", True, ("opens_1d", "prices_1d", "turnover_1d"), "Repair quality after a failed breakout attempt reclaims the pivot and stabilizes with lighter supply."),
    FactorDefinition("failed_breakout_recoil_repair_15", "trend", True, ("prices_1d", "turnover_1d"), "Failed breakout that gets reclaimed and then holds tightly above the old ceiling with calmer post-failure churn."),
    FactorDefinition("shrink_volume_nonbreak_5_20", "liquidity", True, ("prices_1d", "turnover_1d"), "Recent low-volume pullback resilience without meaningful price breakage."),
    FactorDefinition("pullback_energy_decay_20", "liquidity", True, ("prices_1d", "turnover_1d"), "Decay in negative pullback energy while price holds near the upper range."),
    FactorDefinition("distribution_decay_after_inflow_spike_8", "flow", True, ("stock_flows", "prices_1d", "turnover_1d"), "Whether a strong recent inflow spike avoids turning into immediate heavy distribution over the next few sessions."),
    FactorDefinition("recrowding_reacceleration_quality_12", "flow", True, ("stock_flows", "prices_1d", "turnover_1d"), "Whether a stock can reaccelerate after a negative-flow reset with constructive follow-through and without instantly re-entering high-churn crowding."),
    FactorDefinition("serial_outflow_damage_dampening_12", "flow", True, ("stock_flows", "prices_1d", "turnover_1d"), "Whether repeated negative flow shocks are causing progressively less price damage, indicating selling-pressure dampening."),
    FactorDefinition("highs_stepup_consistency_20", "trend", True, ("prices_1d",), "Consistency of rising overhead ceilings with limited failed closes below the prior step-up."),
    FactorDefinition("failed_breakout_scarcity_20", "trend", True, ("opens_1d", "prices_1d", "turnover_1d"), "Scarcity of failed intraday breakouts over the recent structure window."),
    FactorDefinition("advance_participation_uniformity_20", "trend", True, ("prices_1d",), "Whether recent upside participation is broad instead of concentrated in one or two impulse days."),
    FactorDefinition("reaction_high_retake_quality_15", "trend", True, ("prices_1d", "turnover_1d"), "Quality of reclaiming the prior reaction-high span after a recent local washout, with midpoint retention and calmer turnover."),
    FactorDefinition("gap_failure_containment_10", "flow", True, ("opens_1d", "prices_1d", "turnover_1d"), "Whether a recent negative gap or flush day avoids multi-day follow-through damage and quickly regains control."),
    FactorDefinition("repair_liquidity_retention_10", "flow", True, ("prices_1d", "turnover_1d"), "Whether a recent repair move retains enough follow-through liquidity and structure instead of turning into a hollow bounce."),
    FactorDefinition("repair_base_tightness_12", "trend", True, ("prices_1d", "turnover_1d"), "Whether a recent reclaim or false-break repair transitions into a tight, low-churn base above the repaired pivot."),
    FactorDefinition("breakout_pressure_10", "momentum", True, ("prices_1d",), "Recent burst vs baseline return pressure."),
    FactorDefinition("abnormal_turnover_z20", "liquidity", True, ("turnover_1d",), "Latest turnover shock relative to 20-day history."),
    FactorDefinition("turnover_impulse_5_20", "liquidity", True, ("turnover_1d",), "Short-vs-long turnover impulse."),
    FactorDefinition("turnover_stability_20", "liquidity", True, ("turnover_1d",), "Stability of turnover over 20 sessions."),
    FactorDefinition("volume_price_confirmation_10", "liquidity", True, ("prices_1d", "turnover_1d"), "Up-move turnover support net of distribution pressure over 10 sessions."),
    FactorDefinition("amihud_illiquidity_20", "liquidity", False, ("prices_1d", "turnover_1d"), "Price impact per unit of traded cash turnover."),
    FactorDefinition("liquidity_impact_relief_10", "liquidity", True, ("prices_1d", "turnover_1d"), "Daily microstructure proxy rewarding falling price impact, better upside efficiency, and calmer turnover in the recent repair window."),
    FactorDefinition("exhaustion_risk_10", "crowding", False, ("prices_1d", "turnover_1d"), "Blow-off risk from spikes, turnover and highs."),
    FactorDefinition("return_skew_20", "risk", False, ("prices_1d",), "Skewness of daily returns over 20 sessions."),
    FactorDefinition("intraday_strength_1", "intraday", True, ("opens_1d", "prices_1d"), "Close-versus-open strength for the latest session."),
    FactorDefinition("gap_strength_1", "intraday", True, ("opens_1d", "prices_1d"), "Opening gap versus the prior close."),
    FactorDefinition("close_drive_strength_30m", "intraday", True, ("bars_30m",), "Closing drive strength in 30-minute bars."),
    FactorDefinition("vwap_dislocation_30m", "intraday", True, ("bars_30m", "vwap_30m"), "Close relative to intraday VWAP."),
    FactorDefinition("auction_gap_strength_1d", "intraday", True, ("opens_1d", "prices_1d"), "Opening gap strength versus prior close."),
    FactorDefinition("auction_turnover_imbalance_30m", "intraday", True, ("turnover_30m",), "Opening turnover imbalance."),
    FactorDefinition("opening_impulse_retention_30m", "intraday", True, ("opens_30m", "closes_30m"), "Retention of first-bar impulse through the day."),
    FactorDefinition("intraday_trend_consistency_30m", "intraday", True, ("bars_30m",), "Directional consistency of intraday trend."),
    FactorDefinition("intraday_range_close_location_30m", "intraday", True, ("bars_30m",), "End-of-day close location within intraday range."),
    FactorDefinition("large_candle_followthrough_5", "momentum", True, ("prices_1d",), "Fraction of recent large up candles."),
    FactorDefinition("distribution_warning_5", "crowding", False, ("prices_1d", "turnover_1d"), "Distribution days with price down and volume up."),
    FactorDefinition("sector_rs_20", "rotation", True, ("sector_closes", "market_closes"), "Sector relative strength versus market."),
    FactorDefinition("sector_breadth_10", "rotation", True, ("sector_up_counts", "sector_total_counts"), "Sector breadth over recent sessions."),
    FactorDefinition("sector_flow_accel_5", "rotation", True, ("sector_flows",), "Acceleration in sector fund flow."),
    FactorDefinition("sector_flow_breadth_confirm_5_10", "rotation", True, ("sector_flows", "sector_up_counts", "sector_total_counts"), "Sector fund-flow acceleration confirmed by improving internal breadth."),
    FactorDefinition("rotation_early_diffusion_10_20", "rotation", True, ("sector_closes", "market_closes", "sector_up_counts", "sector_total_counts"), "Early sector diffusion proxy rewarding improving breadth and excess-return acceleration before the group becomes broadly crowded."),
    FactorDefinition("sector_flow_breadth_divergence_crowding_15", "rotation", False, ("sector_flows", "sector_up_counts", "sector_total_counts"), "Crowding risk when recent sector fund inflow remains positive while internal breadth narrows instead of expanding."),
    FactorDefinition("leader_laggard_spread_10", "rotation", True, ("leader_prices", "laggard_prices"), "Leader minus laggard spread."),
    FactorDefinition("stock_net_inflow_5_20", "flow", True, ("stock_flows",), "Short-vs-long stock net inflow."),
    FactorDefinition("stock_sector_flow_resonance_10", "flow", True, ("stock_flows", "sector_flows"), "Stock inflow resonance with the parent sector flow regime."),
    FactorDefinition("stock_flow_residual_vs_sector_10", "flow", True, ("stock_flows", "sector_flows", "prices_1d"), "Residual stock flow strength after neutralizing common sector flow exposure."),
    FactorDefinition("stock_flow_residual_acceleration_5_20", "flow", True, ("stock_flows", "sector_flows", "prices_1d"), "Acceleration in stock-versus-sector flow residuals, emphasizing improving relative sponsorship."),
    FactorDefinition("stock_defensive_residual_on_sector_selloff_15", "flow", True, ("stock_flows", "sector_flows", "sector_closes", "prices_1d"), "Residual stock sponsorship and relative return resilience specifically during parent-sector selloff days."),
    FactorDefinition("relative_drawdown_resilience_15", "flow", True, ("prices_1d", "sector_closes", "market_closes"), "Relative drawdown resilience versus sector and market during recent weak windows, emphasizing smaller damage and faster recapture."),
    FactorDefinition("residual_momentum_20_60", "momentum", True, ("prices_1d", "sector_closes", "market_closes"), "Market- and sector-neutral residual momentum over 20/60 sessions, emphasizing stock-specific drift instead of common beta or industry chase."),
    FactorDefinition("stock_sector_breadth_capture_10", "flow", True, ("stock_flows", "sector_flows", "sector_closes", "sector_up_counts", "sector_total_counts", "prices_1d"), "Whether the stock captures sector flow and breadth expansion better than its parent sector."),
    FactorDefinition("stock_sector_breadth_lead_10", "flow", True, ("stock_flows", "sector_flows", "sector_up_counts", "sector_total_counts"), "Whether stock-level residual sponsorship leads later breadth expansion in the parent sector."),
    FactorDefinition("stock_relative_sponsorship_under_sector_crowding_10", "flow", True, ("stock_flows", "sector_flows", "sector_up_counts", "sector_total_counts"), "Residual stock sponsorship on days when the parent sector is getting crowded, helping separate genuine sponsorship from passive sector chasing."),
    FactorDefinition("selloff_absorption_flow_10", "flow", True, ("stock_flows", "prices_1d"), "Down-day flow absorption quality over the last 10 sessions."),
    FactorDefinition("downside_impact_decay_15", "flow", True, ("stock_flows", "prices_1d"), "Decay in downside price impact under comparable selling pressure."),
    FactorDefinition("negative_flow_impact_beta_10", "flow", True, ("stock_flows", "prices_1d"), "Lower slope of downside damage versus negative flow shocks over recent sessions."),
    FactorDefinition("downside_impact_improvement_5_15", "flow", True, ("stock_flows", "prices_1d"), "Recent downside impact improves versus the prior window under comparable flow pressure."),
    FactorDefinition("panic_followthrough_repair_8", "flow", True, ("opens_1d", "prices_1d", "turnover_1d"), "Whether a recent panic gap or flush day is followed by real multi-day repair instead of a one-bar bounce."),
    FactorDefinition("panic_followthrough_decay_12", "flow", True, ("opens_1d", "prices_1d", "turnover_1d"), "Whether a sharp selloff or panic day quickly loses follow-through damage, range expansion, and turnover stress over the next few sessions."),
    FactorDefinition("outflow_decay_price_repair_3_10", "flow", True, ("stock_flows", "prices_1d"), "Outflow relief coinciding with short-horizon price repair from a recent floor."),
    FactorDefinition("outflow_repair_without_inflow_10", "flow", True, ("stock_flows", "prices_1d"), "Price repair emerges even while the recent flow window remains net negative."),
    FactorDefinition("outflow_reversal_efficiency_10", "flow", True, ("stock_flows", "prices_1d"), "How much of an outflow-day intraday selloff gets repaired by the close."),
    FactorDefinition("turnover_adjusted_outflow_damage_10", "flow", True, ("stock_flows", "prices_1d", "turnover_1d"), "Damage per unit of negative flow after adjusting for contemporaneous turnover capacity."),
    FactorDefinition("severe_outflow_nonbreak_rate_15", "flow", True, ("stock_flows", "prices_1d"), "Hit rate of holding structure after severe outflow events in the following sessions."),
    FactorDefinition("outflow_retest_absorption_15", "flow", True, ("stock_flows", "prices_1d", "turnover_1d"), "Whether heavy outflow shocks get retested without meaningful new lows while closes and turnover stabilize over the next few sessions."),
    FactorDefinition("shock_relaxation_convexity_15", "flow", True, ("stock_flows", "prices_1d", "turnover_1d"), "Nonlinear shock-response proxy rewarding names where damage after negative-flow impulses relaxes convexly instead of compounding into multi-day stress."),
    FactorDefinition("sector_net_inflow_5_20", "flow", True, ("sector_flows",), "Short-vs-long sector net inflow."),
    FactorDefinition("sector_flow_conversion_efficiency_20", "rotation", True, ("sector_flows", "sector_closes", "market_closes"), "How efficiently positive sector flow converts into sector excess return versus the market."),
    FactorDefinition("sector_negative_flow_breadth_resilience_10", "rotation", True, ("sector_flows", "sector_up_counts", "sector_total_counts"), "Whether sector breadth remains resilient despite recent negative sector flow pressure."),
    FactorDefinition("flow_persistence_10", "flow", True, ("stock_flows",), "Persistence of flow direction."),
    FactorDefinition("flow_price_divergence_10", "flow", False, ("stock_flows", "prices_1d"), "Mismatch between fund flow and price trend."),
    FactorDefinition("flow_correlation_10", "flow", True, ("stock_flows", "prices_1d"), "Correlation between returns and flow series."),
    FactorDefinition("lower_shadow_absorption_5", "intraday", True, ("opens_1d", "prices_1d"), "Recent lower-shadow reclaim strength after intraday selloffs."),
    FactorDefinition("upper_half_churn_trap_20", "risk", False, ("opens_1d", "prices_1d", "turnover_1d"), "Repeated upper-half failed pushes with elevated churn, indicating unresolved supply near the upper structure."),
    FactorDefinition("limit_up_continuation_rate", "sentiment", True, ("sentiment_proxy",), "Continuation intensity in speculative sentiment."),
    FactorDefinition("board_break_rate", "sentiment", False, ("sentiment_proxy",), "Board-break stress level."),
    FactorDefinition("gap_exhaustion_score", "sentiment", False, ("sentiment_proxy",), "Positive gap exhaustion accumulation."),
    FactorDefinition("panic_reversal_score", "sentiment", True, ("sentiment_proxy",), "Reversal support after panic conditions."),
    FactorDefinition("market_breadth_support_10", "regime", True, ("sentiment_proxy",), "Average breadth support around neutral line."),
    FactorDefinition("sentiment_tailwind_10", "regime", True, ("sentiment_proxy",), "Broad sentiment minus board-break stress."),
    FactorDefinition("crowding_risk_score", "crowding", False, ("stock_flows", "turnover_1d", "sentiment_proxy"), "Crowding composite from flow, turnover and sentiment."),
    FactorDefinition("event_surprise_score_5", "event", True, ("event_direction_series", "event_intensity_series", "event_confidence_series"), "Signed surprise from recent events."),
    FactorDefinition("event_recency_decay_30m", "event", True, ("event_age_minutes", "event_impact_series"), "Recency-weighted event impact."),
    FactorDefinition("event_consistency_20d", "event", True, ("event_impact_series",), "Directional consistency of recent event impacts."),
    FactorDefinition("event_sector_diffusion_5d", "event", True, ("event_symbol_hits", "event_sector_hits"), "Breadth of event diffusion across symbols and sectors."),
    FactorDefinition("regulatory_risk_penalty_20d", "event", False, ("regulatory_event_series",), "Penalty from negative regulatory events."),
    FactorDefinition("shareholder_alignment_60d", "event", True, ("shareholder_event_series",), "Alignment of shareholder actions with longs."),
    FactorDefinition("policy_alignment_score_10d", "event", True, ("policy_event_series", "sector_closes"), "Policy impact aligned with sector leadership."),
    FactorDefinition("crowding_event_conflict_5d", "event", False, ("sentiment_proxy", "hotword_series"), "Conflict between crowding and event heat."),
    FactorDefinition("cashflow_profit_match_ttm", "fundamental", True, ("financial_income", "financial_cashflow"), "Whether operating cash flow genuinely matches trailing net income."),
    FactorDefinition("receivables_revenue_gap_yoy", "fundamental", True, ("financial_income", "financial_balancesheet"), "Revenue growth outpacing receivables growth, signaling cleaner top-line quality."),
    FactorDefinition("operating_margin_reacceleration_ttm", "fundamental", True, ("financial_income",), "Re-acceleration of trailing operating margin versus the same period last year."),
    FactorDefinition("inventory_revenue_gap_yoy", "fundamental", True, ("financial_income", "financial_balancesheet"), "Revenue growth outpacing inventory growth, helping filter inventory-led low-quality expansion."),
    FactorDefinition("report_lag_shock_4q", "fundamental", True, ("financial_income", "financial_balancesheet", "financial_cashflow"), "Recent disclosure arrives faster than the symbol's own recent reporting pattern, helping flag cleaner and less stressed reporting cadence."),
    FactorDefinition("net_leverage_relief_yoy", "fundamental", True, ("financial_balancesheet",), "Net leverage improves versus the same period last year after offsetting liabilities by on-balance-sheet cash."),
    FactorDefinition("cash_coverage_relief_yoy", "fundamental", True, ("financial_balancesheet",), "Cash coverage of total liabilities improves versus the same period last year, indicating easing balance-sheet pressure."),
    FactorDefinition("operating_cashflow_margin_delta_yoy", "fundamental", True, ("financial_income", "financial_cashflow"), "Trailing operating cashflow margin improves versus the same period last year, signaling better cash conversion quality."),
    FactorDefinition("fundamental_freshness_decay_120d", "fundamental", True, ("financial_income", "financial_balancesheet", "financial_cashflow"), "Exponential decay weight based on days since the latest effective financial disclosure."),
    FactorDefinition("growth_quality_underreaction_20", "fundamental", True, ("financial_income", "financial_balancesheet", "financial_cashflow"), "Composite fundamental quality signal favoring clean growth, margin reacceleration, and fresher disclosures without aggressive receivable or inventory expansion."),
    FactorDefinition("cash_conversion_balance_sheet_momentum_20", "fundamental", True, ("financial_income", "financial_balancesheet", "financial_cashflow"), "Composite balance-sheet repair and cash-conversion momentum signal that rewards stronger cash realization, leverage relief, and liability coverage improvement."),
    FactorDefinition("quality_balance_sheet_repair_20", "fundamental", True, ("financial_income", "financial_balancesheet", "financial_cashflow"), "Publicly reproducible quality-minus-fragility style composite emphasizing cleaner growth, cash conversion, leverage relief, and fresh disclosures."),
    FactorDefinition("accrual_cashflow_quality_20", "fundamental", True, ("financial_income", "financial_balancesheet", "financial_cashflow"), "Public earnings-quality style composite that rewards stronger cash realization, cleaner accrual behavior, and less balance-sheet-stretched growth."),
    FactorDefinition("post_report_quality_drift_20", "fundamental", True, ("financial_income", "financial_balancesheet", "financial_cashflow"), "Public post-disclosure underreaction proxy that favors fresher high-quality reports, improving margin/cashflow evidence, and cleaner reporting cadence."),
    FactorDefinition("cash_to_assets_ttm_proxy_20", "fundamental", True, ("financial_balancesheet",), "Public liquidity-strength proxy rewarding larger cash balances relative to the asset base."),
    FactorDefinition("net_cash_to_assets_20", "fundamental", True, ("financial_balancesheet",), "Public balance-sheet safety proxy rewarding stronger net cash support after offsetting liabilities against cash."),
    FactorDefinition("current_ratio_level_20", "fundamental", True, ("financial_balancesheet",), "Public short-term solvency proxy rewarding stronger current-asset coverage of current liabilities."),
    FactorDefinition("current_ratio_improvement_20", "fundamental", True, ("financial_balancesheet",), "Public short-term solvency trend proxy rewarding year-over-year improvement in current-ratio coverage."),
    FactorDefinition("asset_growth_yoy_20", "fundamental", True, ("financial_balancesheet",), "Public asset-growth anomaly proxy rewarding more conservative year-over-year balance-sheet expansion."),
    FactorDefinition("fixed_asset_growth_yoy_20", "fundamental", True, ("financial_balancesheet",), "Public investment-intensity proxy rewarding slower fixed-asset expansion and more disciplined capex growth."),
    FactorDefinition("retained_earnings_growth_yoy_20", "fundamental", True, ("financial_balancesheet", "financial_cashflow"), "Public internal-financing quality proxy rewarding retained-earnings compounding backed by healthier cash realization and leverage relief."),
    FactorDefinition("ebitda_to_assets_20", "fundamental", True, ("financial_income", "financial_balancesheet"), "Public EBITDA-productivity proxy rewarding stronger trailing EBITDA relative to the asset base."),
    FactorDefinition("external_financing_reliance_20", "fundamental", True, ("financial_cashflow", "financial_balancesheet"), "Public external-financing dependence proxy rewarding lower reliance on financing cashflow and better year-over-year financing relief."),
    FactorDefinition("gross_profitability_assets_20", "fundamental", True, ("financial_income", "financial_balancesheet", "financial_cashflow"), "Public profitability style composite emphasizing gross profitability, EBITDA productivity, and cash-generative asset efficiency."),
    FactorDefinition("gross_margin_level_20", "fundamental", True, ("financial_income",), "Public profitability-level proxy rewarding stronger trailing gross margin."),
    FactorDefinition("gross_margin_improvement_20", "fundamental", True, ("financial_income",), "Public profitability-trend proxy rewarding year-over-year improvement in trailing gross margin."),
    FactorDefinition("conservative_investment_discipline_20", "fundamental", True, ("financial_income", "financial_balancesheet"), "Public conservative-investment style composite rewarding restrained asset growth, cleaner working-capital expansion, and retained-earnings-funded growth."),
    FactorDefinition("financing_discipline_safety_20", "fundamental", True, ("financial_balancesheet", "financial_cashflow"), "Public financing-discipline style composite that favors lower reliance on external financing, improving cash buffers, and safer liability structure."),
    FactorDefinition("operating_profitability_assets_20", "fundamental", True, ("financial_income", "financial_balancesheet"), "Public operating-profitability style proxy rewarding stronger trailing operating profit relative to the asset base."),
    FactorDefinition("operating_profitability_improvement_20", "fundamental", True, ("financial_income", "financial_balancesheet"), "Public profitability-trend proxy rewarding year-over-year improvement in operating profit relative to the asset base."),
    FactorDefinition("cash_profitability_assets_20", "fundamental", True, ("financial_cashflow", "financial_balancesheet"), "Public cash-profitability style proxy rewarding stronger trailing operating cashflow relative to the asset base."),
    FactorDefinition("cash_profitability_improvement_20", "fundamental", True, ("financial_cashflow", "financial_balancesheet"), "Public cash-conversion trend proxy rewarding year-over-year improvement in operating cashflow relative to the asset base."),
    FactorDefinition("return_on_assets_level_20", "fundamental", True, ("financial_income", "financial_balancesheet"), "Public profitability-level proxy rewarding stronger trailing net income relative to the asset base."),
    FactorDefinition("return_on_assets_improvement_20", "fundamental", True, ("financial_income", "financial_balancesheet"), "Public profitability-trend proxy rewarding year-over-year improvement in return on assets."),
    FactorDefinition("asset_turnover_level_20", "fundamental", True, ("financial_income", "financial_balancesheet"), "Public efficiency-level proxy rewarding stronger trailing revenue productivity relative to the asset base."),
    FactorDefinition("operating_cashflow_to_liabilities_20", "fundamental", True, ("financial_cashflow", "financial_balancesheet"), "Public balance-sheet quality proxy rewarding stronger operating cashflow coverage of total liabilities."),
    FactorDefinition("asset_turnover_improvement_20", "fundamental", True, ("financial_income", "financial_balancesheet"), "Public asset-turnover style proxy rewarding stronger revenue productivity and year-over-year turnover improvement."),
    FactorDefinition("net_external_financing_burden_20", "fundamental", True, ("financial_cashflow", "financial_balancesheet"), "Public financing-burden style proxy rewarding lower reliance on external financing and year-over-year financing relief."),
    FactorDefinition("liquidity_safety_buffer_20", "fundamental", True, ("financial_balancesheet",), "Public balance-sheet safety proxy rewarding stronger cash coverage and current-ratio support."),
    FactorDefinition("working_capital_to_assets_20", "fundamental", True, ("financial_balancesheet",), "Public liquidity-strength proxy rewarding larger operating working-capital buffers relative to the asset base."),
    FactorDefinition("retained_earnings_to_assets_20", "fundamental", True, ("financial_balancesheet",), "Public accumulated-profitability proxy rewarding stronger retained earnings support relative to the asset base."),
    FactorDefinition("operating_accruals_assets_20", "fundamental", True, ("financial_income", "financial_balancesheet", "financial_cashflow"), "Public Sloan-style accrual proxy rewarding lower accrual intensity after scaling earnings-minus-cashflow by assets."),
    FactorDefinition("working_capital_accruals_20", "fundamental", True, ("financial_balancesheet",), "Public working-capital accrual proxy rewarding more disciplined operating working-capital expansion relative to assets."),
    FactorDefinition("net_operating_assets_ratio_20", "fundamental", True, ("financial_balancesheet",), "Public net-operating-assets style proxy rewarding lower balance-sheet operating asset intensity after excluding cash support."),
    FactorDefinition("working_capital_discipline_20", "fundamental", True, ("financial_income", "financial_balancesheet", "financial_cashflow"), "Public working-capital quality composite rewarding cleaner receivable, inventory, and operating-accrual behavior."),
    FactorDefinition("beneish_anti_manipulation_20", "fundamental", True, ("financial_income", "financial_balancesheet", "financial_cashflow"), "Public Beneish-style anti-manipulation proxy rewarding cleaner receivable discipline, steadier gross margins, lower accrual pressure, milder asset-quality drift, and restrained leverage expansion."),
    FactorDefinition("piotroski_f_score_proxy_20", "fundamental", True, ("financial_income", "financial_balancesheet", "financial_cashflow"), "Public Piotroski-style proxy scoring profitability, cash support, improving leverage and liquidity, stronger efficiency, and lower external-financing dependence using currently available statement fields."),
    FactorDefinition("quality_strength_persistence_20", "fundamental", True, ("financial_income", "financial_balancesheet", "financial_cashflow"), "Public institutional-style quality persistence composite combining Piotroski-style financial-strength evidence, profitability/safety, lower accrual pressure, and fresher high-quality reporting."),
    FactorDefinition("distress_resilience_public_20", "fundamental", True, ("financial_income", "financial_balancesheet", "financial_cashflow"), "Public distress-resilience proxy rewarding balance-sheet slack, cashflow coverage, lower financing strain, and improving short-term solvency."),
    FactorDefinition("franchise_power_quality_20", "fundamental", True, ("financial_income", "financial_balancesheet", "financial_cashflow"), "Public franchise-quality proxy rewarding persistent profitability, cash-generative asset efficiency, margin stability, and self-funded growth."),
    FactorDefinition("residual_sponsorship_resilience_10", "flow", True, ("stock_flows", "sector_flows", "sector_closes", "sector_up_counts", "sector_total_counts", "prices_1d"), "Residual sponsorship composite that rewards stock-specific flow resilience, sector-relative sponsorship, and defensive behavior during sector stress."),
    FactorDefinition("panic_repair_sponsorship_15", "flow", True, ("opens_1d", "prices_1d", "turnover_1d", "stock_flows"), "Post-shock repair composite combining false-break reclaim quality, downside absorption, outflow retest control, and repair liquidity retention."),
    FactorDefinition("quality_sponsorship_underreaction_20", "fundamental", True, ("financial_income", "financial_balancesheet", "financial_cashflow", "stock_flows", "sector_flows", "sector_closes", "sector_up_counts", "sector_total_counts", "prices_1d"), "Cross-domain underreaction composite that blends public quality-repair evidence with residual sponsorship and sector-stress resilience."),
    FactorDefinition("qmj_defensive_sponsorship_20", "fundamental", True, ("financial_income", "financial_balancesheet", "financial_cashflow", "stock_flows", "sector_flows", "sector_closes", "sector_up_counts", "sector_total_counts", "prices_1d", "turnover_1d"), "Public QMJ-style cross-domain proxy that combines cleaner reported quality with lower downside fragility and more defensive sponsorship behavior."),
    FactorDefinition("qmj_profitability_safety_20", "fundamental", True, ("financial_income", "financial_balancesheet", "financial_cashflow", "stock_flows", "sector_flows", "sector_closes", "sector_up_counts", "sector_total_counts", "prices_1d", "turnover_1d"), "Public QMJ-style profitability-and-safety composite blending profitability, conservative investment discipline, financing safety, and defensive sponsorship confirmation."),
    FactorDefinition("institutional_quality_compound_20", "fundamental", True, ("financial_income", "financial_balancesheet", "financial_cashflow", "stock_flows", "sector_flows", "sector_closes", "sector_up_counts", "sector_total_counts", "prices_1d", "turnover_1d"), "Public institutional-style compound quality proxy blending franchise power, distress resilience, QMJ-style profitability, anti-manipulation quality, and sponsorship resilience."),
    FactorDefinition("quality_residual_sponsorship_spread_20", "fundamental", True, ("financial_income", "financial_balancesheet", "financial_cashflow", "stock_flows", "sector_flows", "sector_closes", "sector_up_counts", "sector_total_counts", "prices_1d", "benchmark_prices", "turnover_1d"), "Quality-minus-beta sponsorship spread combining public institutional quality, residual stock sponsorship, and beta-neutral relative alpha."),
    FactorDefinition("crowding_adjusted_sponsorship_escape_10", "flow", True, ("stock_flows", "sector_flows", "sector_up_counts", "sector_total_counts", "prices_1d", "benchmark_prices"), "Crowding-adjusted sponsorship escape proxy that rewards stock-specific sponsorship above sector crowding and penalizes narrowing sector breadth under inflow."),
    FactorDefinition("residual_trend_sponsorship_20", "flow", True, ("stock_flows", "sector_flows", "sector_closes", "market_closes", "prices_1d", "turnover_1d"), "Stock-specific residual momentum confirmed by sponsorship acceleration, beta-neutral alpha, trend quality, and stable turnover."),
    FactorDefinition("clean_acceptance_liquidity_bridge_20", "flow", True, ("institutional_trend_acceptance_clean_20", "liquidity_sponsorship_20"), "Interaction factor rewarding clean institutional acceptance only when it is backed by liquidity sponsorship."),
    FactorDefinition("clean_acceptance_outflow_dampening_20", "flow", True, ("institutional_trend_acceptance_clean_20", "serial_outflow_damage_dampening_12"), "Interaction factor rewarding clean institutional acceptance when repeated outflow shocks show declining price damage."),
    FactorDefinition("support_lift_flow_resilience_20", "flow", True, ("support_lift_strength_20", "flow_shock_half_life_resilience_20"), "Interaction factor rewarding lifting support that is confirmed by fast repair after negative-flow shocks."),
    FactorDefinition("compression_breakout_sponsorship_20", "trend", True, ("prices_1d", "turnover_1d", "sector_flows", "sector_up_counts", "sector_total_counts"), "Quiet-base compression and breakout readiness confirmed by retest absorption, multiscale trend coherence, and sector breadth capture."),
    FactorDefinition("supply_absorption_repair_20", "flow", True, ("opens_1d", "prices_1d", "turnover_1d", "stock_flows", "sector_closes", "market_closes"), "Supply absorption and repair composite rewarding outflow retest absorption, damage dampening, liquidity relief, and low upper-shadow distribution."),
    FactorDefinition("impact_absorption_resilience_20", "flow", True, ("prices_1d", "lows_1d", "turnover_1d", "stock_flows"), "Post-shock impact absorption proxy rewarding lower damage per negative flow, low undercut risk, reclaim strength, and cooled turnover after selloff shocks."),
    FactorDefinition("institutional_alpha_core_20", "flow", True, ("prices_1d", "turnover_1d", "stock_flows", "sector_flows", "sector_closes", "market_closes", "sector_up_counts", "sector_total_counts", "financial_income", "financial_balancesheet", "financial_cashflow"), "Composite daily alpha core combining residual trend sponsorship, compression breakout readiness, quality-minus-beta sponsorship, supply absorption repair, and latent trend signal-to-noise."),
    FactorDefinition("tail_adjusted_residual_alpha_20", "flow", True, ("prices_1d", "turnover_1d", "stock_flows", "sector_flows", "sector_closes", "market_closes", "sector_up_counts", "sector_total_counts", "financial_income", "financial_balancesheet", "financial_cashflow"), "Tail-risk-adjusted residual alpha composite combining residual sponsorship, public quality, beta-neutral drift, state-space trend signal, and penalties for downside volatility, crowding, exhaustion, beta, and distribution pressure."),
    FactorDefinition("weak_market_resilience_20", "risk", True, ("prices_1d", "opens_1d", "turnover_1d", "stock_flows", "sector_flows", "sector_closes", "sector_up_counts", "sector_total_counts", "financial_income", "financial_balancesheet", "financial_cashflow"), "Weak-market resilience composite rewarding quality residual sponsorship, drawdown resilience, negative-flow damage dampening, and stable support while penalizing distribution churn."),
    FactorDefinition("weak_market_reaccumulation_alpha_20", "flow", True, ("prices_1d", "opens_1d", "turnover_1d", "stock_flows", "sector_flows", "sector_closes", "market_closes", "sector_up_counts", "sector_total_counts", "financial_income", "financial_balancesheet", "financial_cashflow"), "Weak-market reaccumulation alpha composite combining quality residual sponsorship, sector diffusion repair, stock-specific breadth leadership, selloff absorption, and explicit tail/crowding/beta penalties."),
    FactorDefinition("orthogonalized_residual_alpha_20", "flow", True, ("prices_1d", "turnover_1d", "stock_flows", "sector_flows", "sector_closes", "market_closes", "sector_up_counts", "sector_total_counts", "financial_income", "financial_balancesheet", "financial_cashflow"), "Residual alpha composite that rewards quality sponsorship and reaccumulation after penalizing broad momentum, beta, volatility, and crowding exposures."),
    FactorDefinition("flow_absorption_half_life_15", "flow", True, ("prices_1d", "opens_1d", "turnover_1d", "stock_flows", "sector_closes", "market_closes"), "Supply absorption half-life proxy rewarding names where repeated outflows do progressively less price damage and repair follows through without distribution."),
    FactorDefinition("institutional_orthogonal_reaccumulation_20", "flow", True, ("prices_1d", "opens_1d", "turnover_1d", "stock_flows", "sector_flows", "sector_closes", "market_closes", "sector_up_counts", "sector_total_counts", "financial_income", "financial_balancesheet", "financial_cashflow"), "Institutional reaccumulation alpha combining orthogonal residual alpha, supply absorption speed, repair quality, trend signal-to-noise, and diffusion while penalizing failed breakouts."),
    FactorDefinition("oscillation_repair_quality_20", "trend", True, ("prices_1d", "opens_1d", "turnover_1d", "stock_flows"), "Oscillation repair composite for range-bound markets, rewarding low-volume support, failed-breakdown reclaim, outflow retest absorption, and low failed-reclaim persistence."),
    FactorDefinition("defensive_alpha_core_20", "risk", True, ("prices_1d", "opens_1d", "turnover_1d", "stock_flows", "sector_flows", "sector_closes", "sector_up_counts", "sector_total_counts", "financial_income", "financial_balancesheet", "financial_cashflow"), "Defensive daily alpha core combining weak-market resilience, oscillation repair quality, supply absorption, quality sponsorship, and public distress resilience."),
    FactorDefinition("calm_residual_alpha_20", "flow", True, ("prices_1d", "turnover_1d", "stock_flows", "sector_flows", "sector_closes", "market_closes", "financial_income", "financial_balancesheet", "financial_cashflow"), "Calm residual alpha composite rewarding quality residual sponsorship, stable turnover, support strength, flow absorption, and lower realized/downside volatility."),
)

FACTOR_DEFINITIONS = {item.name: item for item in _FACTOR_DEFINITIONS}

_FACTOR_ALIASES = {
    "momentum_return_5": "momentum_5",
    "momentum_20": "price_momentum_20",
    "momentum_return_20": "price_momentum_20",
    "momentum_60": "price_momentum_60",
    "momentum_return_60": "price_momentum_60",
    "price_vs_ma20": "ma_gap_20",
    "price_vs_ma_20": "ma_gap_20",
    "price_vs_ma60": "ma_gap_60",
    "price_vs_ma_60": "ma_gap_60",
    "compression_ratio_20_60": "volatility_compression_20",
    "realized_vol_20": "realized_volatility_20",
    "downside_vol_20": "downside_volatility_20",
    "turnover_ratio_5_20": "turnover_impulse_5_20",
    "turnover_zscore_20": "abnormal_turnover_z20",
    "amihud_liquidity_20": "amihud_illiquidity_20",
}

ALPHA_DISCOVERY_EXCLUDED_FAMILIES = {
    "crowding",
    "event",
    "regime",
    "risk",
    "sentiment",
}

ALPHA_DISCOVERY_EXCLUDED_FACTORS = {
    "relative_strength_20",
    "turnover_impulse_5_20",
    "abnormal_turnover_z20",
    "flow_proxy_strength_10",
    "turnover_return_corr_20",
    "flow_correlation_10",
    "flow_price_divergence_10",
    "sector_net_inflow_5_20",
    "market_breadth_support_10",
    "sentiment_tailwind_10",
    "event_consistency_20d",
}

ALPHA_DISCOVERY_INCLUDED_RISK_FACTORS = {
    "weak_market_resilience_20",
    "defensive_alpha_core_20",
}


def canonical_factor_name(name: str) -> str:
    key = str(name or "").strip()
    if not key:
        return ""
    return _FACTOR_ALIASES.get(key, key)


def factor_definition(name: str) -> FactorDefinition | None:
    return FACTOR_DEFINITIONS.get(canonical_factor_name(name))


def factor_family(name: str) -> str:
    definition = factor_definition(name)
    return str(definition.family).strip().lower() if definition is not None else ""


def factor_names_requiring_dependencies(*dependencies: str) -> set[str]:
    required = {str(item).strip() for item in dependencies if str(item).strip()}
    if not required:
        return set()
    return {
        item.name
        for item in _FACTOR_DEFINITIONS
        if required.intersection(item.dependencies)
    }


def is_alpha_discovery_factor(name: str) -> bool:
    normalized = canonical_factor_name(name)
    if not normalized:
        return False
    if factor_definition(normalized) is None:
        return False
    if normalized in ALPHA_DISCOVERY_EXCLUDED_FACTORS:
        return False
    if normalized in ALPHA_DISCOVERY_INCLUDED_RISK_FACTORS:
        return True
    return factor_family(normalized) not in ALPHA_DISCOVERY_EXCLUDED_FAMILIES


def factor_catalog() -> list[dict[str, object]]:
    return [
        {
            "name": item.name,
            "family": item.family,
            "higher_is_better": item.higher_is_better,
            "dependencies": list(item.dependencies),
            "description": item.description,
        }
        for item in _FACTOR_DEFINITIONS
    ]
