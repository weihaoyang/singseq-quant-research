# ruff: noqa
from __future__ import annotations

from factors_catalog.r441_oracle.factor_math import (
    _max_drawdown,
    _mean,
    _returns,
    _safe_div,
    _zscore,
)
from factors_catalog.r441_oracle.factor_models import FactorInputs
from factors_catalog.r441_oracle.factors import (
    advance_participation_uniformity_20,
    amihud_illiquidity_20,
    box_ceiling_residence_20,
    bullish_engulfing_hold_10,
    breakout_failure_repair_15,
    breakout_digestion_hold_10,
    breakout_close_control_10,
    breakout_pressure_10,
    breakout_quality_10,
    breakout_retest_absorption_20,
    asymmetric_autocorr_repair_20,
    drawdown_recovery_convexity_20,
    entropy_compression_breakout_potential_20,
    steady_trend_advancement_20,
    fractal_trend_persistence_20,
    spectral_trend_coherence_20,
    state_space_trend_snr_20,
    compression_persistence_20,
    compression_release_quality_15,
    delayed_undercut_reclaim_12,
    distribution_warning_5,
    drawdown_recovery_20,
    exhaustion_risk_10,
    failed_breakout_recoil_repair_15,
    failed_breakout_reversal_intensity_12,
    failed_breakout_scarcity_20,
    failed_reclaim_persistence_15,
    false_breakdown_repair_10,
    gap_fill_acceptance_8,
    gap_failure_containment_10,
    highs_stepup_consistency_20,
    innovation_drift_absorption_20,
    path_efficiency_tail_resilience_20,
    tail_repair_skew_balance_20,
    trend_liquidity_persistence_20,
    volatility_of_volatility_contraction_20,
    weak_trend_reaccumulation_spring_20,
    large_candle_followthrough_5,
    low_volume_box_support_20,
    lower_shadow_absorption_5,
    liquidity_sponsorship_20,
    momentum_efficiency_20,
    panic_followthrough_decay_12,
    panic_followthrough_repair_8,
    pullback_absorption_10,
    pullback_energy_decay_20,
    reaction_high_retake_quality_15,
    breakout_retention_15,
    recrowding_reacceleration_quality_12,
    resistance_memory_decay_20,
    recovery_efficiency_20,
    repair_base_tightness_12,
    repair_liquidity_retention_10,
    shrink_volume_nonbreak_5_20,
    support_lift_strength_20,
    support_test_reclaim_15,
    turnover_impulse_5_20,
    trend_acceleration_20,
    trend_quality_20,
    trend_stability_15,
    upper_half_churn_trap_20,
    upper_shadow_distribution_pressure_10,
    volume_price_confirmation_10,
    flat_top_higher_low_coil_20,
)
from factors_catalog.r441_oracle.factor_legacy_indicators import (
    _alpha,
    _beta,
    _breakout_distance,
    _compression_ratio,
    _drawdown_from_high,
    _drawdown_resilience,
    _downside_volatility,
    _flow_proxy_strength,
    _momentum,
    _price_vs_ma,
    _range_ratio,
    _realized_volatility,
    _relative_strength,
    _sharpe_like,
    _turnover_return_correlation,
    _turnover_stability,
    _turnover_zscore,
    _vol_adjusted_momentum,
)


def _resolve_end(length: int, end_index: int | None) -> int:
    if end_index is None:
        return length
    return max(0, min(int(end_index), length))


def _slice_window(values: list[float], end: int, window: int) -> list[float]:
    if window <= 0 or end <= 0:
        return []
    start = max(0, end - window)
    return values[start:end]


def _window_start(end: int, window: int) -> int:
    return max(0, end - window)


def _momentum_at(closes: list[float], end: int, window: int) -> float:
    if end <= window:
        return 0.0
    base = closes[end - 1 - window]
    return _safe_div(closes[end - 1] - base, base)


def _price_vs_ma_at(closes: list[float], end: int, window: int) -> float:
    if end < window:
        return 0.0
    start = _window_start(end, window)
    count = end - start
    if count <= 0:
        return 0.0
    total = 0.0
    for index in range(start, end):
        total += closes[index]
    ma = total / count
    return _safe_div(closes[end - 1] - ma, ma)


def _ma_spread_at(closes: list[float], end: int, short_window: int, long_window: int) -> float:
    short_ma = _price_vs_ma_at(closes, end, short_window)
    long_ma = _price_vs_ma_at(closes, end, long_window)
    if end < long_window:
        return 0.0
    short_start = _window_start(end, short_window)
    long_start = _window_start(end, long_window)
    short_avg = _mean(closes[short_start:end])
    long_avg = _mean(closes[long_start:end])
    return _safe_div(short_avg - long_avg, long_avg)


def _breakout_distance_at(closes: list[float], highs: list[float], end: int, window: int) -> float:
    if end <= window:
        return 0.0
    reference_window = highs[end - 1 - window : end - 1]
    if not reference_window:
        return 0.0
    reference = max(reference_window)
    return _safe_div(closes[end - 1] - reference, reference)


def _drawdown_from_high_at(closes: list[float], end: int, window: int) -> float:
    if end < window:
        return 0.0
    start = _window_start(end, window)
    high_water = closes[start]
    for index in range(start + 1, end):
        value = closes[index]
        if value > high_water:
            high_water = value
    return _safe_div(closes[end - 1] - high_water, high_water)


def _realized_volatility_from_returns(returns: list[float], end: int, window: int) -> float:
    if end - 1 < window:
        return 0.0
    stop = end - 1
    start = _window_start(stop, window)
    count = stop - start
    if count < 2:
        return 0.0
    total = 0.0
    for index in range(start, stop):
        total += returns[index]
    mean = total / count
    variance = 0.0
    for index in range(start, stop):
        delta = returns[index] - mean
        variance += delta * delta
    variance /= count
    return variance**0.5


def _downside_volatility_from_returns(returns: list[float], end: int, window: int) -> float:
    if end - 1 < window:
        return 0.0
    stop = end - 1
    start = _window_start(stop, window)
    count = 0
    mean = 0.0
    m2 = 0.0
    for index in range(start, stop):
        value = returns[index]
        if value < 0.0:
            count += 1
            delta = value - mean
            mean += delta / count
            delta2 = value - mean
            m2 += delta * delta2
    if count < 2:
        return 0.0
    return (m2 / count) ** 0.5


def _range_ratio_at(highs: list[float], lows: list[float], closes: list[float], end: int, window: int) -> float:
    if end < window:
        return 0.0
    start = end - window
    total = 0.0
    count = 0
    for index in range(start, end):
        close = closes[index]
        total += _safe_div(highs[index] - lows[index], close if close > 0 else 1.0)
        count += 1
    return _safe_div(total, count)


def _range_position_at(highs: list[float], lows: list[float], closes: list[float], end: int, window: int) -> float:
    if end < window:
        return 0.0
    recent_highs = highs[end - window : end]
    recent_lows = lows[end - window : end]
    high = max(recent_highs)
    low = min(recent_lows)
    if high <= low:
        return 0.0
    return _safe_div(closes[end - 1] - low, high - low) - 0.5


def _rebound_from_low_at(lows: list[float], closes: list[float], end: int, window: int) -> float:
    if end < window:
        return 0.0
    low = min(lows[end - window : end])
    return _safe_div(closes[end - 1] - low, low)


def _channel_width_at(highs: list[float], lows: list[float], closes: list[float], end: int, window: int) -> float:
    if end < window:
        return 0.0
    high = max(highs[end - window : end])
    low = min(lows[end - window : end])
    close = closes[end - 1]
    return _safe_div(high - low, close if close > 0 else 1.0)


def _compression_ratio_at(highs: list[float], lows: list[float], closes: list[float], end: int, short: int, long: int) -> float:
    if end < long:
        return 0.0
    short_range = _range_ratio_at(highs, lows, closes, end, short)
    long_range = _range_ratio_at(highs, lows, closes, end, long)
    return _safe_div(long_range - short_range, abs(long_range) + 1e-12)


def _volatility_ratio_at(close_returns: list[float], end: int, short_window: int, long_window: int) -> float:
    short_vol = _realized_volatility_from_returns(close_returns, end, short_window)
    long_vol = _realized_volatility_from_returns(close_returns, end, long_window)
    return _safe_div(short_vol - long_vol, long_vol if long_vol > 1e-12 else 1.0)


def _turnover_zscore_at(turnover: list[float], end: int, window: int) -> float:
    if end <= window:
        return 0.0
    base = turnover[end - 1 - window : end - 1]
    return _zscore(turnover[end - 1], base)


def _turnover_stability_at(turnover: list[float], end: int, window: int) -> float:
    if end < window:
        return 0.0
    start = _window_start(end, window)
    count = end - start
    if count <= 0:
        return 0.0
    total = 0.0
    for index in range(start, end):
        total += turnover[index]
    avg = total / count
    variance = 0.0
    for index in range(start, end):
        delta = turnover[index] - avg
        variance += delta * delta
    variance /= count
    std = variance**0.5
    return _safe_div(avg, std if std > 1e-12 else 1.0)


def _trend_streak_at(closes: list[float], end: int, window: int) -> float:
    recent_returns = _slice_window(_returns(closes[:end]), len(_returns(closes[:end])), window)
    if len(recent_returns) < window:
        return 0.0
    positive = sum(1 for value in recent_returns if value > 0)
    negative = sum(1 for value in recent_returns if value < 0)
    return _safe_div(positive - negative, len(recent_returns))


def _intraday_strength_at(opens: list[float], closes: list[float], end: int) -> float:
    if end <= 0 or len(opens) < end or len(closes) < end:
        return 0.0
    open_px = opens[end - 1]
    return _safe_div(closes[end - 1] - open_px, open_px)


def _gap_strength_at(opens: list[float], closes: list[float], end: int) -> float:
    if end < 2 or len(opens) < end or len(closes) < end:
        return 0.0
    prev_close = closes[end - 2]
    return _safe_div(opens[end - 1] - prev_close, prev_close)


def _flow_proxy_strength_at(close_returns: list[float], turnover: list[float], end: int, window: int) -> float:
    if end - 1 < window or end < window:
        return 0.0
    stop_returns = end - 1
    start_returns = _window_start(stop_returns, window)
    stop_turnover = end
    start_turnover = _window_start(stop_turnover, window)
    count = min(stop_returns - start_returns, stop_turnover - start_turnover)
    if count <= 0:
        return 0.0
    total_flow = 0.0
    turnover_sum = 0.0
    for offset in range(count):
        ret_value = close_returns[start_returns + offset]
        turnover_value = turnover[start_turnover + offset]
        total_flow += ret_value * turnover_value
        turnover_sum += turnover_value
    mean_turnover = turnover_sum / count
    mean_flow = total_flow / count
    return _safe_div(mean_flow, abs(mean_turnover) + 1.0)


def _turnover_return_correlation_at(close_returns: list[float], turnover_returns: list[float], end: int, window: int) -> float:
    if end - 1 < window:
        return 0.0
    stop = end - 1
    close_start = _window_start(stop, window)
    turnover_start = _window_start(stop, window)
    count = min(stop - close_start, stop - turnover_start)
    if count < 3:
        return 0.0
    close_mean = 0.0
    turnover_mean = 0.0
    for offset in range(count):
        close_mean += close_returns[close_start + offset]
        turnover_mean += turnover_returns[turnover_start + offset]
    close_mean /= count
    turnover_mean /= count
    cov = 0.0
    close_var = 0.0
    turnover_var = 0.0
    for offset in range(count):
        close_delta = close_returns[close_start + offset] - close_mean
        turnover_delta = turnover_returns[turnover_start + offset] - turnover_mean
        cov += close_delta * turnover_delta
        close_var += close_delta * close_delta
        turnover_var += turnover_delta * turnover_delta
    if close_var <= 1e-12 or turnover_var <= 1e-12:
        return 0.0
    return cov / ((close_var**0.5) * (turnover_var**0.5))


def _relative_strength_at(closes: list[float], benchmark_closes: list[float], end: int, benchmark_end: int, window: int) -> float:
    if end <= window or benchmark_end <= window:
        return 0.0
    stock_ret = _momentum_at(closes, end, window)
    bench_ret = _momentum_at(benchmark_closes, benchmark_end, window)
    return stock_ret - bench_ret


def _beta_from_returns(close_returns: list[float], benchmark_returns: list[float], end: int, benchmark_end: int, window: int) -> float:
    if end - 1 < window or benchmark_end - 1 < window:
        return 0.0
    stock_stop = end - 1
    bench_stop = benchmark_end - 1
    stock_start = _window_start(stock_stop, window)
    bench_start = _window_start(bench_stop, window)
    count = min(stock_stop - stock_start, bench_stop - bench_start)
    if count < 5:
        return 0.0
    stock_mean = 0.0
    bench_mean = 0.0
    for offset in range(count):
        stock_mean += close_returns[stock_start + offset]
        bench_mean += benchmark_returns[bench_start + offset]
    stock_mean /= count
    bench_mean /= count
    cov = 0.0
    bench_var = 0.0
    for offset in range(count):
        stock_delta = close_returns[stock_start + offset] - stock_mean
        bench_delta = benchmark_returns[bench_start + offset] - bench_mean
        cov += stock_delta * bench_delta
        bench_var += bench_delta * bench_delta
    if bench_var <= 1e-12:
        return 0.0
    cov /= count
    bench_var /= count
    return cov / bench_var


def _alpha_at(
    closes: list[float],
    benchmark_closes: list[float],
    close_returns: list[float],
    benchmark_returns: list[float],
    end: int,
    benchmark_end: int,
    window: int,
    beta_window: int,
) -> float:
    if end <= window or benchmark_end <= window:
        return 0.0
    beta = _beta_from_returns(close_returns, benchmark_returns, end, benchmark_end, beta_window)
    return _momentum_at(closes, end, window) - beta * _momentum_at(benchmark_closes, benchmark_end, window)


def _vol_adjusted_momentum_at(closes: list[float], close_returns: list[float], end: int, window: int) -> float:
    momentum = _momentum_at(closes, end, window)
    volatility = _realized_volatility_from_returns(close_returns, end, window)
    return _safe_div(momentum, volatility if volatility > 1e-12 else 1.0)


def _sharpe_like_from_returns(close_returns: list[float], end: int, window: int) -> float:
    if end - 1 < window:
        return 0.0
    stop = end - 1
    start = _window_start(stop, window)
    count = stop - start
    if count < window:
        return 0.0
    total = 0.0
    for index in range(start, stop):
        total += close_returns[index]
    mean = total / count
    variance = 0.0
    for index in range(start, stop):
        delta = close_returns[index] - mean
        variance += delta * delta
    variance /= count
    std = variance**0.5
    return _safe_div(mean, std if std > 1e-12 else 1.0)


def _drawdown_resilience_at(closes: list[float], end: int, window: int) -> float:
    if end < window:
        return 0.0
    return -_max_drawdown(_slice_window(closes, end, window))


def _return_skew_at(close_returns: list[float], end: int, window: int) -> float:
    if end - 1 < window:
        return 0.0
    stop = end - 1
    start = _window_start(stop, window)
    segment = close_returns[start:stop]
    if len(segment) < window:
        return 0.0
    mean = _mean(segment)
    std = _realized_volatility_from_returns(close_returns, end, window)
    if std <= 1e-12:
        return 0.0
    return _mean([((value - mean) / std) ** 3 for value in segment])


def _with_legacy_aliases(values: dict[str, float]) -> dict[str, float]:
    aliased = dict(values)
    aliases = {
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
    for alias, canonical in aliases.items():
        if canonical in values:
            aliased[alias] = values[canonical]
    return aliased


def compute_common_factor_values(inputs: FactorInputs) -> dict[str, float]:
    closes = inputs.closes
    highs = inputs.highs if len(inputs.highs) == len(closes) else list(closes)
    lows = inputs.lows if len(inputs.lows) == len(closes) else list(closes)
    turnover = inputs.turnover
    benchmark_closes = inputs.benchmark_closes
    end = _resolve_end(len(closes), inputs.end_index)
    benchmark_end = _resolve_end(len(benchmark_closes), inputs.benchmark_end_index)
    if inputs.close_returns is not None:
        close_returns = inputs.close_returns
        close_returns_end = _resolve_end(len(close_returns), inputs.close_returns_end_index)
    else:
        close_returns = _returns(closes[:end])
        close_returns_end = len(close_returns)
    if inputs.turnover_returns is not None:
        turnover_returns = inputs.turnover_returns
        turnover_returns_end = _resolve_end(len(turnover_returns), inputs.turnover_returns_end_index)
    else:
        turnover_returns = _returns(turnover[:end])
        turnover_returns_end = len(turnover_returns)
    if inputs.benchmark_returns is not None:
        benchmark_returns = inputs.benchmark_returns
        benchmark_returns_end = _resolve_end(len(benchmark_returns), inputs.benchmark_returns_end_index)
    else:
        benchmark_returns = _returns(benchmark_closes[:benchmark_end])
        benchmark_returns_end = len(benchmark_returns)
    close_end = min(end, close_returns_end + 1)
    turnover_returns_end_as_close = min(end, turnover_returns_end + 1)
    benchmark_returns_end_as_close = min(benchmark_end, benchmark_returns_end + 1)
    turnover_correlation_end = min(close_end, turnover_returns_end_as_close)
    values = {
        **_price_factor_values(opens=inputs.opens, closes=closes, highs=highs, lows=lows, turnover=turnover, end=end),
        **_risk_factor_values(closes=closes, highs=highs, lows=lows, close_returns=close_returns, close_end=close_end, end=end),
        **_liquidity_factor_values(
            closes=closes,
            turnover=turnover,
            close_returns=close_returns,
            turnover_returns=turnover_returns,
            end=end,
            close_end=close_end,
            turnover_correlation_end=turnover_correlation_end,
        ),
        **_relative_factor_values(
            closes,
            benchmark_closes,
            close_returns,
            benchmark_returns,
            end,
            benchmark_end,
            close_end,
            benchmark_returns_end_as_close,
        ),
    }
    return _with_legacy_aliases(values)


def _price_factor_values(
    *,
    opens: list[float],
    closes: list[float],
    highs: list[float],
    lows: list[float],
    turnover: list[float],
    end: int,
) -> dict[str, float]:
    open_20 = _slice_window(opens, end, 20)
    open_18 = _slice_window(opens, end, 18)
    open_16 = _slice_window(opens, end, 16)
    open_15 = _slice_window(opens, end, 15)
    open_12 = _slice_window(opens, end, 12)
    open_10 = _slice_window(opens, end, 10)
    open_8 = _slice_window(opens, end, 8)
    open_5 = _slice_window(opens, end, 5)
    high_20 = _slice_window(highs, end, 20)
    high_18 = _slice_window(highs, end, 18)
    high_16 = _slice_window(highs, end, 16)
    high_15 = _slice_window(highs, end, 15)
    high_12 = _slice_window(highs, end, 12)
    high_10 = _slice_window(highs, end, 10)
    high_8 = _slice_window(highs, end, 8)
    high_5 = _slice_window(highs, end, 5)
    low_20 = _slice_window(lows, end, 20)
    low_18 = _slice_window(lows, end, 18)
    low_16 = _slice_window(lows, end, 16)
    low_15 = _slice_window(lows, end, 15)
    low_12 = _slice_window(lows, end, 12)
    low_10 = _slice_window(lows, end, 10)
    low_8 = _slice_window(lows, end, 8)
    low_5 = _slice_window(lows, end, 5)
    close_41 = _slice_window(closes, end, 41)
    close_22 = _slice_window(closes, end, 22)
    close_21 = _slice_window(closes, end, 21)
    close_20 = _slice_window(closes, end, 20)
    close_18 = _slice_window(closes, end, 18)
    close_16 = _slice_window(closes, end, 16)
    close_15 = _slice_window(closes, end, 15)
    close_12 = _slice_window(closes, end, 12)
    close_11 = _slice_window(closes, end, 11)
    close_10 = _slice_window(closes, end, 10)
    close_8 = _slice_window(closes, end, 8)
    close_6 = _slice_window(closes, end, 6)
    close_5 = _slice_window(closes, end, 5)
    turnover_20 = _slice_window(turnover, end, 20)
    turnover_18 = _slice_window(turnover, end, 18)
    turnover_16 = _slice_window(turnover, end, 16)
    turnover_15 = _slice_window(turnover, end, 15)
    turnover_12 = _slice_window(turnover, end, 12)
    turnover_11 = _slice_window(turnover, end, 11)
    turnover_10 = _slice_window(turnover, end, 10)
    turnover_8 = _slice_window(turnover, end, 8)
    return {
        "momentum_5": _momentum_at(closes, end, 5),
        "price_momentum_20": _momentum_at(closes, end, 20),
        "price_momentum_60": _momentum_at(closes, end, 60),
        "ma_gap_20": _price_vs_ma_at(closes, end, 20),
        "ma_gap_60": _price_vs_ma_at(closes, end, 60),
        "ma_spread_20_60": _ma_spread_at(closes, end, 20, 60),
        "trend_streak_10": _trend_streak_at(closes, end, 10),
        "intraday_strength_1": _intraday_strength_at(opens, closes, end),
        "gap_strength_1": _gap_strength_at(opens, closes, end),
        "price_position_20": _range_position_at(highs, lows, closes, end, 20),
        "range_position_20": _range_position_at(highs, lows, closes, end, 20),
        "breakout_distance_20": _breakout_distance_at(closes, closes, end, 20),
        "rebound_from_low_20": _rebound_from_low_at(lows, closes, end, 20),
        "drawdown_20": _drawdown_from_high_at(closes, end, 20),
        "channel_width_20": _channel_width_at(highs, lows, closes, end, 20),
        "breakout_distance_55": _breakout_distance_at(closes, highs, end, 55),
        "drawdown_from_high_20": _drawdown_from_high_at(closes, end, 20),
        "range_ratio_20": _range_ratio_at(highs, lows, closes, end, 20),
        "volatility_compression_20": _compression_ratio_at(highs, lows, closes, end, 20, 60),
        "liquidity_sponsorship_20": liquidity_sponsorship_20(turnover_20),
        "trend_acceleration_20": trend_acceleration_20(close_21),
        "breakout_quality_10": breakout_quality_10(close_11),
        "pullback_absorption_10": pullback_absorption_10(close_11),
        "momentum_efficiency_20": momentum_efficiency_20(close_21),
        "trend_quality_20": trend_quality_20(close_21),
        "trend_stability_15": trend_stability_15(close_16),
        "drawdown_recovery_20": drawdown_recovery_20(close_20),
        "recovery_efficiency_20": recovery_efficiency_20(close_20),
        "support_lift_strength_20": support_lift_strength_20(
            high_20,
            low_20,
            close_20,
        ),
        "box_ceiling_residence_20": box_ceiling_residence_20(close_20),
        "resistance_memory_decay_20": resistance_memory_decay_20(
            high_20,
            low_20,
            close_20,
            turnover_20,
        ),
        "breakout_retention_15": breakout_retention_15(
            high_15,
            close_15,
            turnover_15,
        ),
        "breakout_retest_absorption_20": breakout_retest_absorption_20(
            high_20,
            low_20,
            close_20,
            turnover_20,
        ),
        "steady_trend_advancement_20": steady_trend_advancement_20(
            high_20,
            low_20,
            close_20,
            turnover_20,
        ),
        "fractal_trend_persistence_20": fractal_trend_persistence_20(close_21),
        "spectral_trend_coherence_20": spectral_trend_coherence_20(close_21),
        "state_space_trend_snr_20": state_space_trend_snr_20(close_21),
        "path_efficiency_tail_resilience_20": path_efficiency_tail_resilience_20(close_21, turnover_20),
        "trend_liquidity_persistence_20": trend_liquidity_persistence_20(close_21, turnover_20),
        "weak_trend_reaccumulation_spring_20": weak_trend_reaccumulation_spring_20(
            high_20,
            low_20,
            close_20,
            turnover_20,
        ),
        "innovation_drift_absorption_20": innovation_drift_absorption_20(close_21, turnover_20),
        "entropy_compression_breakout_potential_20": entropy_compression_breakout_potential_20(
            high_20,
            low_20,
            close_20,
            turnover_20,
        ),
        "volatility_of_volatility_contraction_20": volatility_of_volatility_contraction_20(close_41),
        "asymmetric_autocorr_repair_20": asymmetric_autocorr_repair_20(close_22),
        "drawdown_recovery_convexity_20": drawdown_recovery_convexity_20(close_21),
        "tail_repair_skew_balance_20": tail_repair_skew_balance_20(close_21),
        "low_volume_box_support_20": low_volume_box_support_20(
            high_20,
            low_20,
            close_20,
            turnover_20,
        ),
        "compression_release_quality_15": compression_release_quality_15(
            high_15,
            low_15,
            close_15,
            turnover_15,
        ),
        "compression_persistence_20": compression_persistence_20(
            high_20,
            low_20,
            close_20,
            turnover_20,
        ),
        "upper_shadow_distribution_pressure_10": upper_shadow_distribution_pressure_10(
            open_10,
            high_10,
            low_10,
            close_10,
            turnover_10,
        ),
        "breakout_close_control_10": breakout_close_control_10(
            open_10,
            high_10,
            low_10,
            close_10,
        ),
        "gap_fill_acceptance_8": gap_fill_acceptance_8(
            open_8,
            high_8,
            low_8,
            close_8,
            turnover_8,
        ),
        "bullish_engulfing_hold_10": bullish_engulfing_hold_10(
            open_10,
            high_10,
            low_10,
            close_10,
        ),
        "breakout_digestion_hold_10": breakout_digestion_hold_10(
            high_15,
            low_15,
            close_15,
            turnover_15,
        ),
        "failed_reclaim_persistence_15": failed_reclaim_persistence_15(
            high_15,
            low_15,
            close_15,
            turnover_15,
        ),
        "support_test_reclaim_15": support_test_reclaim_15(
            high_15,
            low_15,
            close_15,
        ),
        "false_breakdown_repair_10": false_breakdown_repair_10(
            high_15,
            low_15,
            close_15,
        ),
        "shrink_volume_nonbreak_5_20": shrink_volume_nonbreak_5_20(
            close_20,
            turnover_20,
        ),
        "pullback_energy_decay_20": pullback_energy_decay_20(
            close_21,
            turnover_20,
        ),
        "highs_stepup_consistency_20": highs_stepup_consistency_20(
            high_20,
            low_20,
            close_20,
        ),
        "failed_breakout_scarcity_20": failed_breakout_scarcity_20(
            open_20,
            high_20,
            low_20,
            close_20,
            turnover_20,
        ),
        "advance_participation_uniformity_20": advance_participation_uniformity_20(close_21),
        "large_candle_followthrough_5": large_candle_followthrough_5(close_6),
        "flat_top_higher_low_coil_20": flat_top_higher_low_coil_20(
            high_20,
            low_20,
            close_20,
        ),
        "lower_shadow_absorption_5": lower_shadow_absorption_5(
            open_5,
            high_5,
            low_5,
            close_5,
        ),
        "delayed_undercut_reclaim_12": delayed_undercut_reclaim_12(
            high_15,
            low_15,
            close_15,
            turnover_15,
        ),
        "failed_breakout_reversal_intensity_12": failed_breakout_reversal_intensity_12(
            open_15,
            high_15,
            low_15,
            close_15,
            turnover_15,
        ),
        "breakout_failure_repair_15": breakout_failure_repair_15(
            open_18,
            high_18,
            low_18,
            close_18,
            turnover_18,
        ),
        "failed_breakout_recoil_repair_15": failed_breakout_recoil_repair_15(
            high_18,
            low_18,
            close_18,
            turnover_18,
        ),
        "reaction_high_retake_quality_15": reaction_high_retake_quality_15(
            high_18,
            low_18,
            close_18,
            turnover_18,
        ),
        "gap_failure_containment_10": gap_failure_containment_10(
            open_12,
            high_12,
            low_12,
            close_12,
            turnover_12,
        ),
        "repair_liquidity_retention_10": repair_liquidity_retention_10(
            high_12,
            low_12,
            close_12,
            turnover_12,
        ),
        "repair_base_tightness_12": repair_base_tightness_12(
            high_15,
            low_15,
            close_15,
            turnover_15,
        ),
        "breakout_pressure_10": breakout_pressure_10(close_12),
        "exhaustion_risk_10": exhaustion_risk_10(
            close_11,
            turnover_11,
        ),
        "upper_half_churn_trap_20": upper_half_churn_trap_20(
            open_20,
            high_20,
            low_20,
            close_20,
            turnover_20,
        ),
        "panic_followthrough_repair_8": panic_followthrough_repair_8(
            open_12,
            high_12,
            low_12,
            close_12,
            turnover_12,
        ),
        "panic_followthrough_decay_12": panic_followthrough_decay_12(
            open_16,
            high_16,
            low_16,
            close_16,
            turnover_16,
        ),
    }


def _risk_factor_values(
    *,
    closes: list[float],
    highs: list[float],
    lows: list[float],
    close_returns: list[float],
    close_end: int,
    end: int,
) -> dict[str, float]:
    return {
        "realized_volatility_20": _realized_volatility_from_returns(close_returns, close_end, 20),
        "downside_volatility_20": _downside_volatility_from_returns(close_returns, close_end, 20),
        "volatility_ratio_20_60": _volatility_ratio_at(close_returns, close_end, 20, 60),
        "return_skew_20": _return_skew_at(close_returns, close_end, 20),
        "vol_adjusted_momentum_20": _vol_adjusted_momentum_at(closes, close_returns, close_end, 20),
        "sharpe_like_20": _sharpe_like_from_returns(close_returns, close_end, 20),
        "drawdown_resilience_20": _drawdown_resilience_at(closes, end, 20),
    }


def _liquidity_factor_values(
    *,
    closes: list[float],
    turnover: list[float],
    close_returns: list[float],
    turnover_returns: list[float],
    end: int,
    close_end: int,
    turnover_correlation_end: int,
) -> dict[str, float]:
    return {
        "turnover_impulse_5_20": turnover_impulse_5_20(_slice_window(turnover, end, 20)),
        "abnormal_turnover_z20": _turnover_zscore_at(turnover, end, 20),
        "turnover_stability_20": _turnover_stability_at(turnover, end, 20),
        "amihud_illiquidity_20": amihud_illiquidity_20(
            _slice_window(closes, end, 21),
            _slice_window(turnover, end, 21),
        ),
        "flow_proxy_strength_10": _flow_proxy_strength_at(close_returns, turnover, close_end, 10),
        "turnover_return_corr_20": _turnover_return_correlation_at(close_returns, turnover_returns, turnover_correlation_end, 20),
        "volume_price_confirmation_10": volume_price_confirmation_10(
            _slice_window(closes, end, 11),
            _slice_window(turnover, end, 11),
        ),
        "distribution_warning_5": distribution_warning_5(
            _slice_window(closes, end, 6),
            _slice_window(turnover, end, 5),
        ),
    }


def _relative_factor_values(
    closes: list[float],
    benchmark_closes: list[float],
    close_returns: list[float],
    benchmark_returns: list[float],
    end: int,
    benchmark_end: int,
    close_end: int,
    benchmark_returns_end_as_close: int,
) -> dict[str, float]:
    return {
        "relative_strength_20": _relative_strength_at(closes, benchmark_closes, end, benchmark_end, 20),
        "relative_strength_60": _relative_strength_at(closes, benchmark_closes, end, benchmark_end, 60),
        "beta_60": _beta_from_returns(close_returns, benchmark_returns, close_end, benchmark_returns_end_as_close, 60),
        "alpha_20_beta_60": _alpha_at(closes, benchmark_closes, close_returns, benchmark_returns, close_end, benchmark_returns_end_as_close, 20, 60),
    }
