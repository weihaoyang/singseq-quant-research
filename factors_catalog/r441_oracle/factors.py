# ruff: noqa
from __future__ import annotations

from collections.abc import Sequence
from math import log

from factors_catalog.r441_oracle.event_factors import (
    board_break_rate,
    crowding_event_conflict_5d,
    crowding_risk_score,
    event_consistency_20d,
    event_recency_decay_30m,
    event_sector_diffusion_5d,
    event_surprise_score_5,
    gap_exhaustion_score,
    limit_up_continuation_rate,
    market_breadth_support_10,
    panic_reversal_score,
    policy_alignment_score_10d,
    regulatory_risk_penalty_20d,
    sentiment_tailwind_10,
    shareholder_alignment_60d,
)

def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    return (_mean([(value - avg) ** 2 for value in values])) ** 0.5


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    size = min(len(left), len(right))
    if size < 2:
        return 0.0
    x = [float(value) for value in left[-size:]]
    y = [float(value) for value in right[-size:]]
    mean_x = _mean(x)
    mean_y = _mean(y)
    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    for x_value, y_value in zip(x, y):
        x_delta = x_value - mean_x
        y_delta = y_value - mean_y
        cov += x_delta * y_delta
        var_x += x_delta * x_delta
        var_y += y_delta * y_delta
    if var_x <= 1e-12 or var_y <= 1e-12:
        return 0.0
    return cov / ((var_x ** 0.5) * (var_y ** 0.5))


def trend_acceleration_20(prices: Sequence[float]) -> float:
    if len(prices) < 21:
        return 0.0
    returns = [_safe_div(prices[index + 1] - prices[index], prices[index]) for index in range(len(prices) - 1)]
    recent = returns[-20:]
    early = recent[:10]
    late = recent[10:]
    early_slope = _safe_div(early[-1] - early[0], max(len(early) - 1, 1))
    late_slope = _safe_div(late[-1] - late[0], max(len(late) - 1, 1))
    return late_slope - early_slope


def breakout_quality_10(prices: Sequence[float]) -> float:
    if len(prices) < 11:
        return 0.0
    window = prices[-11:]
    breakout = max(window[:-1])
    close = window[-1]
    return _safe_div(close - breakout, breakout)


def price_momentum_20(prices: Sequence[float]) -> float:
    if len(prices) < 21:
        return 0.0
    base = prices[-21]
    return _safe_div(prices[-1] - base, base)


def price_momentum_60(prices: Sequence[float]) -> float:
    if len(prices) < 61:
        return 0.0
    base = prices[-61]
    return _safe_div(prices[-1] - base, base)


def ma_gap_20(prices: Sequence[float]) -> float:
    if len(prices) < 20:
        return 0.0
    average = _mean(prices[-20:])
    return _safe_div(prices[-1] - average, average)


def ma_gap_60(prices: Sequence[float]) -> float:
    if len(prices) < 60:
        return 0.0
    average = _mean(prices[-60:])
    return _safe_div(prices[-1] - average, average)


def price_position_20(prices: Sequence[float]) -> float:
    if len(prices) < 20:
        return 0.0
    window = prices[-20:]
    high = max(window)
    low = min(window)
    if high <= low:
        return 0.0
    position = _safe_div(window[-1] - low, high - low)
    return max(min(position - 0.5, 0.5), -0.5)


def pullback_absorption_10(prices: Sequence[float]) -> float:
    if len(prices) < 11:
        return 0.0
    window = prices[-11:]
    max_price = max(window)
    min_price = min(window)
    close = window[-1]
    if max_price == min_price:
        return 0.0
    retrace_ratio = _safe_div(max_price - close, max_price - min_price)
    return 1.0 - retrace_ratio


def liquidity_sponsorship_20(turnover: Sequence[float]) -> float:
    if len(turnover) < 20:
        return 0.0
    recent = turnover[-20:]
    baseline = sum(recent[:10]) / 10.0
    latest = sum(recent[10:]) / 10.0
    return _safe_div(latest - baseline, baseline)


def exit_deterioration_score_5(prices: Sequence[float]) -> float:
    if len(prices) < 6:
        return 0.0
    window = prices[-6:]
    returns = [_safe_div(window[index + 1] - window[index], window[index]) for index in range(5)]
    negative = [value for value in returns if value < 0]
    if not negative:
        return 0.0
    return abs(sum(negative))


def volatility_compression_20(prices: Sequence[float]) -> float:
    if len(prices) < 40:
        return 0.0
    returns = [_safe_div(prices[index + 1] - prices[index], prices[index]) for index in range(len(prices) - 1)]
    if len(returns) < 40:
        return 0.0
    pre = returns[-40:-20]
    recent = returns[-20:]
    pre_vol = _std(pre)
    recent_vol = _std(recent)
    return max(_safe_div(pre_vol - recent_vol, pre_vol), 0.0)


def realized_volatility_20(prices: Sequence[float]) -> float:
    if len(prices) < 21:
        return 0.0
    returns = [_safe_div(prices[index + 1] - prices[index], prices[index]) for index in range(len(prices) - 1)]
    return _std(returns[-20:])


def downside_volatility_20(prices: Sequence[float]) -> float:
    if len(prices) < 21:
        return 0.0
    returns = [_safe_div(prices[index + 1] - prices[index], prices[index]) for index in range(len(prices) - 1)]
    downside = [min(value, 0.0) for value in returns[-20:]]
    if not downside:
        return 0.0
    return (_mean([value * value for value in downside])) ** 0.5


def relative_strength_60(prices: Sequence[float], benchmark_prices: Sequence[float]) -> float:
    if len(prices) < 60 or len(benchmark_prices) < 60:
        return 0.0
    stock_ret = _safe_div(prices[-1] - prices[-60], prices[-60])
    bench_ret = _safe_div(benchmark_prices[-1] - benchmark_prices[-60], benchmark_prices[-60])
    return stock_ret - bench_ret


def residual_momentum_20_60(
    prices: Sequence[float],
    sector_closes: Sequence[float],
    market_closes: Sequence[float],
) -> float:
    size = min(len(prices), len(sector_closes), len(market_closes))
    if size < 61:
        return 0.0
    stock = [float(value) for value in prices[-61:]]
    sector = [float(value) for value in sector_closes[-61:]]
    market = [float(value) for value in market_closes[-61:]]
    if min(stock) <= 0.0 or min(sector) <= 0.0 or min(market) <= 0.0:
        return 0.0
    stock_returns = [_safe_div(stock[index + 1] - stock[index], stock[index]) for index in range(60)]
    sector_returns = [_safe_div(sector[index + 1] - sector[index], sector[index]) for index in range(60)]
    market_returns = [_safe_div(market[index + 1] - market[index], market[index]) for index in range(60)]
    sector_excess = [sector_ret - market_ret for sector_ret, market_ret in zip(sector_returns, market_returns)]
    market_mean = _mean(market_returns)
    sector_mean = _mean(sector_excess)
    stock_mean = _mean(stock_returns)
    a00 = sum((value - market_mean) ** 2 for value in market_returns) + 1e-6
    a01 = sum((market_value - market_mean) * (sector_value - sector_mean) for market_value, sector_value in zip(market_returns, sector_excess))
    a11 = sum((value - sector_mean) ** 2 for value in sector_excess) + 1e-6
    b0 = sum((market_value - market_mean) * (stock_value - stock_mean) for market_value, stock_value in zip(market_returns, stock_returns))
    b1 = sum((sector_value - sector_mean) * (stock_value - stock_mean) for sector_value, stock_value in zip(sector_excess, stock_returns))
    determinant = a00 * a11 - a01 * a01
    if abs(determinant) <= 1e-12:
        beta_market = _safe_div(b0, a00)
        beta_sector = 0.0
    else:
        beta_market = (b0 * a11 - b1 * a01) / determinant
        beta_sector = (a00 * b1 - a01 * b0) / determinant
    residuals = [
        stock_ret - stock_mean - beta_market * (market_ret - market_mean) - beta_sector * (sector_ret - sector_mean)
        for stock_ret, market_ret, sector_ret in zip(stock_returns, market_returns, sector_excess)
    ]
    recent_residual_drift = sum(residuals[-20:])
    longer_residual_drift = sum(residuals[-60:])
    residual_vol = _std(residuals[-60:])
    raw = 0.70 * recent_residual_drift + 0.30 * longer_residual_drift
    score = _safe_div(raw, residual_vol * (20 ** 0.5) + 1e-6)
    return max(min(score, 1.5), -1.5)


def momentum_efficiency_20(prices: Sequence[float]) -> float:
    if len(prices) < 21:
        return 0.0
    window = prices[-21:]
    net = window[-1] - window[0]
    path = 0.0
    for index in range(1, len(window)):
        path += abs(window[index] - window[index - 1])
    return _safe_div(net, path if path > 1e-12 else 1.0)


def trend_quality_20(prices: Sequence[float]) -> float:
    if len(prices) < 21:
        return 0.0
    momentum = price_momentum_20(prices)
    if abs(momentum) <= 1e-12:
        return 0.0
    efficiency = momentum_efficiency_20(prices)
    realized_vol = realized_volatility_20(prices)
    if realized_vol <= 1e-12:
        return max(efficiency, 0.0) if momentum > 0 else min(efficiency, 0.0)
    quality = _safe_div(momentum, realized_vol)
    return quality * max(efficiency, 0.0)


def drawdown_recovery_20(prices: Sequence[float]) -> float:
    if len(prices) < 20:
        return 0.0
    window = [float(value) for value in prices[-20:]]
    peak_index = max(range(len(window) - 1), key=lambda index: window[index])
    peak = window[peak_index]
    if peak <= 0:
        return 0.0
    trough_window = window[peak_index:]
    trough = min(trough_window)
    close = window[-1]
    if close >= peak:
        return 1.0
    if trough >= peak:
        return 0.0
    return max(0.0, min(_safe_div(close - trough, peak - trough), 1.0))


def recovery_efficiency_20(prices: Sequence[float]) -> float:
    if len(prices) < 20:
        return 0.0
    window = [float(value) for value in prices[-20:]]
    peak_index = max(range(len(window) - 1), key=lambda index: window[index])
    trough_window = window[peak_index:]
    if len(trough_window) < 2:
        return 0.0
    trough_offset = min(range(len(trough_window)), key=lambda index: trough_window[index])
    trough_index = peak_index + trough_offset
    if trough_index >= len(window) - 1:
        return 0.0
    peak = window[peak_index]
    trough = window[trough_index]
    close = window[-1]
    recovery_ratio = max(0.0, min(_safe_div(close - trough, peak - trough), 1.0))
    path = 0.0
    segment = window[trough_index:]
    for index in range(1, len(segment)):
        path += abs(segment[index] - segment[index - 1])
    path_efficiency = max(0.0, min(_safe_div(close - trough, path if path > 1e-12 else 1.0), 1.0))
    speed = max(0.0, min(1.0 - _safe_div(len(window) - 1 - trough_index, len(window)), 1.0))
    return recovery_ratio * path_efficiency * speed


def support_lift_strength_20(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]) -> float:
    if len(highs) < 20 or len(lows) < 20 or len(closes) < 20:
        return 0.0
    recent_highs = [float(value) for value in highs[-20:]]
    recent_lows = [float(value) for value in lows[-20:]]
    recent_closes = [float(value) for value in closes[-20:]]
    support_series = [min(recent_lows[index - 4 : index + 1]) for index in range(4, len(recent_lows))]
    if len(support_series) < 10:
        return 0.0
    early = _mean(support_series[:5])
    late = _mean(support_series[-5:])
    if early <= 0.0:
        return 0.0
    avg_range = _mean(
        [
            _safe_div(recent_highs[index] - recent_lows[index], recent_closes[index] if recent_closes[index] > 0 else 1.0)
            for index in range(len(recent_closes))
        ]
    )
    return max(min(_safe_div(_safe_div(late - early, early), avg_range + 1e-12), 3.0), -3.0)


def box_ceiling_residence_20(prices: Sequence[float], *, width_cap: float = 0.20) -> float:
    if len(prices) < 20:
        return 0.0
    window = [float(value) for value in prices[-20:]]
    high20 = max(window)
    low20 = min(window)
    if high20 <= low20 or low20 <= 0.0:
        return 0.0
    width = _safe_div(high20 - low20, low20)
    threshold = high20 - 0.2 * (high20 - low20)
    residence = _safe_div(sum(1 for value in window[-10:] if value >= threshold), 10.0)
    compression = max(0.0, 1.0 - _safe_div(width, width_cap))
    return residence * compression


def resistance_memory_decay_20(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(highs) < 20 or len(lows) < 20 or len(closes) < 20 or len(turnover) < 20:
        return 0.0
    recent_highs = [float(value) for value in highs[-20:]]
    recent_lows = [float(value) for value in lows[-20:]]
    recent_closes = [float(value) for value in closes[-20:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-20:]]
    box_high = max(recent_highs)
    box_low = min(recent_lows)
    width = box_high - box_low
    if width <= 1e-12:
        return 0.0
    top_threshold = box_high - width * 0.20
    turnover_baseline = _mean(recent_turnover) + 1e-12
    rejection_scores: list[float] = []
    for high_value, low_value, close_value, turnover_value in zip(
        recent_highs,
        recent_lows,
        recent_closes,
        recent_turnover,
    ):
        if high_value < top_threshold and close_value < top_threshold:
            continue
        intraday_range = high_value - low_value
        if intraday_range <= 1e-12:
            continue
        rejection = _safe_div(high_value - close_value, intraday_range)
        turnover_rel = min(max(_safe_div(turnover_value, turnover_baseline), 0.7), 1.5)
        rejection_scores.append(rejection * turnover_rel)
    if len(rejection_scores) < 4:
        return 0.0
    split = len(rejection_scores) // 2
    early = _mean(rejection_scores[:split])
    late = _mean(rejection_scores[split:])
    decay = max(_safe_div(early - late, abs(early) + abs(late) + 1e-12), 0.0)
    visit_score = min(_safe_div(len(rejection_scores), 6.0), 1.0)
    return max(min(decay * visit_score, 1.0), 0.0)


def breakout_retention_15(
    highs: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(highs) < 15 or len(closes) < 15 or len(turnover) < 15:
        return 0.0
    recent_highs = [float(value) for value in highs[-15:]]
    recent_closes = [float(value) for value in closes[-15:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-15:]]
    scores: list[float] = []
    for index in range(10, len(recent_closes)):
        prior_high = max(recent_highs[index - 10 : index])
        if prior_high <= 0.0 or recent_closes[index] <= prior_high:
            continue
        extension = max(recent_highs[index] - prior_high, recent_closes[index] - prior_high, 1e-12)
        hold = max(min(_safe_div(recent_closes[index] - prior_high, extension), 1.0), 0.0)
        turnover_base = _mean(recent_turnover[max(0, index - 5) : index]) + 1e-12
        turnover_confirm = min(max(_safe_div(recent_turnover[index], turnover_base), 0.8), 1.3)
        scores.append(hold * turnover_confirm)
    if not scores:
        return 0.0
    return max(min(_mean(scores[-3:]), 1.0), 0.0)


def breakout_retest_absorption_20(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(highs) < 20 or len(lows) < 20 or len(closes) < 20 or len(turnover) < 20:
        return 0.0
    recent_highs = [float(value) for value in highs[-20:]]
    recent_lows = [float(value) for value in lows[-20:]]
    recent_closes = [float(value) for value in closes[-20:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-20:]]
    best_score = 0.0
    for index in range(10, len(recent_closes) - 1):
        prior_high = max(recent_highs[index - 10 : index])
        if prior_high <= 0.0 or recent_closes[index] <= prior_high:
            continue
        follow_end = min(len(recent_closes), index + 6)
        follow_lows = recent_lows[index + 1 : follow_end]
        follow_closes = recent_closes[index + 1 : follow_end]
        follow_turnover = recent_turnover[index + 1 : follow_end]
        if len(follow_lows) < 2 or min(follow_lows) > prior_high * 1.01:
            continue
        close_hold = 1.0 - min(max(_safe_div(prior_high - min(follow_closes), 0.03 * prior_high), 0.0), 1.0)
        wick_hold = 1.0 - min(max(_safe_div(prior_high - min(follow_lows), 0.04 * prior_high), 0.0), 1.0)
        breakout_turnover = max(
            recent_turnover[index],
            _mean(recent_turnover[max(0, index - 5) : index]) if index > 0 else recent_turnover[index],
            1e-12,
        )
        follow_turnover_mean = _mean(follow_turnover)
        turnover_cool = max(1.0 - min(_safe_div(follow_turnover_mean, breakout_turnover), 1.0), 0.0)
        rebound = min(max(_safe_div(recent_closes[-1] - prior_high, 0.05 * prior_high), 0.0), 1.0)
        score = wick_hold * (0.4 * close_hold + 0.3 * turnover_cool + 0.3 * rebound)
        best_score = max(best_score, score)
    return max(min(best_score, 1.0), 0.0)


def steady_trend_advancement_20(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(highs) < 20 or len(lows) < 20 or len(closes) < 20 or len(turnover) < 20:
        return 0.0
    recent_highs = [float(value) for value in highs[-20:]]
    recent_lows = [float(value) for value in lows[-20:]]
    recent_closes = [float(value) for value in closes[-20:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-20:]]
    high20 = max(recent_highs)
    low20 = min(recent_lows)
    if high20 <= low20 or recent_closes[0] <= 0.0:
        return 0.0
    positive_returns = []
    negative_returns = []
    for left, right in zip(recent_closes[:-1], recent_closes[1:]):
        ret_value = _safe_div(right - left, left)
        if ret_value >= 0.0:
            positive_returns.append(ret_value)
        else:
            negative_returns.append(abs(ret_value))
    if len(positive_returns) < 5:
        return 0.0
    net_progress = max(_safe_div(recent_closes[-1] - recent_closes[0], recent_closes[0]), 0.0)
    pullback_penalty = _mean(negative_returns) if negative_returns else 0.0
    participation = min(_safe_div(len(positive_returns), 12.0), 1.0)
    upper_band_residence = _mean(
        [
            1.0 if _safe_div(close_px - low20, high20 - low20) >= 0.58 else 0.0
            for close_px in recent_closes[-8:]
        ]
    )
    turnover_base = _mean(recent_turnover[:10]) + 1e-12
    turnover_late = _mean(recent_turnover[-5:])
    turnover_discipline = max(1.0 - min(_safe_div(turnover_late, turnover_base), 1.0), 0.0)
    score = (
        min(net_progress / 0.18, 1.0)
        * (0.34 * participation + 0.33 * upper_band_residence + 0.18 * turnover_discipline + 0.15 * max(1.0 - min(pullback_penalty / 0.03, 1.0), 0.0))
    )
    return max(min(score, 1.0), 0.0)


def fractal_trend_persistence_20(prices: Sequence[float]) -> float:
    if len(prices) < 21:
        return 0.0
    window = [float(value) for value in prices[-21:]]
    if min(window) <= 0.0:
        return 0.0
    horizons = (1, 2, 4)
    scale_scores: list[float] = []
    direction_votes = 0.0
    for step in horizons:
        sampled = window[::step]
        if len(sampled) < 6:
            continue
        returns = [
            _safe_div(sampled[index + 1] - sampled[index], sampled[index])
            for index in range(len(sampled) - 1)
        ]
        if len(returns) < 4:
            continue
        net_move = max(_safe_div(sampled[-1] - sampled[0], sampled[0]), 0.0)
        path = sum(abs(sampled[index + 1] - sampled[index]) for index in range(len(sampled) - 1))
        efficiency = max(_safe_div(sampled[-1] - sampled[0], path if path > 1e-12 else 1.0), 0.0)
        persistence = _safe_div(sum(1 for value in returns if value > 0.0), len(returns))
        scale_scores.append(min(net_move / 0.12, 1.0) * (0.55 * efficiency + 0.45 * persistence))
        direction_votes += 1.0 if returns[-1] >= 0.0 and _mean(returns) >= -1e-12 else 0.0
    if not scale_scores:
        return 0.0
    multiscale_agreement = _safe_div(direction_votes, len(scale_scores))
    return max(min(_mean(scale_scores) * (0.65 + 0.35 * multiscale_agreement), 1.0), 0.0)


def spectral_trend_coherence_20(prices: Sequence[float]) -> float:
    if len(prices) < 21:
        return 0.0
    window = [float(value) for value in prices[-21:]]
    if min(window) <= 0.0:
        return 0.0
    returns = [
        _safe_div(window[index + 1] - window[index], window[index])
        for index in range(len(window) - 1)
    ]
    if len(returns) < 8:
        return 0.0
    mean_ret = _mean(returns)
    centered = [value - mean_ret for value in returns]
    total_energy = sum(value * value for value in centered)
    if total_energy <= 1e-12:
        return 0.0
    lag1 = sum(centered[index] * centered[index - 1] for index in range(1, len(centered)))
    lag2 = sum(centered[index] * centered[index - 2] for index in range(2, len(centered)))
    low_freq_share = max((lag1 + lag2) / (total_energy + 1e-12), 0.0)
    sign_consistency = _safe_div(sum(1 for value in returns if value > 0.0), len(returns))
    smoothness = max(1.0 - min(_std(returns) / (abs(mean_ret) + 0.01), 1.0), 0.0)
    trend_strength = min(max(_safe_div(window[-1] - window[0], window[0]), 0.0) / 0.10, 1.0)
    score = trend_strength * (0.45 * min(low_freq_share, 1.0) + 0.30 * sign_consistency + 0.25 * smoothness)
    return max(min(score, 1.0), 0.0)


def state_space_trend_snr_20(prices: Sequence[float]) -> float:
    if len(prices) < 21:
        return 0.0
    window = [float(value) for value in prices[-21:]]
    if min(window) <= 0.0:
        return 0.0
    returns = [
        _safe_div(window[index + 1] - window[index], window[index])
        for index in range(len(window) - 1)
    ]
    if len(returns) < 8:
        return 0.0
    level = returns[0]
    trend = 0.0
    process_gain = 0.24
    measurement_gain = 0.42
    filtered: list[float] = []
    for value in returns:
        prediction = level + trend
        innovation = value - prediction
        level = prediction + measurement_gain * innovation
        trend = trend + process_gain * measurement_gain * innovation
        filtered.append(level)
    signal = max(_mean(filtered[-8:]), 0.0)
    residuals = [raw - smooth for raw, smooth in zip(returns[-len(filtered):], filtered)]
    residual_noise = _std(residuals[-12:]) + 1e-12
    raw_noise = _std(returns[-12:]) + 1e-12
    snr = signal / residual_noise
    noise_reduction = max(1.0 - min(residual_noise / raw_noise, 1.0), 0.0)
    positive_filtered_share = _safe_div(sum(1 for value in filtered[-10:] if value > 0.0), 10.0)
    score = min(snr / 2.4, 1.0) * (0.45 + 0.30 * noise_reduction + 0.25 * positive_filtered_share)
    return max(min(score, 1.0), 0.0)


def path_efficiency_tail_resilience_20(prices: Sequence[float], turnover: Sequence[float]) -> float:
    if len(prices) < 21:
        return 0.0
    window = [float(value) for value in prices[-21:]]
    if min(window) <= 0.0:
        return 0.0
    returns = [
        _safe_div(window[index + 1] - window[index], window[index])
        for index in range(len(window) - 1)
    ]
    if len(returns) < 12:
        return 0.0
    net_return = _safe_div(window[-1] - window[0], window[0])
    if net_return <= 0.0:
        return 0.0
    path_length = sum(abs(value) for value in returns)
    path_efficiency = max(min(_safe_div(abs(net_return), path_length + 1e-12), 1.0), 0.0)
    positive_share = _safe_div(sum(1 for value in returns if value > 0.0), len(returns))
    top_two = sum(sorted([max(value, 0.0) for value in returns], reverse=True)[:2])
    upside_total = sum(max(value, 0.0) for value in returns) + 1e-12
    upside_uniformity = max(1.0 - min(_safe_div(top_two, upside_total), 1.0), 0.0)
    peak = window[0]
    worst_drawdown = 0.0
    for price in window:
        peak = max(peak, price)
        worst_drawdown = min(worst_drawdown, _safe_div(price - peak, peak))
    drawdown_penalty = min(abs(worst_drawdown) / 0.12, 1.0)
    downside_returns = sorted(value for value in returns if value < 0.0)
    tail_loss = abs(_mean(downside_returns[: max(1, len(downside_returns) // 3)])) if downside_returns else 0.0
    tail_resilience = max(1.0 - min(tail_loss / 0.035, 1.0), 0.0)
    turnover_control = 0.5
    if len(turnover) >= 20:
        recent_turnover = [max(float(value), 0.0) for value in turnover[-20:]]
        first_half = _mean(recent_turnover[:10])
        last_half = _mean(recent_turnover[-10:])
        turnover_control = max(1.0 - min(max(_safe_div(last_half, first_half + 1e-12) - 1.0, 0.0) / 1.5, 1.0), 0.0)
    trend_strength = min(net_return / 0.14, 1.0)
    score = trend_strength * (
        0.30 * path_efficiency
        + 0.22 * tail_resilience
        + 0.18 * positive_share
        + 0.14 * upside_uniformity
        + 0.10 * turnover_control
        + 0.06 * (1.0 - drawdown_penalty)
    )
    return max(min(score, 1.0), 0.0)


def trend_liquidity_persistence_20(prices: Sequence[float], turnover: Sequence[float]) -> float:
    if len(prices) < 21 or len(turnover) < 20:
        return 0.0
    window = [float(value) for value in prices[-21:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-20:]]
    if min(window) <= 0.0:
        return 0.0
    returns = [
        _safe_div(window[index + 1] - window[index], window[index])
        for index in range(len(window) - 1)
    ]
    if len(returns) < 12:
        return 0.0
    positive_flow_days = [turn for ret, turn in zip(returns, recent_turnover) if ret > 0.0]
    negative_flow_days = [turn for ret, turn in zip(returns, recent_turnover) if ret < 0.0]
    up_participation = _safe_div(sum(positive_flow_days), sum(recent_turnover) + 1e-12)
    down_absorption = 1.0 - min(_safe_div(sum(negative_flow_days), sum(recent_turnover) + 1e-12), 1.0)
    flow_return_corr = max(_correlation(returns, recent_turnover), 0.0)
    early_turnover = _mean(recent_turnover[:10]) + 1e-12
    late_turnover = _mean(recent_turnover[-10:])
    turnover_persistence = max(1.0 - min(abs(_safe_div(late_turnover, early_turnover) - 1.0) / 1.2, 1.0), 0.0)
    trend = max(min(_safe_div(window[-1] - window[0], window[0]) / 0.14, 1.0), 0.0)
    score = trend * (
        0.34 * up_participation
        + 0.26 * down_absorption
        + 0.22 * flow_return_corr
        + 0.18 * turnover_persistence
    )
    return max(min(score, 1.0), 0.0)


def flow_shock_half_life_resilience_20(
    prices: Sequence[float],
    turnover: Sequence[float],
    flows: Sequence[float],
) -> float:
    size = min(len(prices), len(turnover), len(flows))
    if size < 20:
        return 0.0
    recent_prices = [float(value) for value in prices[-size:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-size:]]
    recent_flows = [float(value) for value in flows[-size:]]
    if min(recent_prices) <= 0.0:
        return 0.0
    returns = [
        _safe_div(recent_prices[index + 1] - recent_prices[index], recent_prices[index])
        for index in range(len(recent_prices) - 1)
    ]
    aligned_flows = recent_flows[-len(returns):]
    aligned_turnover = recent_turnover[-len(returns):]
    if len(returns) < 12:
        return 0.0
    flow_std = _std(aligned_flows) + 1e-12
    shock_indices = [idx for idx, value in enumerate(aligned_flows) if value < -0.65 * flow_std]
    if not shock_indices:
        return 0.0
    event_scores: list[float] = []
    turnover_base = _mean(aligned_turnover) + 1e-12
    for idx in shock_indices[-4:]:
        follow_returns = returns[idx : min(len(returns), idx + 5)]
        if not follow_returns:
            continue
        immediate_damage = abs(min(follow_returns[0], 0.0))
        later_damage = abs(min(_mean(follow_returns[1:]) if len(follow_returns) > 1 else 0.0, 0.0))
        damage_decay = 1.0 - min(_safe_div(later_damage, immediate_damage + 0.006), 1.0)
        repair = max(sum(follow_returns), 0.0)
        repair_score = min(repair / 0.05, 1.0)
        turnover_capacity = min(_safe_div(_mean(aligned_turnover[idx : min(len(aligned_turnover), idx + 5)]), turnover_base), 1.5) / 1.5
        event_scores.append(0.46 * damage_decay + 0.34 * repair_score + 0.20 * turnover_capacity)
    if not event_scores:
        return 0.0
    return max(min(_mean(event_scores), 1.0), 0.0)


def institutional_trend_acceptance_20(
    prices: Sequence[float],
    turnover: Sequence[float],
    flows: Sequence[float],
) -> float:
    if len(prices) < 21 or len(turnover) < 20:
        return 0.0
    snr = state_space_trend_snr_20(prices)
    path = path_efficiency_tail_resilience_20(prices, turnover)
    liquidity = trend_liquidity_persistence_20(prices, turnover)
    shock_resilience = flow_shock_half_life_resilience_20(prices[-20:], turnover[-20:], flows[-20:])
    momentum = max(price_momentum_20(prices), 0.0)
    momentum_score = min(momentum / 0.16, 1.0)
    score = (
        0.28 * snr
        + 0.24 * path
        + 0.22 * liquidity
        + 0.16 * shock_resilience
        + 0.10 * momentum_score
    )
    return max(min(score, 1.0), 0.0)


def institutional_trend_acceptance_clean_20(
    prices: Sequence[float],
    turnover: Sequence[float],
    flows: Sequence[float],
) -> float:
    if len(prices) < 61 or len(turnover) < 20:
        return 0.0
    snr = state_space_trend_snr_20(prices)
    path = path_efficiency_tail_resilience_20(prices, turnover)
    liquidity = trend_liquidity_persistence_20(prices, turnover)
    shock_resilience = flow_shock_half_life_resilience_20(prices[-20:], turnover[-20:], flows[-20:])
    momentum_20 = max(price_momentum_20(prices), 0.0)
    momentum_60 = max(price_momentum_60(prices), 0.0)
    momentum_balance = 1.0 - min(abs(momentum_20 - momentum_60) / 0.18, 1.0)
    overextension = max(min((momentum_20 - 0.18) / 0.16, 1.0), 0.0)
    turnover = [max(float(value), 0.0) for value in turnover[-20:]]
    turnover_chase = max(min((_safe_div(_mean(turnover[-5:]), _mean(turnover[:10]) + 1e-12) - 1.0) / 1.5, 1.0), 0.0)
    acceptance = (
        0.30 * snr
        + 0.24 * path
        + 0.20 * liquidity
        + 0.16 * shock_resilience
        + 0.10 * momentum_balance
    )
    penalty = 0.55 * overextension + 0.45 * turnover_chase
    return max(min(acceptance * (1.0 - 0.55 * penalty), 1.0), 0.0)


def institutional_trend_false_acceptance_risk_20(
    prices: Sequence[float],
    turnover: Sequence[float],
    flows: Sequence[float],
) -> float:
    if len(prices) < 61 or len(turnover) < 20:
        return 0.0
    snr = state_space_trend_snr_20(prices)
    path = path_efficiency_tail_resilience_20(prices, turnover)
    liquidity = trend_liquidity_persistence_20(prices, turnover)
    shock_resilience = flow_shock_half_life_resilience_20(prices[-20:], turnover[-20:], flows[-20:])
    momentum_20 = max(price_momentum_20(prices), 0.0)
    residual_proxy = max(price_momentum_20(prices) - price_momentum_60(prices), 0.0)
    turnover_window = [max(float(value), 0.0) for value in turnover[-20:]]
    turnover_chase = max(min((_safe_div(_mean(turnover_window[-5:]), _mean(turnover_window[:10]) + 1e-12) - 1.0) / 1.3, 1.0), 0.0)
    overextension = max(min((momentum_20 - 0.16) / 0.16, 1.0), 0.0)
    weak_acceptance = max(snr - 0.45 * path - 0.35 * liquidity - 0.20 * shock_resilience, 0.0)
    residual_weakness = max(1.0 - min(residual_proxy / 0.12, 1.0), 0.0)
    risk = (
        0.34 * min(weak_acceptance / 0.45, 1.0)
        + 0.26 * overextension
        + 0.22 * turnover_chase
        + 0.18 * residual_weakness * max(snr, 0.0)
    )
    return max(min(risk, 1.0), 0.0)


def weak_trend_reaccumulation_spring_20(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(highs) < 20 or len(lows) < 20 or len(closes) < 20 or len(turnover) < 20:
        return 0.0
    recent_highs = [float(value) for value in highs[-20:]]
    recent_lows = [float(value) for value in lows[-20:]]
    recent_closes = [float(value) for value in closes[-20:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-20:]]
    if min(recent_closes) <= 0.0:
        return 0.0
    high20 = max(recent_highs)
    low20 = min(recent_lows)
    box = high20 - low20
    if box <= 1e-12:
        return 0.0
    returns = [
        _safe_div(recent_closes[index + 1] - recent_closes[index], recent_closes[index])
        for index in range(len(recent_closes) - 1)
    ]
    if len(returns) < 12:
        return 0.0
    range_width = _safe_div(box, recent_closes[-1])
    compression = max(1.0 - min(range_width / 0.24, 1.0), 0.0)
    low_lift = max(_safe_div(_mean(recent_lows[-5:]) - _mean(recent_lows[:5]), box), 0.0)
    low_lift_score = min(low_lift / 0.22, 1.0)
    upper_reclaim = _mean([_safe_div(close_px - low20, box) for close_px in recent_closes[-5:]])
    upper_reclaim_score = max(min((upper_reclaim - 0.48) / 0.32, 1.0), 0.0)
    early_vol = _std(returns[:10]) + 1e-12
    late_vol = _std(returns[-8:]) + 1e-12
    vol_contraction = max(1.0 - min(late_vol / early_vol, 1.0), 0.0)
    turnover_base = _mean(recent_turnover[:10]) + 1e-12
    turnover_late = _mean(recent_turnover[-5:])
    turnover_calm = max(1.0 - min(turnover_late / turnover_base, 1.0), 0.0)
    net_return = _safe_div(recent_closes[-1] - recent_closes[0], recent_closes[0])
    constructive_drift = max(min((net_return + 0.015) / 0.095, 1.0), 0.0)
    chase_penalty = max(min((net_return - 0.14) / 0.10, 1.0), 0.0)
    downside_tail = abs(_mean(sorted(value for value in returns if value < 0.0)[:3])) if any(value < 0.0 for value in returns) else 0.0
    tail_absorption = max(1.0 - min(downside_tail / 0.035, 1.0), 0.0)
    score = (
        0.22 * compression
        + 0.20 * low_lift_score
        + 0.18 * upper_reclaim_score
        + 0.14 * vol_contraction
        + 0.12 * turnover_calm
        + 0.08 * constructive_drift
        + 0.06 * tail_absorption
    ) * (1.0 - 0.45 * chase_penalty)
    return max(min(score, 1.0), 0.0)


def innovation_drift_absorption_20(prices: Sequence[float], turnover: Sequence[float]) -> float:
    if len(prices) < 21 or len(turnover) < 20:
        return 0.0
    window = [float(value) for value in prices[-21:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-20:]]
    if min(window) <= 0.0:
        return 0.0
    returns = [
        _safe_div(window[index + 1] - window[index], window[index])
        for index in range(len(window) - 1)
    ]
    if len(returns) < 12:
        return 0.0
    state = returns[0]
    innovations: list[float] = []
    filtered: list[float] = []
    gain = 0.32
    for value in returns:
        innovation = value - state
        state += gain * innovation
        innovations.append(innovation)
        filtered.append(state)
    latest_innovation = innovations[-1]
    positive_shock = max(latest_innovation, 0.0)
    if positive_shock <= 1e-12:
        return 0.0
    drift_after_shock = max(_mean(filtered[-5:]), 0.0)
    residual_noise = _std(innovations[-12:]) + 1e-12
    raw_noise = _std(returns[-12:]) + 1e-12
    absorption = max(1.0 - min(residual_noise / raw_noise, 1.0), 0.0)
    follow_through = _safe_div(sum(1 for value in returns[-5:] if value >= -0.001), 5.0)
    turnover_base = _mean(recent_turnover[:10]) + 1e-12
    turnover_latest = _mean(recent_turnover[-3:])
    turnover_control = max(1.0 - min(max(_safe_div(turnover_latest, turnover_base) - 1.0, 0.0) / 1.6, 1.0), 0.0)
    shock_strength = min(positive_shock / 0.035, 1.0)
    drift_strength = min(drift_after_shock / 0.018, 1.0)
    score = shock_strength * (
        0.34 * drift_strength
        + 0.26 * absorption
        + 0.22 * follow_through
        + 0.18 * turnover_control
    )
    return max(min(score, 1.0), 0.0)


def entropy_compression_breakout_potential_20(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(highs) < 20 or len(lows) < 20 or len(closes) < 20 or len(turnover) < 20:
        return 0.0
    recent_highs = [float(value) for value in highs[-20:]]
    recent_lows = [float(value) for value in lows[-20:]]
    recent_closes = [float(value) for value in closes[-20:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-20:]]
    if min(recent_closes) <= 0.0:
        return 0.0
    returns = [
        _safe_div(recent_closes[index + 1] - recent_closes[index], recent_closes[index])
        for index in range(len(recent_closes) - 1)
    ]
    if len(returns) < 10:
        return 0.0
    signs = [1 if value > 0.001 else -1 if value < -0.001 else 0 for value in returns]
    counts = {state: signs.count(state) for state in (-1, 0, 1)}
    entropy = 0.0
    for count in counts.values():
        probability = _safe_div(count, len(signs))
        if probability > 1e-12:
            entropy -= probability * log(probability, 3)
    entropy_compression = max(1.0 - min(entropy, 1.0), 0.0)
    range_width = _safe_div(max(recent_highs) - min(recent_lows), recent_closes[-1])
    early_range = _safe_div(max(recent_highs[:10]) - min(recent_lows[:10]), recent_closes[9])
    late_range = _safe_div(max(recent_highs[-10:]) - min(recent_lows[-10:]), recent_closes[-1])
    range_compression = max(1.0 - min(_safe_div(late_range, early_range + 1e-12), 1.0), 0.0)
    upper_residence = _mean(
        [
            1.0 if _safe_div(close_px - min(recent_lows), max(recent_highs) - min(recent_lows)) >= 0.58 else 0.0
            for close_px in recent_closes[-8:]
        ]
    )
    turnover_calm = max(1.0 - min(_safe_div(_mean(recent_turnover[-5:]), _mean(recent_turnover[:10]) + 1e-12), 1.0), 0.0)
    constructive_drift = min(max(_safe_div(recent_closes[-1] - recent_closes[0], recent_closes[0]), 0.0) / 0.08, 1.0)
    width_penalty = max(1.0 - min(range_width / 0.22, 1.0), 0.0)
    score = (
        0.24 * entropy_compression
        + 0.24 * range_compression
        + 0.22 * upper_residence
        + 0.16 * turnover_calm
        + 0.14 * constructive_drift
    ) * (0.65 + 0.35 * width_penalty)
    return max(min(score, 1.0), 0.0)


def volatility_of_volatility_contraction_20(prices: Sequence[float]) -> float:
    if len(prices) < 41:
        return 0.0
    window = [float(value) for value in prices[-41:]]
    if min(window) <= 0.0:
        return 0.0
    returns = [
        _safe_div(window[index + 1] - window[index], window[index])
        for index in range(len(window) - 1)
    ]
    short_vols = [_std(returns[index - 4 : index + 1]) for index in range(4, len(returns))]
    if len(short_vols) < 20:
        return 0.0
    early_vov = _std(short_vols[:16])
    late_vov = _std(short_vols[-12:])
    if early_vov <= 1e-12:
        return 0.0
    contraction = max(1.0 - min(late_vov / early_vov, 1.0), 0.0)
    late_vol = _std(returns[-10:])
    base_vol = _std(returns[-30:-10]) + 1e-12
    calm_not_dead = max(1.0 - min(abs(_safe_div(late_vol, base_vol) - 0.72) / 0.72, 1.0), 0.0)
    positive_drift = min(max(_safe_div(window[-1] - window[-11], window[-11]), 0.0) / 0.08, 1.0)
    return max(min(0.50 * contraction + 0.28 * calm_not_dead + 0.22 * positive_drift, 1.0), 0.0)


def asymmetric_autocorr_repair_20(prices: Sequence[float]) -> float:
    if len(prices) < 22:
        return 0.0
    window = [float(value) for value in prices[-22:]]
    if min(window) <= 0.0:
        return 0.0
    returns = [
        _safe_div(window[index + 1] - window[index], window[index])
        for index in range(len(window) - 1)
    ]
    up_follow: list[float] = []
    down_follow: list[float] = []
    for index in range(1, len(returns)):
        previous = returns[index - 1]
        current = returns[index]
        if previous > 0.0:
            up_follow.append(current)
        elif previous < 0.0:
            down_follow.append(current)
    if len(up_follow) < 3 or len(down_follow) < 3:
        return 0.0
    downside = sorted(value for value in returns if value < 0.0)
    down_tail = abs(_mean(downside[:3])) if downside else 0.0
    tail_control = max(1.0 - min(down_tail / 0.035, 1.0), 0.0)
    non_chase = max(1.0 - min(max(max(returns[-5:]) - 0.055, 0.0) / 0.055, 1.0), 0.0)
    score = (
        0.38 * min(max(_mean(up_follow), 0.0) / 0.018, 1.0)
        + 0.34 * min(max(_mean(down_follow), 0.0) / 0.014, 1.0)
        + 0.18 * tail_control
        + 0.10 * non_chase
    )
    return max(min(score, 1.0), 0.0)


def drawdown_recovery_convexity_20(prices: Sequence[float]) -> float:
    if len(prices) < 21:
        return 0.0
    window = [float(value) for value in prices[-21:]]
    if min(window) <= 0.0:
        return 0.0
    trough_index = min(range(len(window)), key=lambda index: window[index])
    if trough_index >= len(window) - 4:
        return 0.0
    pre_peak = max(window[: trough_index + 1])
    trough = window[trough_index]
    drawdown = _safe_div(pre_peak - trough, pre_peak)
    if drawdown <= 0.015:
        return 0.0
    post = window[trough_index:]
    progress = [_safe_div(value - trough, pre_peak - trough) for value in post]
    split = max(2, len(progress) // 2)
    convexity = max(_mean(progress[split:]) - _mean(progress[1:split]), 0.0)
    hold_score = _safe_div(sum(1 for value in post[-5:] if value >= trough * 1.015), min(5, len(post)))
    max_one_bar = max(_safe_div(post[index + 1] - post[index], post[index]) for index in range(len(post) - 1))
    one_bar_penalty = max(0.0, min(max_one_bar / 0.09, 1.0))
    recovery = _safe_div(window[-1] - trough, pre_peak - trough)
    score = min(recovery, 1.25) / 1.25
    score *= 0.45 * min(convexity / 0.32, 1.0) + 0.35 * hold_score + 0.20 * min(drawdown / 0.18, 1.0)
    score *= 1.0 - 0.35 * one_bar_penalty
    return max(min(score, 1.0), 0.0)


def tail_repair_skew_balance_20(prices: Sequence[float]) -> float:
    if len(prices) < 21:
        return 0.0
    window = [float(value) for value in prices[-21:]]
    if min(window) <= 0.0:
        return 0.0
    returns = [
        _safe_div(window[index + 1] - window[index], window[index])
        for index in range(len(window) - 1)
    ]
    upside = sorted([value for value in returns if value > 0.0], reverse=True)
    downside = sorted([value for value in returns if value < 0.0])
    if len(upside) < 3 or len(downside) < 2:
        return 0.0
    upside_tail = _mean(upside[:3])
    downside_tail = abs(_mean(downside[:3]))
    skew_balance = _safe_div(upside_tail - downside_tail, upside_tail + downside_tail + 1e-12)
    repair_after_down = [
        returns[index + 1]
        for index in range(len(returns) - 1)
        if returns[index] < 0.0 and returns[index + 1] > 0.0
    ]
    repair_frequency = _safe_div(len(repair_after_down), len(downside))
    repair_strength = min(max(_mean(repair_after_down), 0.0) / 0.018, 1.0) if repair_after_down else 0.0
    net_drift = min(max(_safe_div(window[-1] - window[0], window[0]), 0.0) / 0.12, 1.0)
    score = (
        0.36 * max(min((skew_balance + 0.25) / 0.85, 1.0), 0.0)
        + 0.28 * min(repair_frequency, 1.0)
        + 0.22 * repair_strength
        + 0.14 * net_drift
    )
    return max(min(score, 1.0), 0.0)


def low_volume_box_support_20(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(highs) < 20 or len(lows) < 20 or len(closes) < 20 or len(turnover) < 20:
        return 0.0
    recent_highs = [float(value) for value in highs[-20:]]
    recent_lows = [float(value) for value in lows[-20:]]
    recent_closes = [float(value) for value in closes[-20:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-20:]]
    support_level = _mean(recent_lows[-5:])
    if support_level <= 0.0:
        return 0.0
    support_hold = max(0.0, min(1.0 - max(0.0, _safe_div(support_level - recent_closes[-1], support_level)), 1.0))
    turnover_baseline = _mean(recent_turnover)
    volume_shrink = max(0.0, min(1.0 - _safe_div(_mean(recent_turnover[-5:]), turnover_baseline + 1e-12), 1.0))
    returns = [
        _safe_div(recent_closes[index + 1] - recent_closes[index], recent_closes[index])
        for index in range(len(recent_closes) - 1)
    ]
    if len(returns) < 15:
        return 0.0
    range_compress = max(0.0, min(1.0 - _safe_div(_std(returns[-5:]), _std(returns[-15:]) + 1e-12), 1.0))
    high20 = max(recent_highs)
    low20 = min(recent_lows)
    if high20 <= low20:
        return 0.0
    base_location = _safe_div(recent_closes[-1] - low20, high20 - low20)
    mid_band_fit = max(0.0, 1.0 - abs(base_location - 0.65) / 0.65)
    return support_hold * (0.45 * volume_shrink + 0.35 * range_compress + 0.20 * mid_band_fit)


def support_test_reclaim_15(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> float:
    if len(highs) < 15 or len(lows) < 15 or len(closes) < 15:
        return 0.0
    recent_highs = [float(value) for value in highs[-15:]]
    recent_lows = [float(value) for value in lows[-15:]]
    recent_closes = [float(value) for value in closes[-15:]]
    tests = 0
    successful_tests = 0
    breach_depth = 0.0
    close_below_count = 0
    for index in range(5, len(recent_closes)):
        baseline_lows = sorted(recent_lows[index - 5 : index])
        if not baseline_lows:
            continue
        support = _mean(baseline_lows[:2])
        if support <= 0.0:
            continue
        low_value = recent_lows[index]
        if low_value > support * 1.01:
            continue
        tests += 1
        day_range = recent_highs[index] - recent_lows[index]
        close_location = _safe_div(
            recent_closes[index] - recent_lows[index],
            day_range if day_range > 1e-12 else 1.0,
        )
        if recent_closes[index] >= support and close_location >= 0.6:
            successful_tests += 1
        if recent_closes[index] < support:
            close_below_count += 1
            breach_depth += max(_safe_div(support - recent_closes[index], support), 0.0)
    if tests <= 0:
        return 0.0
    success_ratio = _safe_div(successful_tests, tests)
    breach_penalty = min(_safe_div(breach_depth, max(tests, 1) * 0.04), 1.0)
    close_below_penalty = _safe_div(close_below_count, tests)
    return max(success_ratio * (1.0 - 0.65 * breach_penalty) * (1.0 - 0.45 * close_below_penalty), 0.0)


def false_breakdown_repair_10(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    *,
    break_delta: float = 0.01,
) -> float:
    if len(highs) < 15 or len(lows) < 15 or len(closes) < 15:
        return 0.0
    recent_highs = [float(value) for value in highs[-15:]]
    recent_lows = [float(value) for value in lows[-15:]]
    recent_closes = [float(value) for value in closes[-15:]]
    current_close = recent_closes[-1]
    best_score = 0.0
    for index in range(5, len(recent_lows)):
        support = min(recent_lows[index - 5 : index])
        if support <= 0.0:
            continue
        low_value = recent_lows[index]
        if low_value >= support * (1.0 - break_delta):
            continue
        break_amp = min(_safe_div(support - low_value, support), 0.12) / 0.12
        day_range = recent_highs[index] - recent_lows[index]
        clv = _safe_div(recent_closes[index] - recent_lows[index], day_range if day_range > 1e-12 else 1.0)
        same_day_reclaim = max(clv - 0.45, 0.0) / 0.55
        hold_power = max(_safe_div(current_close - support, support), 0.0)
        hold_score = min(hold_power / 0.05, 1.0)
        recency_score = 1.0 - _safe_div(len(recent_lows) - 1 - index, 10.0)
        best_score = max(best_score, break_amp * same_day_reclaim * hold_score * max(recency_score, 0.0))
    return max(min(best_score, 1.0), 0.0)


def delayed_undercut_reclaim_12(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(highs) < 12 or len(lows) < 12 or len(closes) < 12 or len(turnover) < 12:
        return 0.0
    recent_highs = [float(value) for value in highs[-12:]]
    recent_lows = [float(value) for value in lows[-12:]]
    recent_closes = [float(value) for value in closes[-12:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-12:]]
    best_score = 0.0
    for index in range(5, len(recent_lows) - 1):
        support = min(recent_lows[index - 5 : index])
        if support <= 0.0:
            continue
        low_value = recent_lows[index]
        if low_value >= support * 0.99:
            continue
        reclaim_index = None
        for probe in range(index, min(index + 4, len(recent_closes))):
            day_range = recent_highs[probe] - recent_lows[probe]
            clv = _safe_div(recent_closes[probe] - recent_lows[probe], day_range if day_range > 1e-12 else 1.0)
            if recent_closes[probe] > support and clv >= 0.65:
                reclaim_index = probe
                break
        if reclaim_index is None:
            continue
        undercut = min(max(_safe_div(support - low_value, support), 0.0) / 0.08, 1.0)
        reclaim_strength = min(max(_safe_div(recent_closes[reclaim_index] - support, support), 0.0) / 0.04, 1.0)
        hold_window = recent_closes[reclaim_index : min(reclaim_index + 3, len(recent_closes))]
        hold_floor = min(hold_window) if hold_window else recent_closes[reclaim_index]
        hold_score = max(1.0 - min(max(_safe_div(support - hold_floor, support), 0.0) / 0.03, 1.0), 0.0)
        pre_start = max(index - 3, 0)
        pre_turnover = recent_turnover[pre_start : index + 1]
        post_turnover = recent_turnover[reclaim_index : min(reclaim_index + 3, len(recent_turnover))]
        pre_mean = _mean(pre_turnover)
        post_mean = _mean(post_turnover)
        turnover_cool = max(1.0 - min(_safe_div(post_mean, pre_mean + 1e-12), 1.0), 0.0)
        recency_score = 1.0 - _safe_div(len(recent_lows) - 1 - reclaim_index, 12.0)
        score = undercut * reclaim_strength * (0.6 * hold_score + 0.4 * turnover_cool) * max(recency_score, 0.0)
        best_score = max(best_score, score)
    return max(min(best_score, 1.0), 0.0)


def failed_breakout_reversal_intensity_12(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(opens) < 12 or len(highs) < 12 or len(lows) < 12 or len(closes) < 12 or len(turnover) < 12:
        return 0.0
    recent_opens = [float(value) for value in opens[-12:]]
    recent_highs = [float(value) for value in highs[-12:]]
    recent_lows = [float(value) for value in lows[-12:]]
    recent_closes = [float(value) for value in closes[-12:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-12:]]
    best_score = 0.0
    for index in range(10, len(recent_highs) - 1):
        prior_high = max(recent_highs[index - 10 : index])
        if prior_high <= 0.0:
            continue
        high_value = recent_highs[index]
        if high_value <= prior_high * 1.005:
            continue
        ext = min(max(_safe_div(high_value - prior_high, prior_high), 0.0) / 0.06, 1.0)
        close_back = min(max(_safe_div(prior_high - recent_closes[index], prior_high), 0.0) / 0.04, 1.0)
        day_range = high_value - recent_lows[index]
        upper_reject = _safe_div(high_value - max(recent_opens[index], recent_closes[index]), day_range if day_range > 1e-12 else 1.0)
        follow_window = recent_closes[index + 1 : min(index + 4, len(recent_closes))]
        follow_floor = min(follow_window) if follow_window else recent_closes[index]
        follow_fail = min(max(_safe_div(prior_high - follow_floor, prior_high), 0.0) / 0.04, 1.0)
        turnover_ratio = _safe_div(recent_turnover[index], _mean(recent_turnover[max(index - 5, 0) : index]) + 1e-12)
        turn = min(max(turnover_ratio, 0.8), 1.5)
        score = ext * (0.45 * close_back + 0.25 * upper_reject + 0.30 * follow_fail) * turn
        best_score = max(best_score, score)
    return max(min(best_score, 1.5), 0.0)


def breakout_failure_repair_15(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(opens) < 15 or len(highs) < 15 or len(lows) < 15 or len(closes) < 15 or len(turnover) < 15:
        return 0.0
    recent_opens = [float(value) for value in opens[-15:]]
    recent_highs = [float(value) for value in highs[-15:]]
    recent_lows = [float(value) for value in lows[-15:]]
    recent_closes = [float(value) for value in closes[-15:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-15:]]
    best_score = 0.0
    for index in range(9, len(recent_closes) - 2):
        pivot_high = max(recent_highs[index - 9 : index])
        if pivot_high <= 0.0 or recent_highs[index] <= pivot_high * 1.004 or recent_closes[index] >= pivot_high:
            continue
        breakout_extension = min(max(_safe_div(recent_highs[index] - pivot_high, pivot_high), 0.0) / 0.05, 1.0)
        failure_close_back = min(max(_safe_div(pivot_high - recent_closes[index], pivot_high), 0.0) / 0.04, 1.0)
        repair_window_end = min(index + 5, len(recent_closes))
        repair_closes = recent_closes[index + 1 : repair_window_end]
        repair_lows = recent_lows[index + 1 : repair_window_end]
        repair_turnover = recent_turnover[index + 1 : repair_window_end]
        if len(repair_closes) < 2:
            continue
        reclaim_index = next((probe for probe, close_px in enumerate(repair_closes, start=index + 1) if close_px >= pivot_high), None)
        if reclaim_index is None:
            continue
        reclaim_close = recent_closes[reclaim_index]
        reclaim_strength = min(max(_safe_div(reclaim_close - pivot_high, pivot_high), 0.0) / 0.04, 1.0)
        follow_closes = recent_closes[reclaim_index : min(reclaim_index + 3, len(recent_closes))]
        follow_lows = recent_lows[reclaim_index : min(reclaim_index + 3, len(recent_lows))]
        hold_floor = min(follow_lows) if follow_lows else recent_lows[reclaim_index]
        hold_score = max(1.0 - min(max(_safe_div(pivot_high - hold_floor, pivot_high), 0.0) / 0.03, 1.0), 0.0)
        close_cluster = _mean([max(_safe_div(close_px - pivot_high, pivot_high * 0.03), 0.0) for close_px in follow_closes])
        pre_turnover = _mean(recent_turnover[max(index - 4, 0) : index + 1]) + 1.0
        post_turnover = _mean(repair_turnover) if repair_turnover else pre_turnover
        turnover_cool = max(1.0 - min(_safe_div(post_turnover, pre_turnover), 1.0), 0.0)
        recency = max(1.0 - _safe_div(len(recent_closes) - 1 - reclaim_index, 15.0), 0.2)
        score = (
            breakout_extension
            * (0.25 + 0.75 * (1.0 - failure_close_back))
            * (0.40 * reclaim_strength + 0.35 * hold_score + 0.15 * min(close_cluster, 1.0) + 0.10 * turnover_cool)
            * recency
        )
        best_score = max(best_score, score)
    return max(min(best_score, 1.25), 0.0)


def failed_breakout_recoil_repair_15(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(highs) < 15 or len(lows) < 15 or len(closes) < 15 or len(turnover) < 15:
        return 0.0
    recent_highs = [float(value) for value in highs[-15:]]
    recent_lows = [float(value) for value in lows[-15:]]
    recent_closes = [float(value) for value in closes[-15:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-15:]]
    best_score = 0.0
    for index in range(9, len(recent_closes) - 3):
        pivot_high = max(recent_highs[index - 9 : index])
        if pivot_high <= 0.0:
            continue
        if recent_highs[index] <= pivot_high * 1.004 or recent_closes[index] >= pivot_high:
            continue
        reclaim_index = None
        for probe in range(index + 1, min(index + 6, len(recent_closes))):
            if recent_closes[probe] >= pivot_high:
                reclaim_index = probe
                break
        if reclaim_index is None:
            continue
        post_closes = recent_closes[reclaim_index : min(reclaim_index + 4, len(recent_closes))]
        post_highs = recent_highs[reclaim_index : min(reclaim_index + 4, len(recent_highs))]
        post_lows = recent_lows[reclaim_index : min(reclaim_index + 4, len(recent_lows))]
        post_turnover = recent_turnover[reclaim_index : min(reclaim_index + 4, len(recent_turnover))]
        if len(post_closes) < 2:
            continue
        hold_ratio = _mean([1.0 if close_px >= pivot_high else 0.0 for close_px in post_closes])
        post_returns = [
            _safe_div(post_closes[pos] - post_closes[pos - 1], post_closes[pos - 1])
            for pos in range(1, len(post_closes))
            if post_closes[pos - 1] > 0.0
        ]
        repair_returns = [
            _safe_div(recent_closes[pos] - recent_closes[pos - 1], recent_closes[pos - 1])
            for pos in range(index + 1, reclaim_index + 1)
            if pos > 0 and recent_closes[pos - 1] > 0.0
        ]
        post_std = _std(post_returns) if post_returns else 0.0
        repair_std = _std(repair_returns) if repair_returns else post_std
        tightness = max(1.0 - min(_safe_div(post_std, repair_std + 1e-12), 1.0), 0.0)
        turnover_fail = recent_turnover[index]
        turnover_cool = max(1.0 - min(_safe_div(_mean(post_turnover), turnover_fail + 1.0), 1.0), 0.0)
        hold_floor = min(post_lows) if post_lows else recent_lows[reclaim_index]
        hold_score = max(1.0 - min(max(_safe_div(pivot_high - hold_floor, pivot_high), 0.0) / 0.03, 1.0), 0.0)
        score = (0.35 * hold_ratio + 0.25 * tightness + 0.20 * turnover_cool + 0.20 * hold_score)
        best_score = max(best_score, score)
    return max(min(best_score, 1.0), 0.0)


def reaction_high_retake_quality_15(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(highs) < 15 or len(lows) < 15 or len(closes) < 15 or len(turnover) < 15:
        return 0.0
    recent_highs = [float(value) for value in highs[-15:]]
    recent_lows = [float(value) for value in lows[-15:]]
    recent_closes = [float(value) for value in closes[-15:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-15:]]
    trough_index = min(range(len(recent_lows)), key=lambda idx: recent_lows[idx])
    if trough_index < 4 or trough_index >= len(recent_closes) - 2:
        return 0.0
    trough_low = recent_lows[trough_index]
    reaction_high = max(recent_highs[max(trough_index - 5, 0) : trough_index])
    if reaction_high <= trough_low:
        return 0.0
    rebound_high = max(recent_highs[trough_index + 1 :])
    reclaim_span = reaction_high - trough_low
    retake_ratio = min(max(_safe_div(rebound_high - trough_low, reclaim_span), 0.0), 1.2)
    midpoint = (reaction_high + trough_low) * 0.5
    stable_rate = _mean([1.0 if close_px >= midpoint else 0.0 for close_px in recent_closes[-3:]])
    post_turnover = _mean(recent_turnover[trough_index + 1 :]) if trough_index + 1 < len(recent_turnover) else recent_turnover[-1]
    pre_turnover = _mean(recent_turnover[max(trough_index - 5, 0) : trough_index + 1]) + 1.0
    turnover_cool = max(1.0 - min(_safe_div(post_turnover, pre_turnover), 1.0), 0.0)
    recency = max(1.0 - _safe_div(len(recent_closes) - 1 - trough_index, 15.0), 0.3)
    score = (0.60 * min(retake_ratio, 1.0) + 0.25 * stable_rate + 0.15 * turnover_cool) * recency
    return max(min(score, 1.2), 0.0)


def gap_failure_containment_10(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(opens) < 10 or len(highs) < 10 or len(lows) < 10 or len(closes) < 10 or len(turnover) < 10:
        return 0.0
    recent_opens = [float(value) for value in opens[-10:]]
    recent_highs = [float(value) for value in highs[-10:]]
    recent_lows = [float(value) for value in lows[-10:]]
    recent_closes = [float(value) for value in closes[-10:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-10:]]
    best_score = 0.0
    for index in range(1, len(recent_closes) - 2):
        prior_close = recent_closes[index - 1]
        if prior_close <= 0.0:
            continue
        gap_down = _safe_div(prior_close - recent_opens[index], prior_close)
        same_day_loss = _safe_div(prior_close - recent_closes[index], prior_close)
        shock = max(gap_down, same_day_loss)
        if shock < 0.02:
            continue
        day_range = recent_highs[index] - recent_lows[index]
        same_day_repair = _safe_div(
            recent_closes[index] - recent_lows[index],
            day_range if day_range > 1e-12 else 1.0,
        )
        follow_lows = recent_lows[index + 1 : index + 4]
        follow_closes = recent_closes[index + 1 : index + 4]
        if len(follow_lows) < 2 or len(follow_closes) < 2:
            continue
        containment = max(
            1.0 - min(max(_safe_div(recent_lows[index] - min(follow_lows), recent_lows[index]), 0.0) / 0.04, 1.0),
            0.0,
        )
        midpoint = (recent_opens[index] + prior_close) * 0.5
        mid_reclaim = _mean([1.0 if close_px >= midpoint else 0.0 for close_px in follow_closes])
        turnover_base = _mean(recent_turnover[max(index - 4, 0) : index]) + 1.0
        turnover_confirm = min(max(_safe_div(recent_turnover[index], turnover_base), 0.8), 1.4)
        shock_weight = min(max(shock, 0.0) / 0.08, 1.0)
        score = shock_weight * (0.35 * same_day_repair + 0.40 * containment + 0.25 * mid_reclaim) * turnover_confirm
        best_score = max(best_score, score)
    return max(min(best_score, 1.2), 0.0)


def repair_liquidity_retention_10(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(highs) < 10 or len(lows) < 10 or len(closes) < 10 or len(turnover) < 10:
        return 0.0
    recent_highs = [float(value) for value in highs[-10:]]
    recent_lows = [float(value) for value in lows[-10:]]
    recent_closes = [float(value) for value in closes[-10:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-10:]]
    best_score = 0.0
    for index in range(3, len(recent_closes) - 3):
        repair_floor = min(recent_lows[max(index - 3, 0) : index + 1])
        if repair_floor <= 0.0:
            continue
        repair_gain = _safe_div(recent_closes[index] - repair_floor, repair_floor)
        if repair_gain < 0.05:
            continue
        repair_turnover = _mean(recent_turnover[max(index - 2, 0) : index + 1]) + 1.0
        follow_turnover = recent_turnover[index + 1 : index + 4]
        follow_lows = recent_lows[index + 1 : index + 4]
        follow_closes = recent_closes[index + 1 : index + 4]
        if len(follow_turnover) < 2 or len(follow_lows) < 2 or len(follow_closes) < 2:
            continue
        turnover_retention = max(
            1.0 - min(max(_safe_div(repair_turnover - _mean(follow_turnover), repair_turnover), 0.0), 1.0),
            0.0,
        )
        hold_score = max(
            1.0 - min(max(_safe_div(recent_closes[index] - min(follow_lows), recent_closes[index]), 0.0) / 0.04, 1.0),
            0.0,
        )
        repair_mid = repair_floor + (recent_closes[index] - repair_floor) * 0.5
        mid_hold = _mean([1.0 if close_px >= repair_mid else 0.0 for close_px in follow_closes])
        gain_weight = min(repair_gain / 0.08, 1.2)
        score = gain_weight * (0.40 * turnover_retention + 0.35 * hold_score + 0.25 * mid_hold)
        best_score = max(best_score, score)
    return max(min(best_score, 1.2), 0.0)


def repair_base_tightness_12(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(highs) < 12 or len(lows) < 12 or len(closes) < 12 or len(turnover) < 12:
        return 0.0
    recent_highs = [float(value) for value in highs[-12:]]
    recent_lows = [float(value) for value in lows[-12:]]
    recent_closes = [float(value) for value in closes[-12:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-12:]]
    best_score = 0.0
    for index in range(5, len(recent_closes) - 3):
        support = min(recent_lows[index - 5 : index])
        if support <= 0.0:
            continue
        if recent_lows[index] >= support * 0.99 and recent_closes[index] < support:
            continue
        day_range = recent_highs[index] - recent_lows[index]
        clv = _safe_div(recent_closes[index] - recent_lows[index], day_range if day_range > 1e-12 else 1.0)
        if recent_lows[index] < support * 0.99:
            if recent_closes[index] <= support or clv < 0.55:
                continue
            repair_pivot = support
        else:
            if recent_closes[index] < support or clv < 0.6:
                continue
            repair_pivot = support
        post_slice = slice(index + 1, min(index + 4, len(recent_closes)))
        post_closes = recent_closes[post_slice]
        post_highs = recent_highs[post_slice]
        post_lows = recent_lows[post_slice]
        post_turnover = recent_turnover[post_slice]
        if len(post_closes) < 2:
            continue
        hold = _mean([1.0 if close_px >= repair_pivot else 0.0 for close_px in post_closes])
        pre_slice = slice(max(index - 3, 0), index + 1)
        pre_ranges = [
            _safe_div(high_px - low_px, close_px if close_px > 0.0 else 1.0)
            for high_px, low_px, close_px in zip(
                recent_highs[pre_slice],
                recent_lows[pre_slice],
                recent_closes[pre_slice],
            )
        ]
        post_ranges = [
            _safe_div(high_px - low_px, close_px if close_px > 0.0 else 1.0)
            for high_px, low_px, close_px in zip(post_highs, post_lows, post_closes)
        ]
        range_contract = max(1.0 - min(_safe_div(_mean(post_ranges), _mean(pre_ranges) + 1e-12), 1.0), 0.0)
        turnover_cool = max(1.0 - min(_safe_div(_mean(post_turnover), recent_turnover[index] + 1.0), 1.0), 0.0)
        hold_floor = min(post_lows) if post_lows else recent_lows[index]
        hold_penalty = min(max(_safe_div(repair_pivot - hold_floor, repair_pivot), 0.0) / 0.03, 1.0)
        score = (0.45 * hold + 0.35 * range_contract + 0.20 * turnover_cool) * (1.0 - 0.5 * hold_penalty)
        best_score = max(best_score, score)
    return max(min(best_score, 1.0), 0.0)


def flat_top_higher_low_coil_20(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> float:
    if len(highs) < 20 or len(lows) < 20 or len(closes) < 20:
        return 0.0
    recent_highs = [float(value) for value in highs[-20:]]
    recent_lows = [float(value) for value in lows[-20:]]
    recent_closes = [float(value) for value in closes[-20:]]
    upper_band = max(recent_highs)
    lower_band = min(recent_lows)
    if upper_band <= 0.0 or upper_band <= lower_band:
        return 0.0
    total_width = upper_band - lower_band
    touch_threshold = upper_band - total_width * 0.12
    touch_count = sum(1 for value in recent_highs[-10:] if value >= touch_threshold)
    touch_score = min(_safe_div(touch_count, 4.0), 1.0)
    early_low = _mean(recent_lows[:5])
    late_low = _mean(recent_lows[-5:])
    rising_floor = min(max(_safe_div(late_low - early_low, max(early_low, 1.0)) / 0.08, 0.0), 1.0)
    early_range = _mean(
        [
            _safe_div(high_px - low_px, close_px if close_px > 0.0 else 1.0)
            for high_px, low_px, close_px in zip(recent_highs[:10], recent_lows[:10], recent_closes[:10])
        ]
    )
    late_range = _mean(
        [
            _safe_div(high_px - low_px, close_px if close_px > 0.0 else 1.0)
            for high_px, low_px, close_px in zip(recent_highs[-10:], recent_lows[-10:], recent_closes[-10:])
        ]
    )
    compression_score = max(1.0 - min(_safe_div(late_range, early_range + 1e-12), 1.0), 0.0)
    ceiling_drift = max(recent_highs[-10:]) - min(recent_highs[-10:])
    flatness_score = max(1.0 - min(_safe_div(ceiling_drift, total_width * 0.45 + 1e-12), 1.0), 0.0)
    close_position = _safe_div(_mean(recent_closes[-5:]) - lower_band, total_width)
    close_score = max(min((close_position - 0.55) / 0.35, 1.0), 0.0)
    return max(min((0.28 * touch_score + 0.28 * rising_floor + 0.22 * compression_score + 0.12 * flatness_score + 0.10 * close_score), 1.0), 0.0)


def lower_shadow_absorption_5(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> float:
    if len(opens) < 5 or len(highs) < 5 or len(lows) < 5 or len(closes) < 5:
        return 0.0
    scores: list[float] = []
    for open_px, high_px, low_px, close_px in zip(opens[-5:], highs[-5:], lows[-5:], closes[-5:]):
        day_range = high_px - low_px
        if day_range <= 1e-12:
            continue
        lower_shadow = max(min(open_px, close_px) - low_px, 0.0)
        reclaim = _safe_div(close_px - low_px, day_range)
        body_sign = 1.0 if close_px >= open_px else 0.5
        intraday_drawdown = _safe_div(max(open_px, close_px) - low_px, max(open_px, close_px) if max(open_px, close_px) > 0 else 1.0)
        if intraday_drawdown < 0.015:
            continue
        scores.append(_safe_div(lower_shadow, day_range) * reclaim * body_sign)
    if not scores:
        return 0.0
    return _mean(scores)


def compression_release_quality_15(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(highs) < 15 or len(lows) < 15 or len(closes) < 15 or len(turnover) < 15:
        return 0.0
    recent_highs = [float(value) for value in highs[-15:]]
    recent_lows = [float(value) for value in lows[-15:]]
    recent_closes = [float(value) for value in closes[-15:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-15:]]
    returns = [
        _safe_div(recent_closes[index + 1] - recent_closes[index], recent_closes[index])
        for index in range(len(recent_closes) - 1)
        if recent_closes[index] > 0.0
    ]
    if len(returns) < 14:
        return 0.0
    early_std = _std(returns[:9])
    late_std = _std(returns[9:])
    compression_score = max(1.0 - _safe_div(late_std, early_std + 1e-12), 0.0)
    release_base = max(recent_closes[-6:-1]) if recent_closes[-6:-1] else 0.0
    if release_base <= 0.0:
        return 0.0
    release_score = max(_safe_div(recent_closes[-1] - release_base, release_base), 0.0)
    breakout_range = max(recent_highs[-5:]) - min(recent_lows[-5:])
    normalization = max(_mean(recent_closes[-5:]), 1.0)
    release_score = min(_safe_div(release_score, max(_safe_div(breakout_range, normalization), 0.02)), 1.5)
    turnover_prev = _mean(recent_turnover[-10:-3])
    turnover_recent = _mean(recent_turnover[-3:])
    turnover_confirm = min(max(_safe_div(turnover_recent, turnover_prev + 1.0), 0.6), 1.8)
    return max(min(compression_score * release_score * turnover_confirm, 1.8), 0.0)


def compression_persistence_20(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(highs) < 20 or len(lows) < 20 or len(closes) < 20 or len(turnover) < 20:
        return 0.0
    recent_highs = [float(value) for value in highs[-20:]]
    recent_lows = [float(value) for value in lows[-20:]]
    recent_closes = [float(value) for value in closes[-20:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-20:]]
    ranges = [
        _safe_div(high_px - low_px, close_px if close_px > 0.0 else 1.0)
        for high_px, low_px, close_px in zip(recent_highs, recent_lows, recent_closes)
    ]
    sorted_ranges = sorted(ranges)
    median_range = sorted_ranges[len(sorted_ranges) // 2] if sorted_ranges else 0.0
    upper_high = max(recent_highs)
    lower_low = min(recent_lows)
    band = max(upper_high - lower_low, 1e-12)
    turnover_baseline = _mean(recent_turnover[:10]) + 1.0
    scores: list[float] = []
    for index in range(12, len(recent_closes)):
        quiet_score = max(1.0 - _safe_div(max(ranges[index] - median_range, 0.0), median_range + 1e-12), 0.0)
        hold_score = max(_safe_div(recent_closes[index] - (lower_low + band * 0.55), band * 0.45), 0.0)
        hold_score = min(hold_score, 1.0)
        turnover_calm = max(1.0 - _safe_div(recent_turnover[index], turnover_baseline), 0.0)
        scores.append(quiet_score * hold_score * (0.6 + 0.4 * min(turnover_calm, 1.0)))
    return _mean(scores) if scores else 0.0


def upper_shadow_distribution_pressure_10(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(opens) < 10 or len(highs) < 10 or len(lows) < 10 or len(closes) < 10 or len(turnover) < 10:
        return 0.0
    recent_opens = [float(value) for value in opens[-10:]]
    recent_highs = [float(value) for value in highs[-10:]]
    recent_lows = [float(value) for value in lows[-10:]]
    recent_closes = [float(value) for value in closes[-10:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-10:]]
    baseline_turnover = _mean(recent_turnover) + 1.0
    high10 = max(recent_highs)
    low10 = min(recent_lows)
    band = max(high10 - low10, 1e-12)
    scores: list[float] = []
    for open_px, high_px, low_px, close_px, turnover_px in zip(
        recent_opens,
        recent_highs,
        recent_lows,
        recent_closes,
        recent_turnover,
    ):
        day_range = high_px - low_px
        if day_range <= 1e-12:
            continue
        near_upper_structure = 1.0 if max(open_px, close_px) >= high10 - band * 0.35 else 0.0
        if near_upper_structure <= 0.0:
            continue
        upper_shadow_ratio = _safe_div(high_px - max(open_px, close_px), day_range)
        close_location = _safe_div(close_px - low_px, day_range)
        turnover_rel = max(_safe_div(turnover_px, baseline_turnover) - 1.0, 0.0)
        scores.append(upper_shadow_ratio * max(1.0 - close_location, 0.0) * min(turnover_rel, 2.0))
    return _mean(scores) if scores else 0.0


def upper_half_churn_trap_20(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(opens) < 20 or len(highs) < 20 or len(lows) < 20 or len(closes) < 20 or len(turnover) < 20:
        return 0.0
    recent_opens = [float(value) for value in opens[-20:]]
    recent_highs = [float(value) for value in highs[-20:]]
    recent_lows = [float(value) for value in lows[-20:]]
    recent_closes = [float(value) for value in closes[-20:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-20:]]
    upper_band = max(recent_highs)
    lower_band = min(recent_lows)
    total_width = upper_band - lower_band
    if upper_band <= 0.0 or total_width <= 1e-12:
        return 0.0
    upper_threshold = lower_band + total_width * 0.65
    baseline_turnover = _mean(recent_turnover) + 1.0
    scores: list[float] = []
    for open_px, high_px, low_px, close_px, turnover_px in zip(
        recent_opens[-10:],
        recent_highs[-10:],
        recent_lows[-10:],
        recent_closes[-10:],
        recent_turnover[-10:],
    ):
        if high_px < upper_threshold or close_px >= upper_threshold:
            continue
        day_range = high_px - low_px
        if day_range <= 1e-12:
            continue
        upper_reject = _safe_div(high_px - close_px, day_range)
        failed_push = 1.0 if max(open_px, close_px) >= upper_threshold else 0.65
        turnover_rel = min(max(_safe_div(turnover_px, baseline_turnover), 0.7), 1.8)
        scores.append(upper_reject * failed_push * turnover_rel)
    return max(min(_mean(scores), 1.5), 0.0) if scores else 0.0


def breakout_close_control_10(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> float:
    if len(opens) < 10 or len(highs) < 10 or len(lows) < 10 or len(closes) < 10:
        return 0.0
    recent_opens = [float(value) for value in opens[-10:]]
    recent_highs = [float(value) for value in highs[-10:]]
    recent_lows = [float(value) for value in lows[-10:]]
    recent_closes = [float(value) for value in closes[-10:]]
    scores: list[float] = []
    for index in range(3, len(recent_closes)):
        prior_high = max(recent_closes[max(0, index - 3) : index])
        if prior_high <= 0.0 or recent_closes[index] < prior_high:
            continue
        day_range = recent_highs[index] - recent_lows[index]
        if day_range <= 1e-12:
            continue
        close_location = _safe_div(recent_closes[index] - recent_lows[index], day_range)
        body_push = _safe_div(abs(recent_closes[index] - recent_opens[index]), day_range)
        upper_shadow_penalty = _safe_div(
            recent_highs[index] - max(recent_opens[index], recent_closes[index]),
            day_range,
        )
        score = close_location * 0.5 + body_push * 0.35 - upper_shadow_penalty * 0.45
        scores.append(score)
    if not scores:
        return 0.0
    return max(min(_mean(scores[-3:]), 1.0), -1.0)


def gap_fill_acceptance_8(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(opens) < 8 or len(highs) < 8 or len(lows) < 8 or len(closes) < 8 or len(turnover) < 8:
        return 0.0
    recent_opens = [float(value) for value in opens[-8:]]
    recent_highs = [float(value) for value in highs[-8:]]
    recent_lows = [float(value) for value in lows[-8:]]
    recent_closes = [float(value) for value in closes[-8:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-8:]]
    best_score = 0.0
    for index in range(1, len(recent_closes) - 2):
        prior_close = recent_closes[index - 1]
        open_value = recent_opens[index]
        if prior_close <= 0.0 or open_value >= prior_close * 0.995:
            continue
        gap_size = max(_safe_div(prior_close - open_value, prior_close), 0.0)
        if gap_size < 0.015:
            continue
        fill_window_end = min(index + 3, len(recent_closes))
        follow_closes = recent_closes[index:fill_window_end]
        follow_highs = recent_highs[index:fill_window_end]
        follow_lows = recent_lows[index:fill_window_end]
        follow_turnover = recent_turnover[index + 1 : fill_window_end]
        gap_ref = max(prior_close - open_value, prior_close * 0.01)
        fill_ratio = min(max(_safe_div(max(follow_highs) - open_value, gap_ref), 0.0), 1.2)
        hold_ratio = _mean([1.0 if close_px >= prior_close else 0.0 for close_px in follow_closes])
        undercut = max(_safe_div(prior_close - min(follow_lows), prior_close), 0.0)
        hold_penalty = min(undercut / 0.03, 1.0)
        turnover_cool = (
            max(1.0 - min(_safe_div(_mean(follow_turnover), recent_turnover[index] + 1.0), 1.0), 0.0)
            if follow_turnover
            else 0.0
        )
        gap_weight = min(gap_size / 0.05, 1.0)
        score = gap_weight * (0.45 * fill_ratio + 0.30 * hold_ratio + 0.25 * turnover_cool) * (1.0 - 0.7 * hold_penalty)
        best_score = max(best_score, score)
    return max(min(best_score, 1.2), 0.0)


def bullish_engulfing_hold_10(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> float:
    if len(opens) < 10 or len(highs) < 10 or len(lows) < 10 or len(closes) < 10:
        return 0.0
    recent_opens = [float(value) for value in opens[-10:]]
    recent_highs = [float(value) for value in highs[-10:]]
    recent_lows = [float(value) for value in lows[-10:]]
    recent_closes = [float(value) for value in closes[-10:]]
    best_score = 0.0
    for index in range(1, len(recent_closes) - 2):
        prior_open = recent_opens[index - 1]
        prior_close = recent_closes[index - 1]
        open_value = recent_opens[index]
        close_value = recent_closes[index]
        if prior_close >= prior_open:
            continue
        if close_value <= open_value:
            continue
        if open_value > prior_close or close_value < prior_open:
            continue
        day_range = recent_highs[index] - recent_lows[index]
        if day_range <= 1e-12:
            continue
        close_location = _safe_div(close_value - recent_lows[index], day_range)
        if close_location < 0.70:
            continue
        body_size = abs(close_value - open_value)
        prior_body = abs(prior_close - prior_open) + 1e-12
        body_push = min(_safe_div(body_size, prior_body), 1.5)
        engulf_strength = min(
            _safe_div((close_value - prior_open) + (prior_close - open_value), prior_body),
            1.5,
        )
        midpoint = (open_value + close_value) * 0.5
        follow_end = min(index + 3, len(recent_closes))
        follow_closes = recent_closes[index + 1 : follow_end]
        follow_lows = recent_lows[index + 1 : follow_end]
        if len(follow_closes) < 2:
            continue
        mid_hold = _mean([1.0 if close_px >= midpoint else 0.0 for close_px in follow_closes])
        low_hold = max(1.0 - min(max(_safe_div(midpoint - min(follow_lows), midpoint), 0.0) / 0.03, 1.0), 0.0)
        continuation = min(max(_safe_div(max(follow_closes) - close_value, close_value), 0.0) / 0.04, 1.0)
        score = min(body_push, 1.0) * (0.30 * min(engulf_strength, 1.0) + 0.30 * mid_hold + 0.25 * low_hold + 0.15 * continuation)
        best_score = max(best_score, score)
    return max(min(best_score, 1.1), 0.0)


def breakout_digestion_hold_10(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(highs) < 15 or len(lows) < 15 or len(closes) < 15 or len(turnover) < 15:
        return 0.0
    recent_highs = [float(value) for value in highs[-15:]]
    recent_lows = [float(value) for value in lows[-15:]]
    recent_closes = [float(value) for value in closes[-15:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-15:]]
    best_score = 0.0
    for index in range(5, 11):
        base_high = max(recent_highs[max(0, index - 10) : index]) if index > 0 else 0.0
        if base_high <= 0.0 or recent_closes[index] <= base_high * 1.002:
            continue
        follow_end = min(index + 5, len(recent_closes))
        if follow_end - index < 3:
            continue
        follow_closes = recent_closes[index + 1 : follow_end]
        follow_lows = recent_lows[index + 1 : follow_end]
        follow_turnover = recent_turnover[index + 1 : follow_end]
        hold_score = max(_safe_div(min(follow_closes) - base_high * 0.995, base_high * 0.04), 0.0)
        nonbreak_score = max(_safe_div(min(follow_lows) - base_high * 0.985, base_high * 0.05), 0.0)
        advance_score = max(_safe_div(follow_closes[-1] - base_high, base_high * 0.06), 0.0)
        digestion_score = max(1.0 - _safe_div(_mean(follow_turnover), recent_turnover[index] + 1.0), 0.0)
        candidate = (
            min(hold_score, 1.0) * 0.35
            + min(nonbreak_score, 1.0) * 0.30
            + min(advance_score, 1.0) * 0.20
            + min(digestion_score, 1.0) * 0.15
        )
        recency_boost = 1.0 - _safe_div((len(recent_closes) - 1) - index, 10.0)
        best_score = max(best_score, candidate * max(recency_boost, 0.3))
    return max(min(best_score, 1.25), 0.0)


def failed_reclaim_persistence_15(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(highs) < 15 or len(lows) < 15 or len(closes) < 15 or len(turnover) < 15:
        return 0.0
    recent_highs = [float(value) for value in highs[-15:]]
    recent_lows = [float(value) for value in lows[-15:]]
    recent_closes = [float(value) for value in closes[-15:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-15:]]
    turnover_baseline = _mean(recent_turnover[:10]) + 1.0
    scores: list[float] = []
    for index in range(5, len(recent_closes)):
        support = min(recent_lows[max(0, index - 5) : index])
        if support <= 0.0 or recent_lows[index] >= support * 0.998:
            continue
        if recent_highs[index] < support or recent_closes[index] >= support:
            continue
        breach_depth = max(_safe_div(support - recent_closes[index], support), 0.0)
        intraday_fail = max(_safe_div(recent_highs[index] - recent_closes[index], support), 0.0)
        turnover_rel = _safe_div(recent_turnover[index], turnover_baseline)
        scores.append(min((breach_depth * 0.7 + intraday_fail * 0.3) * min(turnover_rel, 2.0) * 12.0, 1.5))
    return _mean(scores) if scores else 0.0


def shrink_volume_nonbreak_5_20(prices: Sequence[float], turnover: Sequence[float]) -> float:
    if len(prices) < 20 or len(turnover) < 20:
        return 0.0
    closes = [float(value) for value in prices[-20:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-20:]]
    returns = [_safe_div(closes[index + 1] - closes[index], closes[index]) for index in range(len(closes) - 1)]
    if len(returns) < 6:
        return 0.0
    baseline_abs_return = _mean([abs(value) for value in returns])
    turnover_baseline = _mean(recent_turnover)
    scores: list[float] = []
    for ret_value, turnover_value in zip(returns[-5:], recent_turnover[-5:]):
        tvrel = _safe_div(turnover_value, turnover_baseline + 1e-12)
        damage = max(-ret_value, 0.0)
        low_volume = max(1.0 - tvrel, 0.0)
        nonbreak = max(1.0 - _safe_div(damage, baseline_abs_return + 1e-12), 0.0)
        scores.append(low_volume * nonbreak)
    return _mean(scores) if scores else 0.0


def pullback_energy_decay_20(prices: Sequence[float], turnover: Sequence[float]) -> float:
    if len(prices) < 21 or len(turnover) < 20:
        return 0.0
    closes = [float(value) for value in prices[-21:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-20:]]
    returns = [_safe_div(closes[index + 1] - closes[index], closes[index]) for index in range(len(closes) - 1)]
    if len(returns) < 20:
        return 0.0
    turnover_baseline = _mean(recent_turnover) + 1e-12
    neg_energy = [
        abs(min(ret_value, 0.0)) * _safe_div(turnover_value, turnover_baseline)
        for ret_value, turnover_value in zip(returns[-20:], recent_turnover[-20:])
    ]
    early = _mean(neg_energy[:10])
    late = _mean(neg_energy[10:])
    if early <= 1e-12:
        return 0.0
    high20 = max(closes[-20:])
    low20 = min(closes[-20:])
    if high20 <= low20:
        return 0.0
    top_pos = max(_safe_div(closes[-1] - low20, high20 - low20) - 0.5, 0.0) / 0.5
    decay = max(_safe_div(early - late, early + late + 1e-12), 0.0)
    return max(min(decay * top_pos, 1.0), 0.0)


def distribution_decay_after_inflow_spike_8(
    closes: Sequence[float],
    turnover: Sequence[float],
    flows: Sequence[float],
) -> float:
    if len(closes) < 9 or len(turnover) < 8 or len(flows) < 8:
        return 0.0
    recent_closes = [float(value) for value in closes[-9:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-8:]]
    recent_flows = [float(value) for value in flows[-8:]]
    flow_scale = _mean([abs(value) for value in recent_flows]) + 1e-12
    turnover_baseline = _mean(recent_turnover) + 1e-12
    best_score = 0.0
    for index, flow_value in enumerate(recent_flows[:-2]):
        if flow_value <= 0.0:
            continue
        spike_strength = _safe_div(flow_value, flow_scale) * max(_safe_div(recent_turnover[index], turnover_baseline), 0.5)
        if spike_strength < 0.8:
            continue
        follow_closes = recent_closes[index + 1 : index + 4]
        follow_turnover = recent_turnover[index + 1 : index + 3]
        if len(follow_closes) < 3:
            continue
        post_distribution = 0.0
        for offset, close_px in enumerate(follow_closes):
            prev_close = recent_closes[index + offset + 1]
            if prev_close <= 0.0:
                continue
            ret_value = _safe_div(close_px - prev_close, prev_close)
            if ret_value >= 0.0:
                continue
            turn_rel = _safe_div(follow_turnover[min(offset, len(follow_turnover) - 1)], turnover_baseline)
            post_distribution += abs(ret_value) * max(turn_rel, 0.0)
        hold_ratio = _mean([1.0 if close_px >= recent_closes[index + 1] * 0.985 else 0.0 for close_px in follow_closes])
        score = min(spike_strength, 1.5) * hold_ratio * max(1.0 - min(post_distribution / 0.06, 1.0), 0.0)
        best_score = max(best_score, score)
    return max(min(best_score, 1.2), 0.0)


def recrowding_reacceleration_quality_12(
    closes: Sequence[float],
    turnover: Sequence[float],
    flows: Sequence[float],
) -> float:
    if len(closes) < 13 or len(turnover) < 12 or len(flows) < 12:
        return 0.0
    recent_closes = [float(value) for value in closes[-13:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-12:]]
    recent_flows = [float(value) for value in flows[-12:]]
    turnover_base = _mean(recent_turnover[:6]) + 1e-12
    flow_scale = _mean([abs(value) for value in recent_flows]) + 1e-12
    best_score = 0.0
    for index in range(3, len(recent_flows) - 2):
        prior_flow_mean = _mean(recent_flows[max(0, index - 3) : index])
        if prior_flow_mean >= 0.0:
            continue
        rebound_flow = recent_flows[index]
        if rebound_flow <= 0.0:
            continue
        rebound_strength = _safe_div(rebound_flow - prior_flow_mean, flow_scale)
        if rebound_strength < 0.9:
            continue
        prev_close = recent_closes[index]
        rebound_close = recent_closes[index + 1]
        if prev_close <= 0.0:
            continue
        rebound_return = _safe_div(rebound_close - prev_close, prev_close)
        if rebound_return <= 0.0:
            continue
        follow_closes = recent_closes[index + 1 : index + 4]
        follow_turnover = recent_turnover[index : index + 3]
        if len(follow_closes) < 3 or len(follow_turnover) < 3:
            continue
        hold = _mean([1.0 if close_px >= rebound_close * 0.985 else 0.0 for close_px in follow_closes])
        turnover_reacc = min(max(_safe_div(_mean(follow_turnover), turnover_base), 0.6), 1.8)
        churn_penalty = max(turnover_reacc - 1.35, 0.0) / 0.45
        score = min(rebound_strength / 1.6, 1.0) * min(rebound_return / 0.06, 1.0) * hold * max(1.0 - min(churn_penalty, 1.0), 0.0)
        best_score = max(best_score, score)
    return max(min(best_score, 1.2), 0.0)


def serial_outflow_damage_dampening_12(
    closes: Sequence[float],
    turnover: Sequence[float],
    flows: Sequence[float],
) -> float:
    if len(closes) < 13 or len(turnover) < 12 or len(flows) < 12:
        return 0.0
    recent_closes = [float(value) for value in closes[-13:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-12:]]
    recent_flows = [float(value) for value in flows[-12:]]
    turnover_baseline = _mean(recent_turnover) + 1e-12
    impacts: list[float] = []
    for index, flow_value in enumerate(recent_flows):
        if flow_value >= 0.0:
            continue
        prev_close = recent_closes[index]
        close_value = recent_closes[index + 1]
        if prev_close <= 0.0:
            continue
        damage = max(_safe_div(prev_close - close_value, prev_close), 0.0)
        pressure = max(-flow_value, 0.0) / (_safe_div(recent_turnover[index], turnover_baseline) + 0.2)
        impacts.append(_safe_div(damage, pressure + 1e-12))
    if len(impacts) < 4:
        return 0.0
    split = max(len(impacts) // 2, 1)
    early = _mean(impacts[:split])
    late = _mean(impacts[split:])
    dampening = max(_safe_div(early - late, early + late + 1e-12), 0.0)
    base_floor = min(recent_closes[-6:-1]) if len(recent_closes) >= 6 else min(recent_closes[:-1])
    repair = max(_safe_div(recent_closes[-1] - base_floor, base_floor if base_floor > 0.0 else 1.0), 0.0)
    return max(min(dampening * min(repair / 0.05 + 0.5, 1.2), 1.2), 0.0)


def volume_price_confirmation_10(prices: Sequence[float], turnover: Sequence[float]) -> float:
    if len(prices) < 11 or len(turnover) < 11:
        return 0.0
    size = min(len(prices), len(turnover))
    closes = [float(value) for value in prices[-size:]]
    traded = [max(float(value), 0.0) for value in turnover[-size:]]
    returns = [_safe_div(closes[index] - closes[index - 1], closes[index - 1]) for index in range(1, len(closes))]
    if len(returns) < 10:
        return 0.0
    recent_returns = returns[-10:]
    recent_turnover = traded[-10:]
    turnover_baseline = _mean(traded[-11:-1])
    turnover_scale = abs(turnover_baseline) + 1.0
    support = 0.0
    distribution = 0.0
    for ret_value, turnover_value in zip(recent_returns, recent_turnover):
        turnover_impulse = _safe_div(turnover_value - turnover_baseline, turnover_scale)
        if ret_value > 0.0:
            support += ret_value * max(turnover_impulse, 0.0)
        elif ret_value < 0.0:
            distribution += abs(ret_value) * max(turnover_impulse, 0.0)
    return support - distribution


def trend_stability_15(prices: Sequence[float]) -> float:
    if len(prices) < 16:
        return 0.0
    returns = [_safe_div(prices[index + 1] - prices[index], prices[index]) for index in range(len(prices) - 1)]
    window = returns[-15:]
    sign_changes = 0
    prev_sign = 0
    for value in window:
        sign = 1 if value > 0 else -1 if value < 0 else 0
        if prev_sign != 0 and sign != 0 and sign != prev_sign:
            sign_changes += 1
        if sign != 0:
            prev_sign = sign
    stability = 1.0 - _safe_div(sign_changes, 14.0)
    direction = 1.0 if _mean(window) > 0 else -1.0 if _mean(window) < 0 else 0.0
    return stability * direction


def highs_stepup_consistency_20(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> float:
    if len(highs) < 20 or len(lows) < 20 or len(closes) < 20:
        return 0.0
    recent_highs = [float(value) for value in highs[-20:]]
    recent_lows = [float(value) for value in lows[-20:]]
    recent_closes = [float(value) for value in closes[-20:]]
    ceilings = [max(recent_highs[max(0, index - 4) : index + 1]) for index in range(len(recent_highs))]
    early_ceiling = _mean(ceilings[:5])
    late_ceiling = _mean(ceilings[-5:])
    if early_ceiling <= 0.0:
        return 0.0
    lift = max(_safe_div(late_ceiling - early_ceiling, early_ceiling), 0.0)
    avg_range = _mean(
        [
            _safe_div(high_px - low_px, close_px if close_px > 0.0 else 1.0)
            for high_px, low_px, close_px in zip(recent_highs, recent_lows, recent_closes)
        ]
    )
    violation_rate = _mean(
        [
            1.0 if recent_closes[index] < ceilings[index - 1] * 0.985 else 0.0
            for index in range(10, len(recent_closes))
        ]
    )
    score = _safe_div(lift, avg_range + 0.02) * max(1.0 - violation_rate, 0.0)
    return max(min(score, 1.5), 0.0)


def failed_breakout_scarcity_20(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(opens) < 20 or len(highs) < 20 or len(lows) < 20 or len(closes) < 20 or len(turnover) < 20:
        return 0.0
    recent_opens = [float(value) for value in opens[-20:]]
    recent_highs = [float(value) for value in highs[-20:]]
    recent_lows = [float(value) for value in lows[-20:]]
    recent_closes = [float(value) for value in closes[-20:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-20:]]
    failure_pressure = 0.0
    failure_count = 0
    for index in range(10, len(recent_closes)):
        reference = max(recent_highs[max(0, index - 10) : index])
        if reference <= 0.0 or recent_highs[index] <= reference * 1.002 or recent_closes[index] >= reference * 0.998:
            continue
        day_range = recent_highs[index] - recent_lows[index]
        if day_range <= 1e-12:
            continue
        turnover_baseline = _mean(recent_turnover[max(0, index - 5) : index]) + 1.0
        upper_shadow_ratio = _safe_div(recent_highs[index] - max(recent_opens[index], recent_closes[index]), day_range)
        turnover_rel = _safe_div(recent_turnover[index], turnover_baseline)
        breakdown_back_under = max(_safe_div(reference - recent_closes[index], reference), 0.0)
        failure_pressure += (upper_shadow_ratio * 0.45 + breakdown_back_under * 8.0 * 0.55) * min(turnover_rel, 2.0)
        failure_count += 1
    if failure_count <= 0:
        return 1.0
    penalty = min(failure_pressure / max(float(failure_count), 1.0), 1.0)
    return max(1.0 - penalty, 0.0)


def advance_participation_uniformity_20(prices: Sequence[float]) -> float:
    if len(prices) < 21:
        return 0.0
    returns = [_safe_div(prices[index + 1] - prices[index], prices[index]) for index in range(len(prices) - 1)]
    recent = [float(value) for value in returns[-20:]]
    positives = sorted([value for value in recent if value > 0.0], reverse=True)
    if not positives:
        return 0.0
    positive_sum = sum(positives)
    if positive_sum <= 1e-12:
        return 0.0
    concentration = _safe_div(sum(positives[:2]), positive_sum)
    breadth = _safe_div(len(positives), 20.0)
    return max(1.0 - concentration, 0.0) * breadth


def breakout_pressure_10(prices: Sequence[float]) -> float:
    if len(prices) < 12:
        return 0.0
    returns = [_safe_div(prices[index + 1] - prices[index], prices[index]) for index in range(len(prices) - 1)]
    window = returns[-10:]
    baseline = _mean(window[:7])
    burst = _mean(window[7:])
    return burst - baseline


def turnover_impulse_5_20(turnover: Sequence[float]) -> float:
    if len(turnover) < 20:
        return 0.0
    short = _mean(turnover[-5:])
    long = _mean(turnover[-20:])
    return _safe_div(short - long, abs(long) + 1.0)


def turnover_stability_20(turnover: Sequence[float]) -> float:
    if len(turnover) < 20:
        return 0.0
    window = [max(float(value), 0.0) for value in turnover[-20:]]
    mean_turnover = _mean(window)
    std_turnover = _std(window)
    variation = _safe_div(std_turnover, mean_turnover + 1.0)
    return _safe_div(1.0, 1.0 + variation)


def amihud_illiquidity_20(prices: Sequence[float], turnover: Sequence[float]) -> float:
    if len(prices) < 21 or len(turnover) < 21:
        return 0.0
    size = min(len(prices), len(turnover))
    values: list[float] = []
    for index in range(size - 20, size):
        if index <= 0:
            continue
        base = float(prices[index - 1])
        ret = _safe_div(float(prices[index]) - base, base)
        cash_turnover = max(float(turnover[index]), 1.0)
        values.append(abs(ret) / cash_turnover)
    return _mean(values)


def liquidity_impact_relief_10(prices: Sequence[float], turnover: Sequence[float]) -> float:
    if len(prices) < 21 or len(turnover) < 21:
        return 0.0
    size = min(len(prices), len(turnover))
    recent_prices = [float(value) for value in prices[-(min(size, 21)):]]
    recent_turnover = [max(float(value), 1.0) for value in turnover[-(min(size, 21)):]]
    if len(recent_prices) < 21 or min(recent_prices) <= 0.0:
        return 0.0
    returns = [
        _safe_div(recent_prices[index + 1] - recent_prices[index], recent_prices[index])
        for index in range(len(recent_prices) - 1)
    ]
    turnover_baseline = _mean(recent_turnover[:-1]) + 1.0
    impact = [
        _safe_div(abs(ret_value), _safe_div(turnover_value, turnover_baseline) + 0.2)
        for ret_value, turnover_value in zip(returns, recent_turnover[1:])
    ]
    early_impact = _mean(impact[:10])
    late_impact = _mean(impact[-5:])
    if early_impact <= 1e-12:
        return 0.0
    impact_relief = max(_safe_div(early_impact - late_impact, early_impact), -1.0)
    up_efficiency_values: list[float] = []
    down_damage_values: list[float] = []
    for ret_value, turnover_value in zip(returns[-10:], recent_turnover[-10:]):
        turnover_rel = _safe_div(turnover_value, turnover_baseline)
        if ret_value > 0.0:
            up_efficiency_values.append(_safe_div(ret_value, turnover_rel + 0.2))
        elif ret_value < 0.0:
            down_damage_values.append(_safe_div(abs(ret_value), turnover_rel + 0.2))
    up_efficiency = _mean(up_efficiency_values) if up_efficiency_values else 0.0
    down_damage = _mean(down_damage_values) if down_damage_values else 0.0
    asymmetry = _safe_div(up_efficiency - down_damage, up_efficiency + down_damage + 1e-12)
    turnover_calm = max(1.0 - min(_safe_div(_std(recent_turnover[-10:]), _mean(recent_turnover[-10:]) + 1.0), 1.0), 0.0)
    score = 0.50 * impact_relief + 0.35 * asymmetry + 0.15 * turnover_calm
    return max(min(score, 1.5), -1.5)


def flow_correlation_10(prices: Sequence[float], flows: Sequence[float]) -> float:
    if len(prices) < 11 or len(flows) < 10:
        return 0.0
    returns = [_safe_div(prices[index + 1] - prices[index], prices[index]) for index in range(len(prices) - 1)]
    return _correlation(returns[-10:], flows[-10:])


def exhaustion_risk_10(prices: Sequence[float], turnover: Sequence[float]) -> float:
    if len(prices) < 11 or len(turnover) < 11:
        return 0.0
    returns = [_safe_div(prices[index + 1] - prices[index], prices[index]) for index in range(len(prices) - 1)]
    recent_returns = returns[-10:]
    up_spikes = _safe_div(len([value for value in recent_returns if value >= 0.03]), 10.0)
    recent_turnover = turnover[-11:]
    baseline = recent_turnover[:-1]
    std = _std(baseline)
    turnover_z = _safe_div(recent_turnover[-1] - _mean(baseline), std if std > 1e-12 else 1.0)
    recent_window = [float(value) for value in prices[-10:]]
    near_high = 1.0 if recent_window[-1] >= max(recent_window) * 0.995 else 0.0
    pullback = _safe_div(max(recent_window) - recent_window[-1], max(recent_window))
    trend_run = _safe_div(recent_window[-1] - recent_window[0], recent_window[0])
    extension = max(trend_run - 0.12, 0.0) / 0.18
    weighted = (
        up_spikes * 0.30
        + max(min(turnover_z / 2.5, 1.0), 0.0) * 0.20
        + near_high * max(min(extension, 1.0), 0.0) * 0.30
        + max(0.015 - pullback, 0.0) / 0.015 * 0.20
    )
    return max(0.0, min(weighted, 1.0))


def abnormal_turnover_z20(turnover: Sequence[float]) -> float:
    if len(turnover) < 21:
        return 0.0
    window = turnover[-21:-1]
    std = _std(window)
    return _safe_div(turnover[-1] - _mean(window), std if std > 1e-12 else 1.0)


def close_drive_strength_30m(closes_30m: Sequence[float]) -> float:
    if len(closes_30m) < 8:
        return 0.0
    tail = closes_30m[-2:]
    early = closes_30m[-8:-2]
    return _safe_div(_mean(tail) - _mean(early), _mean(early))


def vwap_dislocation_30m(closes_30m: Sequence[float], vwap_30m: Sequence[float]) -> float:
    if len(closes_30m) < 4 or len(vwap_30m) < 4:
        return 0.0
    return _safe_div(closes_30m[-1] - vwap_30m[-1], vwap_30m[-1])


def auction_gap_strength_1d(opens_1d: Sequence[float], closes_1d: Sequence[float]) -> float:
    if len(opens_1d) < 1 or len(closes_1d) < 2:
        return 0.0
    return _safe_div(opens_1d[-1] - closes_1d[-2], closes_1d[-2])


def auction_turnover_imbalance_30m(turnover_30m: Sequence[float]) -> float:
    if len(turnover_30m) < 2:
        return 0.0
    opening = max(float(turnover_30m[0]), 0.0)
    baseline_window = turnover_30m[1:5] if len(turnover_30m) >= 5 else turnover_30m[1:]
    baseline = _mean([max(float(value), 0.0) for value in baseline_window])
    return _safe_div(opening - baseline, abs(baseline) + 1.0)


def opening_impulse_retention_30m(opens_30m: Sequence[float], closes_30m: Sequence[float]) -> float:
    if len(opens_30m) < 1 or len(closes_30m) < 2:
        return 0.0
    start_open = opens_30m[0]
    if start_open <= 0:
        return 0.0
    opening_impulse = _safe_div(closes_30m[0] - start_open, start_open)
    full_day_move = _safe_div(closes_30m[-1] - start_open, start_open)
    if abs(opening_impulse) < 1e-8:
        return 0.0
    return _safe_div(full_day_move, abs(opening_impulse))


def intraday_trend_consistency_30m(closes_30m: Sequence[float]) -> float:
    if len(closes_30m) < 4:
        return 0.0
    returns = [_safe_div(closes_30m[index + 1] - closes_30m[index], closes_30m[index]) for index in range(len(closes_30m) - 1)]
    mean_ret = _mean(returns)
    if abs(mean_ret) < 1e-10:
        return 0.0
    dominant_sign = 1 if mean_ret > 0 else -1
    aligned = sum(1 for value in returns if (value > 0 and dominant_sign > 0) or (value < 0 and dominant_sign < 0))
    consistency = _safe_div(aligned, len(returns))
    return consistency * dominant_sign


def intraday_range_close_location_30m(
    highs_30m: Sequence[float],
    lows_30m: Sequence[float],
    closes_30m: Sequence[float],
) -> float:
    if len(highs_30m) < 1 or len(lows_30m) < 1 or len(closes_30m) < 1:
        return 0.0
    day_high = max(highs_30m)
    day_low = min(lows_30m)
    if day_high <= day_low:
        return 0.0
    position = _safe_div(closes_30m[-1] - day_low, day_high - day_low)
    return max(min(position - 0.5, 0.5), -0.5)


def large_candle_followthrough_5(prices: Sequence[float]) -> float:
    if len(prices) < 6:
        return 0.0
    returns = [_safe_div(prices[index + 1] - prices[index], prices[index]) for index in range(len(prices) - 1)]
    recent = returns[-5:]
    strong = [value for value in recent if value > 0.02]
    return _safe_div(len(strong), len(recent))


def distribution_warning_5(prices: Sequence[float], turnover: Sequence[float]) -> float:
    if len(prices) < 6 or len(turnover) < 6:
        return 0.0
    warnings = 0
    for index in range(1, 6):
        price_down = prices[-index] < prices[-index - 1]
        volume_up = turnover[-index] > turnover[-index - 1]
        if price_down and volume_up:
            warnings += 1
    return _safe_div(warnings, 5.0)


def sector_rs_20(sector_closes: Sequence[float], market_closes: Sequence[float]) -> float:
    if len(sector_closes) < 20 or len(market_closes) < 20:
        return 0.0
    sector_ret = _safe_div(sector_closes[-1] - sector_closes[-20], sector_closes[-20])
    market_ret = _safe_div(market_closes[-1] - market_closes[-20], market_closes[-20])
    return sector_ret - market_ret


def sector_breadth_10(up_count_series: Sequence[float], total_count_series: Sequence[float]) -> float:
    if len(up_count_series) < 10 or len(total_count_series) < 10:
        return 0.0
    up_recent = _mean(up_count_series[-10:])
    total_recent = max(_mean(total_count_series[-10:]), 1.0)
    return _safe_div(up_recent, total_recent)


def sector_flow_accel_5(flows: Sequence[float]) -> float:
    if len(flows) < 6:
        return 0.0
    prev = _mean(flows[-6:-3])
    recent = _mean(flows[-3:])
    baseline = max(abs(prev), abs(recent), 1.0)
    return _safe_div(recent - prev, baseline)


def sector_flow_breadth_confirm_5_10(
    flows: Sequence[float],
    up_count_series: Sequence[float],
    total_count_series: Sequence[float],
) -> float:
    if len(flows) < 10 or len(up_count_series) < 10 or len(total_count_series) < 10:
        return 0.0
    recent_flow = _mean([float(value) for value in flows[-5:]])
    base_flow = _mean([float(value) for value in flows[-10:]])
    flow_impulse = _safe_div(recent_flow - base_flow, abs(base_flow) + 1.0)
    recent_breadth = _safe_div(_mean(up_count_series[-5:]), max(_mean(total_count_series[-5:]), 1.0))
    base_breadth = _safe_div(_mean(up_count_series[-10:]), max(_mean(total_count_series[-10:]), 1.0))
    breadth_impulse = recent_breadth - base_breadth
    return max(min(flow_impulse * max(breadth_impulse + 0.05, 0.0) * 4.0, 1.5), 0.0)


def rotation_early_diffusion_10_20(
    sector_closes: Sequence[float],
    market_closes: Sequence[float],
    up_count_series: Sequence[float],
    total_count_series: Sequence[float],
) -> float:
    if len(sector_closes) < 21 or len(market_closes) < 21 or len(up_count_series) < 20 or len(total_count_series) < 20:
        return 0.0
    sector = [float(value) for value in sector_closes[-21:]]
    market = [float(value) for value in market_closes[-21:]]
    if min(sector) <= 0.0 or min(market) <= 0.0:
        return 0.0
    breadth = [
        _safe_div(float(up_count), max(float(total_count), 1.0))
        for up_count, total_count in zip(up_count_series[-20:], total_count_series[-20:])
    ]
    if len(breadth) < 20:
        return 0.0
    base_breadth = _mean(breadth[:10])
    recent_breadth = _mean(breadth[-5:])
    breadth_impulse = recent_breadth - base_breadth
    breadth_accel = (breadth[-1] - breadth[-5]) - (breadth[-10] - breadth[-14])
    sector_ret_10 = _safe_div(sector[-1] - sector[-11], sector[-11])
    market_ret_10 = _safe_div(market[-1] - market[-11], market[-11])
    sector_ret_20 = _safe_div(sector[-1] - sector[0], sector[0])
    market_ret_20 = _safe_div(market[-1] - market[0], market[0])
    excess_10 = sector_ret_10 - market_ret_10
    excess_20 = sector_ret_20 - market_ret_20
    excess_accel = excess_10 - 0.5 * excess_20
    early_window = max(0.0, 1.0 - max(recent_breadth - 0.68, 0.0) / 0.22)
    base_reset = max(0.0, min((0.62 - base_breadth) / 0.25, 1.0))
    diffusion = max(breadth_impulse, 0.0) * 2.4 + max(breadth_accel, 0.0) * 1.5
    trend_confirmation = max(excess_accel, 0.0) * 4.0 + max(excess_10, 0.0) * 1.2
    score = (0.58 * diffusion + 0.42 * trend_confirmation) * early_window * (0.55 + 0.45 * base_reset)
    return max(min(score, 1.5), 0.0)


def leader_laggard_spread_10(leaders: Sequence[float], laggards: Sequence[float]) -> float:
    if len(leaders) < 10 or len(laggards) < 10:
        return 0.0
    leader_ret = _safe_div(leaders[-1] - leaders[-10], leaders[-10])
    laggard_ret = _safe_div(laggards[-1] - laggards[-10], laggards[-10])
    return leader_ret - laggard_ret


def stock_net_inflow_5_20(flows: Sequence[float]) -> float:
    if len(flows) < 20:
        return 0.0
    return _safe_div(_mean(flows[-5:]) - _mean(flows[-20:]), abs(_mean(flows[-20:])) + 1.0)


def stock_sector_flow_resonance_10(stock_flows: Sequence[float], sector_flows: Sequence[float]) -> float:
    if len(stock_flows) < 10 or len(sector_flows) < 10:
        return 0.0
    stock_recent = [float(value) for value in stock_flows[-10:]]
    sector_recent = [float(value) for value in sector_flows[-10:]]
    flow_corr = max(_correlation(stock_recent, sector_recent), 0.0)
    stock_impulse = max(
        _safe_div(_mean(stock_recent[-5:]) - _mean(stock_recent), abs(_mean(stock_recent)) + 1.0),
        0.0,
    )
    sector_impulse = max(
        _safe_div(_mean(sector_recent[-5:]) - _mean(sector_recent), abs(_mean(sector_recent)) + 1.0),
        0.0,
    )
    return max(min(flow_corr * (1.0 + stock_impulse) * (1.0 + sector_impulse), 2.0), 0.0)


def stock_flow_residual_vs_sector_10(
    prices: Sequence[float],
    stock_flows: Sequence[float],
    sector_flows: Sequence[float],
) -> float:
    if len(prices) < 11 or len(stock_flows) < 10 or len(sector_flows) < 10:
        return 0.0
    stock_recent = [float(value) for value in stock_flows[-10:]]
    sector_recent = [float(value) for value in sector_flows[-10:]]
    stock_scale = _mean([abs(value) for value in stock_recent]) + 1.0
    sector_scale = _mean([abs(value) for value in sector_recent]) + 1.0
    stock_norm = [value / stock_scale for value in stock_recent]
    sector_norm = [value / sector_scale for value in sector_recent]
    sector_mean = _mean(sector_norm)
    stock_mean = _mean(stock_norm)
    cov = 0.0
    sector_var = 0.0
    for stock_value, sector_value in zip(stock_norm, sector_norm):
        sector_delta = sector_value - sector_mean
        cov += (stock_value - stock_mean) * sector_delta
        sector_var += sector_delta * sector_delta
    beta = _safe_div(cov, sector_var if sector_var > 1e-12 else 1.0)
    residuals = [stock_value - beta * sector_value for stock_value, sector_value in zip(stock_norm, sector_norm)]
    return _mean(residuals[-5:])


def stock_flow_residual_acceleration_5_20(
    prices: Sequence[float],
    stock_flows: Sequence[float],
    sector_flows: Sequence[float],
) -> float:
    if len(prices) < 21 or len(stock_flows) < 20 or len(sector_flows) < 20:
        return 0.0
    stock_recent = [float(value) for value in stock_flows[-20:]]
    sector_recent = [float(value) for value in sector_flows[-20:]]
    stock_scale = _mean([abs(value) for value in stock_recent]) + 1.0
    sector_scale = _mean([abs(value) for value in sector_recent]) + 1.0
    stock_norm = [value / stock_scale for value in stock_recent]
    sector_norm = [value / sector_scale for value in sector_recent]
    sector_mean = _mean(sector_norm)
    stock_mean = _mean(stock_norm)
    cov = 0.0
    sector_var = 0.0
    for stock_value, sector_value in zip(stock_norm, sector_norm):
        sector_delta = sector_value - sector_mean
        cov += (stock_value - stock_mean) * sector_delta
        sector_var += sector_delta * sector_delta
    beta = _safe_div(cov, sector_var if sector_var > 1e-12 else 1.0)
    residuals = [stock_value - beta * sector_value for stock_value, sector_value in zip(stock_norm, sector_norm)]
    recent_resid = _mean(residuals[-5:])
    base_resid = _mean(residuals[:-5])
    scale = _mean([abs(value) for value in residuals]) + 1e-12
    return max(min(_safe_div(recent_resid - base_resid, scale), 1.5), -1.5)


def stock_defensive_residual_on_sector_selloff_15(
    prices: Sequence[float],
    sector_closes: Sequence[float],
    stock_flows: Sequence[float],
    sector_flows: Sequence[float],
) -> float:
    if len(prices) < 16 or len(sector_closes) < 16 or len(stock_flows) < 15 or len(sector_flows) < 15:
        return 0.0
    stock_recent = [float(value) for value in stock_flows[-15:]]
    sector_recent = [float(value) for value in sector_flows[-15:]]
    stock_scale = _mean([abs(value) for value in stock_recent]) + 1.0
    sector_scale = _mean([abs(value) for value in sector_recent]) + 1.0
    stock_norm = [value / stock_scale for value in stock_recent]
    sector_norm = [value / sector_scale for value in sector_recent]
    sector_mean = _mean(sector_norm)
    stock_mean = _mean(stock_norm)
    cov = 0.0
    sector_var = 0.0
    for stock_value, sector_value in zip(stock_norm, sector_norm):
        sector_delta = sector_value - sector_mean
        cov += (stock_value - stock_mean) * sector_delta
        sector_var += sector_delta * sector_delta
    beta = _safe_div(cov, sector_var if sector_var > 1e-12 else 1.0)
    stock_returns = [_safe_div(prices[index + 1] - prices[index], prices[index]) for index in range(len(prices) - 1)][-15:]
    sector_returns = [
        _safe_div(sector_closes[index + 1] - sector_closes[index], sector_closes[index])
        for index in range(len(sector_closes) - 1)
    ][-15:]
    weighted_sum = 0.0
    total_weight = 0.0
    for stock_ret, sector_ret, stock_value, sector_value in zip(stock_returns, sector_returns, stock_norm, sector_norm):
        if sector_value >= -1e-12:
            continue
        weight = abs(sector_value)
        residual = stock_value - beta * sector_value
        weighted_sum += (0.5 * residual + 0.5 * (stock_ret - sector_ret)) * weight
        total_weight += weight
    if total_weight <= 1e-12:
        return 0.0
    return max(min(weighted_sum / total_weight, 1.5), -1.5)


def relative_drawdown_resilience_15(
    prices: Sequence[float],
    sector_closes: Sequence[float],
    market_closes: Sequence[float],
) -> float:
    if len(prices) < 16 or len(sector_closes) < 16 or len(market_closes) < 16:
        return 0.0
    stock_recent = [float(value) for value in prices[-16:]]
    sector_recent = [float(value) for value in sector_closes[-16:]]
    market_recent = [float(value) for value in market_closes[-16:]]
    if min(stock_recent) <= 0.0 or min(sector_recent) <= 0.0 or min(market_recent) <= 0.0:
        return 0.0
    def local_drawdown(series: Sequence[float]) -> float:
        peak = 1.0
        equity = 1.0
        worst = 0.0
        for index in range(len(series) - 1):
            base = float(series[index])
            nxt = float(series[index + 1])
            ret = _safe_div(nxt - base, base) if base > 0.0 else 0.0
            equity *= 1.0 + ret
            peak = max(peak, equity)
            worst = max(worst, _safe_div(peak - equity, peak))
        return worst

    stock_drawdown = local_drawdown(stock_recent)
    sector_drawdown = local_drawdown(sector_recent)
    market_drawdown = local_drawdown(market_recent)
    if sector_drawdown <= 1e-12 and market_drawdown <= 1e-12:
        return 0.0
    sector_resilience = max(_safe_div(sector_drawdown - stock_drawdown, sector_drawdown + 1e-12), -1.0)
    market_resilience = max(_safe_div(market_drawdown - stock_drawdown, market_drawdown + 1e-12), -1.0)
    recent_stock_return = _safe_div(stock_recent[-1] - stock_recent[-6], stock_recent[-6]) if stock_recent[-6] > 0.0 else 0.0
    recent_sector_return = _safe_div(sector_recent[-1] - sector_recent[-6], sector_recent[-6]) if sector_recent[-6] > 0.0 else 0.0
    recent_market_return = _safe_div(market_recent[-1] - market_recent[-6], market_recent[-6]) if market_recent[-6] > 0.0 else 0.0
    recapture = 0.5 * (recent_stock_return - recent_sector_return) + 0.5 * (recent_stock_return - recent_market_return)
    score = sector_resilience * 0.45 + market_resilience * 0.35 + max(recapture, -0.08) * 2.0
    return max(min(score, 1.5), -1.5)


def stock_sector_breadth_capture_10(
    prices: Sequence[float],
    sector_closes: Sequence[float],
    stock_flows: Sequence[float],
    sector_flows: Sequence[float],
    sector_up_counts: Sequence[float],
    sector_total_counts: Sequence[float],
) -> float:
    if (
        len(prices) < 11
        or len(sector_closes) < 11
        or len(stock_flows) < 10
        or len(sector_flows) < 10
        or len(sector_up_counts) < 10
        or len(sector_total_counts) < 10
    ):
        return 0.0
    stock_returns = [_safe_div(prices[index + 1] - prices[index], prices[index]) for index in range(len(prices) - 1)][-10:]
    sector_returns = [
        _safe_div(sector_closes[index + 1] - sector_closes[index], sector_closes[index])
        for index in range(len(sector_closes) - 1)
    ][-10:]
    stock_recent = [float(value) for value in stock_flows[-10:]]
    sector_recent = [float(value) for value in sector_flows[-10:]]
    stock_scale = _mean([abs(value) for value in stock_recent]) + 1.0
    sector_scale = _mean([abs(value) for value in sector_recent]) + 1.0
    stock_norm = [value / stock_scale for value in stock_recent]
    sector_norm = [value / sector_scale for value in sector_recent]
    sector_mean = _mean(sector_norm)
    stock_mean = _mean(stock_norm)
    cov = 0.0
    sector_var = 0.0
    for stock_value, sector_value in zip(stock_norm, sector_norm):
        sector_delta = sector_value - sector_mean
        cov += (stock_value - stock_mean) * sector_delta
        sector_var += sector_delta * sector_delta
    beta = _safe_div(cov, sector_var if sector_var > 1e-12 else 1.0)
    weighted_sum = 0.0
    total_weight = 0.0
    for stock_ret, sector_ret, stock_value, sector_value, up_count, total_count in zip(
        stock_returns,
        sector_returns,
        stock_norm,
        sector_norm,
        sector_up_counts[-10:],
        sector_total_counts[-10:],
    ):
        breadth = _safe_div(float(up_count), max(float(total_count), 1.0))
        tailwind = max(sector_value, 0.0) * max(breadth - 0.5, 0.0)
        if tailwind <= 1e-12:
            continue
        residual = stock_value - beta * sector_value
        excess_return = stock_ret - sector_ret
        weighted_sum += (excess_return + 0.5 * residual) * tailwind
        total_weight += tailwind
    if total_weight <= 1e-12:
        return 0.0
    return weighted_sum / total_weight


def stock_sector_breadth_lead_10(
    stock_flows: Sequence[float],
    sector_flows: Sequence[float],
    sector_up_counts: Sequence[float],
    sector_total_counts: Sequence[float],
) -> float:
    if len(stock_flows) < 11 or len(sector_flows) < 11 or len(sector_up_counts) < 11 or len(sector_total_counts) < 11:
        return 0.0
    stock_recent = [float(value) for value in stock_flows[-11:]]
    sector_recent = [float(value) for value in sector_flows[-11:]]
    stock_scale = _mean([abs(value) for value in stock_recent]) + 1.0
    sector_scale = _mean([abs(value) for value in sector_recent]) + 1.0
    stock_norm = [value / stock_scale for value in stock_recent]
    sector_norm = [value / sector_scale for value in sector_recent]
    sector_mean = _mean(sector_norm)
    stock_mean = _mean(stock_norm)
    cov = 0.0
    sector_var = 0.0
    for stock_value, sector_value in zip(stock_norm, sector_norm):
        sector_delta = sector_value - sector_mean
        cov += (stock_value - stock_mean) * sector_delta
        sector_var += sector_delta * sector_delta
    beta = _safe_div(cov, sector_var if sector_var > 1e-12 else 1.0)
    residuals = [stock_value - beta * sector_value for stock_value, sector_value in zip(stock_norm, sector_norm)]
    breadth = [
        _safe_div(float(up_count), max(float(total_count), 1.0))
        for up_count, total_count in zip(sector_up_counts[-11:], sector_total_counts[-11:])
    ]
    breadth_delta = [breadth[index + 1] - breadth[index] for index in range(len(breadth) - 1)]
    lead_pairs = list(zip(residuals[:-1], breadth_delta))
    if len(lead_pairs) < 5:
        return 0.0
    lead_residuals = [item[0] for item in lead_pairs]
    lead_breadth_delta = [item[1] for item in lead_pairs]
    correlation_score = max(_correlation(lead_residuals, lead_breadth_delta), 0.0)
    hit_rate = _safe_div(
        sum(1 for residual, delta in lead_pairs if residual > 0.0 and delta > 0.0),
        sum(1 for residual in lead_residuals if residual > 0.0),
    )
    tail_strength = max(_mean(residuals[-4:]), 0.0)
    return max(min((0.45 * correlation_score + 0.35 * hit_rate + 0.20 * min(tail_strength * 2.5, 1.0)), 1.2), 0.0)


def stock_relative_sponsorship_under_sector_crowding_10(
    stock_flows: Sequence[float],
    sector_flows: Sequence[float],
    sector_up_counts: Sequence[float],
    sector_total_counts: Sequence[float],
) -> float:
    if len(stock_flows) < 10 or len(sector_flows) < 10 or len(sector_up_counts) < 10 or len(sector_total_counts) < 10:
        return 0.0
    stock_recent = [float(value) for value in stock_flows[-10:]]
    sector_recent = [float(value) for value in sector_flows[-10:]]
    breadth = [
        _safe_div(float(up_count), max(float(total_count), 1.0))
        for up_count, total_count in zip(sector_up_counts[-10:], sector_total_counts[-10:])
    ]
    stock_scale = _mean([abs(value) for value in stock_recent]) + 1.0
    sector_scale = _mean([abs(value) for value in sector_recent]) + 1.0
    stock_norm = [value / stock_scale for value in stock_recent]
    sector_norm = [value / sector_scale for value in sector_recent]
    sector_mean = _mean(sector_norm)
    stock_mean = _mean(stock_norm)
    cov = 0.0
    sector_var = 0.0
    for stock_value, sector_value in zip(stock_norm, sector_norm):
        sector_delta = sector_value - sector_mean
        cov += (stock_value - stock_mean) * sector_delta
        sector_var += sector_delta * sector_delta
    beta = _safe_div(cov, sector_var if sector_var > 1e-12 else 1.0)
    weighted_sum = 0.0
    total_weight = 0.0
    for stock_value, sector_value, breadth_value in zip(stock_norm, sector_norm, breadth):
        if sector_value <= 0.0 or breadth_value >= 0.55:
            continue
        crowded_weight = sector_value * max(0.55 - breadth_value, 0.0)
        if crowded_weight <= 1e-12:
            continue
        residual = stock_value - beta * sector_value
        weighted_sum += residual * crowded_weight
        total_weight += crowded_weight
    if total_weight <= 1e-12:
        return 0.0
    return max(min(weighted_sum / total_weight, 1.5), -1.5)


def sector_flow_conversion_efficiency_20(
    sector_flows: Sequence[float],
    sector_closes: Sequence[float],
    market_closes: Sequence[float],
) -> float:
    if len(sector_flows) < 20 or len(sector_closes) < 21 or len(market_closes) < 21:
        return 0.0
    recent_flows = [float(value) for value in sector_flows[-20:]]
    sector_returns = [
        _safe_div(sector_closes[index + 1] - sector_closes[index], sector_closes[index])
        for index in range(len(sector_closes) - 1)
    ][-20:]
    market_returns = [
        _safe_div(market_closes[index + 1] - market_closes[index], market_closes[index])
        for index in range(len(market_closes) - 1)
    ][-20:]
    positive_flow = [max(value, 0.0) for value in recent_flows]
    flow_scale = _mean([abs(value) for value in recent_flows]) + 1.0
    realized_conversion = 0.0
    penalty = 0.0
    weight_total = 0.0
    for flow_value, sector_ret, market_ret in zip(positive_flow, sector_returns, market_returns):
        if flow_value <= 0.0:
            continue
        weight = flow_value / flow_scale
        excess_ret = sector_ret - market_ret
        realized_conversion += weight * excess_ret
        if excess_ret < 0.0:
            penalty += weight * abs(excess_ret)
        weight_total += weight
    if weight_total <= 1e-12:
        return 0.0
    normalized = realized_conversion / weight_total
    penalty_norm = penalty / weight_total
    return max(min((normalized * 18.0) - (penalty_norm * 12.0), 1.5), -1.5)


def sector_negative_flow_breadth_resilience_10(
    sector_flows: Sequence[float],
    up_count_series: Sequence[float],
    total_count_series: Sequence[float],
) -> float:
    if len(sector_flows) < 20 or len(up_count_series) < 20 or len(total_count_series) < 20:
        return 0.0
    breadth = [
        _safe_div(float(up_count), max(float(total_count), 1.0))
        for up_count, total_count in zip(up_count_series[-20:], total_count_series[-20:])
    ]
    recent_flows = [float(value) for value in sector_flows[-10:]]
    breadth_base = _mean(breadth[:10])
    weighted_sum = 0.0
    total_weight = 0.0
    flow_scale = _mean([abs(value) for value in recent_flows]) + 1.0
    for flow_value, breadth_value in zip(recent_flows, breadth[-10:]):
        if flow_value >= -1e-12:
            continue
        normalized_flow = abs(flow_value) / flow_scale
        resilience = _safe_div(breadth_value - breadth_base, normalized_flow + 0.2)
        weighted_sum += resilience * normalized_flow
        total_weight += normalized_flow
    if total_weight <= 1e-12:
        return 0.0
    return max(min(weighted_sum / total_weight, 1.5), -1.5)


def sector_flow_breadth_divergence_crowding_15(
    sector_flows: Sequence[float],
    up_count_series: Sequence[float],
    total_count_series: Sequence[float],
) -> float:
    if len(sector_flows) < 15 or len(up_count_series) < 15 or len(total_count_series) < 15:
        return 0.0
    recent_flows = [float(value) for value in sector_flows[-15:]]
    breadth = [
        _safe_div(float(up_count), max(float(total_count), 1.0))
        for up_count, total_count in zip(up_count_series[-15:], total_count_series[-15:])
    ]
    recent_flow_mean = _mean(recent_flows[-5:])
    flow_scale = _mean([abs(value) for value in recent_flows]) + 1.0
    flow_tailwind = max(_safe_div(recent_flow_mean, flow_scale), 0.0)
    breadth_recent = _mean(breadth[-5:])
    breadth_base = _mean(breadth[:10])
    breadth_gap = max(breadth_base - breadth_recent, 0.0)
    narrow_breadth_penalty = max(0.55 - breadth_recent, 0.0)
    crowding = flow_tailwind * (breadth_gap * 4.0 + narrow_breadth_penalty * 2.5)
    return max(min(crowding, 1.8), 0.0)


def sector_net_inflow_5_20(flows: Sequence[float]) -> float:
    return stock_net_inflow_5_20(flows)


def flow_persistence_10(flows: Sequence[float]) -> float:
    if len(flows) < 10:
        return 0.0
    same_direction = 0
    signs = [1 if value > 0 else -1 if value < 0 else 0 for value in flows[-10:]]
    for index in range(1, len(signs)):
        if signs[index] != 0 and signs[index] == signs[index - 1]:
            same_direction += 1
    return _safe_div(same_direction, 9.0)


def selloff_absorption_flow_10(prices: Sequence[float], flows: Sequence[float]) -> float:
    if len(prices) < 11 or len(flows) < 10:
        return 0.0
    returns = [_safe_div(prices[index + 1] - prices[index], prices[index]) for index in range(len(prices) - 1)]
    recent_returns = returns[-10:]
    recent_flows = [float(value) for value in flows[-10:]]
    numerator = 0.0
    denominator = 0.0
    for ret_value, flow_value in zip(recent_returns, recent_flows):
        selloff = abs(min(ret_value, 0.0))
        if selloff <= 0.0:
            continue
        numerator += max(flow_value, 0.0) * selloff
        numerator -= max(-flow_value, 0.0) * selloff
        denominator += abs(flow_value) * selloff
    if denominator <= 1e-12:
        return 0.0
    return max(min(numerator / denominator, 1.0), -1.0)


def downside_impact_decay_15(prices: Sequence[float], flows: Sequence[float]) -> float:
    if len(prices) < 16 or len(flows) < 15:
        return 0.0
    returns = [_safe_div(prices[index + 1] - prices[index], prices[index]) for index in range(len(prices) - 1)]
    recent_returns = returns[-15:]
    recent_flows = [float(value) for value in flows[-15:]]
    impact_series: list[float] = []
    for ret_value, flow_value in zip(recent_returns, recent_flows):
        if ret_value >= 0.0 and flow_value >= 0.0:
            continue
        impact_series.append(abs(min(ret_value, 0.0)) / (abs(min(flow_value, 0.0)) + 1.0))
    if len(impact_series) < 6:
        return 0.0
    split = len(impact_series) // 2
    early = _mean(impact_series[:split])
    late = _mean(impact_series[split:])
    if early <= 1e-12:
        return 0.0
    return max(min(_safe_div(early - late, early), 1.0), -1.0)


def negative_flow_impact_beta_10(prices: Sequence[float], flows: Sequence[float]) -> float:
    if len(prices) < 11 or len(flows) < 10:
        return 0.0
    returns = [_safe_div(prices[index + 1] - prices[index], prices[index]) for index in range(len(prices) - 1)]
    recent_returns = returns[-10:]
    recent_flows = [float(value) for value in flows[-10:]]
    flow_scale = _mean([abs(value) for value in recent_flows]) + 1e-12
    damage_scale = _mean([abs(value) for value in recent_returns]) + 1e-12
    points: list[tuple[float, float]] = []
    for ret_value, flow_value in zip(recent_returns, recent_flows):
        neg_flow = max(-flow_value, 0.0) / flow_scale
        if neg_flow <= 0.0:
            continue
        downside_damage = max(-ret_value, 0.0) / damage_scale
        points.append((neg_flow, downside_damage))
    if len(points) < 4:
        return 0.0
    mean_x = _mean([point[0] for point in points])
    mean_y = _mean([point[1] for point in points])
    var_x = 0.0
    cov_xy = 0.0
    for x_value, y_value in points:
        x_delta = x_value - mean_x
        cov_xy += x_delta * (y_value - mean_y)
        var_x += x_delta * x_delta
    if var_x <= 1e-12:
        return 0.0
    slope = max(cov_xy / var_x, 0.0)
    return max(1.0 - min(slope, 1.5) / 1.5, 0.0)


def impact_absorption_resilience_20(
    prices: Sequence[float],
    lows: Sequence[float],
    turnover: Sequence[float],
    flows: Sequence[float],
) -> float:
    if len(prices) < 21 or len(lows) < 21 or len(turnover) < 20 or len(flows) < 20:
        return 0.0
    closes = [float(value) for value in prices[-21:]]
    low_window = [float(value) for value in lows[-21:]]
    turnover_window = [float(value) for value in turnover[-20:]]
    flow_window = [float(value) for value in flows[-20:]]
    if min(closes) <= 0.0 or min(low_window) <= 0.0:
        return 0.0

    returns = [_safe_div(closes[index + 1] - closes[index], closes[index]) for index in range(20)]
    flow_scale = _mean([abs(value) for value in flow_window]) + 1.0
    turnover_base = _mean([max(value, 0.0) for value in turnover_window]) + 1e-12
    shock_indices = [
        index
        for index, (ret_value, flow_value) in enumerate(zip(returns, flow_window))
        if flow_value / flow_scale < -0.85 or ret_value < -0.025
    ]
    if not shock_indices:
        return 0.0

    impact_series = [
        _safe_div(max(-returns[index], 0.0), max(-flow_window[index] / flow_scale, 0.0) + 0.2)
        for index in shock_indices
    ]
    split = max(1, len(impact_series) // 2)
    early_impact = _mean(impact_series[:split])
    late_impact = _mean(impact_series[split:]) if len(impact_series) > split else impact_series[-1]
    damage_decay = 0.0 if early_impact <= 1e-12 else max(min((early_impact - late_impact) / early_impact, 1.0), -1.0)

    first_shock = shock_indices[0]
    shock_close = closes[first_shock]
    shock_low = low_window[first_shock + 1]
    post_lows = low_window[first_shock + 1 :]
    post_turnover = turnover_window[first_shock:]
    if shock_close <= 0.0 or shock_low <= 0.0 or not post_lows:
        return 0.0
    deepest_low = min(post_lows)
    undercut = max(_safe_div(shock_low - deepest_low, shock_low), 0.0)
    nonbreak = max(1.0 - undercut / 0.03, 0.0)
    reclaim = max(min(_safe_div(closes[-1] - deepest_low, shock_close) / 0.08, 1.0), 0.0)

    shock_turnover = max(turnover_window[first_shock], turnover_base)
    later_turnover = _mean([max(value, 0.0) for value in post_turnover[-5:]]) if post_turnover else turnover_base
    turnover_cool = max(1.0 - min(_safe_div(later_turnover, shock_turnover), 1.4) / 1.4, 0.0)
    liquidity_capacity = max(min(_safe_div(later_turnover, turnover_base), 1.5) / 1.5, 0.0)

    score = (
        0.30 * damage_decay
        + 0.24 * nonbreak
        + 0.20 * reclaim
        + 0.14 * turnover_cool
        + 0.12 * liquidity_capacity
    )
    return max(min(score, 1.5), -1.0)


def downside_impact_improvement_5_15(prices: Sequence[float], flows: Sequence[float]) -> float:
    if len(prices) < 16 or len(flows) < 15:
        return 0.0
    returns = [_safe_div(prices[index + 1] - prices[index], prices[index]) for index in range(len(prices) - 1)]
    recent_returns = returns[-15:]
    recent_flows = [float(value) for value in flows[-15:]]
    flow_scale = _mean([abs(value) for value in recent_flows]) + 1e-12
    impact_series: list[float] = []
    for ret_value, flow_value in zip(recent_returns, recent_flows):
        neg_flow = max(-flow_value, 0.0) / flow_scale
        if neg_flow <= 0.0 and ret_value >= 0.0:
            impact_series.append(0.0)
            continue
        downside_damage = max(-ret_value, 0.0)
        impact_series.append(_safe_div(downside_damage, neg_flow + 0.2))
    early = _mean(impact_series[:10])
    late = _mean(impact_series[10:])
    if early <= 1e-12:
        return 0.0
    return max(min(_safe_div(early - late, early + late + 1e-12), 1.0), -1.0)


def panic_followthrough_repair_8(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(opens) < 8 or len(highs) < 8 or len(lows) < 8 or len(closes) < 8 or len(turnover) < 8:
        return 0.0
    recent_opens = [float(value) for value in opens[-8:]]
    recent_highs = [float(value) for value in highs[-8:]]
    recent_lows = [float(value) for value in lows[-8:]]
    recent_closes = [float(value) for value in closes[-8:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-8:]]
    best_score = 0.0
    for index in range(1, len(recent_closes) - 2):
        prior_close = recent_closes[index - 1]
        if prior_close <= 0.0:
            continue
        gap_down = _safe_div(prior_close - recent_opens[index], prior_close)
        same_day_loss = _safe_div(prior_close - recent_closes[index], prior_close)
        if gap_down < 0.02 and same_day_loss < 0.035:
            continue
        day_range = recent_highs[index] - recent_lows[index]
        clv = _safe_div(recent_closes[index] - recent_lows[index], day_range if day_range > 1e-12 else 1.0)
        panic_weight = min(max(max(gap_down, same_day_loss), 0.0) / 0.08, 1.0)
        follow_window = recent_closes[index + 1 : min(index + 4, len(recent_closes))]
        follow_highs = recent_highs[index + 1 : min(index + 4, len(recent_highs))]
        follow_lows = recent_lows[index + 1 : min(index + 4, len(recent_lows))]
        follow_turnover = recent_turnover[index + 1 : min(index + 4, len(recent_turnover))]
        if len(follow_window) < 2:
            continue
        reclaim_ratio = max(
            _safe_div(max(follow_highs) - recent_closes[index], max(prior_close - recent_closes[index], prior_close * 0.01)),
            0.0,
        )
        hold_floor = min(follow_lows) if follow_lows else recent_lows[index]
        hold_score = max(1.0 - min(max(_safe_div(recent_closes[index] - hold_floor, recent_closes[index]), 0.0) / 0.04, 1.0), 0.0)
        turnover_shift = max(1.0 - min(_safe_div(_mean(follow_turnover), recent_turnover[index] + 1.0), 1.0), 0.0)
        score = panic_weight * (0.30 * clv + 0.35 * min(reclaim_ratio, 1.2) + 0.20 * hold_score + 0.15 * turnover_shift)
        best_score = max(best_score, score)
    return max(min(best_score, 1.2), 0.0)


def panic_followthrough_decay_12(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(opens) < 12 or len(highs) < 12 or len(lows) < 12 or len(closes) < 12 or len(turnover) < 12:
        return 0.0
    recent_opens = [float(value) for value in opens[-12:]]
    recent_highs = [float(value) for value in highs[-12:]]
    recent_lows = [float(value) for value in lows[-12:]]
    recent_closes = [float(value) for value in closes[-12:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-12:]]
    best_score = 0.0
    for index in range(1, len(recent_closes) - 3):
        prior_close = recent_closes[index - 1]
        if prior_close <= 0.0:
            continue
        ret_value = _safe_div(recent_closes[index] - prior_close, prior_close)
        day_range_ratio = _safe_div(recent_highs[index] - recent_lows[index], prior_close)
        close_loc = _safe_div(
            recent_closes[index] - recent_lows[index],
            recent_highs[index] - recent_lows[index] if recent_highs[index] - recent_lows[index] > 1e-12 else 1.0,
        )
        if ret_value > -0.025 and day_range_ratio < 0.05:
            continue
        if close_loc > 0.45 and ret_value > -0.03:
            continue
        follow_lows = recent_lows[index + 1 : index + 4]
        follow_highs = recent_highs[index + 1 : index + 4]
        follow_closes = recent_closes[index + 1 : index + 4]
        follow_turnover = recent_turnover[index + 1 : index + 4]
        if len(follow_lows) < 3 or len(follow_closes) < 3 or len(follow_turnover) < 3:
            continue
        extra_damage = max(_safe_div(recent_closes[index] - min(follow_lows), recent_closes[index]), 0.0)
        shock_range = max(day_range_ratio, 1e-12)
        follow_ranges = [
            _safe_div(high_value - low_value, close_value if close_value > 0.0 else 1.0)
            for high_value, low_value, close_value in zip(follow_highs, follow_lows, follow_closes)
        ]
        range_decay = max(1.0 - min(_safe_div(_mean(follow_ranges), shock_range), 1.0), 0.0)
        turnover_decay = max(1.0 - min(_safe_div(_mean(follow_turnover), recent_turnover[index] + 1.0), 1.0), 0.0)
        reclaim = _mean([1.0 if close_px >= recent_closes[index] else 0.0 for close_px in follow_closes])
        shock_weight = min(max(max(abs(ret_value), day_range_ratio), 0.0) / 0.08, 1.0)
        control = max(1.0 - min(extra_damage / 0.05, 1.0), 0.0)
        score = shock_weight * control * (0.40 * range_decay + 0.35 * turnover_decay + 0.25 * reclaim)
        best_score = max(best_score, score)
    return max(min(best_score, 1.2), 0.0)


def outflow_decay_price_repair_3_10(prices: Sequence[float], flows: Sequence[float]) -> float:
    if len(prices) < 11 or len(flows) < 10:
        return 0.0
    recent_prices = [float(value) for value in prices[-11:]]
    recent_flows = [float(value) for value in flows[-10:]]
    flow_scale = _mean([abs(value) for value in recent_flows]) + 1e-12
    normalized_outflow = [max(-value, 0.0) / flow_scale for value in recent_flows]
    out_recent = _mean(normalized_outflow[-3:])
    out_prev = _mean(normalized_outflow[:-3])
    flow_relief = _safe_div(out_prev - out_recent, out_prev + out_recent + 1e-12)
    reference_floor = min(recent_prices[:-1])
    if reference_floor <= 0.0:
        return 0.0
    repair = max(_safe_div(recent_prices[-1] - reference_floor, reference_floor), 0.0)
    repair_score = min(repair / 0.08, 1.5)
    return max(min(flow_relief * repair_score, 1.5), -1.0)


def outflow_repair_without_inflow_10(prices: Sequence[float], flows: Sequence[float]) -> float:
    if len(prices) < 11 or len(flows) < 10:
        return 0.0
    recent_prices = [float(value) for value in prices[-11:]]
    recent_flows = [float(value) for value in flows[-10:]]
    mean_recent_flow = _mean(recent_flows[-3:])
    if mean_recent_flow >= 0.0:
        return 0.0
    flow_scale = _mean([abs(value) for value in recent_flows]) + 1e-12
    reference_floor = min(recent_prices[:-1])
    if reference_floor <= 0.0:
        return 0.0
    headwind = max(-mean_recent_flow, 0.0) / flow_scale
    repair = max(_safe_div(recent_prices[-1] - reference_floor, reference_floor), 0.0)
    return max(min(min(headwind, 1.5) * min(repair / 0.08, 1.5), 1.5), 0.0)


def outflow_reversal_efficiency_10(
    lows: Sequence[float],
    closes: Sequence[float],
    flows: Sequence[float],
) -> float:
    if len(lows) < 11 or len(closes) < 11 or len(flows) < 10:
        return 0.0
    recent_lows = [float(value) for value in lows[-10:]]
    recent_closes = [float(value) for value in closes[-11:]]
    recent_flows = [float(value) for value in flows[-10:]]
    flow_scale = _mean([abs(value) for value in recent_flows]) + 1e-12
    weighted_score = 0.0
    total_weight = 0.0
    for index, flow_value in enumerate(recent_flows):
        neg_flow = max(-flow_value, 0.0)
        if neg_flow <= 0.0:
            continue
        prev_close = recent_closes[index]
        close_value = recent_closes[index + 1]
        excursion = max(prev_close - recent_lows[index], 0.0)
        if excursion <= 1e-12:
            continue
        damage = max(prev_close - close_value, 0.0)
        repair = max(1.0 - _safe_div(damage, excursion + 1e-12), 0.0)
        weight = neg_flow / flow_scale
        weighted_score += repair * weight
        total_weight += weight
    if total_weight <= 1e-12:
        return 0.0
    return max(min(weighted_score / total_weight, 1.0), 0.0)


def turnover_adjusted_outflow_damage_10(
    closes: Sequence[float],
    turnover: Sequence[float],
    flows: Sequence[float],
) -> float:
    if len(closes) < 11 or len(turnover) < 10 or len(flows) < 10:
        return 0.0
    recent_closes = [float(value) for value in closes[-11:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-10:]]
    recent_flows = [float(value) for value in flows[-10:]]
    baseline_turnover = _mean(recent_turnover) + 1e-12
    scores: list[float] = []
    for index, flow_value in enumerate(recent_flows):
        neg_flow = max(-flow_value, 0.0)
        if neg_flow <= 0.0:
            continue
        pressure = neg_flow / (recent_turnover[index] + baseline_turnover * 0.10)
        damage = max(_safe_div(recent_closes[index] - recent_closes[index + 1], recent_closes[index]), 0.0)
        scores.append(1.0 - min(_safe_div(damage, pressure + 1e-12), 1.0))
    if not scores:
        return 0.0
    return max(min(_mean(scores), 1.0), 0.0)


def severe_outflow_nonbreak_rate_15(
    lows: Sequence[float],
    closes: Sequence[float],
    flows: Sequence[float],
) -> float:
    if len(lows) < 18 or len(closes) < 18 or len(flows) < 17:
        return 0.0
    recent_lows = [float(value) for value in lows[-18:]]
    recent_closes = [float(value) for value in closes[-18:]]
    recent_flows = [float(value) for value in flows[-17:]]
    flow_scale = _mean([abs(value) for value in recent_flows[-15:]]) + 1e-12
    weighted_hits = 0.0
    total_weight = 0.0
    for index in range(0, len(recent_flows) - 3):
        severity = max(-recent_flows[index], 0.0) / flow_scale
        if severity <= 1.0:
            continue
        follow_lows = recent_lows[index + 1 : index + 4]
        follow_closes = recent_closes[index + 2 : index + 5]
        if len(follow_lows) < 3 or len(follow_closes) < 3:
            continue
        nonbreak = min(follow_lows) >= recent_lows[index + 1]
        repair = max(follow_closes) >= recent_closes[index + 1]
        hit = 1.0 if nonbreak or repair else 0.0
        weighted_hits += hit * severity
        total_weight += severity
    if total_weight <= 1e-12:
        return 0.0
    return max(min(weighted_hits / total_weight, 1.0), 0.0)


def outflow_retest_absorption_15(
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
    flows: Sequence[float],
) -> float:
    if len(lows) < 18 or len(closes) < 18 or len(turnover) < 18 or len(flows) < 17:
        return 0.0
    recent_lows = [float(value) for value in lows[-18:]]
    recent_closes = [float(value) for value in closes[-18:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-18:]]
    recent_flows = [float(value) for value in flows[-17:]]
    flow_scale = _mean([abs(value) for value in recent_flows[-15:]]) + 1e-12
    best_score = 0.0
    for index in range(0, len(recent_flows) - 4):
        severity = max(-recent_flows[index], 0.0) / flow_scale
        if severity <= 1.0:
            continue
        shock_low = recent_lows[index + 1]
        shock_close = recent_closes[index + 1]
        shock_turnover = recent_turnover[index + 1] + 1.0
        if shock_low <= 0.0 or shock_close <= 0.0:
            continue
        follow_lows = recent_lows[index + 2 : index + 6]
        follow_closes = recent_closes[index + 2 : index + 6]
        follow_turnover = recent_turnover[index + 2 : index + 6]
        if len(follow_lows) < 4 or len(follow_closes) < 4 or len(follow_turnover) < 4:
            continue
        retest_depth = max(_safe_div(shock_low - min(follow_lows), shock_low), 0.0)
        low_hold = max(1.0 - min(retest_depth / 0.03, 1.0), 0.0)
        shock_mid = shock_low + (shock_close - shock_low) * 0.5
        mid_reclaim = _mean([1.0 if close_px >= shock_mid else 0.0 for close_px in follow_closes])
        turnover_cool = max(1.0 - min(_safe_div(_mean(follow_turnover), shock_turnover), 1.0), 0.0)
        close_reclaim = _mean([1.0 if close_px >= shock_close else 0.0 for close_px in follow_closes])
        severity_weight = min(max(severity - 1.0, 0.0), 1.5)
        score = severity_weight * low_hold * (0.40 * mid_reclaim + 0.35 * turnover_cool + 0.25 * close_reclaim)
        best_score = max(best_score, score)
    return max(min(best_score, 1.2), 0.0)


def shock_relaxation_convexity_15(
    closes: Sequence[float],
    turnover: Sequence[float],
    flows: Sequence[float],
) -> float:
    if len(closes) < 18 or len(turnover) < 17 or len(flows) < 17:
        return 0.0
    recent_closes = [float(value) for value in closes[-18:]]
    recent_turnover = [max(float(value), 0.0) for value in turnover[-17:]]
    recent_flows = [float(value) for value in flows[-17:]]
    returns = [
        _safe_div(recent_closes[index + 1] - recent_closes[index], recent_closes[index])
        for index in range(len(recent_closes) - 1)
    ]
    flow_scale = _mean([abs(value) for value in recent_flows]) + 1e-12
    turnover_scale = _mean(recent_turnover) + 1e-12
    best_score = 0.0
    for index in range(0, len(recent_flows) - 3):
        shock = max(-recent_flows[index], 0.0) / flow_scale
        if shock <= 1.0:
            continue
        immediate_damage = max(-returns[index], 0.0)
        if immediate_damage <= 1e-12:
            continue
        decay_path = [max(-returns[index + offset], 0.0) for offset in range(1, 4)]
        if len(decay_path) < 3:
            continue
        first_leg, second_leg, third_leg = decay_path
        convex_decay = max((first_leg - second_leg), 0.0) + max((second_leg - third_leg), 0.0)
        residual_tail = third_leg
        turnover_follow = _mean(recent_turnover[index + 1 : index + 4])
        turnover_cool = max(1.0 - min(_safe_div(turnover_follow, max(recent_turnover[index], turnover_scale)), 1.0), 0.0)
        rebound = max(_safe_div(recent_closes[index + 4] - recent_closes[index + 1], recent_closes[index + 1]), 0.0)
        convexity_score = _safe_div(convex_decay, immediate_damage + 1e-12)
        tail_penalty = min(_safe_div(residual_tail, immediate_damage + 1e-12), 1.0)
        repair_score = min(rebound / 0.05, 1.0)
        score = min(max(shock - 1.0, 0.0), 1.5) * max(convexity_score - 0.6 * tail_penalty, 0.0) * (0.65 + 0.20 * turnover_cool + 0.15 * repair_score)
        best_score = max(best_score, score)
    return max(min(best_score, 1.2), 0.0)


def flow_price_divergence_10(flows: Sequence[float], prices: Sequence[float]) -> float:
    if len(flows) < 10 or len(prices) < 10:
        return 0.0
    flow_sign = 1 if _mean(flows[-10:]) > 0 else -1
    price_ret = _safe_div(prices[-1] - prices[-10], prices[-10])
    return abs(flow_sign - (1 if price_ret > 0 else -1)) / 2.0
