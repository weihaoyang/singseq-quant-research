"""Exact daily-bar formulas migrated from research/factors.py at commit 9501b921.

This module deliberately contains only formulas whose complete inputs are adjusted
OHLC, turnover, and the daily opening auction gap.  It has no research-runtime,
benchmark, sector, flow, fundamental, event, or intraday dependency.
"""

from __future__ import annotations

from collections.abc import Sequence


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

def price_momentum_20(prices: Sequence[float]) -> float:
    if len(prices) < 21:
        return 0.0
    base = prices[-21]
    return _safe_div(prices[-1] - base, base)

def ma_gap_20(prices: Sequence[float]) -> float:
    if len(prices) < 20:
        return 0.0
    average = _mean(prices[-20:])
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

def realized_volatility_20(prices: Sequence[float]) -> float:
    if len(prices) < 21:
        return 0.0
    returns = [_safe_div(prices[index + 1] - prices[index], prices[index]) for index in range(len(prices) - 1)]
    return _std(returns[-20:])


def volatility_compression_20(prices: Sequence[float]) -> float:
    """Exact r441 20-day volatility compression score.

    The frozen oracle compares the preceding 20 daily returns with the most
    recent 20 daily returns.  This is deliberately not an OHLC range-ratio
    measure; the historical package exposes the same value under both
    ``volatility_compression_20`` and ``compression_ratio_20_60``.
    """

    if len(prices) < 41:
        return 0.0
    returns = [
        _safe_div(prices[index + 1] - prices[index], prices[index])
        for index in range(len(prices) - 1)
    ]
    pre = returns[-40:-20]
    recent = returns[-20:]
    pre_vol = _std(pre)
    recent_vol = _std(recent)
    return max(_safe_div(pre_vol - recent_vol, pre_vol), 0.0)

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
        post_highs = recent_highs[reclaim_index : min(reclaim_index + 4, len(recent_highs))]  # noqa: F841
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

def repair_liquidity_retention_10(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    turnover: Sequence[float],
) -> float:
    if len(highs) < 10 or len(lows) < 10 or len(closes) < 10 or len(turnover) < 10:
        return 0.0
    recent_highs = [float(value) for value in highs[-10:]]  # noqa: F841
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

def turnover_impulse_5_20(turnover: Sequence[float]) -> float:
    if len(turnover) < 20:
        return 0.0
    short = _mean(turnover[-5:])
    long = _mean(turnover[-20:])
    return _safe_div(short - long, abs(long) + 1.0)

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

def abnormal_turnover_z20(turnover: Sequence[float]) -> float:
    if len(turnover) < 21:
        return 0.0
    window = turnover[-21:-1]
    std = _std(window)
    return _safe_div(turnover[-1] - _mean(window), std if std > 1e-12 else 1.0)

def auction_gap_strength_1d(opens_1d: Sequence[float], closes_1d: Sequence[float]) -> float:
    if len(opens_1d) < 1 or len(closes_1d) < 2:
        return 0.0
    return _safe_div(opens_1d[-1] - closes_1d[-2], closes_1d[-2])

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
