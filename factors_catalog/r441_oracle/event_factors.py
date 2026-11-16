# ruff: noqa
from __future__ import annotations

from collections.abc import Sequence


def _safe_div(num: float, den: float) -> float:
    return num / den if den else float("nan")


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def limit_up_continuation_rate(series: Sequence[float]) -> float:
    if not series:
        return 0.0
    return max(0.0, min(_mean(series[-10:]), 1.0))


def board_break_rate(series: Sequence[float]) -> float:
    if not series:
        return 0.0
    return max(0.0, min(_mean(series[-10:]), 1.0))


def gap_exhaustion_score(gaps: Sequence[float]) -> float:
    if len(gaps) < 5:
        return 0.0
    positive = [value for value in gaps[-5:] if value > 0]
    return _safe_div(sum(positive), len(gaps[-5:]))


def panic_reversal_score(reversal_series: Sequence[float]) -> float:
    if len(reversal_series) < 5:
        return 0.0
    return max(0.0, min(_mean(reversal_series[-5:]), 1.0))


def market_breadth_support_10(sentiment_series: Sequence[float]) -> float:
    if len(sentiment_series) < 10:
        return 0.0
    return _mean(sentiment_series[-10:]) - 0.5


def sentiment_tailwind_10(sentiment_series: Sequence[float], board_break_series: Sequence[float]) -> float:
    size = min(len(sentiment_series), len(board_break_series), 10)
    if size <= 0:
        return 0.0
    sentiment = _mean(sentiment_series[-size:])
    board_break = _mean(board_break_series[-size:])
    return sentiment - board_break


def crowding_risk_score(flow_zscore: float, turnover_zscore: float, sentiment_score: float) -> float:
    weighted = 0.4 * max(flow_zscore, 0.0) + 0.3 * max(turnover_zscore, 0.0) + 0.3 * max(sentiment_score, 0.0)
    return max(0.0, min(weighted, 1.0))


def event_surprise_score_5(
    direction_series: Sequence[float],
    intensity_series: Sequence[float],
    confidence_series: Sequence[float],
) -> float:
    size = min(len(direction_series), len(intensity_series), len(confidence_series), 5)
    if size <= 0:
        return 0.0
    weighted = 0.0
    for index in range(1, size + 1):
        direction = float(direction_series[-index])
        intensity = max(float(intensity_series[-index]), 0.0)
        confidence = max(min(float(confidence_series[-index]), 1.0), 0.0)
        weighted += direction * intensity * confidence
    return _safe_div(weighted, size)


def event_recency_decay_30m(age_minutes: Sequence[float], impact_series: Sequence[float]) -> float:
    size = min(len(age_minutes), len(impact_series))
    if size <= 0:
        return 0.0
    weighted = 0.0
    weights = 0.0
    for index in range(size):
        age = max(float(age_minutes[index]), 0.0)
        impact = float(impact_series[index])
        decay = 1.0 / (1.0 + age / 30.0)
        weighted += impact * decay
        weights += decay
    return _safe_div(weighted, weights if weights > 1e-9 else 1.0)


def event_consistency_20d(impact_series: Sequence[float]) -> float:
    if len(impact_series) < 5:
        return 0.0
    window = impact_series[-20:] if len(impact_series) >= 20 else impact_series
    signs = [1 if value > 0 else -1 if value < 0 else 0 for value in window]
    non_zero = [value for value in signs if value != 0]
    if len(non_zero) < 2:
        return 0.0
    aligned = sum(1 for index in range(1, len(non_zero)) if non_zero[index] == non_zero[index - 1])
    direction = 1.0 if _mean(window) > 0 else -1.0
    return _safe_div(aligned, max(len(non_zero) - 1, 1)) * direction


def event_sector_diffusion_5d(symbol_hits: Sequence[float], sector_hits: Sequence[float]) -> float:
    size = min(len(symbol_hits), len(sector_hits), 5)
    if size <= 0:
        return 0.0
    symbol_recent = _mean(symbol_hits[-size:])
    sector_recent = _mean(sector_hits[-size:])
    return _safe_div(symbol_recent, max(sector_recent, 1e-9))


def regulatory_risk_penalty_20d(regulatory_impact_series: Sequence[float]) -> float:
    if not regulatory_impact_series:
        return 0.0
    recent = regulatory_impact_series[-20:]
    negatives = [abs(min(value, 0.0)) for value in recent]
    return max(0.0, min(_mean(negatives), 1.0))


def shareholder_alignment_60d(shareholder_direction_series: Sequence[float]) -> float:
    if not shareholder_direction_series:
        return 0.0
    recent = shareholder_direction_series[-60:]
    positive = sum(1 for value in recent if value > 0)
    negative = sum(1 for value in recent if value < 0)
    return _safe_div(positive - negative, max(len(recent), 1))


def policy_alignment_score_10d(policy_impact_series: Sequence[float], sector_rs_series: Sequence[float]) -> float:
    size = min(len(policy_impact_series), len(sector_rs_series), 10)
    if size <= 0:
        return 0.0
    policy = _mean(policy_impact_series[-size:])
    sector_rs = _mean(sector_rs_series[-size:])
    return policy if policy * sector_rs >= 0 else -abs(policy)


def crowding_event_conflict_5d(crowding_series: Sequence[float], hotword_series: Sequence[float]) -> float:
    size = min(len(crowding_series), len(hotword_series), 5)
    if size <= 0:
        return 0.0
    products = [max(crowding_series[-index], 0.0) * max(hotword_series[-index], 0.0) for index in range(1, size + 1)]
    return max(0.0, min(_mean(products), 1.0))
