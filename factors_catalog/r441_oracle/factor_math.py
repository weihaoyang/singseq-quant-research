# ruff: noqa
from __future__ import annotations


def _safe_div(num: float, den: float) -> float:
    return num / den if den else float("nan")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return variance**0.5


def _returns(prices: list[float]) -> list[float]:
    if len(prices) < 2:
        return []
    rows: list[float] = []
    for index in range(1, len(prices)):
        prev = prices[index - 1]
        rows.append(_safe_div(prices[index] - prev, prev))
    return rows


def _rank(values: list[float]) -> list[float]:
    if not values:
        return []
    value_at = values.__getitem__
    ordered_indices = sorted(range(len(values)), key=value_at)
    ranks = [0.0] * len(values)
    cursor = 0
    total = len(ordered_indices)
    while cursor < total:
        tail = cursor
        anchor = value_at(ordered_indices[cursor])
        while tail + 1 < total and value_at(ordered_indices[tail + 1]) == anchor:
            tail += 1
        avg_rank = (cursor + tail) / 2.0 + 1.0
        for index in range(cursor, tail + 1):
            ranks[ordered_indices[index]] = avg_rank
        cursor = tail + 1
    return ranks


def _rank_with_sorted(values: list[float]) -> tuple[list[float], list[float]]:
    if not values:
        return [], []
    value_at = values.__getitem__
    ordered_indices = sorted(range(len(values)), key=value_at)
    ranks = [0.0] * len(values)
    sorted_values = [0.0] * len(values)
    cursor = 0
    total = len(ordered_indices)
    while cursor < total:
        tail = cursor
        anchor = value_at(ordered_indices[cursor])
        while tail + 1 < total and value_at(ordered_indices[tail + 1]) == anchor:
            tail += 1
        avg_rank = (cursor + tail) / 2.0 + 1.0
        for index in range(cursor, tail + 1):
            original_index = ordered_indices[index]
            ranks[original_index] = avg_rank
            sorted_values[index] = value_at(original_index)
        cursor = tail + 1
    return ranks, sorted_values


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 3:
        return 0.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    cov = 0.0
    left_var = 0.0
    right_var = 0.0
    for left_value, right_value in zip(left, right):
        left_delta = left_value - left_mean
        right_delta = right_value - right_mean
        cov += left_delta * right_delta
        left_var += left_delta * left_delta
        right_var += right_delta * right_delta
    if left_var <= 1e-12 or right_var <= 1e-12:
        return 0.0
    return cov / ((left_var**0.5) * (right_var**0.5))


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 3:
        return 0.0
    return _pearson(_rank(left), _rank(right))


def _quantile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    normalized = max(0.0, min(1.0, ratio))
    sorted_values = sorted(values)
    position = int(round((len(sorted_values) - 1) * normalized))
    return sorted_values[position]


def _window(values: list[float], size: int) -> list[float]:
    if size <= 0:
        return []
    return values[-size:]


def _slice_until(values: list[float], end_index: int | None) -> list[float]:
    if end_index is None:
        return values
    bounded = max(0, min(int(end_index), len(values)))
    return values[:bounded]


def _moving_average(values: list[float], size: int) -> float:
    segment = _window(values, size)
    return _mean(segment)


def _zscore(value: float, values: list[float]) -> float:
    std = _std(values)
    if std <= 1e-12:
        return 0.0
    return (value - _mean(values)) / std


def _max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst
