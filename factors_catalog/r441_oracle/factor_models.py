# ruff: noqa
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FactorInputs:
    opens: list[float]
    highs: list[float]
    lows: list[float]
    closes: list[float]
    turnover: list[float]
    benchmark_closes: list[float]
    close_returns: list[float] | None = None
    turnover_returns: list[float] | None = None
    benchmark_returns: list[float] | None = None
    close_returns_end_index: int | None = None
    turnover_returns_end_index: int | None = None
    benchmark_returns_end_index: int | None = None
    end_index: int | None = None
    benchmark_end_index: int | None = None


@dataclass(frozen=True)
class FactorObservation:
    day: str
    symbol: str
    forward_return: float
    factor_values: dict[str, float]
    proxy_forward_return: float | None = None


@dataclass(frozen=True)
class FactorDayBucket:
    day: str
    values: list[float]
    row_indices: list[int] | None = None
    ranked_values: list[float] | None = None
    sorted_values: list[float] | None = None
    forward_returns: list[float] | None = None
    ranked_forward_returns: list[float] | None = None


@dataclass(frozen=True)
class FactorEvaluationFrame:
    factor_names: list[str]
    total_observations: int
    buckets_by_factor: dict[str, list[FactorDayBucket]]
    forward_returns_by_day: dict[str, list[float]]
    forward_return_ranks_by_day: dict[str, list[float]]
    overall_factor_values_by_factor: dict[str, list[float]]
    overall_forward_returns_by_factor: dict[str, list[float]]
    overall_forward_return_ranks_by_factor: dict[str, list[float]]


@dataclass(frozen=True)
class FactorDaySummary:
    day: str
    sample_size: int
    ic: float
    top_quantile_return: float
    bottom_quantile_return: float
    top_bottom_spread: float


@dataclass(frozen=True)
class FactorZScoreDayBucket:
    day: str
    symbols: list[str]
    forward_returns: list[float]
    forward_return_ranks: list[float]
    zscores_by_factor: dict[str, list[float]]


@dataclass(frozen=True)
class FactorZScoreFrame:
    factor_names: list[str]
    day_buckets: list[FactorZScoreDayBucket]


@dataclass(frozen=True)
class FactorMetric:
    factor: str
    observation_count: int
    coverage_ratio: float
    valid_days: int
    mean_daily_ic: float
    overall_spearman: float
    ic_ir: float
    top_quantile_return: float
    bottom_quantile_return: float
    top_bottom_spread: float
    stability_ratio: float
    direction: int
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FactorMetricInputs:
    values: list[float]
    daily_ics: list[float]
    daily_spreads: list[float]
    top_returns: list[float]
    bottom_returns: list[float]


@dataclass(frozen=True)
class FactorSpec:
    factor: str
    weight: float
    direction: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
