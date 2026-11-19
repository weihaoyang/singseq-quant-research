"""Causal panel operators for local factor research.

The canonical input is a pandas ``Series`` aligned to a long panel with one
``stocks`` Series and, for cross-sectional operations, one ``dates`` Series.
Outputs preserve the input index. Rolling operators only look backward.

Example::

    from quant_sdk.operators import cross_rank, delay, safe_divide, ts_mean

    close = df["close"].astype(float)
    ret = safe_divide(close, delay(close, df["ts_code"], 1)) - 1.0
    factor = -cross_rank(ts_mean(ret, df["ts_code"], 20), df["trade_date"])
"""
from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pandas as pd


def _as_series(value: Any, name: str) -> pd.Series:
    result = value if isinstance(value, pd.Series) else pd.Series(value)
    if not result.index.is_unique:
        raise ValueError(f"{name} 的索引必须唯一")
    return result.astype(float)


def _as_labels(value: Any, name: str) -> pd.Series:
    result = value if isinstance(value, pd.Series) else pd.Series(value)
    if not result.index.is_unique:
        raise ValueError(f"{name} 的索引必须唯一")
    return result


def _aligned(*values: pd.Series) -> list[pd.Series]:
    first = values[0]
    if any(not other.index.equals(first.index) for other in values[1:]):
        raise ValueError("输入序列必须使用相同且顺序一致的索引")
    return list(values)


def _binary(left: Any, right: Any) -> tuple[pd.Series, pd.Series]:
    """Align two inputs while broadcasting scalar operands over a Series."""
    if not isinstance(left, pd.Series) and isinstance(right, pd.Series) and np.isscalar(left):
        left = pd.Series(float(left), index=right.index, dtype=float)
    if not isinstance(right, pd.Series) and isinstance(left, pd.Series) and np.isscalar(right):
        right = pd.Series(float(right), index=left.index, dtype=float)
    return tuple(_aligned(_as_series(left, "left"), _as_series(right, "right")))


def _minimum(window: int, min_periods: int | None) -> int:
    if not isinstance(window, int) or window < 1:
        raise ValueError("window 必须是正整数")
    minimum = window if min_periods is None else int(min_periods)
    if minimum < 1 or minimum > window:
        raise ValueError("min_periods 必须在 1 和 window 之间")
    return minimum


def _rolling(values: pd.Series, stocks: pd.Series, window: int,
             method: str, min_periods: int | None = None) -> pd.Series:
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    minimum = _minimum(window, min_periods)
    return values.groupby(stocks, sort=False).transform(
        lambda group: getattr(
            group.rolling(window, min_periods=minimum),
            method,
        )(),
    )


def _rolling_apply(values: pd.Series, stocks: pd.Series, window: int,
                   function: Any, min_periods: int | None = None) -> pd.Series:
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    minimum = _minimum(window, min_periods)
    return values.groupby(stocks, sort=False).transform(
        lambda group: group.rolling(window, min_periods=minimum).apply(function, raw=True),
    )


def safe_divide(left: Any, right: Any) -> pd.Series:
    """Element-wise division returning NaN when the denominator is zero."""
    left, right = _binary(left, right)
    numerator = left.to_numpy(dtype=float)
    denominator = right.to_numpy(dtype=float)
    output = np.full(len(left), np.nan, dtype=float)
    np.divide(numerator, denominator, out=output,
              where=np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0.0))
    return pd.Series(output, index=left.index)


def add(left: Any, right: Any) -> pd.Series:
    """Element-wise sum of two aligned series."""
    left, right = _binary(left, right)
    return left + right


def subtract(left: Any, right: Any) -> pd.Series:
    """Element-wise difference of two aligned series."""
    left, right = _binary(left, right)
    return left - right


def multiply(left: Any, right: Any) -> pd.Series:
    """Element-wise product of two aligned series."""
    left, right = _binary(left, right)
    return left * right


def inverse(values: Any) -> pd.Series:
    """Reciprocal with zero values mapped to NaN."""
    values = _as_series(values, "values")
    return safe_divide(1.0, values)


def safe_abs(values: Any) -> pd.Series:
    """Absolute value with the same index and floating-point contract."""
    return _as_series(values, "values").abs()


def sign(values: Any) -> pd.Series:
    """Return -1, 0, or 1 while preserving missing values."""
    values = _as_series(values, "values")
    return pd.Series(np.sign(values.to_numpy(dtype=float)), index=values.index)


def clip(values: Any, *, lower: float | None = None,
         upper: float | None = None) -> pd.Series:
    """Clip values to optional finite bounds."""
    if lower is not None and not np.isfinite(float(lower)):
        raise ValueError("lower 必须是有限数")
    if upper is not None and not np.isfinite(float(upper)):
        raise ValueError("upper 必须是有限数")
    values = _as_series(values, "values")
    return values.clip(lower=lower, upper=upper)


def relu(values: Any) -> pd.Series:
    """Positive-part transform."""
    return _as_series(values, "values").clip(lower=0.0)


def square(values: Any) -> pd.Series:
    """Element-wise square."""
    return _as_series(values, "values") ** 2.0


def cube(values: Any) -> pd.Series:
    """Element-wise cube."""
    return _as_series(values, "values") ** 3.0


def signed_sqrt(values: Any) -> pd.Series:
    """Square root of the absolute value with the original sign."""
    return signed_power(values, 0.5)


def signed_log(values: Any) -> pd.Series:
    """Logarithm of one plus absolute value with the original sign."""
    values = _as_series(values, "values")
    raw = values.to_numpy(dtype=float)
    return pd.Series(np.sign(raw) * np.log1p(np.abs(raw)), index=values.index)


def exp(values: Any) -> pd.Series:
    """Element-wise exponential with overflow converted to infinity."""
    values = _as_series(values, "values")
    with np.errstate(over="ignore", invalid="ignore"):
        return pd.Series(np.exp(values.to_numpy(dtype=float)), index=values.index)


def tanh(values: Any) -> pd.Series:
    """Element-wise hyperbolic tangent."""
    values = _as_series(values, "values")
    return pd.Series(np.tanh(values.to_numpy(dtype=float)), index=values.index)


def sigmoid(values: Any) -> pd.Series:
    """Numerically stable logistic transform."""
    values = _as_series(values, "values")
    raw = values.to_numpy(dtype=float)
    output = np.empty(len(raw), dtype=float)
    positive = raw >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-raw[positive]))
    exponent = np.exp(raw[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return pd.Series(output, index=values.index)


def min2(left: Any, right: Any) -> pd.Series:
    """Element-wise minimum of two aligned series."""
    left, right = _binary(left, right)
    return pd.Series(np.minimum(left.to_numpy(dtype=float), right.to_numpy(dtype=float)),
                     index=left.index)


def max2(left: Any, right: Any) -> pd.Series:
    """Element-wise maximum of two aligned series."""
    left, right = _binary(left, right)
    return pd.Series(np.maximum(left.to_numpy(dtype=float), right.to_numpy(dtype=float)),
                     index=left.index)


def logical_and(left: Any, right: Any) -> pd.Series:
    """Element-wise conjunction; missing inputs are false."""
    if np.isscalar(left) and isinstance(right, pd.Series):
        left = pd.Series(bool(left), index=right.index)
    if np.isscalar(right) and isinstance(left, pd.Series):
        right = pd.Series(bool(right), index=left.index)
    left, right = _aligned(
        left if isinstance(left, pd.Series) else pd.Series(left),
        right if isinstance(right, pd.Series) else pd.Series(right),
    )
    return left.fillna(False).astype(bool) & right.fillna(False).astype(bool)


def logical_or(left: Any, right: Any) -> pd.Series:
    """Element-wise disjunction; missing inputs are false."""
    if np.isscalar(left) and isinstance(right, pd.Series):
        left = pd.Series(bool(left), index=right.index)
    if np.isscalar(right) and isinstance(left, pd.Series):
        right = pd.Series(bool(right), index=left.index)
    left, right = _aligned(
        left if isinstance(left, pd.Series) else pd.Series(left),
        right if isinstance(right, pd.Series) else pd.Series(right),
    )
    return left.fillna(False).astype(bool) | right.fillna(False).astype(bool)


def logical_not(values: Any) -> pd.Series:
    """Element-wise negation; missing inputs are false."""
    values = values if isinstance(values, pd.Series) else pd.Series(values)
    return ~values.fillna(False).astype(bool)


def _comparison(left: Any, right: Any, operation: str) -> pd.Series:
    left, right = _binary(left, right)
    return getattr(left, operation)(right)


def greater(left: Any, right: Any) -> pd.Series:
    """Element-wise greater-than comparison."""
    return _comparison(left, right, "gt")


def greater_equal(left: Any, right: Any) -> pd.Series:
    """Element-wise greater-than-or-equal comparison."""
    return _comparison(left, right, "ge")


def less(left: Any, right: Any) -> pd.Series:
    """Element-wise less-than comparison."""
    return _comparison(left, right, "lt")


def less_equal(left: Any, right: Any) -> pd.Series:
    """Element-wise less-than-or-equal comparison."""
    return _comparison(left, right, "le")


def equal(left: Any, right: Any) -> pd.Series:
    """Element-wise equality comparison."""
    return _comparison(left, right, "eq")


def not_equal(left: Any, right: Any) -> pd.Series:
    """Element-wise inequality comparison."""
    return _comparison(left, right, "ne")


def fillna(values: Any, fill: Any) -> pd.Series:
    """Fill missing values from an aligned fallback series."""
    values, fill = _binary(values, fill)
    return values.fillna(fill)


def isfinite(values: Any) -> pd.Series:
    """Return a boolean mask for finite observations."""
    values = _as_series(values, "values")
    return pd.Series(np.isfinite(values.to_numpy(dtype=float)), index=values.index)


def safe_log(values: Any) -> pd.Series:
    """Natural logarithm defined only for strictly positive values."""
    values = _as_series(values, "values")
    output = np.full(len(values), np.nan, dtype=float)
    positive = np.isfinite(values.to_numpy(dtype=float)) & (values.to_numpy(dtype=float) > 0.0)
    output[positive] = np.log(values.to_numpy(dtype=float)[positive])
    return pd.Series(output, index=values.index)


def safe_log1p(values: Any) -> pd.Series:
    """Natural logarithm of one plus values greater than -1."""
    values = _as_series(values, "values")
    output = np.full(len(values), np.nan, dtype=float)
    raw = values.to_numpy(dtype=float)
    valid = np.isfinite(raw) & (raw > -1.0)
    output[valid] = np.log1p(raw[valid])
    return pd.Series(output, index=values.index)


def log_diff(left: Any, right: Any) -> pd.Series:
    """Difference of two natural logarithms, defined only for positive inputs."""
    left_value, right_value = _aligned(
        _as_series(left, "left"), _as_series(right, "right"),
    )
    return safe_log(left_value) - safe_log(right_value)


def signed_power(values: Any, exponent: float) -> pd.Series:
    """Preserve sign while applying a power to the absolute value."""
    values = _as_series(values, "values")
    raw = values.to_numpy(dtype=float)
    output = np.full(len(raw), np.nan, dtype=float)
    valid = np.isfinite(raw)
    output[valid] = np.sign(raw[valid]) * np.power(np.abs(raw[valid]), float(exponent))
    return pd.Series(output, index=values.index)


def power(values: Any, exponent: float) -> pd.Series:
    """Element-wise power with a finite scalar exponent."""
    if not np.isfinite(float(exponent)):
        raise ValueError("exponent 必须是有限数")
    values = _as_series(values, "values")
    with np.errstate(over="ignore", invalid="ignore"):
        return values.pow(float(exponent))


def where(condition: Any, truthy: Any, falsy: Any) -> pd.Series:
    """Vectorized conditional expression with aligned inputs."""
    condition = condition if isinstance(condition, pd.Series) else pd.Series(condition)
    truthy = _as_series(truthy, "truthy")
    falsy = _as_series(falsy, "falsy")
    _aligned(condition, truthy, falsy)
    return pd.Series(
        np.where(condition.fillna(False).to_numpy(dtype=bool),
                 truthy.to_numpy(dtype=float), falsy.to_numpy(dtype=float)),
        index=truthy.index,
    )


def cross_rank(values: Any, dates: Any, *, pct: bool = True) -> pd.Series:
    """Rank each date's cross-section without using future dates."""
    values, dates = _aligned(
        _as_series(values, "values"), _as_labels(dates, "dates"),
    )
    return values.groupby(dates, sort=False).rank(method="average", pct=bool(pct))


def cross_demean(values: Any, dates: Any) -> pd.Series:
    """Subtract each date's cross-sectional mean."""
    values, dates = _aligned(
        _as_series(values, "values"), _as_labels(dates, "dates"),
    )
    return values.groupby(dates, sort=False).transform(lambda group: group - group.mean())


def cross_mean(values: Any, dates: Any) -> pd.Series:
    """Broadcast each date's cross-sectional mean back to every row."""
    values, dates = _aligned(
        _as_series(values, "values"), _as_labels(dates, "dates"),
    )
    return values.groupby(dates, sort=False).transform("mean")


def cross_sum(values: Any, dates: Any) -> pd.Series:
    """Broadcast each date's cross-sectional sum back to every row."""
    values, dates = _aligned(
        _as_series(values, "values"), _as_labels(dates, "dates"),
    )
    return values.groupby(dates, sort=False).transform("sum")


def cross_median(values: Any, dates: Any) -> pd.Series:
    """Broadcast each date's cross-sectional median back to every row."""
    values, dates = _aligned(
        _as_series(values, "values"), _as_labels(dates, "dates"),
    )
    return values.groupby(dates, sort=False).transform("median")


def cross_std(values: Any, dates: Any) -> pd.Series:
    """Broadcast each date's cross-sectional sample standard deviation."""
    values, dates = _aligned(
        _as_series(values, "values"), _as_labels(dates, "dates"),
    )
    return values.groupby(dates, sort=False).transform("std")


def cross_mad(values: Any, dates: Any) -> pd.Series:
    """Broadcast each date's median absolute deviation."""
    values, dates = _aligned(
        _as_series(values, "values"), _as_labels(dates, "dates"),
    )

    def calculate(group: pd.Series) -> pd.Series:
        median = group.median()
        deviation = (group - median).abs().median()
        return pd.Series(deviation, index=group.index, dtype=float)

    return values.groupby(dates, sort=False).transform(calculate)


def cross_zscore(values: Any, dates: Any) -> pd.Series:
    """Standardize each date's cross-section; zero dispersion becomes NaN."""
    values, dates = _aligned(
        _as_series(values, "values"), _as_labels(dates, "dates"),
    )

    def standardize(group: pd.Series) -> pd.Series:
        dispersion = group.std(ddof=0)
        return (group - group.mean()) / dispersion if dispersion > 0.0 else group * np.nan

    return values.groupby(dates, sort=False).transform(standardize)


def cross_scale(values: Any, dates: Any, scale: float = 1.0) -> pd.Series:
    """Scale each date's cross-section by the sum of absolute values."""
    values, dates = _aligned(
        _as_series(values, "values"), _as_labels(dates, "dates"),
    )

    def apply_scale(group: pd.Series) -> pd.Series:
        denominator = group.abs().sum()
        return group * (float(scale) / denominator) if denominator > 0.0 else group * np.nan

    return values.groupby(dates, sort=False).transform(apply_scale)


def cross_bucket(values: Any, dates: Any, *, buckets: int = 5) -> pd.Series:
    """Assign equal-sized cross-sectional buckets from 1..buckets."""
    if not isinstance(buckets, int) or buckets < 2:
        raise ValueError("buckets 必须是大于等于 2 的整数")
    values, dates = _aligned(
        _as_series(values, "values"), _as_labels(dates, "dates"),
    )
    ranked = values.groupby(dates, sort=False).rank(method="first", pct=True)
    return pd.Series(
        np.ceil(ranked.to_numpy(dtype=float) * buckets).astype(float).clip(1, buckets),
        index=values.index,
    )


def group_rank(values: Any, dates: Any, groups: Any, *, pct: bool = True) -> pd.Series:
    """Rank inside each date and supplied category group."""
    values, dates, groups = _aligned(
        _as_series(values, "values"), dates, _as_labels(groups, "groups"),
    )
    return values.groupby([dates, groups], sort=False).rank(method="average", pct=bool(pct))


def group_demean(values: Any, dates: Any, groups: Any) -> pd.Series:
    """Subtract the mean of each date and category group."""
    values, dates, groups = _aligned(
        _as_series(values, "values"), dates, _as_labels(groups, "groups"),
    )
    return values.groupby([dates, groups], sort=False).transform(
        lambda group: group - group.mean(),
    )


def group_zscore(values: Any, dates: Any, groups: Any) -> pd.Series:
    """Standardize inside each date and category group."""
    values, dates, groups = _aligned(
        _as_series(values, "values"), dates, _as_labels(groups, "groups"),
    )

    def standardize(group: pd.Series) -> pd.Series:
        dispersion = group.std(ddof=0)
        return (group - group.mean()) / dispersion if dispersion > 0.0 else group * np.nan

    return values.groupby([dates, groups], sort=False).transform(standardize)


def group_mean(values: Any, dates: Any, groups: Any) -> pd.Series:
    """Broadcast each date and category group's mean."""
    values, dates, groups = _aligned(
        _as_series(values, "values"), dates, _as_labels(groups, "groups"),
    )
    return values.groupby([dates, groups], sort=False).transform("mean")


def group_scale(values: Any, dates: Any, groups: Any, scale: float = 1.0) -> pd.Series:
    """Scale within each date and category group by the sum of absolute values."""
    values, dates, groups = _aligned(
        _as_series(values, "values"), dates, _as_labels(groups, "groups"),
    )

    def apply_scale(group: pd.Series) -> pd.Series:
        denominator = group.abs().sum()
        return group * (float(scale) / denominator) if denominator > 0.0 else group * np.nan

    return values.groupby([dates, groups], sort=False).transform(apply_scale)


def winsorize(values: Any, dates: Any, *, lower: float = 0.01,
              upper: float = 0.99) -> pd.Series:
    """Clip each date's cross-section to empirical quantiles."""
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("要求 0 <= lower < upper <= 1")
    values, dates = _aligned(
        _as_series(values, "values"), _as_labels(dates, "dates"),
    )
    return values.groupby(dates, sort=False).transform(
        lambda group: group.clip(group.quantile(lower), group.quantile(upper)),
    )


def winsorize_mad(values: Any, dates: Any, *, scale: float = 3.0) -> pd.Series:
    """Winsorize each date's cross-section by median absolute deviation."""
    values, dates = _aligned(
        _as_series(values, "values"), _as_labels(dates, "dates"),
    )

    def clip_group(group: pd.Series) -> pd.Series:
        median = group.median()
        deviation = (group - median).abs().median()
        if not np.isfinite(deviation) or deviation == 0.0:
            return group
        bound = float(scale) * 1.4826 * deviation
        return group.clip(median - bound, median + bound)

    return values.groupby(dates, sort=False).transform(clip_group)


def delay(values: Any, stocks: Any, periods: int = 1) -> pd.Series:
    """Shift a stock's series backward by `periods` observations."""
    if not isinstance(periods, int) or periods < 1:
        raise ValueError("periods 必须是正整数")
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    return values.groupby(stocks, sort=False).shift(periods)


def delta(values: Any, stocks: Any, periods: int = 1) -> pd.Series:
    """Return the change versus `periods` observations ago."""
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    return values - delay(values, stocks, periods)


def pct_change(values: Any, stocks: Any, periods: int = 1) -> pd.Series:
    """Return causal percentage change versus a prior observation."""
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    return safe_divide(delta(values, stocks, periods), delay(values, stocks, periods))


def returns(values: Any, stocks: Any, periods: int = 1, *,
            logarithmic: bool = False) -> pd.Series:
    """Causal simple or log return over a number of observations."""
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    current = values
    previous = delay(values, stocks, periods)
    if logarithmic:
        return safe_log(current) - safe_log(previous)
    return safe_divide(delta(values, stocks, periods), previous)


def vwap(amount: Any, volume: Any) -> pd.Series:
    """Volume-weighted average price from aligned amount and volume fields."""
    return safe_divide(amount, volume)


def adv(amount: Any, stocks: Any, window: int,
        min_periods: int | None = None) -> pd.Series:
    """Average daily traded amount over the trailing window."""
    return ts_mean(amount, stocks, window, min_periods)


def ts_mean(values: Any, stocks: Any, window: int,
            min_periods: int | None = None) -> pd.Series:
    """Trailing mean per stock."""
    return _rolling(values, stocks, window, "mean", min_periods)


def ts_sum(values: Any, stocks: Any, window: int,
           min_periods: int | None = None) -> pd.Series:
    """Trailing sum per stock."""
    return _rolling(values, stocks, window, "sum", min_periods)


def ts_product(values: Any, stocks: Any, window: int,
               min_periods: int | None = None) -> pd.Series:
    """Trailing product per stock."""
    return _rolling(values, stocks, window, "prod", min_periods)


def ts_count(values: Any, stocks: Any, window: int, *,
             positive: bool | None = None,
             min_periods: int | None = None) -> pd.Series:
    """Count finite observations, or positive/non-positive observations."""
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    minimum = _minimum(window, min_periods)
    if positive is True:
        prepared = values.gt(0.0).astype(float).where(values.notna())
    elif positive is False:
        prepared = values.le(0.0).astype(float).where(values.notna())
    else:
        prepared = values.notna().astype(float)
    return prepared.groupby(stocks, sort=False).transform(
        lambda group: group.rolling(window, min_periods=minimum).sum(),
    )


def ts_sum_positive(values: Any, stocks: Any, window: int,
                    min_periods: int | None = None) -> pd.Series:
    """Trailing sum of strictly positive observations."""
    values = _as_series(values, "values")
    return ts_sum(values.clip(lower=0.0), stocks, window, min_periods)


def ts_sum_negative(values: Any, stocks: Any, window: int,
                    min_periods: int | None = None) -> pd.Series:
    """Trailing sum of strictly negative observations."""
    values = _as_series(values, "values")
    return ts_sum(values.clip(upper=0.0), stocks, window, min_periods)


def ts_std(values: Any, stocks: Any, window: int,
           min_periods: int | None = None) -> pd.Series:
    """Trailing sample standard deviation per stock."""
    return _rolling(values, stocks, window, "std", min_periods)


def ts_mad(values: Any, stocks: Any, window: int,
           min_periods: int | None = None) -> pd.Series:
    """Trailing median absolute deviation."""
    def calculate(sample: np.ndarray) -> float:
        if np.isnan(sample).all():
            return np.nan
        median = np.nanmedian(sample)
        return float(np.nanmedian(np.abs(sample - median)))

    return _rolling_apply(values, stocks, window, calculate, min_periods)


def ts_range(values: Any, stocks: Any, window: int,
             min_periods: int | None = None) -> pd.Series:
    """Trailing range between the window maximum and minimum."""
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    return ts_max(values, stocks, window, min_periods) - ts_min(
        values, stocks, window, min_periods,
    )


def ts_var(values: Any, stocks: Any, window: int,
           min_periods: int | None = None) -> pd.Series:
    """Trailing sample variance per stock."""
    return _rolling(values, stocks, window, "var", min_periods)


def ts_min(values: Any, stocks: Any, window: int,
           min_periods: int | None = None) -> pd.Series:
    """Trailing minimum per stock."""
    return _rolling(values, stocks, window, "min", min_periods)


def ts_max(values: Any, stocks: Any, window: int,
           min_periods: int | None = None) -> pd.Series:
    """Trailing maximum per stock."""
    return _rolling(values, stocks, window, "max", min_periods)


def ts_median(values: Any, stocks: Any, window: int,
              min_periods: int | None = None) -> pd.Series:
    """Trailing median per stock."""
    return _rolling(values, stocks, window, "median", min_periods)


def ts_skew(values: Any, stocks: Any, window: int,
            min_periods: int | None = None) -> pd.Series:
    """Trailing sample skewness per stock."""
    return _rolling(values, stocks, window, "skew", min_periods)


def ts_kurt(values: Any, stocks: Any, window: int,
            min_periods: int | None = None) -> pd.Series:
    """Trailing sample excess kurtosis per stock."""
    return _rolling(values, stocks, window, "kurt", min_periods)


def ts_downside_std(values: Any, stocks: Any, window: int,
                    min_periods: int | None = None) -> pd.Series:
    """Trailing standard deviation of negative observations only."""
    values = _as_series(values, "values")
    return ts_std(values.where(values.lt(0.0)), stocks, window, min_periods)


def ts_upside_std(values: Any, stocks: Any, window: int,
                  min_periods: int | None = None) -> pd.Series:
    """Trailing standard deviation of positive observations only."""
    values = _as_series(values, "values")
    return ts_std(values.where(values.gt(0.0)), stocks, window, min_periods)


def ts_cv(values: Any, stocks: Any, window: int,
          min_periods: int | None = None) -> pd.Series:
    """Trailing coefficient of variation, using absolute mean as denominator."""
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    return safe_divide(
        ts_std(values, stocks, window, min_periods),
        ts_mean(values, stocks, window, min_periods).abs(),
    )


def ts_ir(values: Any, stocks: Any, window: int,
          min_periods: int | None = None) -> pd.Series:
    """Trailing mean divided by trailing standard deviation."""
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    return safe_divide(
        ts_mean(values, stocks, window, min_periods),
        ts_std(values, stocks, window, min_periods),
    )


def ts_positive_ratio(values: Any, stocks: Any, window: int,
                      min_periods: int | None = None) -> pd.Series:
    """Share of strictly positive finite observations in the trailing window."""
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    return safe_divide(
        ts_count(values, stocks, window, positive=True, min_periods=min_periods),
        ts_count(values, stocks, window, positive=None, min_periods=min_periods),
    )


def ts_negative_ratio(values: Any, stocks: Any, window: int,
                      min_periods: int | None = None) -> pd.Series:
    """Share of non-positive finite observations in the trailing window."""
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    return safe_divide(
        ts_count(values, stocks, window, positive=False, min_periods=min_periods),
        ts_count(values, stocks, window, positive=None, min_periods=min_periods),
    )


def ts_cvar(values: Any, stocks: Any, window: int, *, tail: float = 0.05,
            min_periods: int | None = None) -> pd.Series:
    """Trailing historical lower-tail average return."""
    if not 0.0 < float(tail) <= 1.0:
        raise ValueError("tail 必须在 0 到 1 之间")

    def calculate(sample: np.ndarray) -> float:
        finite = sample[np.isfinite(sample)]
        if len(finite) == 0:
            return np.nan
        threshold = np.quantile(finite, float(tail))
        return float(finite[finite <= threshold].mean())

    return _rolling_apply(values, stocks, window, calculate, min_periods)


def ts_zscore(values: Any, stocks: Any, window: int,
              min_periods: int | None = None) -> pd.Series:
    """Trailing z-score of the latest observation."""
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    mean = ts_mean(values, stocks, window, min_periods)
    dispersion = ts_std(values, stocks, window, min_periods)
    return safe_divide(values - mean, dispersion)


def ts_drawdown(values: Any, stocks: Any, window: int,
                min_periods: int | None = None) -> pd.Series:
    """Current distance below the trailing window maximum."""
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    peak = ts_max(values, stocks, window, min_periods)
    return safe_divide(values, peak) - 1.0


def ts_drawup(values: Any, stocks: Any, window: int,
              min_periods: int | None = None) -> pd.Series:
    """Current distance above the trailing window minimum."""
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    trough = ts_min(values, stocks, window, min_periods)
    return safe_divide(values, trough) - 1.0


def ts_max_drawdown(values: Any, stocks: Any, window: int,
                    min_periods: int | None = None) -> pd.Series:
    """Worst close-to-close drawdown observed inside each trailing window."""
    def calculate(sample: np.ndarray) -> float:
        finite = sample[np.isfinite(sample)]
        if len(finite) == 0:
            return np.nan
        peak = np.maximum.accumulate(finite)
        drawdown = np.divide(
            finite,
            peak,
            out=np.full_like(finite, np.nan, dtype=float),
            where=peak != 0.0,
        ) - 1.0
        return float(np.nanmin(drawdown))

    return _rolling_apply(values, stocks, window, calculate, min_periods)


def ts_ffill(values: Any, stocks: Any) -> pd.Series:
    """Forward-fill only from earlier observations of the same instrument."""
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    return values.groupby(stocks, sort=False).ffill()


def ts_quantile(values: Any, stocks: Any, window: int, quantile: float,
                min_periods: int | None = None) -> pd.Series:
    """Trailing empirical quantile per stock."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile 必须在 0 和 1 之间")
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    minimum = _minimum(window, min_periods)
    return values.groupby(stocks, sort=False).transform(
        lambda group: group.rolling(window, min_periods=minimum).quantile(float(quantile)),
    )


def ts_rank(values: Any, stocks: Any, window: int,
            min_periods: int | None = None) -> pd.Series:
    """Trailing percentile rank of the latest observation."""
    return _rolling(values, stocks, window, "rank", min_periods)


def _rolling_extreme(values: Any, stocks: Any, window: int, *,
                     maximum: bool, min_periods: int | None = None) -> pd.Series:
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    minimum = _minimum(window, min_periods)

    def distance_from_latest(sample: np.ndarray) -> float:
        if np.isnan(sample).all():
            return np.nan
        index = int(np.nanargmax(sample) if maximum else np.nanargmin(sample))
        return float(len(sample) - index)

    return values.groupby(stocks, sort=False).transform(
        lambda group: group.rolling(window, min_periods=minimum)
        .apply(distance_from_latest, raw=True),
    )


def ts_argmax(values: Any, stocks: Any, window: int,
              min_periods: int | None = None) -> pd.Series:
    """Distance in observations from the latest value to the window maximum."""
    return _rolling_extreme(values, stocks, window, maximum=True, min_periods=min_periods)


def ts_argmin(values: Any, stocks: Any, window: int,
              min_periods: int | None = None) -> pd.Series:
    """Distance in observations from the latest value to the window minimum."""
    return _rolling_extreme(values, stocks, window, maximum=False, min_periods=min_periods)


def _rolling_pair(left: Any, right: Any, stocks: Any, window: int,
                  operation: str, min_periods: int | None = None) -> pd.Series:
    left = _as_series(left, "left")
    right = _as_series(right, "right")
    left, right, stocks = _aligned(left, right, stocks)
    minimum = _minimum(window, min_periods)
    output = pd.Series(np.nan, index=left.index, dtype=float)
    frame = pd.DataFrame({"left": left, "right": right, "stock": stocks})
    for _, group in frame.groupby("stock", sort=False):
        rolling = group["left"].rolling(window, min_periods=minimum)
        output.loc[group.index] = getattr(rolling, operation)(group["right"])
    return output


def ts_corr(left: Any, right: Any, stocks: Any, window: int,
            min_periods: int | None = None) -> pd.Series:
    """Trailing correlation between two aligned time series."""
    return _rolling_pair(left, right, stocks, window, "corr", min_periods)


def ts_cov(left: Any, right: Any, stocks: Any, window: int,
           min_periods: int | None = None) -> pd.Series:
    """Trailing sample covariance between two aligned time series."""
    return _rolling_pair(left, right, stocks, window, "cov", min_periods)


def ts_beta(left: Any, right: Any, stocks: Any, window: int,
            min_periods: int | None = None) -> pd.Series:
    """Trailing beta of the first series against the second."""
    left, right, stocks = _aligned(
        _as_series(left, "left"), _as_series(right, "right"), stocks,
    )
    covariance = ts_cov(left, right, stocks, window, min_periods)
    variance = ts_var(right, stocks, window, min_periods)
    return safe_divide(covariance, variance)


def ts_autocorr(values: Any, stocks: Any, window: int, *,
                lag: int = 1, min_periods: int | None = None) -> pd.Series:
    """Trailing autocorrelation of a series with its prior lag."""
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("lag 必须是正整数")
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    prior = delay(values, stocks, lag)
    return ts_corr(values, prior, stocks, window, min_periods)


def _trend_metric(values: Any, stocks: Any, window: int, metric: str,
                  min_periods: int | None = None) -> pd.Series:
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    minimum = _minimum(window, min_periods)
    def calculate(sample: np.ndarray) -> float:
        if np.isnan(sample).any() or len(sample) < 2:
            return np.nan
        x = np.arange(len(sample), dtype=float)
        centered = x - x.mean()
        denominator = float(centered @ centered)
        if denominator == 0.0:
            return np.nan
        mean = float(sample.mean())
        slope = float((sample - mean) @ centered / denominator)
        fitted = mean + slope * centered
        if metric == "slope":
            return slope
        if metric == "residual":
            return float(sample[-1] - fitted[-1])
        total = float(((sample - mean) ** 2).sum())
        return float((slope * slope * denominator) / total) if total > 0.0 else np.nan

    return values.groupby(stocks, sort=False).transform(
        lambda group: group.rolling(window, min_periods=minimum).apply(calculate, raw=True),
    )


def ts_slope(values: Any, stocks: Any, window: int,
             min_periods: int | None = None) -> pd.Series:
    """Trailing ordinary least-squares slope against 0..window-1."""
    return _trend_metric(values, stocks, window, "slope", min_periods)


def ts_residual(values: Any, stocks: Any, window: int,
                min_periods: int | None = None) -> pd.Series:
    """Latest observation minus its trailing regression-fitted value."""
    return _trend_metric(values, stocks, window, "residual", min_periods)


def ts_r2(values: Any, stocks: Any, window: int,
          min_periods: int | None = None) -> pd.Series:
    """R-squared of the trailing linear trend."""
    return _trend_metric(values, stocks, window, "r2", min_periods)


def decay_linear(values: Any, stocks: Any, window: int,
                 min_periods: int | None = None) -> pd.Series:
    """Linearly weighted trailing mean; the latest observation has weight `window`."""
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    minimum = _minimum(window, min_periods)
    def weighted(sample: np.ndarray) -> float:
        weights = np.arange(1.0, len(sample) + 1.0)
        weights /= weights.sum()
        return float(sample @ weights)

    return values.groupby(stocks, sort=False).transform(
        lambda group: group.rolling(window, min_periods=minimum)
        .apply(weighted, raw=True),
    )


def wma(values: Any, stocks: Any, window: int,
        min_periods: int | None = None) -> pd.Series:
    """Alias for linearly weighted moving average."""
    return decay_linear(values, stocks, window, min_periods)


def ema(values: Any, stocks: Any, span: int,
        min_periods: int | None = None) -> pd.Series:
    """Causal exponential moving average."""
    values, stocks = _aligned(_as_series(values, "values"), stocks)
    minimum = _minimum(span, min_periods)
    return values.groupby(stocks, sort=False).transform(
        lambda group: group.ewm(span=span, adjust=False, min_periods=minimum).mean(),
    )


def demean_by_group(values: Any, groups: Any) -> pd.Series:
    """Subtract a supplied categorical group's mean."""
    values, groups = _aligned(
        _as_series(values, "values"), _as_labels(groups, "groups"),
    )
    return values.groupby(groups, sort=False).transform(lambda group: group - group.mean())


def neutralize(values: Any, dates: Any, industries: Any, *,
               log_market_cap: Any | None = None) -> pd.Series:
    """Cross-sectional residual after industry and optional log-cap regression."""
    values, dates, industries = _aligned(
        _as_series(values, "values"), _as_labels(dates, "dates"),
        _as_labels(industries, "industries"),
    )
    market_cap = None if log_market_cap is None else _as_series(
        log_market_cap, "log_market_cap",
    )
    if market_cap is not None:
        values, dates, industries, market_cap = _aligned(
            values, dates, industries, market_cap,
        )

    output = np.full(len(values), np.nan, dtype=float)
    raw_values = values.to_numpy(dtype=float)
    date_values = dates.to_numpy()
    industry_codes, _ = pd.factorize(industries, sort=False)
    cap_values = None if market_cap is None else market_cap.to_numpy(dtype=float)

    if len(date_values) == 0:
        return pd.Series(output, index=values.index)

    order = np.argsort(date_values, kind="stable")
    sorted_dates = date_values[order]
    _, split_indices = np.unique(sorted_dates, return_index=True)
    split_indices = np.append(split_indices, len(date_values))

    for i in range(len(split_indices) - 1):
        positions = order[split_indices[i]:split_indices[i + 1]]
        valid = np.isfinite(raw_values[positions]) & (industry_codes[positions] >= 0)
        if cap_values is not None:
            valid &= np.isfinite(cap_values[positions])
        valid_positions = positions[valid]
        if len(valid_positions) == 0:
            continue
        sample = raw_values[valid_positions]
        columns = [np.ones(len(sample), dtype=float)]
        sample_industries = industry_codes[valid_positions]
        for code in np.unique(sample_industries)[:-1]:
            columns.append((sample_industries == code).astype(float))
        if cap_values is not None:
            columns.append(cap_values[valid_positions])
        design = np.column_stack(columns)
        if design.shape[0] <= design.shape[1]:
            continue
        coefficients, *_ = np.linalg.lstsq(design, sample, rcond=None)
        output[valid_positions] = sample - design @ coefficients
    return pd.Series(output, index=values.index)


def rank(values: Any, dates: Any, *, pct: bool = True) -> pd.Series:
    """WorldQuant-style cross-sectional rank."""
    return cross_rank(values, dates, pct=pct)


def scale(values: Any, dates: Any, scale: float = 1.0) -> pd.Series:
    """WorldQuant-style cross-sectional scale."""
    return cross_scale(values, dates, scale)


def ref(values: Any, stocks: Any, periods: int = 1) -> pd.Series:
    """WorldQuant-style lag reference."""
    return delay(values, stocks, periods)


def log(values: Any) -> pd.Series:
    """Alias for the safe natural logarithm."""
    return safe_log(values)


def sqrt(values: Any) -> pd.Series:
    """Square root of non-negative values."""
    values = _as_series(values, "values")
    raw = values.to_numpy(dtype=float)
    output = np.full(len(raw), np.nan, dtype=float)
    valid = np.isfinite(raw) & (raw >= 0.0)
    output[valid] = np.sqrt(raw[valid])
    return pd.Series(output, index=values.index)


def absolute(values: Any) -> pd.Series:
    """Alias for absolute value."""
    return safe_abs(values)


def ts_decay_linear(values: Any, stocks: Any, window: int,
                    min_periods: int | None = None) -> pd.Series:
    """WorldQuant-style linearly decaying moving average."""
    return decay_linear(values, stocks, window, min_periods)


def ts_arg_max(values: Any, stocks: Any, window: int,
               min_periods: int | None = None) -> pd.Series:
    """Alias for ts_argmax."""
    return ts_argmax(values, stocks, window, min_periods)


def ts_arg_min(values: Any, stocks: Any, window: int,
               min_periods: int | None = None) -> pd.Series:
    """Alias for ts_argmin."""
    return ts_argmin(values, stocks, window, min_periods)


_REGISTRY: dict[str, tuple[str, str, Any]] = {
    "add": ("elementwise", "两条对齐序列逐元素相加", add),
    "subtract": ("elementwise", "两条对齐序列逐元素相减", subtract),
    "multiply": ("elementwise", "两条对齐序列逐元素相乘", multiply),
    "inverse": ("elementwise", "倒数，零值返回缺失值", inverse),
    "absolute": ("elementwise", "绝对值的行业通用名称", absolute),
    "relu": ("elementwise", "只保留正值", relu),
    "square": ("elementwise", "逐元素平方", square),
    "cube": ("elementwise", "逐元素立方", cube),
    "power": ("elementwise", "逐元素幂运算", power),
    "signed_sqrt": ("elementwise", "保留符号的平方根", signed_sqrt),
    "signed_log": ("elementwise", "保留符号的 log1p", signed_log),
    "log": ("elementwise", "安全自然对数的通用别名", log),
    "exp": ("elementwise", "逐元素指数", exp),
    "sqrt": ("elementwise", "非负值平方根", sqrt),
    "tanh": ("elementwise", "双曲正切压缩", tanh),
    "sigmoid": ("elementwise", "数值稳定的逻辑函数", sigmoid),
    "logical_and": ("logical", "逐元素逻辑与", logical_and),
    "logical_or": ("logical", "逐元素逻辑或", logical_or),
    "logical_not": ("logical", "逐元素逻辑非", logical_not),
    "greater": ("comparison", "逐元素大于比较", greater),
    "greater_equal": ("comparison", "逐元素大于等于比较", greater_equal),
    "less": ("comparison", "逐元素小于比较", less),
    "less_equal": ("comparison", "逐元素小于等于比较", less_equal),
    "equal": ("comparison", "逐元素相等比较", equal),
    "not_equal": ("comparison", "逐元素不等比较", not_equal),
    "rank": ("cross_section", "WorldQuant 风格截面排名", rank),
    "scale": ("cross_section", "WorldQuant 风格截面缩放", scale),
    "cross_mean": ("cross_section", "按交易日截面均值广播", cross_mean),
    "cross_sum": ("cross_section", "按交易日截面求和广播", cross_sum),
    "cross_median": ("cross_section", "按交易日截面中位数广播", cross_median),
    "cross_std": ("cross_section", "按交易日截面标准差广播", cross_std),
    "cross_mad": ("cross_section", "按交易日截面中位绝对偏差广播", cross_mad),
    "group_mean": ("cross_section", "按交易日和类别分组均值广播", group_mean),
    "group_scale": ("cross_section", "按交易日和类别分组缩放", group_scale),
    "ref": ("time_series", "WorldQuant 风格滞后引用", ref),
    "ts_mad": ("time_series", "滚动中位绝对偏差", ts_mad),
    "ts_range": ("time_series", "滚动最大最小范围", ts_range),
    "ts_upside_std": ("risk", "滚动上行标准差", ts_upside_std),
    "ts_cv": ("risk", "滚动变异系数", ts_cv),
    "ts_ir": ("risk", "滚动信息比率", ts_ir),
    "ts_positive_ratio": ("time_series", "滚动正值占比", ts_positive_ratio),
    "ts_negative_ratio": ("time_series", "滚动非正值占比", ts_negative_ratio),
    "ts_cvar": ("risk", "滚动历史下尾平均", ts_cvar),
    "ts_max_drawdown": ("risk", "滚动窗口内最大回撤", ts_max_drawdown),
    "ts_ffill": ("time_series", "按个股只用过去值向前填充", ts_ffill),
    "ts_arg_max": ("time_series", "ts_argmax 的兼容别名", ts_arg_max),
    "ts_arg_min": ("time_series", "ts_argmin 的兼容别名", ts_arg_min),
    "wma": ("smoothing", "线性加权均值的通用名称", wma),
    "ts_decay_linear": ("smoothing", "WorldQuant 风格线性衰减均值", ts_decay_linear),
    "safe_divide": ("elementwise", "安全除法，零分母返回 NaN。", safe_divide),
    "safe_abs": ("elementwise", "绝对值。", safe_abs),
    "sign": ("elementwise", "符号函数，返回 -1/0/1。", sign),
    "clip": ("elementwise", "按上下界裁剪。", clip),
    "min2": ("elementwise", "两条序列逐元素最小值。", min2),
    "max2": ("elementwise", "两条序列逐元素最大值。", max2),
    "fillna": ("elementwise", "用对齐的回退序列填补缺失值。", fillna),
    "isfinite": ("elementwise", "有限值掩码。", isfinite),
    "safe_log": ("elementwise", "仅对正值取自然对数。", safe_log),
    "safe_log1p": ("elementwise", "安全 log1p。", safe_log1p),
    "log_diff": ("elementwise", "正数序列的对数差。", log_diff),
    "signed_power": ("elementwise", "保留符号的幂运算。", signed_power),
    "where": ("control", "向量化条件选择。", where),
    "cross_rank": ("cross_section", "按交易日截面排名。", cross_rank),
    "cross_demean": ("cross_section", "按交易日截面去均值。", cross_demean),
    "cross_zscore": ("cross_section", "按交易日截面标准化。", cross_zscore),
    "cross_scale": ("cross_section", "按截面绝对值之和缩放。", cross_scale),
    "cross_bucket": ("cross_section", "按交易日截面等频分组。", cross_bucket),
    "group_rank": ("cross_section", "按交易日和分组类别排名。", group_rank),
    "group_demean": ("cross_section", "按交易日和分组类别去均值。", group_demean),
    "group_zscore": ("cross_section", "按交易日和分组类别标准化。", group_zscore),
    "winsorize": ("cross_section", "按截面分位数缩尾。", winsorize),
    "winsorize_mad": ("cross_section", "按截面 MAD 缩尾。", winsorize_mad),
    "delay": ("time_series", "个股时序滞后。", delay),
    "delta": ("time_series", "个股时序差分。", delta),
    "pct_change": ("time_series", "个股时序收益率。", pct_change),
    "returns": ("time_series", "简单或对数收益率。", returns),
    "vwap": ("time_series", "成交额除以成交量的 VWAP。", vwap),
    "adv": ("time_series", "滚动平均成交额。", adv),
    "ts_mean": ("time_series", "个股滚动均值。", ts_mean),
    "ts_sum": ("time_series", "个股滚动求和。", ts_sum),
    "ts_product": ("time_series", "个股滚动乘积。", ts_product),
    "ts_count": ("time_series", "个股滚动有效值或正负值计数。", ts_count),
    "ts_sum_positive": ("time_series", "个股滚动正收益之和。", ts_sum_positive),
    "ts_sum_negative": ("time_series", "个股滚动负收益之和。", ts_sum_negative),
    "ts_std": ("time_series", "个股滚动标准差。", ts_std),
    "ts_var": ("time_series", "个股滚动方差。", ts_var),
    "ts_min": ("time_series", "个股滚动最小值。", ts_min),
    "ts_max": ("time_series", "个股滚动最大值。", ts_max),
    "ts_median": ("time_series", "个股滚动中位数。", ts_median),
    "ts_quantile": ("time_series", "个股滚动分位数。", ts_quantile),
    "ts_skew": ("time_series", "个股滚动偏度。", ts_skew),
    "ts_kurt": ("time_series", "个股滚动峰度。", ts_kurt),
    "ts_downside_std": ("time_series", "个股滚动下行波动率。", ts_downside_std),
    "ts_zscore": ("time_series", "个股滚动 z-score。", ts_zscore),
    "ts_drawdown": ("time_series", "相对滚动高点的回撤。", ts_drawdown),
    "ts_drawup": ("time_series", "相对滚动低点的上行距离。", ts_drawup),
    "ts_rank": ("time_series", "个股滚动分位排名。", ts_rank),
    "ts_argmax": ("time_series", "距窗口最高值的观测数。", ts_argmax),
    "ts_argmin": ("time_series", "距窗口最低值的观测数。", ts_argmin),
    "ts_corr": ("time_series", "两只序列的滚动相关。", ts_corr),
    "ts_cov": ("time_series", "两只序列的滚动协方差。", ts_cov),
    "ts_beta": ("time_series", "个股对另一序列的滚动 beta。", ts_beta),
    "ts_autocorr": ("time_series", "个股滚动自相关。", ts_autocorr),
    "ts_slope": ("regression", "滚动线性回归斜率。", ts_slope),
    "ts_residual": ("regression", "滚动线性回归最新残差。", ts_residual),
    "ts_r2": ("regression", "滚动线性趋势 R²。", ts_r2),
    "decay_linear": ("smoothing", "线性衰减加权均值。", decay_linear),
    "ema": ("smoothing", "因果指数移动平均。", ema),
    "demean_by_group": ("neutralization", "按指定分组去均值。", demean_by_group),
    "neutralize": ("neutralization", "行业与可选对数市值截面中性化。", neutralize),
}

# Compatibility spellings are part of the protocol surface, but they are not
# additional capabilities.  Keeping this relationship explicit prevents the
# discovery count from overstating the operator library and gives Agents one
# canonical name to emit in new Alpha Graphs.
_ALIASES: dict[str, str] = {
    "absolute": "safe_abs",
    "log": "safe_log",
    "rank": "cross_rank",
    "ref": "delay",
    "scale": "cross_scale",
    "ts_arg_max": "ts_argmax",
    "ts_arg_min": "ts_argmin",
    "ts_decay_linear": "decay_linear",
    "wma": "decay_linear",
}


def catalog() -> dict[str, Any]:
    """Return a machine-readable operator manifest for agents and IDEs."""
    operators = []
    for name, (category, description, function) in sorted(_REGISTRY.items()):
        alias_of = _ALIASES.get(name)
        operators.append({
            "name": name,
            "category": category,
            "description": description,
            "signature": f"{name}{inspect.signature(function)}",
            "causal": True,
            "role": "alias" if alias_of else "canonical",
            "canonical": alias_of is None,
            **({"aliasOf": alias_of} if alias_of else {}),
        })
    from quant_sdk.capabilities import installed_operator_specs

    operators.extend(installed_operator_specs())
    canonical = sum(
        1 for item in operators if item.get("canonical", True)
    )
    return {
        "version": 2,
        "module": "quant_sdk.operators",
        "index_contract": "输入 Series 必须索引唯一并使用相同索引顺序",
        "nan_policy": "保留缺失值；需要填充时显式使用 fillna/where",
        "counts": {
            "entries": len(operators),
            "canonical": canonical,
            "aliases": len(operators) - canonical,
        },
        "operators": operators,
    }


__all__ = [name for name in _REGISTRY] + ["catalog"]
