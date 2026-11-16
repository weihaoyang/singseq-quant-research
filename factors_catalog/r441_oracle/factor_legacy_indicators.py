# ruff: noqa
from __future__ import annotations

from factors_catalog.r441_oracle.factor_math import (
    _max_drawdown,
    _mean,
    _moving_average,
    _pearson,
    _returns,
    _safe_div,
    _std,
    _window,
    _zscore,
)


def _momentum(closes: list[float], window: int) -> float:
    if len(closes) <= window:
        return 0.0
    base = closes[-1 - window]
    return _safe_div(closes[-1] - base, base)


def _price_vs_ma(closes: list[float], window: int) -> float:
    if len(closes) < window:
        return 0.0
    ma = _moving_average(closes, window)
    return _safe_div(closes[-1] - ma, ma)


def _breakout_distance(closes: list[float], highs: list[float], window: int) -> float:
    if len(closes) <= window or len(highs) <= window:
        return 0.0
    reference = max(highs[-1 - window : -1])
    return _safe_div(closes[-1] - reference, reference)


def _drawdown_from_high(closes: list[float], window: int) -> float:
    if len(closes) < window:
        return 0.0
    high_water = max(_window(closes, window))
    return _safe_div(closes[-1] - high_water, high_water)


def _realized_volatility(closes: list[float], window: int) -> float:
    returns = _window(_returns(closes), window)
    return _std(returns)


def _downside_volatility(closes: list[float], window: int) -> float:
    returns = _window(_returns(closes), window)
    negatives = [value for value in returns if value < 0.0]
    return _std(negatives)


def _range_ratio(highs: list[float], lows: list[float], closes: list[float], window: int) -> float:
    if len(highs) < window or len(lows) < window or len(closes) < window:
        return 0.0
    ratios: list[float] = []
    for high, low, close in zip(_window(highs, window), _window(lows, window), _window(closes, window)):
        ratios.append(_safe_div(high - low, close if close > 0 else 1.0))
    return _mean(ratios)


def _compression_ratio(highs: list[float], lows: list[float], closes: list[float], short: int, long: int) -> float:
    if len(highs) < long or len(lows) < long or len(closes) < long:
        return 0.0
    short_range = _range_ratio(highs, lows, closes, short)
    long_range = _range_ratio(highs[-long:], lows[-long:], closes[-long:], long)
    return _safe_div(long_range - short_range, abs(long_range) + 1e-12)


def _turnover_zscore(turnover: list[float], window: int) -> float:
    if len(turnover) <= window:
        return 0.0
    base = turnover[-1 - window : -1]
    return _zscore(turnover[-1], base)


def _turnover_stability(turnover: list[float], window: int) -> float:
    segment = _window(turnover, window)
    if len(segment) < window:
        return 0.0
    avg = _mean(segment)
    std = _std(segment)
    return _safe_div(avg, std if std > 1e-12 else 1.0)


def _flow_proxy_strength(closes: list[float], turnover: list[float], window: int) -> float:
    returns = _returns(closes)
    if len(returns) < window or len(turnover) < window + 1:
        return 0.0
    flows: list[float] = []
    for index in range(-window, 0):
        flows.append(returns[index] * turnover[index])
    latest_turnover = _mean(_window(turnover, window))
    return _safe_div(_mean(flows), abs(latest_turnover) + 1.0)


def _turnover_return_correlation(closes: list[float], turnover: list[float], window: int) -> float:
    returns = _window(_returns(closes), window)
    if len(returns) < 3 or len(turnover) < window + 1:
        return 0.0
    turnover_changes = _returns(_window(turnover, window + 1))
    if len(turnover_changes) != len(returns):
        return 0.0
    return _pearson(returns, turnover_changes)


def _relative_strength(closes: list[float], benchmark_closes: list[float], window: int) -> float:
    if len(closes) <= window or len(benchmark_closes) <= window:
        return 0.0
    stock_ret = _momentum(closes, window)
    bench_ret = _momentum(benchmark_closes, window)
    return stock_ret - bench_ret


def _beta(closes: list[float], benchmark_closes: list[float], window: int) -> float:
    stock_returns = _window(_returns(closes), window)
    bench_returns = _window(_returns(benchmark_closes), window)
    if len(stock_returns) != len(bench_returns) or len(stock_returns) < 5:
        return 0.0
    bench_var = _std(bench_returns) ** 2
    if bench_var <= 1e-12:
        return 0.0
    stock_mean = _mean(stock_returns)
    bench_mean = _mean(bench_returns)
    cov = 0.0
    for stock_ret, bench_ret in zip(stock_returns, bench_returns):
        cov += (stock_ret - stock_mean) * (bench_ret - bench_mean)
    cov /= len(stock_returns)
    return cov / bench_var


def _alpha(closes: list[float], benchmark_closes: list[float], window: int, beta_window: int) -> float:
    if len(closes) <= window or len(benchmark_closes) <= window:
        return 0.0
    beta = _beta(closes, benchmark_closes, beta_window)
    return _momentum(closes, window) - beta * _momentum(benchmark_closes, window)


def _vol_adjusted_momentum(closes: list[float], window: int) -> float:
    momentum = _momentum(closes, window)
    volatility = _realized_volatility(closes, window)
    return _safe_div(momentum, volatility if volatility > 1e-12 else 1.0)


def _sharpe_like(closes: list[float], window: int) -> float:
    returns = _window(_returns(closes), window)
    if len(returns) < window:
        return 0.0
    return _safe_div(_mean(returns), _std(returns) if _std(returns) > 1e-12 else 1.0)


def _drawdown_resilience(closes: list[float], window: int) -> float:
    if len(closes) < window:
        return 0.0
    return -_max_drawdown(_window(closes, window))
