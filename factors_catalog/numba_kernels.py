# ruff: noqa
from __future__ import annotations

from os import getenv
from math import sqrt
from typing import Sequence

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

try:
    from numba import njit
except Exception:  # noqa: BLE001
    njit = None  # type: ignore[assignment]

_NUMBA_RUNTIME_OK = bool(njit is not None and np is not None and hasattr(np, "trapz"))
_NUMBA_CACHE_ENABLED = getenv("QUANT_NUMBA_DISK_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}

_CORE_PRICE_TURNOVER_KEYS = (
    "trend_acceleration_20",
    "breakout_quality_10",
    "price_momentum_20",
    "price_momentum_60",
    "ma_gap_20",
    "ma_gap_60",
    "price_position_20",
    "pullback_absorption_10",
    "liquidity_sponsorship_20",
    "exit_deterioration_score_5",
    "volatility_compression_20",
    "realized_volatility_20",
    "downside_volatility_20",
    "momentum_efficiency_20",
    "trend_stability_15",
    "breakout_pressure_10",
    "turnover_impulse_5_20",
    "turnover_stability_20",
    "amihud_illiquidity_20",
    "exhaustion_risk_10",
    "abnormal_turnover_z20",
    "large_candle_followthrough_5",
    "distribution_warning_5",
)


def _core_price_turnover_bundle_from_values(values: Sequence[float]) -> dict[str, float]:
    if len(values) != len(_CORE_PRICE_TURNOVER_KEYS):
        raise ValueError(
            f"core price/turnover factor count mismatch: values={len(values)} keys={len(_CORE_PRICE_TURNOVER_KEYS)}"
        )
    return {key: float(value) for key, value in zip(_CORE_PRICE_TURNOVER_KEYS, values)}


def _std_window_py(values: Sequence[float], window: int) -> float:
    if len(values) < window + 1:
        return 0.0
    returns: list[float] = []
    for i in range(len(values) - window - 1, len(values) - 1):
        prev = values[i]
        curr = values[i + 1]
        if prev > 0:
            returns.append((curr - prev) / prev)
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((item - mean) ** 2 for item in returns) / (len(returns) - 1)
    return sqrt(var)


def _safe_div_py(num: float, den: float) -> float:
    return num / den if den else 0.0


def _mean_py(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std_py(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean_py(values)
    return (_mean_py([(value - avg) ** 2 for value in values])) ** 0.5


def _price_momentum_py(prices: Sequence[float], window: int) -> float:
    return _safe_div_py(prices[-1] - prices[-window - 1], prices[-window - 1]) if len(prices) >= window + 1 else 0.0


def _ma_gap_py(prices: Sequence[float], window: int) -> float:
    if len(prices) < window:
        return 0.0
    avg = _mean_py(prices[-window:])
    return _safe_div_py(prices[-1] - avg, avg)


def _price_position_py(prices: Sequence[float], window: int) -> float:
    if len(prices) < window:
        return 0.0
    window_values = prices[-window:]
    high = max(window_values)
    low = min(window_values)
    if high <= low:
        return 0.0
    position = _safe_div_py(window_values[-1] - low, high - low)
    return max(min(position - 0.5, 0.5), -0.5)


def _pullback_absorption_py(prices: Sequence[float], window: int) -> float:
    if len(prices) < window:
        return 0.0
    window_values = prices[-window:]
    max_price = max(window_values)
    min_price = min(window_values)
    if max_price == min_price:
        return 0.0
    retrace_ratio = _safe_div_py(max_price - window_values[-1], max_price - min_price)
    return 1.0 - retrace_ratio


def _liquidity_sponsorship_py(turnover: Sequence[float], window: int = 20) -> float:
    if len(turnover) < window:
        return 0.0
    recent = turnover[-window:]
    midpoint = window // 2
    baseline = _mean_py(recent[:midpoint])
    latest = _mean_py(recent[midpoint:])
    return _safe_div_py(latest - baseline, baseline)


def _turnover_impulse_py(turnover: Sequence[float], short_window: int = 5, long_window: int = 20) -> float:
    if len(turnover) < long_window:
        return 0.0
    short = _mean_py(turnover[-short_window:])
    long = _mean_py(turnover[-long_window:])
    return _safe_div_py(short - long, abs(long) + 1.0)


def _turnover_stability_py(turnover: Sequence[float], window_size: int = 20) -> float:
    if len(turnover) < window_size:
        return 0.0
    window = [max(value, 0.0) for value in turnover[-window_size:]]
    mean_turnover = _mean_py(window)
    std_turnover = _std_py(window)
    variation = _safe_div_py(std_turnover, mean_turnover + 1.0)
    return _safe_div_py(1.0, 1.0 + variation)


def _abnormal_turnover_z_py(turnover: Sequence[float], window_size: int = 20) -> float:
    if len(turnover) < window_size + 1:
        return 0.0
    window = turnover[-window_size - 1:-1]
    std = _std_py(window)
    return _safe_div_py(turnover[-1] - _mean_py(window), std if std > 1e-12 else 1.0)


def _core_price_turnover_bundle_py(prices: Sequence[float], turnover: Sequence[float]) -> dict[str, float]:
    prices_list = [float(value) for value in prices]
    turnover_list = [float(value) for value in turnover]
    returns = [
        _safe_div_py(prices_list[index + 1] - prices_list[index], prices_list[index])
        for index in range(len(prices_list) - 1)
    ]

    trend_acceleration = 0.0
    if len(prices_list) >= 21:
        recent = returns[-20:]
        early = recent[:10]
        late = recent[10:]
        early_slope = _safe_div_py(early[-1] - early[0], max(len(early) - 1, 1))
        late_slope = _safe_div_py(late[-1] - late[0], max(len(late) - 1, 1))
        trend_acceleration = late_slope - early_slope

    breakout_quality = 0.0
    if len(prices_list) >= 11:
        window = prices_list[-11:]
        breakout = max(window[:-1])
        breakout_quality = _safe_div_py(window[-1] - breakout, breakout)

    price_momentum_20 = _price_momentum_py(prices_list, 20)
    price_momentum_60 = _price_momentum_py(prices_list, 60)
    ma_gap_20 = _ma_gap_py(prices_list, 20)
    ma_gap_60 = _ma_gap_py(prices_list, 60)
    price_position_20 = _price_position_py(prices_list, 20)
    pullback_absorption_10 = _pullback_absorption_py(prices_list, 11)

    liquidity_sponsorship_20 = _liquidity_sponsorship_py(turnover_list)

    exit_deterioration_score_5 = 0.0
    if len(prices_list) >= 6:
        recent_returns = returns[-5:]
        negative = [value for value in recent_returns if value < 0.0]
        exit_deterioration_score_5 = abs(sum(negative)) if negative else 0.0

    volatility_compression_20 = 0.0
    if len(returns) >= 40:
        pre = returns[-40:-20]
        recent = returns[-20:]
        pre_vol = _std_py(pre)
        recent_vol = _std_py(recent)
        volatility_compression_20 = max(_safe_div_py(pre_vol - recent_vol, pre_vol), 0.0)

    realized_volatility_20 = _std_py(returns[-20:]) if len(prices_list) >= 21 else 0.0
    downside_volatility_20 = 0.0
    if len(prices_list) >= 21:
        downside = [min(value, 0.0) for value in returns[-20:]]
        downside_volatility_20 = (_mean_py([value * value for value in downside])) ** 0.5 if downside else 0.0

    momentum_efficiency_20 = 0.0
    if len(prices_list) >= 21:
        window21 = prices_list[-21:]
        net = window21[-1] - window21[0]
        path = 0.0
        for index in range(1, len(window21)):
            path += abs(window21[index] - window21[index - 1])
        momentum_efficiency_20 = _safe_div_py(net, path if path > 1e-12 else 1.0)

    trend_stability_15 = 0.0
    if len(prices_list) >= 16:
        window15 = returns[-15:]
        sign_changes = 0
        prev_sign = 0
        for value in window15:
            sign = 1 if value > 0 else -1 if value < 0 else 0
            if prev_sign != 0 and sign != 0 and sign != prev_sign:
                sign_changes += 1
            if sign != 0:
                prev_sign = sign
        stability = 1.0 - _safe_div_py(sign_changes, 14.0)
        mean_window = _mean_py(window15)
        direction = 1.0 if mean_window > 0 else -1.0 if mean_window < 0 else 0.0
        trend_stability_15 = stability * direction

    breakout_pressure_10 = 0.0
    if len(prices_list) >= 12:
        window10 = returns[-10:]
        breakout_pressure_10 = _mean_py(window10[7:]) - _mean_py(window10[:7])

    turnover_impulse_5_20 = _turnover_impulse_py(turnover_list)
    turnover_stability_20 = _turnover_stability_py(turnover_list)
    abnormal_turnover_z20 = _abnormal_turnover_z_py(turnover_list)

    amihud_illiquidity_20 = 0.0
    if len(prices_list) >= 21 and len(turnover_list) >= 21:
        values: list[float] = []
        size = min(len(prices_list), len(turnover_list))
        for index in range(size - 20, size):
            if index <= 0:
                continue
            base = prices_list[index - 1]
            ret = _safe_div_py(prices_list[index] - base, base)
            cash_turnover = max(turnover_list[index], 1.0)
            values.append(abs(ret) / cash_turnover)
        amihud_illiquidity_20 = _mean_py(values)

    exhaustion_risk_10 = 0.0
    if len(prices_list) >= 11 and len(turnover_list) >= 11:
        recent_returns = returns[-10:]
        up_spikes = _safe_div_py(len([value for value in recent_returns if value >= 0.03]), 10.0)
        recent_turnover = turnover_list[-11:]
        baseline = recent_turnover[:-1]
        std = _std_py(baseline)
        turnover_z = _safe_div_py(recent_turnover[-1] - _mean_py(baseline), std if std > 1e-12 else 1.0)
        recent_window = [float(value) for value in prices_list[-10:]]
        near_high = 1.0 if recent_window[-1] >= max(recent_window) * 0.995 else 0.0
        pullback = _safe_div_py(max(recent_window) - recent_window[-1], max(recent_window))
        trend_run = _safe_div_py(recent_window[-1] - recent_window[0], recent_window[0])
        extension = max(trend_run - 0.12, 0.0) / 0.18
        weighted = (
            up_spikes * 0.30
            + max(min(turnover_z / 2.5, 1.0), 0.0) * 0.20
            + near_high * max(min(extension, 1.0), 0.0) * 0.30
            + max(0.015 - pullback, 0.0) / 0.015 * 0.20
        )
        exhaustion_risk_10 = max(0.0, min(weighted, 1.0))

    large_candle_followthrough_5 = 0.0
    if len(prices_list) >= 6:
        recent = returns[-5:]
        strong = [value for value in recent if value > 0.02]
        large_candle_followthrough_5 = _safe_div_py(len(strong), len(recent))

    distribution_warning_5 = 0.0
    if len(prices_list) >= 6 and len(turnover_list) >= 6:
        warnings = 0
        for index in range(1, 6):
            price_down = prices_list[-index] < prices_list[-index - 1]
            volume_up = turnover_list[-index] > turnover_list[-index - 1]
            if price_down and volume_up:
                warnings += 1
        distribution_warning_5 = _safe_div_py(warnings, 5.0)

    values = (
        trend_acceleration,
        breakout_quality,
        price_momentum_20,
        price_momentum_60,
        ma_gap_20,
        ma_gap_60,
        price_position_20,
        pullback_absorption_10,
        liquidity_sponsorship_20,
        exit_deterioration_score_5,
        volatility_compression_20,
        realized_volatility_20,
        downside_volatility_20,
        momentum_efficiency_20,
        trend_stability_15,
        breakout_pressure_10,
        turnover_impulse_5_20,
        turnover_stability_20,
        amihud_illiquidity_20,
        exhaustion_risk_10,
        abnormal_turnover_z20,
        large_candle_followthrough_5,
        distribution_warning_5,
    )
    return _core_price_turnover_bundle_from_values(values)


def _core_price_turnover_bundle_np(prices: Sequence[float], turnover: Sequence[float]) -> dict[str, float]:
    if np is None:
        return _core_price_turnover_bundle_py(prices, turnover)

    prices_arr = np.asarray(prices, dtype=np.float64)
    turnover_arr = np.asarray(turnover, dtype=np.float64)
    returns = (
        np.divide(
            prices_arr[1:] - prices_arr[:-1],
            prices_arr[:-1],
            out=np.zeros(max(prices_arr.shape[0] - 1, 0), dtype=np.float64),
            where=prices_arr[:-1] != 0.0,
        )
        if prices_arr.shape[0] >= 2
        else np.zeros(0, dtype=np.float64)
    )

    trend_acceleration = 0.0
    if prices_arr.shape[0] >= 21:
        recent = returns[-20:]
        early = recent[:10]
        late = recent[10:]
        early_slope = (early[-1] - early[0]) / max(early.shape[0] - 1, 1)
        late_slope = (late[-1] - late[0]) / max(late.shape[0] - 1, 1)
        trend_acceleration = float(late_slope - early_slope)

    breakout_quality = 0.0
    if prices_arr.shape[0] >= 11:
        window11 = prices_arr[-11:]
        breakout = float(np.max(window11[:-1]))
        breakout_quality = float((window11[-1] - breakout) / breakout) if breakout else 0.0

    prices_scalar = prices_arr.tolist()
    price_momentum_20 = _price_momentum_py(prices_scalar, 20)
    price_momentum_60 = _price_momentum_py(prices_scalar, 60)
    ma_gap_20 = _ma_gap_py(prices_scalar, 20)
    ma_gap_60 = _ma_gap_py(prices_scalar, 60)
    price_position_20 = _price_position_py(prices_scalar, 20)
    pullback_absorption_10 = _pullback_absorption_py(prices_scalar, 11)

    turnover_scalar = turnover_arr.tolist()
    liquidity_sponsorship_20 = _liquidity_sponsorship_py(turnover_scalar)

    exit_deterioration_score_5 = 0.0
    if prices_arr.shape[0] >= 6:
        recent_returns = returns[-5:]
        negative = recent_returns[recent_returns < 0.0]
        exit_deterioration_score_5 = float(abs(np.sum(negative))) if negative.shape[0] else 0.0

    volatility_compression_20 = 0.0
    if returns.shape[0] >= 40:
        pre = returns[-40:-20]
        recent = returns[-20:]
        pre_vol = float(np.std(pre))
        recent_vol = float(np.std(recent))
        volatility_compression_20 = max(float((pre_vol - recent_vol) / pre_vol) if pre_vol else 0.0, 0.0)

    realized_volatility_20 = float(np.std(returns[-20:])) if prices_arr.shape[0] >= 21 else 0.0
    downside_volatility_20 = 0.0
    if prices_arr.shape[0] >= 21:
        downside = np.minimum(returns[-20:], 0.0)
        downside_volatility_20 = float(np.sqrt(np.mean(downside * downside))) if downside.shape[0] else 0.0

    momentum_efficiency_20 = 0.0
    if prices_arr.shape[0] >= 21:
        window21 = prices_arr[-21:]
        net = float(window21[-1] - window21[0])
        path = float(np.sum(np.abs(np.diff(window21))))
        momentum_efficiency_20 = float(net / path) if path > 1e-12 else 0.0

    trend_stability_15 = 0.0
    if prices_arr.shape[0] >= 16:
        window15 = returns[-15:]
        sign_changes = 0
        prev_sign = 0
        for value in window15:
            sign = 1 if value > 0 else -1 if value < 0 else 0
            if prev_sign != 0 and sign != 0 and sign != prev_sign:
                sign_changes += 1
            if sign != 0:
                prev_sign = sign
        stability = 1.0 - (sign_changes / 14.0)
        mean_window = float(np.mean(window15)) if window15.shape[0] else 0.0
        direction = 1.0 if mean_window > 0 else -1.0 if mean_window < 0 else 0.0
        trend_stability_15 = float(stability * direction)

    breakout_pressure_10 = 0.0
    if prices_arr.shape[0] >= 12:
        window10 = returns[-10:]
        breakout_pressure_10 = float(np.mean(window10[7:]) - np.mean(window10[:7]))

    turnover_impulse_5_20 = _turnover_impulse_py(turnover_scalar)
    turnover_stability_20 = _turnover_stability_py(turnover_scalar)
    abnormal_turnover_z20 = _abnormal_turnover_z_py(turnover_scalar)

    amihud_illiquidity_20 = 0.0
    if prices_arr.shape[0] >= 21 and turnover_arr.shape[0] >= 21:
        size = min(prices_arr.shape[0], turnover_arr.shape[0])
        amihud_values = np.zeros(20, dtype=np.float64)
        used = 0
        for index in range(size - 20, size):
            if index <= 0:
                continue
            base = float(prices_arr[index - 1])
            ret = float((prices_arr[index] - base) / base) if base else 0.0
            cash_turnover = max(float(turnover_arr[index]), 1.0)
            amihud_values[used] = abs(ret) / cash_turnover
            used += 1
        if used > 0:
            amihud_illiquidity_20 = float(np.mean(amihud_values[:used]))

    exhaustion_risk_10 = 0.0
    if prices_arr.shape[0] >= 11 and turnover_arr.shape[0] >= 11:
        recent_returns = returns[-10:]
        up_spikes = float(np.sum(recent_returns >= 0.03) / 10.0)
        recent_turnover = turnover_arr[-11:]
        baseline = recent_turnover[:-1]
        std = float(np.std(baseline))
        turnover_z = float((recent_turnover[-1] - float(np.mean(baseline))) / (std if std > 1e-12 else 1.0))
        recent_window = prices_arr[-10:]
        recent_high = float(np.max(recent_window))
        near_high = 1.0 if float(recent_window[-1]) >= recent_high * 0.995 else 0.0
        pullback = float((recent_high - float(recent_window[-1])) / recent_high) if recent_high else 0.0
        trend_run = float((float(recent_window[-1]) - float(recent_window[0])) / float(recent_window[0])) if float(recent_window[0]) else 0.0
        extension = max(trend_run - 0.12, 0.0) / 0.18
        weighted = (
            up_spikes * 0.30
            + max(min(turnover_z / 2.5, 1.0), 0.0) * 0.20
            + near_high * max(min(extension, 1.0), 0.0) * 0.30
            + max(0.015 - pullback, 0.0) / 0.015 * 0.20
        )
        exhaustion_risk_10 = max(0.0, min(weighted, 1.0))

    large_candle_followthrough_5 = 0.0
    if prices_arr.shape[0] >= 6:
        recent = returns[-5:]
        large_candle_followthrough_5 = float(np.sum(recent > 0.02) / max(recent.shape[0], 1))

    distribution_warning_5 = 0.0
    if prices_arr.shape[0] >= 6 and turnover_arr.shape[0] >= 6:
        warnings = 0
        for index in range(1, 6):
            price_down = prices_arr[-index] < prices_arr[-index - 1]
            volume_up = turnover_arr[-index] > turnover_arr[-index - 1]
            if price_down and volume_up:
                warnings += 1
        distribution_warning_5 = warnings / 5.0

    values = (
        trend_acceleration,
        breakout_quality,
        price_momentum_20,
        price_momentum_60,
        ma_gap_20,
        ma_gap_60,
        price_position_20,
        pullback_absorption_10,
        liquidity_sponsorship_20,
        exit_deterioration_score_5,
        volatility_compression_20,
        realized_volatility_20,
        downside_volatility_20,
        momentum_efficiency_20,
        trend_stability_15,
        breakout_pressure_10,
        turnover_impulse_5_20,
        turnover_stability_20,
        amihud_illiquidity_20,
        exhaustion_risk_10,
        abnormal_turnover_z20,
        large_candle_followthrough_5,
        distribution_warning_5,
    )
    return _core_price_turnover_bundle_from_values(values)


if njit is not None:

    @njit(cache=_NUMBA_CACHE_ENABLED, fastmath=True)
    def _std_window_numba(values: "np.ndarray", window: int) -> float:  # type: ignore[name-defined]
        size = values.shape[0]
        if size < window + 1:
            return 0.0
        start = size - window - 1
        count = window
        mean = 0.0
        used = 0
        for i in range(start, start + count):
            prev = values[i]
            curr = values[i + 1]
            if prev <= 0.0:
                continue
            mean += (curr - prev) / prev
            used += 1
        if used < 2:
            return 0.0
        mean /= used
        var = 0.0
        for i in range(start, start + count):
            prev = values[i]
            curr = values[i + 1]
            if prev <= 0.0:
                continue
            ret = (curr - prev) / prev
            diff = ret - mean
            var += diff * diff
        return sqrt(var / (used - 1))

else:

    _std_window_numba = None  # type: ignore[assignment]


def volatility_last_window(
    closes: Sequence[float],
    *,
    window: int = 20,
    enable_numba: bool = True,
) -> float:
    global _NUMBA_RUNTIME_OK
    if len(closes) < window + 1:
        return 0.0
    if np is not None and enable_numba and _NUMBA_RUNTIME_OK and _std_window_numba is not None:
        try:
            arr = np.asarray(closes, dtype=np.float64)
            return float(_std_window_numba(arr, int(window)))
        except Exception:  # noqa: BLE001
            _NUMBA_RUNTIME_OK = False
    return float(_std_window_py(closes, int(window)))


def compute_core_price_turnover_factor_bundle(
    prices: Sequence[float],
    turnover: Sequence[float],
    *,
    enable_numba: bool = True,
) -> dict[str, float]:
    # Small rolling windows are typically faster in pure Python due to numpy conversion overhead.
    use_accelerated_path = max(len(prices), len(turnover)) >= 256
    if enable_numba and use_accelerated_path and np is not None:
        try:
            return _core_price_turnover_bundle_np(prices, turnover)
        except Exception:  # noqa: BLE001
            pass
    return _core_price_turnover_bundle_py(prices, turnover)
