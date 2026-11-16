# ruff: noqa
from __future__ import annotations


def safe_div(num: float, den: float) -> float:
    return num / den if den else float("nan")


def _tail(values: list[float], size: int) -> list[float]:
    if size <= 0:
        return []
    if len(values) <= size:
        return values
    return values[-size:]


def intraday_input_quality(
    *,
    closes_30m: list[float] | None,
    opens_30m: list[float] | None,
    highs_30m: list[float] | None,
    lows_30m: list[float] | None,
    turnover_30m: list[float] | None,
    vwap_30m: list[float] | None,
) -> dict[str, object]:
    missing_fields = [
        name
        for name, values in (
            ("closes_30m", closes_30m),
            ("opens_30m", opens_30m),
            ("highs_30m", highs_30m),
            ("lows_30m", lows_30m),
            ("turnover_30m", turnover_30m),
            ("vwap_30m", vwap_30m),
        )
        if not values
    ]
    synthetic_intraday = False
    return {
        "intraday_source": "30m" if not missing_fields else "unavailable",
        "synthetic_intraday": synthetic_intraday,
        "partial_intraday_fallback": bool(missing_fields) and not synthetic_intraday,
        "missing_fields": missing_fields,
    }


def price_returns(prices: list[float]) -> list[float]:
    if len(prices) < 2:
        return []
    returns: list[float] = []
    for index in range(1, len(prices)):
        returns.append(safe_div(prices[index] - prices[index - 1], prices[index - 1]))
    return returns


def signed_flow_proxy(prices: list[float], turnover: list[float]) -> list[float]:
    length = min(len(prices), len(turnover))
    if length < 2:
        return []
    flows: list[float] = []
    for index in range(1, length):
        ret = safe_div(prices[index] - prices[index - 1], prices[index - 1])
        flows.append(float(turnover[index]) * ret)
    return flows


def signed_flow_proxy_tail(prices: list[float], turnover: list[float], size: int) -> list[float]:
    length = min(len(prices), len(turnover))
    if length < 2 or size <= 0:
        return []
    result_size = min(size, length - 1)
    start = length - result_size
    flows: list[float] = []
    for index in range(start, length):
        prev = prices[index - 1]
        curr = prices[index]
        ret = safe_div(curr - prev, prev)
        flows.append(float(turnover[index]) * ret)
    return flows


def _build_sentiment_proxies(
    returns: list[float],
    sentiment_series: list[float] | None,
    board_break_series: list[float] | None,
    gap_series: list[float] | None,
    reversal_series: list[float] | None,
) -> tuple[list[float], list[float], list[float], list[float]]:
    _ = returns
    return (
        list(sentiment_series) if sentiment_series is not None else [],
        list(board_break_series) if board_break_series is not None else [],
        list(gap_series) if gap_series is not None else [],
        list(reversal_series) if reversal_series is not None else [],
    )


def _build_event_proxies(
    sentiment_proxy: list[float],
    event_direction_series: list[float] | None,
    event_intensity_series: list[float] | None,
    event_confidence_series: list[float] | None,
    event_age_minutes: list[float] | None,
    event_impact_series: list[float] | None,
    event_sector_hits: list[float] | None,
    event_symbol_hits: list[float] | None,
    regulatory_event_series: list[float] | None,
    shareholder_event_series: list[float] | None,
    policy_event_series: list[float] | None,
    hotword_series: list[float] | None,
) -> dict[str, list[float]]:
    _ = sentiment_proxy
    return {
        "event_direction_series": list(event_direction_series) if event_direction_series is not None else [],
        "event_intensity_series": list(event_intensity_series) if event_intensity_series is not None else [],
        "event_confidence_series": list(event_confidence_series) if event_confidence_series is not None else [],
        "event_age_minutes": list(event_age_minutes) if event_age_minutes is not None else [],
        "event_impact_series": list(event_impact_series) if event_impact_series is not None else [],
        "event_sector_hits": list(event_sector_hits) if event_sector_hits is not None else [],
        "event_symbol_hits": list(event_symbol_hits) if event_symbol_hits is not None else [],
        "regulatory_event_series": list(regulatory_event_series) if regulatory_event_series is not None else [],
        "shareholder_event_series": list(shareholder_event_series) if shareholder_event_series is not None else [],
        "policy_event_series": list(policy_event_series) if policy_event_series is not None else [],
        "hotword_series": list(hotword_series) if hotword_series is not None else [],
    }


def normalize_inputs(
    prices: list[float],
    turnover: list[float],
    benchmark_prices: list[float] | None,
    opens_1d: list[float] | None,
    highs_1d: list[float] | None,
    lows_1d: list[float] | None,
    closes_30m: list[float] | None,
    opens_30m: list[float] | None,
    highs_30m: list[float] | None,
    lows_30m: list[float] | None,
    turnover_30m: list[float] | None,
    vwap_30m: list[float] | None,
    stock_flows: list[float] | None,
    sector_flows: list[float] | None,
    sector_closes: list[float] | None,
    market_closes: list[float] | None,
    sector_up_counts: list[float] | None,
    sector_total_counts: list[float] | None,
    leader_prices: list[float] | None,
    laggard_prices: list[float] | None,
    sentiment_series: list[float] | None,
    board_break_series: list[float] | None,
    gap_series: list[float] | None,
    reversal_series: list[float] | None,
    event_direction_series: list[float] | None,
    event_intensity_series: list[float] | None,
    event_confidence_series: list[float] | None,
    event_age_minutes: list[float] | None,
    event_impact_series: list[float] | None,
    event_sector_hits: list[float] | None,
    event_symbol_hits: list[float] | None,
    regulatory_event_series: list[float] | None,
    shareholder_event_series: list[float] | None,
    policy_event_series: list[float] | None,
    hotword_series: list[float] | None,
    official_clean_mode: bool = False,
    allow_flow_proxy_fallback: bool = True,
) -> dict[str, list[float]]:
    return _normalize_inputs_impl(locals())


def _normalize_inputs_impl(payload: dict[str, object]) -> dict[str, list[float]]:
    # Most downstream helper proxies consume <=60 lookback windows; keep close/high/low defaults
    # aligned with full `prices` length to preserve factor input shape consistency.
    max_window = 64
    prices = payload["prices"]
    turnover = payload["turnover"]
    prices_view = _tail(prices, max_window + 1)
    turnover_view = _tail(turnover, max_window + 1)
    returns_source = payload["gap_series"] if payload["gap_series"] is not None else price_returns(prices_view)
    returns = list(returns_source)
    stock_flow_series, sector_flow_series = _build_flow_proxies(
        prices_view=prices_view,
        turnover_view=turnover_view,
        stock_flows=payload["stock_flows"],
        sector_flows=payload["sector_flows"],
        allow_flow_proxy_fallback=bool(payload.get("allow_flow_proxy_fallback", True)),
    )
    sentiment_proxy, board_break_proxy, gap_proxy, reversal_proxy = _build_sentiment_proxies(
        returns,
        payload["sentiment_series"],
        payload["board_break_series"],
        payload["gap_series"],
        payload["reversal_series"],
    )
    event_inputs = _build_event_proxies(
        sentiment_proxy,
        payload["event_direction_series"],
        payload["event_intensity_series"],
        payload["event_confidence_series"],
        payload["event_age_minutes"],
        payload["event_impact_series"],
        payload["event_sector_hits"],
        payload["event_symbol_hits"],
        payload["regulatory_event_series"],
        payload["shareholder_event_series"],
        payload["policy_event_series"],
        payload["hotword_series"],
    )
    benchmark_series = list(payload["benchmark_prices"]) if payload["benchmark_prices"] is not None else []
    return {
        **_build_daily_market_inputs(
            benchmark_series=benchmark_series,
            opens_1d=payload["opens_1d"],
            highs_1d=payload["highs_1d"],
            lows_1d=payload["lows_1d"],
            sector_closes=payload["sector_closes"],
            market_closes=payload["market_closes"],
            leader_prices=payload["leader_prices"],
            laggard_prices=payload["laggard_prices"],
        ),
        **_build_intraday_market_inputs(
            prices_view=prices_view,
            turnover_view=turnover_view,
            closes_30m=payload["closes_30m"],
            opens_30m=payload["opens_30m"],
            highs_30m=payload["highs_30m"],
            lows_30m=payload["lows_30m"],
            turnover_30m=payload["turnover_30m"],
            vwap_30m=payload["vwap_30m"],
        ),
        "stock_flows": stock_flow_series,
        "sector_flows": sector_flow_series,
        "sector_up_counts": list(payload["sector_up_counts"]) if payload["sector_up_counts"] is not None else [],
        "sector_total_counts": list(payload["sector_total_counts"]) if payload["sector_total_counts"] is not None else [],
        "sentiment_series": sentiment_proxy,
        "board_break_series": board_break_proxy,
        "gap_series": gap_proxy,
        "reversal_series": reversal_proxy,
        **event_inputs,
    }


def _build_flow_proxies(
    *,
    prices_view: list[float],
    turnover_view: list[float],
    stock_flows: list[float] | None,
    sector_flows: list[float] | None,
    allow_flow_proxy_fallback: bool,
) -> tuple[list[float], list[float]]:
    if stock_flows:
        stock_flow_series = list(stock_flows)
    elif allow_flow_proxy_fallback:
        stock_flow_series = signed_flow_proxy_tail(prices_view, turnover_view, 20) or _tail(turnover_view, 20)
    else:
        stock_flow_series = []
    if sector_flows:
        sector_flow_series = list(sector_flows)
    elif allow_flow_proxy_fallback:
        sector_flow_series = stock_flow_series
    else:
        sector_flow_series = []
    return stock_flow_series, sector_flow_series


def _build_daily_market_inputs(
    *,
    benchmark_series: list[float],
    opens_1d: list[float] | None,
    highs_1d: list[float] | None,
    lows_1d: list[float] | None,
    sector_closes: list[float] | None,
    market_closes: list[float] | None,
    leader_prices: list[float] | None,
    laggard_prices: list[float] | None,
) -> dict[str, list[float]]:
    sector_close_series = list(sector_closes) if sector_closes is not None else []
    market_close_series = list(market_closes) if market_closes is not None else []
    return {
        "benchmark_prices": benchmark_series,
        "opens_1d": list(opens_1d) if opens_1d is not None else [],
        "highs_1d": list(highs_1d) if highs_1d is not None else [],
        "lows_1d": list(lows_1d) if lows_1d is not None else [],
        "sector_closes": sector_close_series,
        "market_closes": market_close_series,
        "leader_prices": list(leader_prices) if leader_prices is not None else [],
        "laggard_prices": list(laggard_prices) if laggard_prices is not None else [],
    }


def _build_intraday_market_inputs(
    *,
    prices_view: list[float],
    turnover_view: list[float],
    closes_30m: list[float] | None,
    opens_30m: list[float] | None,
    highs_30m: list[float] | None,
    lows_30m: list[float] | None,
    turnover_30m: list[float] | None,
    vwap_30m: list[float] | None,
) -> dict[str, list[float]]:
    closes_30m_series = list(closes_30m) if closes_30m is not None else []
    return {
        "closes_30m": closes_30m_series,
        "opens_30m": list(opens_30m) if opens_30m is not None else [],
        "highs_30m": list(highs_30m) if highs_30m is not None else [],
        "lows_30m": list(lows_30m) if lows_30m is not None else [],
        "turnover_30m": list(turnover_30m) if turnover_30m is not None else [],
        "vwap_30m": list(vwap_30m) if vwap_30m is not None else [],
    }
