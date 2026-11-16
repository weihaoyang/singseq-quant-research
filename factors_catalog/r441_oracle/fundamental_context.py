# ruff: noqa: BLE001
"""Frozen point-in-time fundamental factor oracle from commit 9267c326b."""

from __future__ import annotations

from bisect import bisect_left
from datetime import date
from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _present_float(row: dict[str, Any], key: str) -> float | None:
    """Return a finite reported value; missing fields remain unavailable."""
    if key not in row or row.get(key) is None or row.get(key) == "":
        return None
    try:
        value = float(row[key])
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def _next_trading_day_iso(value: str, trading_days: tuple[str, ...]) -> str:
    target = str(value or "")[:10]
    index = bisect_left(trading_days, target)
    if index < len(trading_days):
        return trading_days[index]
    raise ValueError(f"r441_financial_next_trading_day_missing:{target}")


def _financial_effective_day(
    row: dict[str, Any], trading_days: tuple[str, ...]
) -> str:
    raw_day = str(row.get("f_ann_date") or row.get("ann_date") or row.get("report_period") or "").strip()[:10]
    if not raw_day:
        return ""
    try:
        return _next_trading_day_iso(raw_day, trading_days)
    except Exception:
        return raw_day


def _quarter_value_by_period(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    ordered = sorted(
        (
            row for row in rows
            if str(row.get("report_period") or "").strip()
        ),
        key=lambda item: str(item.get("report_period") or "").strip(),
    )
    cumulative_by_period: dict[str, float] = {}
    for row in ordered:
        period = str(row.get("report_period") or "").strip()
        value = _present_float(row, key)
        if value is None:
            continue
        cumulative_by_period[period] = value
    quarter_by_period: dict[str, float] = {}
    for period, cumulative in cumulative_by_period.items():
        month = period[5:7]
        if month == "03":
            quarter_by_period[period] = cumulative
            continue
        prev_period = {
            "06": f"{period[:4]}-03-31",
            "09": f"{period[:4]}-06-30",
            "12": f"{period[:4]}-09-30",
        }.get(month, "")
        if prev_period not in cumulative_by_period:
            continue
        prev_value = cumulative_by_period[prev_period]
        quarter_by_period[period] = cumulative - prev_value
    return quarter_by_period


def _ttm_value_by_period(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    quarter_values = _quarter_value_by_period(rows, key)
    ordered_periods = sorted(quarter_values)
    ttm_by_period: dict[str, float] = {}
    for index, period in enumerate(ordered_periods):
        if index < 3:
            continue
        window = ordered_periods[index - 3 : index + 1]
        ttm_by_period[period] = sum(float(quarter_values.get(item, 0.0) or 0.0) for item in window)
    return ttm_by_period


def _yoy_value(ttm_by_period: dict[str, float], period: str) -> float | None:
    current = ttm_by_period.get(period)
    if current is None:
        return None
    prior = ttm_by_period.get(f"{int(period[:4]) - 1}{period[4:]}")
    if prior is None or abs(float(prior)) <= 1e-12:
        return None
    return float(current - prior) / abs(float(prior))


def _safe_ratio(num: float, den: float) -> float:
    return num / den if abs(float(den)) > 1e-12 else 0.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _freshness_decay(effective_day: str, current_day: str, half_life_days: float = 120.0) -> float:
    if not effective_day or not current_day:
        return 0.0
    try:
        age = max((date.fromisoformat(current_day) - date.fromisoformat(effective_day)).days, 0)
    except Exception:
        return 0.0
    return float(2.718281828459045 ** (-float(age) / max(float(half_life_days), 1.0)))


def _disclosure_lag_days(report_period: str, effective_day: str) -> int | None:
    if not report_period or not effective_day:
        return None
    try:
        return max((date.fromisoformat(effective_day) - date.fromisoformat(report_period)).days, 0)
    except Exception:
        return None


def load_symbol_fundamental_factor_context(
    data_repo: Any,
    *,
    symbols: list[str],
    start_day: str,
    end_day: str,
    trading_days: tuple[str, ...],
) -> dict[str, dict[str, dict[str, float]]]:
    fundamental_by_symbol: dict[str, dict[str, dict[str, float]]] = {}
    start_cut = str(start_day or "").strip()[:10]
    end_cut = str(end_day or "").strip()[:10]
    if hasattr(data_repo, "list_financial_rows_bulk"):
        income_rows_by_symbol = data_repo.list_financial_rows_bulk(symbols, "income", limit=24)
        balancesheet_rows_by_symbol = data_repo.list_financial_rows_bulk(symbols, "balancesheet", limit=24)
        cashflow_rows_by_symbol = data_repo.list_financial_rows_bulk(symbols, "cashflow", limit=24)
    else:
        income_rows_by_symbol = {}
        balancesheet_rows_by_symbol = {}
        cashflow_rows_by_symbol = {}
    for symbol in symbols:
        income_rows = income_rows_by_symbol.get(symbol) or data_repo.list_financial_rows(symbol, "income", limit=24)
        balancesheet_rows = balancesheet_rows_by_symbol.get(symbol) or data_repo.list_financial_rows(symbol, "balancesheet", limit=24)
        cashflow_rows = cashflow_rows_by_symbol.get(symbol) or data_repo.list_financial_rows(symbol, "cashflow", limit=24)
        if not income_rows and not balancesheet_rows and not cashflow_rows:
            continue
        revenue_ttm = _ttm_value_by_period(income_rows, "revenue")
        operate_profit_ttm = _ttm_value_by_period(income_rows, "operate_profit")
        net_income_parent_ttm = _ttm_value_by_period(income_rows, "net_income_parent")
        operating_cashflow_ttm = _ttm_value_by_period(cashflow_rows, "operating_cashflow")
        operating_cost_ttm = _ttm_value_by_period(income_rows, "oper_cost")
        ebitda_ttm = _ttm_value_by_period(income_rows, "ebitda")
        financing_cashflow_ttm = _ttm_value_by_period(cashflow_rows, "financing_cashflow")
        income_map = {str(row.get("report_period") or "").strip(): row for row in income_rows}
        balancesheet_map = {str(row.get("report_period") or "").strip(): row for row in balancesheet_rows}
        cashflow_map = {str(row.get("report_period") or "").strip(): row for row in cashflow_rows}
        period_candidates = sorted(set(income_map) | set(balancesheet_map) | set(cashflow_map))
        effective_rows: list[tuple[str, dict[str, float]]] = []
        disclosure_lag_by_period: dict[str, int] = {}
        net_leverage_by_period: dict[str, float] = {}
        for period in period_candidates:
            income_row = dict(income_map.get(period) or {})
            balance_row = dict(balancesheet_map.get(period) or {})
            cash_row = dict(cashflow_map.get(period) or {})
            effective_day = _financial_effective_day(
                income_row or balance_row or cash_row, trading_days
            )
            if not effective_day:
                continue
            lag_days = _disclosure_lag_days(period, effective_day)
            if lag_days is not None:
                disclosure_lag_by_period[period] = lag_days
            total_assets = float(balance_row.get("total_assets", 0.0) or 0.0)
            total_liabilities = float(balance_row.get("total_liabilities", 0.0) or 0.0)
            cash_balance = float(balance_row.get("cash", 0.0) or 0.0)
            if total_assets > 1e-12:
                net_leverage_by_period[period] = _safe_ratio(total_liabilities - cash_balance, total_assets)
        for period in period_candidates:
            income_row = dict(income_map.get(period) or {})
            balance_row = dict(balancesheet_map.get(period) or {})
            cash_row = dict(cashflow_map.get(period) or {})
            effective_day = _financial_effective_day(
                income_row or balance_row or cash_row, trading_days
            )
            if not effective_day:
                continue
            revenue_ttm_value = float(revenue_ttm.get(period, income_row.get("revenue", 0.0)) or 0.0)
            operate_profit_ttm_value = float(operate_profit_ttm.get(period, income_row.get("operate_profit", 0.0)) or 0.0)
            net_income_parent_ttm_value = float(net_income_parent_ttm.get(period, income_row.get("net_income_parent", 0.0)) or 0.0)
            operating_cashflow_ttm_value = float(operating_cashflow_ttm.get(period, cash_row.get("operating_cashflow", 0.0)) or 0.0)
            total_assets = float(balance_row.get("total_assets", 0.0) or 0.0)
            total_liabilities = float(balance_row.get("total_liabilities", 0.0) or 0.0)
            cash_balance = float(balance_row.get("cash", 0.0) or 0.0)
            accounts_receivable = float(balance_row.get("accounts_receivable", 0.0) or 0.0)
            inventories = float(balance_row.get("inventories", 0.0) or 0.0)
            fixed_assets = float(balance_row.get("fixed_assets", 0.0) or 0.0)
            current_assets = float(balance_row.get("current_assets", 0.0) or 0.0)
            current_liabilities = float(balance_row.get("current_liabilities", 0.0) or 0.0)
            retained_earnings = float(balance_row.get("retained_earnings", 0.0) or 0.0)
            operating_cost_ttm_value = float(operating_cost_ttm.get(period, income_row.get("oper_cost", 0.0)) or 0.0)
            ebitda_ttm_value = float(ebitda_ttm.get(period, income_row.get("ebitda", 0.0)) or 0.0)
            financing_cashflow_ttm_value = float(financing_cashflow_ttm.get(period, cash_row.get("financing_cashflow", 0.0)) or 0.0)
            revenue_yoy = _yoy_value(revenue_ttm, period)
            receivable_period = float(balance_row.get("accounts_receivable", 0.0) or 0.0)
            prior_receivable_row = balancesheet_map.get(f"{int(period[:4]) - 1}{period[4:]}")
            prior_receivable = float((prior_receivable_row or {}).get("accounts_receivable", 0.0) or 0.0)
            receivable_yoy = None if abs(prior_receivable) <= 1e-12 else _safe_ratio(receivable_period - prior_receivable, abs(prior_receivable))
            prior_inventory_row = balancesheet_map.get(f"{int(period[:4]) - 1}{period[4:]}")
            prior_inventory = float((prior_inventory_row or {}).get("inventories", 0.0) or 0.0)
            inventory_yoy = None if abs(prior_inventory) <= 1e-12 else _safe_ratio(inventories - prior_inventory, abs(prior_inventory))
            margin = _safe_ratio(operate_profit_ttm_value, revenue_ttm_value)
            gross_margin = _safe_ratio(revenue_ttm_value - operating_cost_ttm_value, revenue_ttm_value)
            prior_margin = None
            prior_gross_margin = None
            prior_period = f"{int(period[:4]) - 1}{period[4:]}"
            if prior_period in revenue_ttm and prior_period in operate_profit_ttm:
                prior_margin = _safe_ratio(float(operate_profit_ttm.get(prior_period, 0.0) or 0.0), float(revenue_ttm.get(prior_period, 0.0) or 0.0))
            if prior_period in revenue_ttm and prior_period in operating_cost_ttm:
                prior_revenue_for_margin = float(revenue_ttm.get(prior_period, 0.0) or 0.0)
                prior_operating_cost = float(operating_cost_ttm.get(prior_period, 0.0) or 0.0)
                prior_gross_margin = _safe_ratio(prior_revenue_for_margin - prior_operating_cost, prior_revenue_for_margin)
            report_lag_value = 0.0
            current_lag = disclosure_lag_by_period.get(period)
            if current_lag is not None:
                comparable_lags = [
                    float(value)
                    for candidate_period, value in disclosure_lag_by_period.items()
                    if candidate_period < period
                ][-4:]
                if comparable_lags:
                    report_lag_value = _safe_ratio((_mean(comparable_lags) - float(current_lag)), max(_mean(comparable_lags), 1.0))
            net_leverage_relief = 0.0
            prior_net_leverage = net_leverage_by_period.get(prior_period)
            current_net_leverage = net_leverage_by_period.get(period)
            if prior_net_leverage is not None and current_net_leverage is not None:
                net_leverage_relief = float(prior_net_leverage) - float(current_net_leverage)
            prior_cash_coverage = None
            current_cash_coverage = _safe_ratio(cash_balance, total_liabilities) if total_liabilities > 1e-12 else None
            prior_balance_row = dict(balancesheet_map.get(prior_period) or {})
            prior_total_liabilities = float(prior_balance_row.get("total_liabilities", 0.0) or 0.0)
            prior_cash_balance = float(prior_balance_row.get("cash", 0.0) or 0.0)
            if prior_total_liabilities > 1e-12:
                prior_cash_coverage = _safe_ratio(prior_cash_balance, prior_total_liabilities)
            cash_coverage_relief = (
                float(current_cash_coverage) - float(prior_cash_coverage)
                if current_cash_coverage is not None and prior_cash_coverage is not None
                else 0.0
            )
            ocf_margin = _safe_ratio(operating_cashflow_ttm_value, revenue_ttm_value)
            prior_ocf_margin = None
            if prior_period in revenue_ttm and prior_period in operating_cashflow_ttm:
                prior_ocf_margin = _safe_ratio(
                    float(operating_cashflow_ttm.get(prior_period, 0.0) or 0.0),
                    float(revenue_ttm.get(prior_period, 0.0) or 0.0),
                )
            prior_total_assets = float(prior_balance_row.get("total_assets", 0.0) or 0.0)
            prior_fixed_assets = float(prior_balance_row.get("fixed_assets", 0.0) or 0.0)
            prior_current_assets = float(prior_balance_row.get("current_assets", 0.0) or 0.0)
            prior_current_liabilities = float(prior_balance_row.get("current_liabilities", 0.0) or 0.0)
            prior_retained_earnings = float(prior_balance_row.get("retained_earnings", 0.0) or 0.0)
            prior_cash_row = dict(cashflow_map.get(prior_period) or {})
            prior_financing_cashflow_ttm_value = float(financing_cashflow_ttm.get(prior_period, prior_cash_row.get("financing_cashflow", 0.0)) or 0.0)
            gross_profitability = _safe_ratio(revenue_ttm_value - operating_cost_ttm_value, total_assets)
            operating_profitability = _safe_ratio(operate_profit_ttm_value, total_assets)
            cash_profitability = _safe_ratio(operating_cashflow_ttm_value, total_assets)
            prior_return_on_assets = 0.0
            if abs(prior_total_assets) > 1e-12:
                prior_return_on_assets = _safe_ratio(float(net_income_parent_ttm.get(prior_period, 0.0) or 0.0), prior_total_assets)
            return_on_assets = _safe_ratio(net_income_parent_ttm_value, total_assets)
            prior_operating_profitability = _safe_ratio(float(operate_profit_ttm.get(prior_period, 0.0) or 0.0), prior_total_assets) if abs(prior_total_assets) > 1e-12 else 0.0
            prior_cash_profitability = _safe_ratio(float(operating_cashflow_ttm.get(prior_period, 0.0) or 0.0), prior_total_assets) if abs(prior_total_assets) > 1e-12 else 0.0
            ebitda_asset_efficiency = _safe_ratio(ebitda_ttm_value, total_assets)
            prior_revenue_ttm_value = float(revenue_ttm.get(prior_period, 0.0) or 0.0)
            current_asset_turnover = _safe_ratio(revenue_ttm_value, total_assets)
            prior_asset_turnover = _safe_ratio(prior_revenue_ttm_value, prior_total_assets) if abs(prior_total_assets) > 1e-12 else 0.0
            asset_turnover_improvement = current_asset_turnover - prior_asset_turnover
            asset_growth_yoy = _safe_ratio(total_assets - prior_total_assets, abs(prior_total_assets)) if abs(prior_total_assets) > 1e-12 else 0.0
            fixed_asset_growth_yoy = _safe_ratio(fixed_assets - prior_fixed_assets, abs(prior_fixed_assets)) if abs(prior_fixed_assets) > 1e-12 else 0.0
            current_ratio = _safe_ratio(current_assets, current_liabilities) if current_liabilities > 1e-12 else 0.0
            prior_current_ratio = _safe_ratio(prior_current_assets, prior_current_liabilities) if prior_current_liabilities > 1e-12 else 0.0
            current_ratio_relief = current_ratio - prior_current_ratio
            retained_earnings_growth_yoy = _safe_ratio(retained_earnings - prior_retained_earnings, abs(prior_retained_earnings)) if abs(prior_retained_earnings) > 1e-12 else 0.0
            external_financing_reliance = _safe_ratio(financing_cashflow_ttm_value, total_assets)
            external_financing_relief = (
                _safe_ratio(prior_financing_cashflow_ttm_value, abs(prior_total_assets)) - external_financing_reliance
                if abs(prior_total_assets) > 1e-12
                else 0.0
            )
            operating_accruals_assets = _safe_ratio(net_income_parent_ttm_value - operating_cashflow_ttm_value, total_assets)
            operating_working_capital = current_assets - cash_balance - current_liabilities
            prior_operating_working_capital = prior_current_assets - prior_cash_balance - prior_current_liabilities
            working_capital_accruals = (
                _safe_ratio(operating_working_capital - prior_operating_working_capital, abs(prior_total_assets))
                if abs(prior_total_assets) > 1e-12
                else 0.0
            )
            net_operating_assets_ratio = _safe_ratio(total_assets - cash_balance - total_liabilities, total_assets)
            working_capital_to_assets = _safe_ratio(operating_working_capital, total_assets)
            retained_earnings_to_assets = _safe_ratio(retained_earnings, total_assets)
            liquidity_safety_buffer = (
                0.55 * _clamp(_safe_ratio(cash_balance, total_liabilities) if total_liabilities > 1e-12 else 0.0, -1.5, 1.5)
                + 0.45 * _clamp(current_ratio, -1.5, 1.5)
            )
            payload = {
                "cashflow_profit_match_ttm": _safe_ratio(operating_cashflow_ttm_value, abs(net_income_parent_ttm_value)) if abs(net_income_parent_ttm_value) > 1e-12 else 0.0,
                "receivables_revenue_gap_yoy": (float(revenue_yoy) - float(receivable_yoy)) if revenue_yoy is not None and receivable_yoy is not None else 0.0,
                "inventory_revenue_gap_yoy": (float(revenue_yoy) - float(inventory_yoy)) if revenue_yoy is not None and inventory_yoy is not None else 0.0,
                "operating_margin_reacceleration_ttm": (margin - float(prior_margin)) if prior_margin is not None else 0.0,
                "report_lag_shock_4q": report_lag_value,
                "net_leverage_relief_yoy": net_leverage_relief,
                "cash_coverage_relief_yoy": cash_coverage_relief,
                "operating_cashflow_margin_delta_yoy": (ocf_margin - float(prior_ocf_margin)) if prior_ocf_margin is not None else 0.0,
                "fundamental_freshness_decay_120d": 1.0,
                "fundamental_report_freshness_90d": 1.0,
                "fundamental_total_assets": total_assets,
                "fundamental_total_liabilities": total_liabilities,
                "fundamental_cash": cash_balance,
                "fundamental_accounts_receivable": accounts_receivable,
                "fundamental_revenue_ttm": revenue_ttm_value,
                "cash_to_assets_ttm_proxy_20": 0.0,
                "net_cash_to_assets_20": 0.0,
                "current_ratio_level_20": 0.0,
                "current_ratio_improvement_20": 0.0,
                "asset_growth_yoy_20": 0.0,
                "fixed_asset_growth_yoy_20": 0.0,
                "retained_earnings_growth_yoy_20": 0.0,
                "ebitda_to_assets_20": 0.0,
                "external_financing_reliance_20": 0.0,
                "gross_profitability_assets_20": 0.0,
                "gross_margin_level_20": 0.0,
                "gross_margin_improvement_20": 0.0,
                "conservative_investment_discipline_20": 0.0,
                "financing_discipline_safety_20": 0.0,
                "operating_profitability_assets_20": 0.0,
                "operating_profitability_improvement_20": 0.0,
                "cash_profitability_assets_20": 0.0,
                "cash_profitability_improvement_20": 0.0,
                "return_on_assets_level_20": 0.0,
                "return_on_assets_improvement_20": 0.0,
                "asset_turnover_level_20": 0.0,
                "operating_cashflow_to_liabilities_20": 0.0,
                "asset_turnover_improvement_20": 0.0,
                "net_external_financing_burden_20": 0.0,
                "liquidity_safety_buffer_20": 0.0,
                "working_capital_to_assets_20": 0.0,
                "retained_earnings_to_assets_20": 0.0,
                "operating_accruals_assets_20": 0.0,
                "working_capital_accruals_20": 0.0,
                "net_operating_assets_ratio_20": 0.0,
                "working_capital_discipline_20": 0.0,
                "beneish_anti_manipulation_20": 0.0,
                "piotroski_f_score_proxy_20": 0.0,
                "quality_strength_persistence_20": 0.0,
                "distress_resilience_public_20": 0.0,
                "franchise_power_quality_20": 0.0,
                "institutional_quality_compound_20": 0.0,
            }
            growth_quality_underreaction = (
                0.26 * _clamp(payload["receivables_revenue_gap_yoy"], -1.5, 1.5)
                + 0.22 * _clamp(payload["inventory_revenue_gap_yoy"], -1.5, 1.5)
                + 0.20 * _clamp(payload["operating_margin_reacceleration_ttm"], -0.6, 0.6)
                + 0.18 * _clamp(payload["report_lag_shock_4q"], -1.0, 1.0)
                + 0.14 * _clamp(payload["fundamental_freshness_decay_120d"], 0.0, 1.0)
            )
            cash_conversion_balance_sheet_momentum = (
                0.32 * _clamp(payload["cashflow_profit_match_ttm"] - 1.0, -2.0, 2.0)
                + 0.24 * _clamp(payload["operating_cashflow_margin_delta_yoy"], -0.6, 0.6)
                + 0.20 * _clamp(payload["net_leverage_relief_yoy"], -1.0, 1.0)
                + 0.14 * _clamp(payload["cash_coverage_relief_yoy"], -1.0, 1.0)
                + 0.10 * _clamp(payload["fundamental_freshness_decay_120d"], 0.0, 1.0)
            )
            payload["growth_quality_underreaction_20"] = growth_quality_underreaction
            payload["cash_conversion_balance_sheet_momentum_20"] = cash_conversion_balance_sheet_momentum
            payload["quality_balance_sheet_repair_20"] = (
                0.34 * _clamp(growth_quality_underreaction, -1.5, 1.5)
                + 0.34 * _clamp(cash_conversion_balance_sheet_momentum, -1.5, 1.5)
                + 0.12 * _clamp(payload["net_leverage_relief_yoy"], -1.0, 1.0)
                + 0.10 * _clamp(payload["cash_coverage_relief_yoy"], -1.0, 1.0)
                + 0.10 * _clamp(payload["fundamental_freshness_decay_120d"], 0.0, 1.0)
            )
            payload["accrual_cashflow_quality_20"] = (
                0.46 * _clamp(payload["cashflow_profit_match_ttm"] - 0.8, -2.0, 2.0)
                + 0.20 * _clamp(payload["receivables_revenue_gap_yoy"], -1.5, 1.5)
                + 0.18 * _clamp(payload["inventory_revenue_gap_yoy"], -1.5, 1.5)
                + 0.10 * _clamp(payload["operating_cashflow_margin_delta_yoy"], -0.6, 0.6)
                + 0.06 * _clamp(payload["fundamental_freshness_decay_120d"], 0.0, 1.0)
            )
            payload["post_report_quality_drift_20"] = (
                0.28 * _clamp(growth_quality_underreaction, -1.5, 1.5)
                + 0.24 * _clamp(cash_conversion_balance_sheet_momentum, -1.5, 1.5)
                + 0.20 * _clamp(payload["operating_margin_reacceleration_ttm"], -0.6, 0.6)
                + 0.14 * _clamp(payload["report_lag_shock_4q"], -1.0, 1.0)
                + 0.14 * _clamp(payload["fundamental_report_freshness_90d"], 0.0, 1.0)
            )
            payload["cash_to_assets_ttm_proxy_20"] = (
                0.74 * _clamp(_safe_ratio(cash_balance, total_assets) * 10.0, -1.5, 1.5)
                + 0.26 * _clamp(payload["fundamental_report_freshness_90d"], 0.0, 1.0)
            )
            payload["net_cash_to_assets_20"] = (
                0.62 * _clamp(_safe_ratio(cash_balance - total_liabilities, total_assets) * 10.0, -1.5, 1.5)
                + 0.22 * _clamp(payload["cash_coverage_relief_yoy"], -1.0, 1.0)
                + 0.16 * _clamp(payload["net_leverage_relief_yoy"], -1.0, 1.0)
            )
            payload["current_ratio_level_20"] = (
                0.66 * _clamp(current_ratio, -1.5, 1.5)
                + 0.20 * _clamp(current_ratio_relief, -1.5, 1.5)
                + 0.14 * _clamp(payload["fundamental_report_freshness_90d"], 0.0, 1.0)
            )
            payload["current_ratio_improvement_20"] = (
                0.82 * _clamp(current_ratio_relief, -1.5, 1.5)
                + 0.18 * _clamp(payload["fundamental_report_freshness_90d"], 0.0, 1.0)
            )
            payload["asset_growth_yoy_20"] = (
                0.72 * _clamp(-asset_growth_yoy, -1.5, 1.5)
                + 0.18 * _clamp(payload["receivables_revenue_gap_yoy"], -1.5, 1.5)
                + 0.10 * _clamp(payload["inventory_revenue_gap_yoy"], -1.5, 1.5)
            )
            payload["fixed_asset_growth_yoy_20"] = (
                0.74 * _clamp(-fixed_asset_growth_yoy, -1.5, 1.5)
                + 0.14 * _clamp(payload["receivables_revenue_gap_yoy"], -1.5, 1.5)
                + 0.12 * _clamp(payload["inventory_revenue_gap_yoy"], -1.5, 1.5)
            )
            payload["retained_earnings_growth_yoy_20"] = (
                0.58 * _clamp(retained_earnings_growth_yoy, -1.5, 1.5)
                + 0.22 * _clamp(payload["cashflow_profit_match_ttm"] - 0.8, -2.0, 2.0)
                + 0.20 * _clamp(payload["net_leverage_relief_yoy"], -1.0, 1.0)
            )
            payload["ebitda_to_assets_20"] = (
                0.70 * _clamp(ebitda_asset_efficiency * 10.0, -1.5, 1.5)
                + 0.18 * _clamp(asset_turnover_improvement * 10.0, -1.5, 1.5)
                + 0.12 * _clamp(payload["fundamental_report_freshness_90d"], 0.0, 1.0)
            )
            payload["external_financing_reliance_20"] = (
                0.70 * _clamp(-external_financing_reliance * 10.0, -1.5, 1.5)
                + 0.30 * _clamp(external_financing_relief * 10.0, -1.5, 1.5)
            )
            payload["gross_profitability_assets_20"] = (
                0.22 * _clamp(gross_profitability * 10.0, -1.5, 1.5)
                + 0.18 * _clamp(ebitda_asset_efficiency * 10.0, -1.5, 1.5)
                + 0.22 * _clamp(payload["cashflow_profit_match_ttm"] - 0.8, -2.0, 2.0)
                + 0.18 * _clamp(payload["operating_margin_reacceleration_ttm"], -0.6, 0.6)
                + 0.12 * _clamp(payload["receivables_revenue_gap_yoy"], -1.5, 1.5)
                + 0.08 * _clamp(payload["fundamental_report_freshness_90d"], 0.0, 1.0)
            )
            payload["gross_margin_level_20"] = (
                0.84 * _clamp(gross_margin, -1.5, 1.5)
                + 0.16 * _clamp(payload["fundamental_report_freshness_90d"], 0.0, 1.0)
            )
            payload["gross_margin_improvement_20"] = (
                0.78 * _clamp((gross_margin - float(prior_gross_margin)) if prior_gross_margin is not None else 0.0, -0.6, 0.6)
                + 0.14 * _clamp(payload["receivables_revenue_gap_yoy"], -1.5, 1.5)
                + 0.08 * _clamp(payload["inventory_revenue_gap_yoy"], -1.5, 1.5)
            )
            payload["conservative_investment_discipline_20"] = (
                0.32 * _clamp(-asset_growth_yoy, -1.5, 1.5)
                + 0.20 * _clamp(-fixed_asset_growth_yoy, -1.5, 1.5)
                + 0.18 * _clamp(payload["receivables_revenue_gap_yoy"], -1.5, 1.5)
                + 0.16 * _clamp(payload["inventory_revenue_gap_yoy"], -1.5, 1.5)
                + 0.14 * _clamp(retained_earnings_growth_yoy, -1.5, 1.5)
            )
            payload["financing_discipline_safety_20"] = (
                0.26 * _clamp(external_financing_relief * 10.0, -1.5, 1.5)
                + 0.22 * _clamp(-external_financing_reliance * 10.0, -1.5, 1.5)
                + 0.18 * _clamp(payload["cash_coverage_relief_yoy"], -1.0, 1.0)
                + 0.18 * _clamp(payload["net_leverage_relief_yoy"], -1.0, 1.0)
                + 0.16 * _clamp(current_ratio_relief, -1.5, 1.5)
            )
            payload["operating_profitability_assets_20"] = (
                0.52 * _clamp(operating_profitability * 10.0, -1.5, 1.5)
                + 0.20 * _clamp(payload["operating_margin_reacceleration_ttm"], -0.6, 0.6)
                + 0.16 * _clamp(asset_turnover_improvement * 10.0, -1.5, 1.5)
                + 0.12 * _clamp(payload["fundamental_report_freshness_90d"], 0.0, 1.0)
            )
            payload["operating_profitability_improvement_20"] = (
                0.72 * _clamp((operating_profitability - prior_operating_profitability) * 10.0, -1.5, 1.5)
                + 0.16 * _clamp(payload["operating_margin_reacceleration_ttm"], -0.6, 0.6)
                + 0.12 * _clamp(payload["fundamental_report_freshness_90d"], 0.0, 1.0)
            )
            payload["cash_profitability_assets_20"] = (
                0.42 * _clamp(cash_profitability * 10.0, -1.5, 1.5)
                + 0.26 * _clamp(payload["cashflow_profit_match_ttm"] - 0.8, -2.0, 2.0)
                + 0.18 * _clamp(payload["operating_cashflow_margin_delta_yoy"], -0.6, 0.6)
                + 0.14 * _clamp(payload["fundamental_report_freshness_90d"], 0.0, 1.0)
            )
            payload["cash_profitability_improvement_20"] = (
                0.70 * _clamp((cash_profitability - prior_cash_profitability) * 10.0, -1.5, 1.5)
                + 0.18 * _clamp(payload["operating_cashflow_margin_delta_yoy"], -0.6, 0.6)
                + 0.12 * _clamp(payload["fundamental_report_freshness_90d"], 0.0, 1.0)
            )
            payload["return_on_assets_level_20"] = (
                0.76 * _clamp(return_on_assets * 10.0, -1.5, 1.5)
                + 0.16 * _clamp(payload["cashflow_profit_match_ttm"] - 0.8, -2.0, 2.0)
                + 0.08 * _clamp(payload["fundamental_report_freshness_90d"], 0.0, 1.0)
            )
            payload["return_on_assets_improvement_20"] = (
                0.68 * _clamp((return_on_assets - prior_return_on_assets) * 10.0, -1.5, 1.5)
                + 0.18 * _clamp(payload["cashflow_profit_match_ttm"] - 0.8, -2.0, 2.0)
                + 0.14 * _clamp(payload["operating_cashflow_margin_delta_yoy"], -0.6, 0.6)
            )
            payload["asset_turnover_level_20"] = (
                0.70 * _clamp(current_asset_turnover * 5.0, -1.5, 1.5)
                + 0.18 * _clamp(payload["receivables_revenue_gap_yoy"], -1.5, 1.5)
                + 0.12 * _clamp(payload["inventory_revenue_gap_yoy"], -1.5, 1.5)
            )
            payload["operating_cashflow_to_liabilities_20"] = (
                0.74 * _clamp(_safe_ratio(operating_cashflow_ttm_value, total_liabilities) * 10.0, -1.5, 1.5)
                + 0.16 * _clamp(payload["cash_coverage_relief_yoy"], -1.0, 1.0)
                + 0.10 * _clamp(payload["fundamental_report_freshness_90d"], 0.0, 1.0)
            )
            payload["asset_turnover_improvement_20"] = (
                0.46 * _clamp(asset_turnover_improvement * 10.0, -1.5, 1.5)
                + 0.24 * _clamp(current_asset_turnover * 5.0, -1.5, 1.5)
                + 0.18 * _clamp(payload["receivables_revenue_gap_yoy"], -1.5, 1.5)
                + 0.12 * _clamp(payload["inventory_revenue_gap_yoy"], -1.5, 1.5)
            )
            payload["net_external_financing_burden_20"] = (
                0.58 * _clamp(-external_financing_reliance * 10.0, -1.5, 1.5)
                + 0.42 * _clamp(external_financing_relief * 10.0, -1.5, 1.5)
            )
            payload["liquidity_safety_buffer_20"] = (
                0.52 * _clamp(liquidity_safety_buffer, -1.5, 1.5)
                + 0.24 * _clamp(payload["cash_coverage_relief_yoy"], -1.0, 1.0)
                + 0.24 * _clamp(current_ratio_relief, -1.5, 1.5)
            )
            payload["working_capital_to_assets_20"] = (
                0.76 * _clamp(working_capital_to_assets * 10.0, -1.5, 1.5)
                + 0.24 * _clamp(current_ratio_relief, -1.5, 1.5)
            )
            payload["retained_earnings_to_assets_20"] = (
                0.74 * _clamp(retained_earnings_to_assets * 10.0, -1.5, 1.5)
                + 0.26 * _clamp(retained_earnings_growth_yoy, -1.5, 1.5)
            )
            payload["operating_accruals_assets_20"] = (
                0.62 * _clamp(-operating_accruals_assets * 10.0, -1.5, 1.5)
                + 0.22 * _clamp(payload["cashflow_profit_match_ttm"] - 0.8, -2.0, 2.0)
                + 0.16 * _clamp(payload["operating_cashflow_margin_delta_yoy"], -0.6, 0.6)
            )
            payload["working_capital_accruals_20"] = (
                0.52 * _clamp(-working_capital_accruals * 10.0, -1.5, 1.5)
                + 0.26 * _clamp(payload["receivables_revenue_gap_yoy"], -1.5, 1.5)
                + 0.22 * _clamp(payload["inventory_revenue_gap_yoy"], -1.5, 1.5)
            )
            payload["net_operating_assets_ratio_20"] = (
                0.68 * _clamp(-net_operating_assets_ratio * 10.0, -1.5, 1.5)
                + 0.18 * _clamp(payload["net_leverage_relief_yoy"], -1.0, 1.0)
                + 0.14 * _clamp(payload["cash_coverage_relief_yoy"], -1.0, 1.0)
            )
            payload["working_capital_discipline_20"] = (
                0.26 * _clamp(payload["working_capital_accruals_20"], -1.5, 1.5)
                + 0.20 * _clamp(payload["operating_accruals_assets_20"], -1.5, 1.5)
                + 0.20 * _clamp(payload["receivables_revenue_gap_yoy"], -1.5, 1.5)
                + 0.18 * _clamp(payload["inventory_revenue_gap_yoy"], -1.5, 1.5)
                + 0.16 * _clamp(payload["cashflow_profit_match_ttm"] - 0.8, -2.0, 2.0)
            )
            asset_quality_drift = (
                _safe_ratio(current_assets + fixed_assets, total_assets)
                - _safe_ratio(prior_current_assets + prior_fixed_assets, prior_total_assets)
                if abs(prior_total_assets) > 1e-12
                else 0.0
            )
            leverage_growth = (
                _safe_ratio(total_liabilities, total_assets) - _safe_ratio(prior_total_liabilities, prior_total_assets)
                if abs(prior_total_assets) > 1e-12
                else 0.0
            )
            payload["beneish_anti_manipulation_20"] = (
                0.24 * _clamp(payload["receivables_revenue_gap_yoy"], -1.5, 1.5)
                + 0.18 * _clamp(-abs((gross_margin - float(prior_gross_margin)) if prior_gross_margin is not None else 0.0) * 4.0, -1.5, 1.5)
                + 0.18 * _clamp(payload["operating_accruals_assets_20"], -1.5, 1.5)
                + 0.16 * _clamp(-asset_quality_drift * 10.0, -1.5, 1.5)
                + 0.12 * _clamp(-leverage_growth * 10.0, -1.5, 1.5)
                + 0.12 * _clamp(payload["cashflow_profit_match_ttm"] - 0.8, -2.0, 2.0)
            )
            piotroski_score_raw = 0.0
            piotroski_score_raw += 1.0 if return_on_assets > 0.0 else 0.0
            piotroski_score_raw += 1.0 if operating_cashflow_ttm_value > 0.0 else 0.0
            piotroski_score_raw += 1.0 if return_on_assets > prior_return_on_assets else 0.0
            piotroski_score_raw += 1.0 if operating_cashflow_ttm_value > net_income_parent_ttm_value else 0.0
            piotroski_score_raw += 1.0 if net_leverage_relief > 0.0 else 0.0
            piotroski_score_raw += 1.0 if current_ratio_relief > 0.0 else 0.0
            piotroski_score_raw += 1.0 if (external_financing_relief > 0.0 or external_financing_reliance <= 0.0) else 0.0
            piotroski_score_raw += 1.0 if (prior_gross_margin is not None and gross_margin > float(prior_gross_margin)) else 0.0
            piotroski_score_raw += 1.0 if asset_turnover_improvement > 0.0 else 0.0
            payload["piotroski_f_score_proxy_20"] = (
                0.84 * _clamp(((piotroski_score_raw / 9.0) - 0.5) * 2.0, -1.5, 1.5)
                + 0.16 * _clamp(payload["fundamental_report_freshness_90d"], 0.0, 1.0)
            )
            payload["quality_strength_persistence_20"] = (
                0.24 * _clamp(payload["piotroski_f_score_proxy_20"], -1.5, 1.5)
                + 0.18 * _clamp(payload["quality_balance_sheet_repair_20"], -1.5, 1.5)
                + 0.14 * _clamp(payload["accrual_cashflow_quality_20"], -1.5, 1.5)
                + 0.12 * _clamp(payload["gross_profitability_assets_20"], -1.5, 1.5)
                + 0.10 * _clamp(payload["return_on_assets_improvement_20"], -1.5, 1.5)
                + 0.10 * _clamp(payload["operating_profitability_assets_20"], -1.5, 1.5)
                + 0.08 * _clamp(payload["retained_earnings_to_assets_20"], -1.5, 1.5)
                + 0.08 * _clamp(payload["asset_turnover_improvement_20"], -1.5, 1.5)
                + 0.08 * _clamp(payload["operating_cashflow_to_liabilities_20"], -1.5, 1.5)
                + 0.08 * _clamp(payload["financing_discipline_safety_20"], -1.5, 1.5)
                + 0.08 * _clamp(payload["beneish_anti_manipulation_20"], -1.5, 1.5)
                + 0.06 * _clamp(payload["working_capital_discipline_20"], -1.5, 1.5)
            )
            payload["distress_resilience_public_20"] = (
                0.24 * _clamp(payload["liquidity_safety_buffer_20"], -1.5, 1.5)
                + 0.18 * _clamp(payload["net_cash_to_assets_20"], -1.5, 1.5)
                + 0.16 * _clamp(payload["operating_cashflow_to_liabilities_20"], -1.5, 1.5)
                + 0.14 * _clamp(payload["financing_discipline_safety_20"], -1.5, 1.5)
                + 0.10 * _clamp(payload["current_ratio_improvement_20"], -1.5, 1.5)
                + 0.10 * _clamp(payload["net_external_financing_burden_20"], -1.5, 1.5)
                + 0.08 * _clamp(payload["cash_coverage_relief_yoy"], -1.0, 1.0)
            )
            payload["franchise_power_quality_20"] = (
                0.18 * _clamp(payload["gross_profitability_assets_20"], -1.5, 1.5)
                + 0.16 * _clamp(payload["operating_profitability_assets_20"], -1.5, 1.5)
                + 0.14 * _clamp(payload["cash_profitability_assets_20"], -1.5, 1.5)
                + 0.12 * _clamp(payload["return_on_assets_improvement_20"], -1.5, 1.5)
                + 0.10 * _clamp(payload["gross_margin_improvement_20"], -1.5, 1.5)
                + 0.10 * _clamp(payload["asset_turnover_improvement_20"], -1.5, 1.5)
                + 0.10 * _clamp(payload["retained_earnings_to_assets_20"], -1.5, 1.5)
                + 0.10 * _clamp(payload["quality_strength_persistence_20"], -1.5, 1.5)
            )
            payload["institutional_quality_compound_20"] = (
                0.24 * _clamp(payload["franchise_power_quality_20"], -1.5, 1.5)
                + 0.22 * _clamp(payload["distress_resilience_public_20"], -1.5, 1.5)
                + 0.18 * _clamp(payload["quality_strength_persistence_20"], -1.5, 1.5)
                + 0.12 * _clamp(payload["quality_balance_sheet_repair_20"], -1.5, 1.5)
                + 0.10 * _clamp(payload["beneish_anti_manipulation_20"], -1.5, 1.5)
                + 0.08 * _clamp(payload["post_report_quality_drift_20"], -1.5, 1.5)
                + 0.06 * _clamp(payload["accrual_cashflow_quality_20"], -1.5, 1.5)
            )
            effective_rows.append((effective_day, payload))
        if not effective_rows:
            continue
        effective_rows.sort(key=lambda item: item[0])
        symbol_payload: dict[str, dict[str, float]] = {}
        current_payload: dict[str, float] | None = None
        current_effective_day = ""
        cursor = 0
        if start_cut and end_cut:
            current_day = date.fromisoformat(start_cut)
            end_day_value = date.fromisoformat(end_cut)
            while current_day <= end_day_value:
                day_key = current_day.isoformat()
                while cursor < len(effective_rows) and effective_rows[cursor][0] <= day_key:
                    current_payload = dict(effective_rows[cursor][1])
                    current_effective_day = str(effective_rows[cursor][0] or "")
                    current_payload["fundamental_freshness_decay_120d"] = _freshness_decay(current_effective_day, day_key, 120.0)
                    current_payload["fundamental_report_freshness_90d"] = _freshness_decay(current_effective_day, day_key, 90.0)
                    cursor += 1
                if current_payload is not None:
                    current_payload["fundamental_freshness_decay_120d"] = _freshness_decay(current_effective_day, day_key, 120.0)
                    current_payload["fundamental_report_freshness_90d"] = _freshness_decay(current_effective_day, day_key, 90.0)
                    symbol_payload[day_key] = dict(current_payload)
                current_day = current_day.fromordinal(current_day.toordinal() + 1)
        if symbol_payload:
            fundamental_by_symbol[str(symbol)] = symbol_payload
    return fundamental_by_symbol
