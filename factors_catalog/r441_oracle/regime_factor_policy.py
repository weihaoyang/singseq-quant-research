# ruff: noqa
from __future__ import annotations

from dataclasses import dataclass
from math import tanh
from typing import Any


@dataclass(frozen=True)
class RegimeFactorRule:
    factor: str
    direction: int
    weight: float


@dataclass(frozen=True)
class RegimeDecisionPolicy:
    overlay_floor: float
    support_floor: float
    threshold_shift_scale: float
    threshold_shift_min: float
    threshold_shift_max: float
    sizing_base: float
    sizing_slope: float
    sizing_min: float
    sizing_max: float


REGIME_FACTOR_POLICY: dict[str, tuple[RegimeFactorRule, ...]] = {
    "trend": (
        RegimeFactorRule("compression_ratio_20_60", -1, 0.10),
        RegimeFactorRule("flow_proxy_strength_10", 1, 0.07),
        RegimeFactorRule("alpha_20_beta_60", 1, 0.06),
        RegimeFactorRule("turnover_return_corr_20", 1, 0.05),
        RegimeFactorRule("momentum_efficiency_20", 1, 0.05),
    ),
    "cautious": (
        RegimeFactorRule("range_ratio_20", -1, 0.08),
        RegimeFactorRule("flow_proxy_strength_10", -1, 0.05),
        RegimeFactorRule("relative_strength_60", 1, 0.06),
        RegimeFactorRule("turnover_stability_20", 1, 0.05),
        RegimeFactorRule("downside_vol_20", -1, 0.05),
    ),
    "risk_off": (
        RegimeFactorRule("turnover_return_corr_20", -1, 0.10),
        RegimeFactorRule("turnover_stability_20", 1, 0.08),
        RegimeFactorRule("compression_ratio_20_60", -1, 0.06),
        RegimeFactorRule("relative_strength_60", 1, 0.06),
        RegimeFactorRule("turnover_zscore_20", -1, 0.05),
    ),
}

REGIME_DECISION_POLICY: dict[str, RegimeDecisionPolicy] = {
    "trend": RegimeDecisionPolicy(
        overlay_floor=-0.02,
        support_floor=0.14,
        threshold_shift_scale=0.35,
        threshold_shift_min=-0.03,
        threshold_shift_max=0.05,
        sizing_base=1.0,
        sizing_slope=1.2,
        sizing_min=0.75,
        sizing_max=1.35,
    ),
    "cautious": RegimeDecisionPolicy(
        overlay_floor=-0.08,
        support_floor=0.0,
        threshold_shift_scale=0.45,
        threshold_shift_min=-0.08,
        threshold_shift_max=0.03,
        sizing_base=0.82,
        sizing_slope=1.1,
        sizing_min=0.45,
        sizing_max=1.0,
    ),
    "risk_off": RegimeDecisionPolicy(
        overlay_floor=-0.02,
        support_floor=0.16,
        threshold_shift_scale=0.55,
        threshold_shift_min=-0.10,
        threshold_shift_max=0.02,
        sizing_base=0.55,
        sizing_slope=0.9,
        sizing_min=0.2,
        sizing_max=0.8,
    ),
}

REGIME_DECISION_POLICY_PRESETS: dict[str, dict[str, RegimeDecisionPolicy]] = {
    "research_v1": REGIME_DECISION_POLICY,
    "trend_capture": {
        "trend": RegimeDecisionPolicy(
            overlay_floor=-0.03,
            support_floor=0.11,
            threshold_shift_scale=0.45,
            threshold_shift_min=-0.04,
            threshold_shift_max=0.06,
            sizing_base=1.06,
            sizing_slope=1.35,
            sizing_min=0.82,
            sizing_max=1.45,
        ),
        "cautious": RegimeDecisionPolicy(
            overlay_floor=0.0,
            support_floor=0.18,
            threshold_shift_scale=0.42,
            threshold_shift_min=-0.07,
            threshold_shift_max=0.03,
            sizing_base=0.86,
            sizing_slope=1.05,
            sizing_min=0.50,
            sizing_max=1.02,
        ),
        "risk_off": REGIME_DECISION_POLICY["risk_off"],
    },
    "defensive_v1": {
        "trend": RegimeDecisionPolicy(
            overlay_floor=-0.01,
            support_floor=0.16,
            threshold_shift_scale=0.30,
            threshold_shift_min=-0.02,
            threshold_shift_max=0.04,
            sizing_base=0.96,
            sizing_slope=1.0,
            sizing_min=0.70,
            sizing_max=1.22,
        ),
        "cautious": RegimeDecisionPolicy(
            overlay_floor=0.015,
            support_floor=0.24,
            threshold_shift_scale=0.50,
            threshold_shift_min=-0.10,
            threshold_shift_max=0.02,
            sizing_base=0.74,
            sizing_slope=0.95,
            sizing_min=0.38,
            sizing_max=0.88,
        ),
        "risk_off": RegimeDecisionPolicy(
            overlay_floor=0.03,
            support_floor=0.34,
            threshold_shift_scale=0.60,
            threshold_shift_min=-0.12,
            threshold_shift_max=0.01,
            sizing_base=0.42,
            sizing_slope=0.72,
            sizing_min=0.15,
            sizing_max=0.62,
        ),
    },
    "liquidity_focus": {
        "trend": RegimeDecisionPolicy(
            overlay_floor=-0.035,
            support_floor=0.10,
            threshold_shift_scale=0.48,
            threshold_shift_min=-0.05,
            threshold_shift_max=0.07,
            sizing_base=1.10,
            sizing_slope=1.45,
            sizing_min=0.86,
            sizing_max=1.50,
        ),
        "cautious": RegimeDecisionPolicy(
            overlay_floor=-0.005,
            support_floor=0.16,
            threshold_shift_scale=0.40,
            threshold_shift_min=-0.06,
            threshold_shift_max=0.04,
            sizing_base=0.90,
            sizing_slope=1.00,
            sizing_min=0.52,
            sizing_max=1.08,
        ),
        "risk_off": RegimeDecisionPolicy(
            overlay_floor=0.01,
            support_floor=0.22,
            threshold_shift_scale=0.45,
            threshold_shift_min=-0.08,
            threshold_shift_max=0.03,
            sizing_base=0.60,
            sizing_slope=0.90,
            sizing_min=0.24,
            sizing_max=0.82,
        ),
    },
}

_NORMALIZATION_SCALE_BY_FACTOR: dict[str, float] = {
    "compression_ratio_20_60": 0.25,
    "volatility_compression_20": 0.25,
    "flow_proxy_strength_10": 0.015,
    "stock_net_inflow_5_20": 0.40,
    "alpha_20_beta_60": 0.20,
    "turnover_return_corr_20": 0.50,
    "flow_correlation_10": 0.50,
    "range_ratio_20": 0.04,
    "realized_volatility_20": 0.03,
    "drawdown_from_high_20": 0.12,
    "downside_vol_20": 0.03,
    "downside_volatility_20": 0.03,
    "beta_60": 2.0,
    "turnover_zscore_20": 2.5,
    "turnover_stability_20": 0.80,
    "relative_strength_60": 0.15,
    "momentum_efficiency_20": 0.60,
}

_POLICY_FACTOR_ALIASES: dict[str, tuple[tuple[str, float], ...]] = {
    "compression_ratio_20_60": (("volatility_compression_20", -1.0),),
    "flow_proxy_strength_10": (("stock_net_inflow_5_20", 1.0),),
    "alpha_20_beta_60": (("relative_strength_60", 1.0),),
    "turnover_return_corr_20": (("flow_correlation_10", 1.0),),
    "range_ratio_20": (("realized_volatility_20", 1.0),),
    "downside_vol_20": (("downside_volatility_20", 1.0),),
    "realized_vol_20": (("realized_volatility_20", 1.0),),
    "momentum_20": (("price_momentum_20", 1.0),),
    "momentum_60": (("price_momentum_60", 1.0),),
    "price_vs_ma_20": (("ma_gap_20", 1.0),),
    "price_vs_ma_60": (("ma_gap_60", 1.0),),
}

REGIME_FACTOR_POLICY_PRESETS: dict[str, dict[str, tuple[RegimeFactorRule, ...]]] = {
    "research_v1": REGIME_FACTOR_POLICY,
    "trend_capture": {
        "trend": (
            RegimeFactorRule("compression_ratio_20_60", -1, 0.12),
            RegimeFactorRule("flow_proxy_strength_10", 1, 0.08),
            RegimeFactorRule("alpha_20_beta_60", 1, 0.08),
            RegimeFactorRule("turnover_return_corr_20", 1, 0.06),
            RegimeFactorRule("momentum_efficiency_20", 1, 0.06),
        ),
        "cautious": REGIME_FACTOR_POLICY["cautious"],
        "risk_off": REGIME_FACTOR_POLICY["risk_off"],
    },
    "defensive_v1": {
        "trend": REGIME_FACTOR_POLICY["trend"],
        "cautious": (
            RegimeFactorRule("range_ratio_20", -1, 0.10),
            RegimeFactorRule("drawdown_from_high_20", 1, 0.10),
            RegimeFactorRule("downside_vol_20", -1, 0.07),
            RegimeFactorRule("beta_60", -1, 0.05),
            RegimeFactorRule("flow_proxy_strength_10", -1, 0.05),
        ),
        "risk_off": (
            RegimeFactorRule("turnover_return_corr_20", -1, 0.12),
            RegimeFactorRule("turnover_stability_20", 1, 0.10),
            RegimeFactorRule("relative_strength_60", 1, 0.08),
            RegimeFactorRule("turnover_zscore_20", -1, 0.06),
            RegimeFactorRule("compression_ratio_20_60", -1, 0.05),
        ),
    },
    "liquidity_focus": {
        "trend": (
            RegimeFactorRule("flow_proxy_strength_10", 1, 0.10),
            RegimeFactorRule("turnover_return_corr_20", 1, 0.08),
            RegimeFactorRule("alpha_20_beta_60", 1, 0.06),
            RegimeFactorRule("momentum_efficiency_20", 1, 0.05),
        ),
        "cautious": (
            RegimeFactorRule("flow_proxy_strength_10", -1, 0.08),
            RegimeFactorRule("beta_60", -1, 0.06),
            RegimeFactorRule("range_ratio_20", -1, 0.05),
            RegimeFactorRule("drawdown_from_high_20", 1, 0.05),
        ),
        "risk_off": (
            RegimeFactorRule("turnover_return_corr_20", -1, 0.12),
            RegimeFactorRule("turnover_stability_20", 1, 0.10),
            RegimeFactorRule("turnover_zscore_20", -1, 0.08),
            RegimeFactorRule("relative_strength_60", 1, 0.05),
        ),
    },
}


def _policy_to_jsonable(policy: dict[str, tuple[RegimeFactorRule, ...]]) -> dict[str, list[dict[str, Any]]]:
    return {
        regime: [{"factor": rule.factor, "direction": int(rule.direction), "weight": float(rule.weight)} for rule in rules]
        for regime, rules in policy.items()
    }


def _decision_policy_to_jsonable(policy: dict[str, RegimeDecisionPolicy]) -> dict[str, dict[str, float]]:
    return {
        regime: {
            "overlay_floor": float(rule.overlay_floor),
            "support_floor": float(rule.support_floor),
            "threshold_shift_scale": float(rule.threshold_shift_scale),
            "threshold_shift_min": float(rule.threshold_shift_min),
            "threshold_shift_max": float(rule.threshold_shift_max),
            "sizing_base": float(rule.sizing_base),
            "sizing_slope": float(rule.sizing_slope),
            "sizing_min": float(rule.sizing_min),
            "sizing_max": float(rule.sizing_max),
        }
        for regime, rule in policy.items()
    }


def available_policy_presets() -> list[str]:
    return sorted(REGIME_FACTOR_POLICY_PRESETS)


def resolve_policy_preset(name: str | None) -> dict[str, list[dict[str, Any]]]:
    normalized = str(name or "research_v1").strip().lower()
    policy = REGIME_FACTOR_POLICY_PRESETS.get(normalized) or REGIME_FACTOR_POLICY_PRESETS["research_v1"]
    return _policy_to_jsonable(policy)


def resolve_decision_policy_preset(name: str | None) -> dict[str, dict[str, float]]:
    normalized = str(name or "research_v1").strip().lower()
    policy = REGIME_DECISION_POLICY_PRESETS.get(normalized) or REGIME_DECISION_POLICY_PRESETS["research_v1"]
    return _decision_policy_to_jsonable(policy)


def normalize_policy_overrides(raw: dict[str, list[dict[str, Any]]] | None) -> dict[str, tuple[RegimeFactorRule, ...]]:
    if not isinstance(raw, dict):
        return {}
    result: dict[str, tuple[RegimeFactorRule, ...]] = {}
    for regime, rows in raw.items():
        normalized_rules: list[RegimeFactorRule] = []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            factor = str(row.get("factor", "")).strip()
            if not factor:
                continue
            direction_raw = int(row.get("direction", 1) or 1)
            direction = 1 if direction_raw >= 0 else -1
            weight = float(row.get("weight", 0.0) or 0.0)
            normalized_rules.append(RegimeFactorRule(factor=factor, direction=direction, weight=weight))
        result[str(regime).strip().lower()] = tuple(normalized_rules)
    return result


def normalize_decision_policy_overrides(raw: dict[str, dict[str, Any]] | None) -> dict[str, RegimeDecisionPolicy]:
    if not isinstance(raw, dict):
        return {}
    result: dict[str, RegimeDecisionPolicy] = {}
    for regime, row in raw.items():
        if not isinstance(row, dict):
            continue
        normalized_regime = str(regime).strip().lower()
        overlay_floor = float(row.get("overlay_floor", 0.0) or 0.0)
        support_floor = float(row.get("support_floor", 0.0) or 0.0)
        if normalized_regime == "cautious":
            overlay_floor = min(overlay_floor, -0.04)
            support_floor = min(support_floor, 0.08)
        elif normalized_regime == "risk_off":
            overlay_floor = min(overlay_floor, -0.02)
            support_floor = min(support_floor, 0.18)
        result[normalized_regime] = RegimeDecisionPolicy(
            overlay_floor=overlay_floor,
            support_floor=support_floor,
            threshold_shift_scale=float(row.get("threshold_shift_scale", 0.0) or 0.0),
            threshold_shift_min=float(row.get("threshold_shift_min", 0.0) or 0.0),
            threshold_shift_max=float(row.get("threshold_shift_max", 0.0) or 0.0),
            sizing_base=float(row.get("sizing_base", 1.0) or 1.0),
            sizing_slope=float(row.get("sizing_slope", 0.0) or 0.0),
            sizing_min=float(row.get("sizing_min", 0.0) or 0.0),
            sizing_max=float(row.get("sizing_max", 1.5) or 1.5),
        )
    return result


def regime_factor_rules(
    regime: str | None,
    overrides: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[RegimeFactorRule, ...]:
    normalized = str(regime or "").strip().lower()
    override_rules = normalize_policy_overrides(overrides).get(normalized)
    if override_rules is not None:
        return override_rules
    return REGIME_FACTOR_POLICY.get(normalized, ())


def regime_decision_policy(
    regime: str | None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> RegimeDecisionPolicy | None:
    normalized = str(regime or "").strip().lower()
    override_policy = normalize_decision_policy_overrides(overrides).get(normalized)
    if override_policy is not None:
        return override_policy
    return REGIME_DECISION_POLICY.get(normalized)


def normalize_policy_factor_value(factor: str, value: float) -> float:
    scale = float(_NORMALIZATION_SCALE_BY_FACTOR.get(str(factor), 1.0) or 1.0)
    bounded = tanh(float(value) / scale)
    return round(max(-1.0, min(1.0, bounded)), 6)


def resolve_policy_factor_value(factors: dict[str, float], factor: str) -> tuple[str, float]:
    normalized_factor = str(factor or "").strip()
    if normalized_factor in factors:
        return normalized_factor, float(factors.get(normalized_factor, 0.0))
    for alias_name, multiplier in _POLICY_FACTOR_ALIASES.get(normalized_factor, ()):
        if alias_name in factors:
            return alias_name, float(factors.get(alias_name, 0.0)) * float(multiplier)
    return normalized_factor, 0.0


def calc_regime_overlay_score(
    factors: dict[str, float],
    regime: str | None,
    overrides: dict[str, list[dict[str, Any]]] | None = None,
) -> float:
    rules = regime_factor_rules(regime, overrides)
    if not rules:
        return 0.0
    total = 0.0
    for rule in rules:
        resolved_factor, resolved_value = resolve_policy_factor_value(factors, rule.factor)
        normalized = normalize_policy_factor_value(resolved_factor, resolved_value)
        total += normalized * float(rule.direction) * float(rule.weight)
    return round(total, 6)


def calc_regime_support_score(
    factors: dict[str, float],
    regime: str | None,
    overrides: dict[str, list[dict[str, Any]]] | None = None,
) -> float:
    rules = regime_factor_rules(regime, overrides)
    total_weight = sum(abs(float(rule.weight)) for rule in rules)
    if total_weight <= 1e-12:
        return 0.0
    support = 0.0
    for rule in rules:
        resolved_factor, resolved_value = resolve_policy_factor_value(factors, rule.factor)
        normalized = normalize_policy_factor_value(resolved_factor, resolved_value)
        aligned = normalized * float(rule.direction)
        support += max(aligned, 0.0) * abs(float(rule.weight))
    return round(max(0.0, min(1.0, support / total_weight)), 6)


def calc_regime_decision_adjustments(
    factors: dict[str, float],
    regime: str | None,
    *,
    overrides: dict[str, list[dict[str, Any]]] | None = None,
    decision_overrides: dict[str, dict[str, Any]] | None = None,
    overlay_scale: float = 1.0,
) -> dict[str, float | bool]:
    normalized = str(regime or "").strip().lower()
    raw_overlay = calc_regime_overlay_score(factors, normalized, overrides=overrides)
    support_score = calc_regime_support_score(factors, normalized, overrides=overrides)
    effective_overlay = float(raw_overlay) * float(overlay_scale)
    policy = regime_decision_policy(normalized, overrides=decision_overrides)
    if policy is None:
        return {
            "raw_overlay_score": round(raw_overlay, 6),
            "effective_overlay_score": round(effective_overlay, 6),
            "support_score": round(support_score, 6),
            "overlay_floor": 0.0,
            "support_floor": 0.0,
            "entry_allowed": True,
            "threshold_bias_shift": 0.0,
            "position_scale": 1.0,
        }
    threshold_shift = max(
        policy.threshold_shift_min,
        min(policy.threshold_shift_max, effective_overlay * policy.threshold_shift_scale),
    )
    overlay_floor = float(policy.overlay_floor)
    support_floor = float(policy.support_floor)
    overlay_gap = effective_overlay - overlay_floor
    support_gap = support_score - support_floor
    if normalized == "risk_off":
        entry_allowed = (
            (overlay_gap >= 0.0 and support_gap >= -0.04)
            or (support_gap >= 0.0 and effective_overlay >= overlay_floor - 0.03)
            or (effective_overlay >= 0.03 and support_score >= max(support_floor - 0.08, 0.0))
        )
    elif normalized == "cautious":
        entry_allowed = (
            (overlay_gap >= 0.0 and support_gap >= -0.03)
            or (support_gap >= 0.0 and effective_overlay >= overlay_floor - 0.03)
            or (overlay_gap >= -0.015 and support_gap >= -0.06)
        )
    else:
        entry_allowed = overlay_gap >= 0.0 and support_gap >= 0.0
    regime_signal = 0.55 * effective_overlay + 0.45 * (support_score - support_floor)
    compressed_signal = tanh(regime_signal * (1.55 if normalized == "risk_off" else 1.30 if normalized == "cautious" else 1.05))
    sizing_multiplier = 0.80 if normalized == "risk_off" else 0.90 if normalized == "cautious" else 1.0
    position_scale = max(
        policy.sizing_min,
        min(
            policy.sizing_max,
            policy.sizing_base + compressed_signal * policy.sizing_slope * sizing_multiplier,
        ),
    )
    return {
        "raw_overlay_score": round(raw_overlay, 6),
        "effective_overlay_score": round(effective_overlay, 6),
        "support_score": round(support_score, 6),
        "overlay_floor": overlay_floor,
        "support_floor": support_floor,
        "entry_allowed": entry_allowed,
        "threshold_bias_shift": round(threshold_shift, 6),
        "position_scale": round(position_scale, 6),
    }
