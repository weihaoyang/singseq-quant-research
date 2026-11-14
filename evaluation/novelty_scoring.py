"""Novelty-weighted competition scoring (versioned).

Quality stays the gate: a high-novelty factor with no predictive power must not
score.  Above the gate, the score is multiplied by a novelty premium so the
competition rewards factors that are *different from the crowded baseline*
without rewarding noise.

    quality  = geometric_mean(ic, icir, oos_retention, monotonicity)   in [0,1]
    final    = quality * (w0 + (1 - w0) * novelty)   if the gate passes, else 0

``w0`` is the "quality floor weight": ``w0=1`` ignores novelty (legacy), and the
premium is at most ``1/w0``.  The version string and the zoo version are part of
the score, so a published ranking is reproducible forever.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

__all__ = ["SCORING_VERSION", "ScoringConfig", "quality_score", "passes_quality_gate", "final_score"]

#: v2 adds the optional DSR component and gate.  ``novelty-score-1`` names the
#: pre-DSR formula and must never be re-used for results produced with the extra
#: component: the same version string used to describe two different formulas.
SCORING_VERSION = "novelty-score-2"


@dataclass(frozen=True)
class ScoringConfig:
    """Frozen scoring parameters (part of the reproducible contract)."""

    w0: float = 0.6                 # quality-floor weight; novelty premium in [w0, 1]
    min_abs_ic: float = 0.02        # |mean IC| floor
    min_icir: float = 0.20          # ICIR floor
    min_oos_retention: float = 0.40  # OOS IC / in-sample IC floor
    min_monotonicity: float = 0.60  # fraction of monotone group means
    min_dsr: float = 0.0            # Deflated Sharpe floor (0 = off; 0.95 = strict)
    ic_ref: float = 0.05            # scale for the IC component
    icir_ref: float = 0.50          # scale for the ICIR component

    def as_dict(self) -> Dict[str, Any]:
        return {"version": SCORING_VERSION, **self.__dict__}


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(min(1.0, max(0.0, x)))


def _geom_mean(parts: Dict[str, float]) -> float:
    values = [max(0.0, v) for v in parts.values()]
    if any(v == 0.0 for v in values):
        return 0.0
    return float(np.exp(np.mean(np.log(values))))


def quality_score(metrics: Dict[str, Any], cfg: ScoringConfig = ScoringConfig()) -> Dict[str, Any]:
    """Quality components in [0,1] and their geometric mean."""
    ic = _clip01(abs(float(metrics.get("ic_mean") or 0.0)) / cfg.ic_ref)
    icir = _clip01(abs(float(metrics.get("icir") or 0.0)) / cfg.icir_ref)
    retention = _clip01(float(metrics.get("oos_retention", 1.0)))
    monotonic = _clip01(float(metrics.get("monotonicity", 0.0)))
    parts = {"ic": ic, "icir": icir, "oos_retention": retention, "monotonicity": monotonic}
    dsr = metrics.get("dsr")
    if dsr is not None:
        # Only include the deflated-Sharpe component when it was actually computed,
        # so a scoring run without it keeps the previous four-component meaning.
        parts["dsr"] = _clip01(float(dsr))
    return {"components": parts, "quality": _geom_mean(parts)}


def passes_quality_gate(metrics: Dict[str, Any], cfg: ScoringConfig = ScoringConfig()) -> bool:
    dsr = metrics.get("dsr")
    return (
        abs(float(metrics.get("ic_mean") or 0.0)) >= cfg.min_abs_ic
        and abs(float(metrics.get("icir") or 0.0)) >= cfg.min_icir
        and float(metrics.get("oos_retention", 1.0)) >= cfg.min_oos_retention
        and float(metrics.get("monotonicity", 0.0)) >= cfg.min_monotonicity
        and (dsr is None or float(dsr) >= cfg.min_dsr)
    )


def final_score(metrics: Dict[str, Any], novelty: float, *,
                cfg: ScoringConfig = ScoringConfig(),
                zoo_version: Optional[str] = None) -> Dict[str, Any]:
    """Compute the novelty-weighted final score with a full audit trail."""
    quality = quality_score(metrics, cfg)
    gate = passes_quality_gate(metrics, cfg)
    novelty = _clip01(float(novelty))
    premium = cfg.w0 + (1.0 - cfg.w0) * novelty
    final = quality["quality"] * premium if gate else 0.0
    return {
        "scoring_version": SCORING_VERSION,
        "zoo_version": zoo_version,
        "config": cfg.as_dict(),
        "gate_passed": gate,
        "quality": round(quality["quality"], 6),
        "quality_components": {k: round(v, 6) for k, v in quality["components"].items()},
        "novelty": round(novelty, 6),
        "novelty_premium": round(premium, 6),
        "final_score": round(final, 6),
    }
