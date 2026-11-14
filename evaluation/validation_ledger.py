"""Validation ledger, holdout registry and a fail-closed promotion gate.

Why
---
The overfitting maths already live in :mod:`evaluation.overfit` (Deflated Sharpe,
PBO/CSCV, multiple-testing haircut).  What was missing to make a mined factor's
claim *auditable* is the **evidence contract**:

* an honest trial count from a ledger (so ``n_trials`` is measured, not guessed);
* an **untouched** holdout that was never used to select the factor;
* cost sensitivity at more than one rate;
* multiple-seed stability.

This module repays the framework half of TECH_DEBT ``FACTOR-HOLDOUT-004``.  It
does **not** run research and never invents numbers: the actual held-out research
runs remain a research activity, while this guarantees a promotion cannot be
claimed without the evidence being present, recorded and internally consistent.

Everything is dependency-light (stdlib only), append-only on disk, and pure so
it can be unit-tested without a database.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "VALIDATION_VERSION",
    "TrialLedger",
    "HoldoutRegistry",
    "multiple_seed_summary",
    "validation_gate",
]

VALIDATION_VERSION = "validation-1"

DEFAULT_DSR_MIN = 0.95
DEFAULT_PBO_MAX = 0.20
DEFAULT_MIN_SEEDS = 3
DEFAULT_MIN_COST_RATES = 2
DEFAULT_SEED_DISPERSION_MAX = 0.50


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# Trial ledger — every factor trial is recorded, so n_trials is measured.
# ---------------------------------------------------------------------------
class TrialLedger:
    """Append-only record of factor trials for multiple-testing control.

    A trial is one (factor, window, config) evaluation.  Re-recording the same
    ``trial_id`` is idempotent so a resumed run does not inflate ``n_trials``
    (which would silently over-penalise the Deflated Sharpe Ratio).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def record(self, *, trial_id: str, factor: str, window: dict[str, str] | None = None,
               metrics: dict[str, Any] | None = None, seed: int | None = None,
               note: str = "") -> dict[str, Any]:
        if not trial_id:
            raise ValueError("trial_id is required")
        existing = {str(r.get("trial_id")) for r in self.records()}
        if trial_id in existing:
            return {"recorded": False, "reason": "duplicate_trial_id", "trial_id": trial_id}
        row = {
            "trial_id": str(trial_id),
            "factor": str(factor),
            "window": dict(window or {}),
            "metrics": dict(metrics or {}),
            "seed": seed,
            "note": str(note),
            "recorded_at": _now(),
        }
        _append_jsonl(self.path, row)
        return {"recorded": True, "trial_id": trial_id}

    def records(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.path)

    def n_trials(self, *, factor: str | None = None) -> int:
        rows = self.records()
        if factor is not None:
            rows = [r for r in rows if str(r.get("factor")) == str(factor)]
        return len({str(r.get("trial_id")) for r in rows})

    def distinct_factors(self) -> list[str]:
        return sorted({str(r.get("factor")) for r in self.records()})


# ---------------------------------------------------------------------------
# Holdout registry — an untouched window stays untouched, or it is recorded.
# ---------------------------------------------------------------------------
class HoldoutRegistry:
    """Declare holdout windows and fail closed if one was already exposed.

    A window is "untouched" until :meth:`touch` records who used it and why.
    ``assert_untouched`` is the gate: a factor selected on a window that was
    already used for selection cannot be validated on it again.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def declare(self, *, name: str, start: str, end: str, note: str = "") -> dict[str, Any]:
        if not name or not start or not end:
            raise ValueError("name/start/end are required")
        for row in self._rows():
            if row.get("event") == "declare" and str(row.get("name")) == name:
                return {"declared": False, "reason": "already_declared", "name": name}
        _append_jsonl(self.path, {"event": "declare", "name": name, "start": start,
                                  "end": end, "note": note, "at": _now()})
        return {"declared": True, "name": name}

    def touch(self, *, name: str, actor: str, reason: str) -> dict[str, Any]:
        if not name or not actor:
            raise ValueError("name/actor are required")
        _append_jsonl(self.path, {"event": "touch", "name": name, "actor": actor,
                                  "reason": reason, "at": _now()})
        return {"touched": True, "name": name}

    def _rows(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.path)

    def status(self, name: str) -> dict[str, Any]:
        declared = next((r for r in self._rows()
                         if r.get("event") == "declare" and str(r.get("name")) == name), None)
        touches = [r for r in self._rows()
                   if r.get("event") == "touch" and str(r.get("name")) == name]
        return {
            "name": name,
            "declared": declared is not None,
            "window": ({"start": declared.get("start"), "end": declared.get("end")}
                       if declared else None),
            "exposed": bool(touches),
            "touches": touches,
        }

    def assert_untouched(self, name: str) -> dict[str, Any]:
        status = self.status(name)
        if not status["declared"]:
            raise ValueError(f"holdout not declared: {name}")
        if status["exposed"]:
            raise ValueError(f"holdout already exposed: {name} "
                             f"({len(status['touches'])} touch point(s))")
        return status


# ---------------------------------------------------------------------------
# Multiple-seed stability + the combined fail-closed gate.
# ---------------------------------------------------------------------------
def multiple_seed_summary(seed_metrics: Sequence[dict[str, Any]], *,
                          key: str = "rank_ic", min_seeds: int = DEFAULT_MIN_SEEDS,
                          max_dispersion: float = DEFAULT_SEED_DISPERSION_MAX) -> dict[str, Any]:
    """Aggregate one metric across seeds and test sign stability.

    ``dispersion`` is the coefficient of variation of the absolute values
    (std/|mean|).  A factor whose metric flips sign across seeds, or whose
    dispersion is large, is not stable.
    """
    values: list[float] = []
    for row in seed_metrics:
        raw = row.get(key) if isinstance(row, dict) else None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    n = len(values)
    if n < int(min_seeds):
        return {"status": "BLOCKED", "reason": f"need >= {min_seeds} finite seeds, got {n}",
                "n_seeds": n, "key": key}
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    dispersion = std / abs(mean) if abs(mean) > 1e-12 else float("inf")
    sign_consistent = all(v > 0 for v in values) or all(v < 0 for v in values)
    stable = sign_consistent and dispersion <= max_dispersion
    return {
        "status": "OK" if stable else "BLOCKED",
        "key": key, "n_seeds": n, "mean": mean, "std": std,
        "min": min(values), "max": max(values),
        "dispersion": dispersion, "sign_consistent": sign_consistent,
        "max_dispersion": max_dispersion,
    }


def validation_gate(evidence: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed promotion gate combining every overfitting control.

    Required evidence (all are checked; any missing/failing item BLOCKS):

    * ``ledger_n_trials`` >= 1 (from :class:`TrialLedger`);
    * ``holdout`` -> ``{"name", "untouched": true}``;
    * ``dsr`` >= ``dsr_min`` (default 0.95);
    * ``pbo`` <= ``pbo_max`` (default 0.20);
    * ``cost_sensitivity`` -> >= 2 rates, and the highest-cost run still has the
      **same sign** as the zero/lowest-cost run (a factor that only works at
      zero cost is not tradable);
    * ``seed_metrics`` -> :func:`multiple_seed_summary` OK.

    Returns ``{"status": "PASS"|"BLOCKED", "checks": {...}, "reasons": [...]}``.
    """
    reasons: list[str] = []
    checks: dict[str, Any] = {}

    def _num(x: Any) -> float | None:
        try:
            value = float(x)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    n_trials = _num(evidence.get("ledger_n_trials"))
    if n_trials is None or n_trials < 1:
        checks["n_trials"] = "MISSING"
        reasons.append("trial ledger has no trials (n_trials must come from the ledger)")
    else:
        checks["n_trials"] = int(n_trials)

    holdout = evidence.get("holdout") or {}
    if not isinstance(holdout, dict) or not holdout.get("untouched"):
        checks["holdout"] = "TOUCHED_OR_MISSING"
        reasons.append("holdout is missing or already touched (selection must happen off-holdout)")
    else:
        checks["holdout"] = {"name": str(holdout.get("name") or ""), "untouched": True}

    dsr = _num(evidence.get("dsr"))
    dsr_min = _num(evidence.get("dsr_min")) or DEFAULT_DSR_MIN
    if dsr is None or dsr < dsr_min:
        checks["dsr"] = {"value": dsr, "min": dsr_min, "pass": False}
        reasons.append(f"DSR {dsr} < min {dsr_min}")
    else:
        checks["dsr"] = {"value": dsr, "min": dsr_min, "pass": True}

    pbo = _num(evidence.get("pbo"))
    pbo_max = _num(evidence.get("pbo_max")) or DEFAULT_PBO_MAX
    if pbo is None or pbo > pbo_max:
        checks["pbo"] = {"value": pbo, "max": pbo_max, "pass": False}
        reasons.append(f"PBO {pbo} > max {pbo_max}")
    else:
        checks["pbo"] = {"value": pbo, "max": pbo_max, "pass": True}

    cost = [row for row in (evidence.get("cost_sensitivity") or [])
            if isinstance(row, dict) and _num(row.get("bps")) is not None]
    if len(cost) < DEFAULT_MIN_COST_RATES:
        checks["cost_sensitivity"] = "INSUFFICIENT_RATES"
        reasons.append(f"cost sensitivity needs >= {DEFAULT_MIN_COST_RATES} cost rates")
    else:
        cost_sorted = sorted(cost, key=lambda r: float(r["bps"]))
        cheap = _num(cost_sorted[0].get("annual"))
        dear = _num(cost_sorted[-1].get("annual"))
        sign_ok = cheap is not None and dear is not None and (cheap > 0) == (dear > 0)
        checks["cost_sensitivity"] = {
            "n_rates": len(cost_sorted),
            "cheapest_annual": cheap, "dearest_annual": dear, "sign_stable": sign_ok,
        }
        if not sign_ok:
            reasons.append("edge does not survive the dearest cost rate")

    seed = multiple_seed_summary(
        evidence.get("seed_metrics") or [],
        key=str(evidence.get("seed_key") or "rank_ic"),
        min_seeds=int(evidence.get("min_seeds") or DEFAULT_MIN_SEEDS),
    )
    checks["multiple_seed"] = seed
    if seed.get("status") != "OK":
        reasons.append(f"multiple-seed instability: {seed.get('reason') or 'unstable'}")

    return {
        "status": "PASS" if not reasons else "BLOCKED",
        "checks": checks,
        "reasons": reasons,
        "version": VALIDATION_VERSION,
        "policy": {
            "dsr_min": dsr_min, "pbo_max": pbo_max,
            "min_seeds": int(evidence.get("min_seeds") or DEFAULT_MIN_SEEDS),
            "min_cost_rates": DEFAULT_MIN_COST_RATES,
        },
    }
