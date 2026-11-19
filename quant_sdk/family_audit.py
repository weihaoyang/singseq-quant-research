"""Sealed local candidate-family evidence for Alpha Graph research.

The family audit evaluates multiple immutable candidate definitions on one
local data slice.  It aligns their held-out, after-cost daily returns, computes
PBO/CSCV, Deflated Sharpe and Benjamini-Hochberg evidence, and binds the result
to the candidate definitions and data identity.

The output is evidence, not a publication action.  A high PBO remains a valid
observation and blocks promotion through ``pbo.gateStatus`` rather than being
hidden or rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from evaluation.overfit import (
    OVERFIT_VERSION,
    benjamini_hochberg,
    deflated_sharpe_ratio,
    pbo_cscv,
)
from quant_sdk.evidence_audit import (
    _causality_probes,
    _evaluate_values,
    _static_source_audit,
    _values_for_frame,
    research_split,
)
from quant_sdk.labels import label_contract

FAMILY_AUDIT_VERSION = "qalpha-family-audit/1"
FAMILY_REQUEST_VERSION = "qalpha-family-request/1"
PBO_GATE_MAX = 0.20
MAX_CANDIDATES = 200
MIN_ALIGNED_OBSERVATIONS = 32


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        value if isinstance(value, bytes) else _canonical_bytes(value)
    ).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_output(path: str | Path | None, value: Mapping[str, Any]) -> None:
    if path is None:
        return
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def _candidate_inputs(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not 2 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError(
            f"candidate family requires 2..{MAX_CANDIDATES} candidates"
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(candidates):
        if not isinstance(raw, Mapping):
            raise ValueError(f"candidate[{index}] must be an object")
        candidate_id = str(raw.get("id") or "").strip()
        if not candidate_id:
            raise ValueError(f"candidate[{index}].id is required")
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate id: {candidate_id}")
        seen.add(candidate_id)
        factor_name = str(raw.get("factorName") or "").strip()
        source = str(raw.get("source") or "")
        if not factor_name and not source.strip():
            raise ValueError(
                f"candidate {candidate_id} requires factorName or source"
            )
        try:
            direction = int(raw.get("direction", 1))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"candidate {candidate_id}.direction must be -1 or 1"
            ) from error
        if direction not in {-1, 1}:
            raise ValueError(
                f"candidate {candidate_id}.direction must be -1 or 1"
            )
        normalized.append({
            "id": candidate_id,
            "factorName": factor_name or None,
            "source": source,
            "direction": direction,
        })
    return normalized


def _daily_curve_returns(result: Mapping[str, Any]) -> tuple[list[str], np.ndarray]:
    curves = result.get("curves")
    curves = curves if isinstance(curves, Mapping) else {}
    dates = [
        str(value).replace("-", "")
        for value in (curves.get("dates") or [])
    ]
    equity = np.asarray(curves.get("ls") or [], dtype=float)
    if len(dates) != len(equity) or len(equity) < 2:
        raise ValueError("candidate backtest did not return an aligned net curve")
    if not np.isfinite(equity).all() or np.any(equity <= 0.0):
        raise ValueError("candidate net curve contains non-positive marks")
    returns = np.diff(equity) / equity[:-1]
    if not np.isfinite(returns).all():
        raise ValueError("candidate net curve contains non-finite returns")
    return dates[1:], returns


def _ic_mean(summary: Mapping[str, Any]) -> float | None:
    value = summary.get("meanIc")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def audit_family(
    *,
    data_path: str | Path,
    start: str,
    end: str,
    candidates: Sequence[Mapping[str, Any]],
    family_id: str = "candidate-family",
    selection_ratio: float = 0.7,
    purge_days: int = 1,
    embargo_days: int = 1,
    horizon: int = 1,
    min_names: int = 5,
    n_groups: int = 5,
    costs: bool = True,
    allow_short: bool = False,
    benchmark_code: str = "000300.SH",
    pbo_splits: int = 16,
    alpha: float = 0.05,
    cut_quantiles: Sequence[float] = (0.55, 0.7, 0.85),
    tolerance: float = 1e-12,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate and seal one local candidate family."""
    from quant_sdk.catalog_factors import compute_named_factors
    from quant_sdk.evidence_pack import _frame_slice_identity
    from quant_sdk.local_backtest import run_local_backtest
    from quant_sdk.research_workflow import (
        _compact_date,
        _factor_matrix,
        _ic_summary,
        _load_frame,
        _rank_ic_daily,
        _window,
    )

    normalized = _candidate_inputs(candidates)
    start = _compact_date(start, "start")
    end = _compact_date(end, "end")
    if not 2 <= int(pbo_splits) <= 32 or int(pbo_splits) % 2:
        raise ValueError("pboSplits must be an even integer between 2 and 32")
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha must be within (0, 1]")

    frame = _load_frame(
        data_path,
        start,
        end,
        benchmark_code=str(benchmark_code),
    )
    split = research_split(
        frame["trade_date"].astype(str).tolist(),
        selection_ratio=selection_ratio,
        purge_days=purge_days,
        embargo_days=embargo_days,
    )
    selection_frame, _ = _window(
        frame,
        _factor_matrix(frame, np.zeros(len(frame), dtype=float)),
        split["selectionStart"],
        split["selectionEnd"],
    )
    holdout_frame, _ = _window(
        frame,
        _factor_matrix(frame, np.zeros(len(frame), dtype=float)),
        split["holdoutStart"],
        split["holdoutEnd"],
    )

    named = [
        candidate["factorName"]
        for candidate in normalized
        if candidate["factorName"]
    ]
    named_values = (
        compute_named_factors(named, frame)
        if named else pd.DataFrame(index=frame.index)
    )

    evaluated: list[dict[str, Any]] = []
    common_returns: dict[str, dict[str, float]] = {}
    for candidate in normalized:
        candidate_id = candidate["id"]
        factor_name = candidate["factorName"] or ""
        source = candidate["source"]
        static = _static_source_audit(source)
        if static.get("status") == "FAILED":
            evaluated.append({
                "id": candidate_id,
                "status": "REJECTED_LEAKAGE",
                "direction": candidate["direction"],
                "factorName": factor_name or None,
                "sourceSha256": _sha256(source) if source else None,
                "staticAudit": static,
            })
            continue
        try:
            if factor_name:
                values = named_values[factor_name].to_numpy(dtype=float)
            else:
                values = np.asarray(
                    _evaluate_values(
                        factor_name="",
                        source=source,
                        frame=frame,
                    ),
                    dtype=float,
                )
            runtime = _causality_probes(
                frame=frame,
                values=values,
                factor_name=factor_name,
                source=source,
                cut_quantiles=cut_quantiles,
                tolerance=tolerance,
            )
            if runtime.get("status") != "PASSED":
                raise ValueError("runtime causality check failed")
            oriented = values * int(candidate["direction"])
            matrix = _factor_matrix(frame, oriented)
            _, selection_matrix = _window(
                frame, matrix,
                split["selectionStart"], split["selectionEnd"],
            )
            _, holdout_matrix = _window(
                frame, matrix,
                split["holdoutStart"], split["holdoutEnd"],
            )
            selection_ic = _ic_summary(_rank_ic_daily(
                selection_frame,
                _values_for_frame(selection_frame, frame, oriented),
                horizon=int(horizon),
                min_names=int(min_names),
            ))
            holdout_ic = _ic_summary(_rank_ic_daily(
                holdout_frame,
                _values_for_frame(holdout_frame, frame, oriented),
                horizon=int(horizon),
                min_names=int(min_names),
            ))
            backtest = run_local_backtest(
                "",
                holdout_frame,
                factor_matrix=holdout_matrix,
                n_groups=int(n_groups),
                costs=bool(costs),
                allow_short=bool(allow_short),
                n_trials=len(normalized),
            )
            dates, returns = _daily_curve_returns(backtest)
            common_returns[candidate_id] = dict(zip(dates, returns, strict=True))
            evaluated.append({
                "id": candidate_id,
                "status": "OBSERVED",
                "direction": int(candidate["direction"]),
                "factorName": factor_name or None,
                "sourceSha256": _sha256(source) if source else None,
                "staticAudit": static,
                "runtimeCausality": {
                    "status": runtime.get("status"),
                    "probeCount": len(runtime.get("probes") or []),
                },
                "selectionMeanIc": _ic_mean(selection_ic),
                "holdoutMeanIc": _ic_mean(holdout_ic),
                "selectionIcDays": int(selection_ic.get("days") or 0),
                "holdoutIcDays": int(holdout_ic.get("days") or 0),
                "holdoutMetrics": backtest.get("metrics"),
                "turnoverMean": backtest.get("turnover_mean"),
                "capacity": backtest.get("capacity"),
            })
        except Exception as error:  # noqa: BLE001 - seal the exact rejection
            evaluated.append({
                "id": candidate_id,
                "status": "BLOCKED",
                "direction": int(candidate["direction"]),
                "factorName": factor_name or None,
                "sourceSha256": _sha256(source) if source else None,
                "staticAudit": static,
                "reason": f"{type(error).__name__}: {error}",
            })

    observed_ids = [item["id"] for item in evaluated if item["status"] == "OBSERVED"]
    common_dates = (
        sorted(set.intersection(
            *(set(common_returns[candidate_id]) for candidate_id in observed_ids)
        ))
        if observed_ids else []
    )
    matrix = (
        np.asarray([
            [common_returns[candidate_id][date] for candidate_id in observed_ids]
            for date in common_dates
        ], dtype=float)
        if common_dates and observed_ids else np.empty((0, 0), dtype=float)
    )

    pbo: dict[str, Any] = {
        "status": "BLOCKED",
        "gateStatus": "BLOCKED",
        "reason": "CANDIDATE_FAMILY_INCOMPLETE",
        "gateMax": PBO_GATE_MAX,
        "version": OVERFIT_VERSION,
    }
    multiple: dict[str, Any] = {
        "status": "BLOCKED",
        "method": "benjamini_hochberg",
        "alpha": float(alpha),
    }
    dsr: list[dict[str, Any]] = []
    if (
        len(observed_ids) >= 2
        and matrix.ndim == 2
        and matrix.shape[1] >= 2
        and matrix.shape[0] >= MIN_ALIGNED_OBSERVATIONS
        and np.isfinite(matrix).all()
    ):
        raw_pbo = pbo_cscv(matrix, n_splits=int(pbo_splits))
        probability = raw_pbo.get("pbo")
        computed = probability is not None
        pbo = {
            **raw_pbo,
            "status": "PASS" if computed else "BLOCKED",
            "gateStatus": (
                "PASS"
                if computed and float(probability) <= PBO_GATE_MAX
                else "BLOCKED"
            ),
            "gateMax": PBO_GATE_MAX,
            "logits": [],
            "logitsOmitted": True,
        }
        p_values = [
            float(stats.ttest_1samp(matrix[:, index], 0.0).pvalue)
            for index in range(matrix.shape[1])
        ]
        adjusted = benjamini_hochberg(p_values, alpha=float(alpha))
        multiple = {
            **adjusted,
            "status": "PASS" if adjusted.get("valid") is True else "BLOCKED",
            "method": "benjamini_hochberg",
            "alpha": float(alpha),
            "pValues": p_values,
            "candidateIds": observed_ids,
        }
        trial_returns = [matrix[:, index].tolist() for index in range(matrix.shape[1])]
        for index, candidate_id in enumerate(observed_ids):
            item = dict(deflated_sharpe_ratio(
                trial_returns[index],
                n_trials=len(observed_ids),
                trial_returns=trial_returns,
            ))
            item["id"] = candidate_id
            dsr.append(item)

    selection_scores = {
        item["id"]: item.get("selectionMeanIc")
        for item in evaluated
        if item.get("status") == "OBSERVED"
        and item.get("selectionMeanIc") is not None
    }
    selected_id = (
        max(selection_scores, key=lambda key: (selection_scores[key], key))
        if selection_scores else None
    )
    rejected_indices = set(multiple.get("rejected") or [])
    selected_survived = bool(
        selected_id
        and selected_id in observed_ids
        and observed_ids.index(selected_id) in rejected_indices
    )
    checks = {
        "candidateCountAtLeastTwo": len(normalized) >= 2,
        "candidateIdsUnique": len({item["id"] for item in normalized}) == len(normalized),
        "allCandidatesObserved": len(observed_ids) == len(normalized),
        "alignedReturnMatrixPresent": matrix.ndim == 2 and matrix.shape[1] >= 2,
        "alignedObservationsSufficient": matrix.shape[0] >= MIN_ALIGNED_OBSERVATIONS,
        "pboComputed": pbo.get("status") == "PASS",
        "multipleTestingComputed": multiple.get("status") == "PASS",
        "pboBelowGate": pbo.get("gateStatus") == "PASS",
        "selectedCandidateSurvivedMultipleTesting": selected_survived,
    }
    evidence_checks = {
        key: value for key, value in checks.items()
        if key not in {"pboBelowGate", "selectedCandidateSurvivedMultipleTesting"}
    }
    evidence_valid = all(evidence_checks.values())
    data_identity = _frame_slice_identity(frame)
    configuration = {
        "horizon": int(horizon),
        "label": label_contract(frame, int(horizon)),
        "selectionRatio": float(selection_ratio),
        "purgeDays": int(purge_days),
        "embargoDays": int(embargo_days),
        "minNames": int(min_names),
        "nGroups": int(n_groups),
        "costs": bool(costs),
        "allowShort": bool(allow_short),
        "benchmarkCode": (
            (frame.attrs.get("benchmarkResolution") or {}).get("code")
            or str(benchmark_code)
        ),
        "benchmarkResolution": frame.attrs.get("benchmarkResolution"),
        "pboSplits": int(pbo_splits),
        "alpha": float(alpha),
        "overfitVersion": OVERFIT_VERSION,
    }
    definitions = [
        {
            "id": item["id"],
            "factorName": item["factorName"],
            "direction": item["direction"],
            "sourceSha256": _sha256(item["source"]) if item["source"] else None,
            "source": item["source"] or None,
        }
        for item in normalized
    ]
    family_identity = {
        "familyId": str(family_id).strip() or "candidate-family",
        "dataSha256": data_identity["sha256"],
        "window": {"start": start, "end": end},
        "definitions": definitions,
        "configuration": configuration,
    }
    result: dict[str, Any] = {
        "format": FAMILY_AUDIT_VERSION,
        "status": "PASS" if evidence_valid else "BLOCKED",
        "validated": bool(evidence_valid and checks["pboBelowGate"]),
        "executionBoundary": "local",
        "familyId": family_identity["familyId"],
        "familySha256": _sha256(family_identity),
        "window": {"start": start, "end": end},
        "split": split,
        "configuration": configuration,
        "candidates": evaluated,
        "candidateDefinitions": definitions,
        "candidateCount": len(normalized),
        "observedCandidateCount": len(observed_ids),
        "selectedCandidateId": selected_id,
        "selectedCandidateSurvivedMultipleTesting": selected_survived,
        "returnMatrix": {
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "dates": common_dates,
            "candidateIds": observed_ids,
            "sha256": (
                _sha256({
                    "dates": common_dates,
                    "candidateIds": observed_ids,
                    "returns": np.round(matrix, 12).tolist(),
                })
                if matrix.size else None
            ),
            "valuesEmbedded": False,
        },
        "pbo": pbo,
        "multipleTesting": multiple,
        "deflatedSharpe": dsr,
        "checks": checks,
        "submitted": False,
        "published": False,
        "disclosure": {
            "dataUploaded": False,
            "automaticSubmission": False,
            "automaticPublication": False,
            "candidateSourcesIncluded": any(
                item["source"] is not None for item in definitions
            ),
        },
        "note": (
            "PASS means the sealed family matrix and statistical controls were "
            "computed. PBO above its gate or a selected candidate that does not "
            "survive multiple testing remains visible and blocks validation."
        ),
    }
    _write_output(output, result)
    return _jsonable(result)


def load_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("candidate-family manifest must be a JSON object")
    if str(value.get("format") or "").strip() != FAMILY_REQUEST_VERSION:
        raise ValueError(f"candidate-family manifest must use {FAMILY_REQUEST_VERSION}")
    candidates = value.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("candidate-family manifest requires a candidates array")
    resolved: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            raise ValueError("candidate-family entries must be objects")
        entry = dict(item)
        source_file = str(entry.pop("sourceFile", "") or "").strip()
        if source_file:
            file_path = (source.parent / source_file).resolve()
            entry["source"] = file_path.read_text(encoding="utf-8")
        resolved.append(entry)
    value = dict(value)
    value["candidates"] = resolved
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit a sealed local candidate family"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--family-id")
    parser.add_argument("--selection-ratio", type=float, default=0.7)
    parser.add_argument("--purge-days", type=int, default=1)
    parser.add_argument("--embargo-days", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--min-names", type=int, default=5)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--no-costs", action="store_true")
    parser.add_argument("--benchmark-code", default="000300.SH")
    parser.add_argument("--pbo-splits", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--cut-quantiles", default="0.55,0.7,0.85")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.candidates)
    quantiles = [
        float(item.strip())
        for item in str(args.cut_quantiles).split(",")
        if item.strip()
    ]
    result = audit_family(
        data_path=args.data,
        start=args.start,
        end=args.end,
        candidates=manifest["candidates"],
        family_id=args.family_id or str(manifest.get("familyId") or "candidate-family"),
        selection_ratio=args.selection_ratio,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
        horizon=args.horizon,
        min_names=args.min_names,
        n_groups=args.groups,
        costs=not args.no_costs,
        allow_short=bool(args.allow_short),
        benchmark_code=args.benchmark_code,
        pbo_splits=args.pbo_splits,
        alpha=args.alpha,
        cut_quantiles=quantiles,
        tolerance=args.tolerance,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
