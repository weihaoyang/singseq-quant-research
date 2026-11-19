"""Fast local screening over large named-factor candidate sets.

This is a prescreen, not a portfolio backtest. It loads one panel, computes
factor families in bounded chunks, ranks candidates by reproducible diagnostics,
and greedily removes highly correlated candidates. Finalists still need the
canonical local backtest, costs and robustness checks.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_sdk.alpha_graph import (
    _autocorrelation_metrics,
    _average_rank_matrix,
    _rank_ic_metrics,
)
from quant_sdk.catalog_factors import (
    compute_named_factor,
    compute_named_factors,
    factor_family,
    runnable_factor_names,
)
from quant_sdk.labels import canonical_forward_returns, label_contract
from quant_sdk.research_workflow import _compact_date, _load_frame, _write_json

SCREEN_VERSION = "alpha-screen-1"


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


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


_TECHNICAL_PRIORITY = {
    "talib": 10,
    "mytt": 20,
    "ta": 30,
    "qtl": 40,
    "pta": 50,
    "talipp": 60,
    "pyti": 70,
}


def _candidate_sort_key(name: str) -> tuple[int, str, str]:
    family = factor_family(name)
    return (_TECHNICAL_PRIORITY.get(family, 0), family, name)


def _compute_chunk(
    names: list[str],
    frame: pd.DataFrame,
    *,
    workers: int | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    try:
        return compute_named_factors(
            names,
            frame,
            workers=workers,
        ), []
    except Exception as batch_error:  # noqa: BLE001 - isolate one bad factor
        columns: dict[str, pd.Series] = {}
        errors = []
        for name in names:
            try:
                columns[name] = compute_named_factor(name, frame)
            except Exception as error:  # noqa: BLE001 - candidate-specific diagnostic
                errors.append({
                    "name": name,
                    "error": f"{type(error).__name__}: {error}",
                    "batchError": str(batch_error),
                })
        if columns:
            return pd.DataFrame(columns), errors
        return pd.DataFrame(index=range(len(frame))), errors


def _score(metric: dict[str, Any]) -> float | None:
    mean_ic = metric.get("meanRankIc")
    if mean_ic is None:
        return None
    return (
        abs(float(mean_ic))
        * math.sqrt(max(int(metric.get("icDays") or 0), 0))
        * math.sqrt(max(float(metric.get("coverage") or 0.0), 0.0))
    )


def _correlation_to_selected(
    candidate: pd.Series,
    selected: dict[str, pd.Series],
    *,
    selected_cache: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[float, str | None]:
    strongest = 0.0
    strongest_name = None
    candidate_values = candidate.to_numpy(dtype=float)
    candidate_valid = np.isfinite(candidate_values)
    remaining: list[tuple[str, pd.Series]] = []
    common = [
        (name, other)
        for name, other in selected.items()
        if (
            selected_cache is not None
            and name in selected_cache
            and np.all(selected_cache[name][1][candidate_valid])
        )
    ]
    if common and candidate_valid.any():
        common_names = [name for name, _ in common]
        common_values = np.column_stack([
            (
                selected_cache[name][0]
                if selected_cache is not None
                else other.to_numpy(dtype=float)
            )[candidate_valid]
            for name, other in common
        ])
        ranks = _average_rank_matrix(
            np.column_stack([
                candidate_values[candidate_valid],
                common_values,
            ]),
        )
        left = ranks[:, 0]
        right = ranks[:, 1:]
        left -= left.mean()
        right -= right.mean(axis=0)
        denominator = np.sqrt(
            (left @ left) * np.einsum("ij,ij->j", right, right),
        )
        correlations = np.divide(
            right.T @ left,
            denominator,
            out=np.zeros(len(common_names), dtype=float),
            where=np.isfinite(denominator) & (denominator > 0.0),
        )
        for index, value in enumerate(correlations):
            correlation = abs(float(value))
            if correlation > strongest:
                strongest = correlation
                strongest_name = common_names[index]
        common_set = set(common_names)
        remaining = [
            (name, other)
            for name, other in selected.items()
            if name not in common_set
        ]
    else:
        remaining = list(selected.items())
    for name, other in remaining:
        other_values = other.to_numpy(dtype=float)
        valid = np.isfinite(candidate_values) & np.isfinite(other_values)
        if int(valid.sum()) < 3:
            continue
        ranks = _average_rank_matrix(
            np.column_stack([
                candidate_values[valid],
                other_values[valid],
            ]),
        )
        left = ranks[:, 0]
        right = ranks[:, 1]
        left -= left.mean()
        right -= right.mean()
        denominator = float(np.sqrt((left @ left) * (right @ right)))
        if denominator == 0.0:
            continue
        correlation = float((left @ right) / denominator)
        if abs(correlation) > strongest:
            strongest = abs(correlation)
            strongest_name = name
    return strongest, strongest_name


def screen_factors(
    *,
    data_path: str,
    start: str,
    end: str,
    factor_names: Sequence[str] | None = None,
    families: Sequence[str] | None = None,
    query: str = "",
    scan_limit: int = 0,
    top: int = 30,
    chunk_size: int = 32,
    max_correlation: float = 0.85,
    minimum_coverage: float = 0.5,
    minimum_ic_days: int = 10,
    min_names: int = 5,
    horizon: int = 1,
    workers: int | None = None,
    benchmark_code: str = "000300.SH",
    output: str | Path | None = None,
) -> dict[str, Any]:
    start = _compact_date(start, "start")
    end = _compact_date(end, "end")
    if not 0.0 <= float(max_correlation) <= 1.0:
        raise ValueError("maxCorrelation must be between 0 and 1")
    if int(top) < 1 or int(top) > 500:
        raise ValueError("top must be between 1 and 500")
    if int(chunk_size) < 1 or int(chunk_size) > 512:
        raise ValueError("chunkSize must be between 1 and 512")
    frame = _load_frame(
        data_path, start, end, benchmark_code=str(benchmark_code),
    )
    from quant_sdk.capabilities import installed_factor_specs

    installed_families = {
        str(item["name"]): str(item.get("family") or factor_family(item["name"])).lower()
        for item in installed_factor_specs(fields=set(frame.columns))
    }

    def family(name: str) -> str:
        return installed_families.get(name, factor_family(name))

    names = [
        str(name).strip() for name in (factor_names or [])
        if str(name).strip()
    ]
    if not names:
        names = runnable_factor_names(frame, families=families)
    names = list(dict.fromkeys(names))
    family_filter = {
        str(value).strip().lower() for value in (families or [])
        if str(value).strip()
    }
    needle = str(query or "").strip().lower()
    names = [
        name for name in names
        if (not needle or needle in name.lower())
        and (not family_filter or family(name).lower() in family_filter)
    ]
    if not names:
        raise ValueError("no factor candidates matched the screen filters")
    names.sort(key=lambda name: (
        _TECHNICAL_PRIORITY.get(family(name), 0),
        family(name),
        name,
    ))
    if int(scan_limit) > 0:
        names = names[:int(scan_limit)]

    work = frame.sort_values(
        ["ts_code", "trade_date"], kind="mergesort",
    ).reset_index(drop=True)
    cache_parent = (
        Path(output).expanduser().resolve().parent
        if output
        else Path(data_path).expanduser().resolve().parent
    )
    cache_holder = None
    try:
        maximum_cached_factors = int(os.environ.get(
            "QUANT_SCREEN_MAX_CACHE_FACTORS", "256",
        ))
    except ValueError:
        maximum_cached_factors = 256
    cache_enabled = (
        int(chunk_size) < len(names) <= max(1, maximum_cached_factors)
    )
    if cache_enabled:
        try:
            cache_parent.mkdir(parents=True, exist_ok=True)
            cache_holder = tempfile.TemporaryDirectory(
                prefix=".quant-alpha-screen-",
                dir=str(cache_parent),
            )
        except OSError:
            cache_holder = tempfile.TemporaryDirectory(
                prefix=".quant-alpha-screen-",
            )
    cache_root = Path(cache_holder.name) if cache_holder else None
    cache_blocks: list[tuple[Path, dict[str, int]]] = []
    forward = canonical_forward_returns(work, int(horizon))
    summaries: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    compute_seconds = 0.0
    metrics_seconds = 0.0
    for group in _chunks(names, int(chunk_size)):
        started = time.perf_counter()
        factors, failed = _compute_chunk(
            group, work, workers=workers,
        )
        compute_seconds += time.perf_counter() - started
        errors.extend(failed)
        if factors.empty:
            continue
        started = time.perf_counter()
        metrics = _rank_ic_metrics(
            factors, work, forward, min_names=int(min_names),
        )
        metrics_seconds += time.perf_counter() - started
        if cache_root is not None:
            stored = [
                name for name in factors.columns
                if float(metrics[name].get("coverage") or 0.0)
                >= float(minimum_coverage)
                and int(metrics[name].get("icDays") or 0)
                >= int(minimum_ic_days)
                and metrics[name].get("meanRankIc") is not None
            ]
            if stored:
                cache_path = cache_root / f"block-{len(cache_blocks):05d}.bin"
                matrix = np.memmap(
                    cache_path,
                    mode="w+",
                    dtype=np.float64,
                    shape=(len(work), len(stored)),
                )
                matrix[:] = factors[stored].to_numpy(dtype=np.float64)
                matrix.flush()
                del matrix
                cache_blocks.append((
                    cache_path,
                    {name: index for index, name in enumerate(stored)},
                ))
        for name in factors.columns:
            metric = metrics[name]
            # Persistence is a heavier full-panel rank. It is not part of the
            # eligibility score, so compute it only for the finalists that
            # survive ranking and correlation de-duplication.
            metric["autocorrelation"] = None
            metric["family"] = family(name)
            metric["selectionScore"] = _score(metric)
            summaries[name] = metric

    ranked = sorted(
        summaries,
        key=lambda name: (
            summaries[name].get("selectionScore") is not None,
            float(summaries[name].get("selectionScore") or -math.inf),
            name,
        ),
        reverse=True,
    )
    eligible = [
        name for name in ranked
        if float(summaries[name].get("coverage") or 0.0) >= float(minimum_coverage)
        and int(summaries[name].get("icDays") or 0) >= int(minimum_ic_days)
        and summaries[name].get("meanRankIc") is not None
    ]

    selected: list[dict[str, Any]] = []
    selected_values: dict[str, pd.Series] = {}
    selected_correlation_values: dict[str, pd.Series] = {}
    selected_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rejected_correlation: list[dict[str, Any]] = []
    persistence_seconds = 0.0
    correlation_seconds = 0.0
    try:
        requested_correlation_rows = int(
            os.environ.get("QUANT_SCREEN_CORRELATION_ROWS", "200000")
        )
    except ValueError:
        requested_correlation_rows = 200_000
    if requested_correlation_rows > 0 and len(work) > requested_correlation_rows:
        correlation_positions = np.linspace(
            0,
            len(work) - 1,
            requested_correlation_rows,
            dtype=np.int64,
        )
        correlation_sampling = "deterministic-even-sample"
    else:
        correlation_positions = np.arange(len(work), dtype=np.int64)
        correlation_sampling = "full-panel"
    for group in _chunks(eligible, int(chunk_size)):
        if len(selected) >= int(top):
            break
        if cache_blocks:
            parts = []
            for cache_path, columns in cache_blocks:
                matched = [
                    name for name in group
                    if name in columns
                ]
                if not matched:
                    continue
                matrix = np.memmap(
                    cache_path,
                    mode="r",
                    dtype=np.float64,
                    shape=(len(work), len(columns)),
                )
                parts.append(pd.DataFrame(
                    np.asarray(matrix[:, [columns[name] for name in matched]]),
                    index=work.index,
                    columns=matched,
                ))
                del matrix
            factors = (
                pd.concat(parts, axis=1).reindex(columns=group)
                if parts else pd.DataFrame(index=work.index)
            )
        else:
            started = time.perf_counter()
            factors, failed = _compute_chunk(
                group, work, workers=workers,
            )
            compute_seconds += time.perf_counter() - started
            errors.extend(failed)
        for name in group:
            if name not in factors.columns or len(selected) >= int(top):
                continue
            started = time.perf_counter()
            candidate = factors[name].iloc[correlation_positions]
            strongest, strongest_name = _correlation_to_selected(
                candidate,
                selected_correlation_values,
                selected_cache=selected_cache,
            )
            correlation_seconds += time.perf_counter() - started
            if strongest > float(max_correlation):
                rejected_correlation.append({
                    "name": name,
                    "against": strongest_name,
                    "absCorrelation": round(strongest, 6),
                })
                continue
            series = factors[name].astype(float)
            selected_values[name] = series
            sampled = candidate.astype(float)
            selected_correlation_values[name] = sampled
            array = sampled.to_numpy(dtype=float)
            selected_cache[name] = (array, np.isfinite(array))
            selected.append({
                "name": name,
                **summaries[name],
                "autocorrelation": None,
                "maxSelectedCorrelation": (
                    round(strongest, 6) if strongest_name else None
                ),
                "mostCorrelatedSelected": strongest_name,
            })

    if selected_values:
        started = time.perf_counter()
        persistence = _autocorrelation_metrics(
            pd.DataFrame(selected_values, index=work.index),
            work["ts_code"],
        )
        persistence_seconds += time.perf_counter() - started
        for item in selected:
            item["autocorrelation"] = persistence.get(item["name"])
    cache_used = bool(cache_blocks)
    if cache_holder is not None:
        cache_holder.cleanup()
        cache_holder = None

    external_candidates = [
        name for name in names
        if name.lower().startswith(("alpha101_", "alpha191_"))
    ]
    technical_candidates = [
        name for name in names
        if name.lower().startswith(
            ("talib_", "mytt_", "pta_", "ta_", "qtl_", "talipp_", "pyti_")
        )
    ]
    if external_candidates and len(external_candidates) == len(names):
        factor_backend = "ta-cn-wide-matrix"
    elif technical_candidates and len(technical_candidates) == len(names):
        factor_backend = "technical-adapter-batch"
    elif external_candidates or technical_candidates:
        factor_backend = "mixed-family-batch"
    else:
        factor_backend = "local-family-batch"

    result = {
        "version": SCREEN_VERSION,
        "status": "LOCAL_SCREEN_COMPLETE",
        "dataPath": str(Path(data_path).expanduser().resolve()),
        "window": {"start": start, "end": end},
        "benchmark": frame.attrs.get("benchmarkResolution"),
        "panel": {
            "rows": len(work),
            "dates": int(work["trade_date"].nunique()),
            "assets": int(work["ts_code"].nunique()),
        },
        "configuration": {
            "horizon": int(horizon),
            "label": label_contract(work, int(horizon)),
            "workers": workers,
            "minNames": int(min_names),
            "chunkSize": int(chunk_size),
            "top": int(top),
            "maxCorrelation": float(max_correlation),
            "minimumCoverage": float(minimum_coverage),
            "minimumIcDays": int(minimum_ic_days),
            "families": sorted(family_filter),
            "query": str(query or ""),
        },
        "engine": {
            "factorBackend": factor_backend,
            "externalFactorWorkers": (
                "QUANT_FACTOR_WORKERS or min(8, CPU count)"
            ),
            "technicalFactorWorkers": (
                "QUANT_FACTOR_WORKERS or min(16, CPU count)"
            ),
            "candidateOrder": (
                "cost-tier-family-grouped"
                if int(scan_limit) > 0
                else "family-grouped"
            ),
            "selectionCache": "temporary-memmap" if cache_used else "none",
            "maximumCachedFactors": max(1, maximum_cached_factors),
            "autocorrelation": "selected candidates only",
            "correlationRows": len(correlation_positions),
            "correlationSampling": correlation_sampling,
        },
        "timings": {
            "computeSeconds": round(compute_seconds, 3),
            "metricsSeconds": round(metrics_seconds, 3),
            "persistenceSeconds": round(persistence_seconds, 3),
            "correlationSeconds": round(correlation_seconds, 3),
        },
        "counts": {
            "requested": len(names),
            "computed": len(summaries),
            "eligible": len(eligible),
            "selected": len(selected),
            "failed": len(errors),
            "correlationRejected": len(rejected_correlation),
        },
        "scoreDefinition": (
            "abs(meanRankIc) * sqrt(icDays) * sqrt(coverage); "
            "descriptive screen only, not risk-adjusted performance"
        ),
        "selected": selected,
        "ranked": [
            {"name": name, **summaries[name]} for name in ranked[:200]
        ],
        "correlationRejected": rejected_correlation[:500],
        "errors": errors[:500],
        "submitted": False,
        "published": False,
        "note": "候选筛选只在本地运行；入选因子仍需 canonical 回测、成本、OOS 和容量检验。",
    }
    if output:
        _write_json(output, _jsonable(result))
        result["output"] = str(Path(output).expanduser().resolve())
    return _jsonable(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local alpha candidate screen")
    parser.add_argument("--data", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--factor-names", default="")
    parser.add_argument("--families", default="")
    parser.add_argument("--query", default="")
    parser.add_argument("--scan-limit", type=int, default=0)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--max-correlation", type=float, default=0.85)
    parser.add_argument("--minimum-coverage", type=float, default=0.5)
    parser.add_argument("--minimum-ic-days", type=int, default=10)
    parser.add_argument("--min-names", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--benchmark-code", default="000300.SH")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    names = [
        item.strip() for item in str(args.factor_names).split(",")
        if item.strip()
    ]
    families = [
        item.strip() for item in str(args.families).split(",")
        if item.strip()
    ]
    result = screen_factors(
        data_path=args.data, start=args.start, end=args.end,
        factor_names=names or None, families=families or None,
        query=args.query, scan_limit=args.scan_limit, top=args.top,
        chunk_size=args.chunk_size, max_correlation=args.max_correlation,
        minimum_coverage=args.minimum_coverage,
        minimum_ic_days=args.minimum_ic_days, min_names=args.min_names,
        horizon=args.horizon, workers=args.workers,
        benchmark_code=args.benchmark_code,
        output=args.out,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
