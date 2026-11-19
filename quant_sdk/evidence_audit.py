"""Candidate-level evidence audit for local Alpha research.

The audit answers four questions before a factor is packaged as evidence:

* was the factor selected without looking at the holdout period?
* does the factor ignore observations that occur after each signal date?
* does the candidate survive after costs in a purged/embargoed holdout?
* can observed orders be mapped to transparent capacity and benchmark stress?

It never submits or publishes a factor.  A successful audit is a local claim
about one factor, one data slice and one published protocol, not an investment
recommendation.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_sdk.labels import LABEL_VERSION, label_contract

AUDIT_VERSION = "qalpha-audit/1"


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


def _write_json(path: str | Path, value: Any) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def research_split(
    dates: Sequence[str],
    *,
    selection_ratio: float = 0.7,
    purge_days: int = 1,
    embargo_days: int = 1,
) -> dict[str, Any]:
    """Return a chronological selection/holdout split with both buffers."""
    ordered = sorted({str(value).replace("-", "") for value in dates})
    if len(ordered) < 6:
        raise ValueError("research split 至少需要 6 个交易日")
    ratio = float(selection_ratio)
    purge = int(purge_days)
    embargo = int(embargo_days)
    if not 0.1 <= ratio <= 0.9:
        raise ValueError("selectionRatio must be between 0.1 and 0.9")
    if purge < 0 or embargo < 0:
        raise ValueError("purgeDays and embargoDays must be non-negative")
    base_index = min(len(ordered) - 1, max(1, int(len(ordered) * ratio)))
    selection_end_index = base_index - 1 - purge
    holdout_start_index = base_index + embargo
    if selection_end_index < 1 or holdout_start_index >= len(ordered):
        raise ValueError("purge/embargo leave an empty selection or holdout window")
    return {
        "selectionRatio": ratio,
        "purgeDays": purge,
        "embargoDays": embargo,
        "selectionStart": ordered[0],
        "selectionEnd": ordered[selection_end_index],
        "holdoutStart": ordered[holdout_start_index],
        "holdoutEnd": ordered[-1],
        "excludedBoundaryDates": ordered[
            selection_end_index + 1:holdout_start_index
        ],
        "selectionDates": selection_end_index + 1,
        "holdoutDates": len(ordered) - holdout_start_index,
    }


def _benchmark_price_series(
    frame: pd.DataFrame,
    *,
    primary_price: str,
) -> tuple[pd.Series, str, str]:
    """Choose one internally consistent benchmark price basis.

    Adjusted prices are all-or-nothing.  Splicing adjusted rows into raw rows
    creates artificial jumps at every missing observation and can manufacture
    implausible benchmark returns.
    """
    raw_column = primary_price if primary_price in frame.columns else "close"
    if raw_column not in frame.columns:
        raise ValueError(
            f"benchmark requires {primary_price} or close"
        )
    raw = pd.to_numeric(frame[raw_column], errors="coerce")
    adjustment_complete = False
    if {"adj_close", "close"}.issubset(frame.columns):
        adjusted_close = pd.to_numeric(frame["adj_close"], errors="coerce")
        close = pd.to_numeric(frame["close"], errors="coerce")
        adjustment_complete = bool(
            adjusted_close.notna().all()
            and close.notna().all()
            and (adjusted_close > 0.0).all()
            and (close > 0.0).all()
        )
        if adjustment_complete:
            return (
                raw * adjusted_close / close,
                f"adj_{raw_column}",
                "next-open" if raw_column == "open" else "close-to-close",
            )
    return (
        raw,
        f"{raw_column}_proxy",
        "next-open" if raw_column == "open" else "close-to-close",
    )


def _mark_benchmark_identity(
    frame: pd.DataFrame,
    *,
    return_basis: str,
    price_basis: str,
    grade: str,
) -> pd.DataFrame:
    work = frame.copy()
    work["benchmark_basis"] = f"{return_basis}:{price_basis}"
    work["benchmark_grade"] = grade
    return work


def attach_equal_weight_benchmark(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach a real equal-weight universe return series when none exists.

    The result uses the same open-to-open return window as the canonical
    strategy.  It is deliberately labelled as a proxy: an index benchmark
    remains preferable and is required for a publication-grade proof.
    """
    if "benchmark_ret" in frame.columns and pd.to_numeric(
        frame["benchmark_ret"], errors="coerce"
    ).notna().any():
        return frame.copy(), {
            "status": "PROVIDED",
            "kind": "panel-benchmark-return",
        }
    required = {"ts_code", "trade_date"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "equal-weight benchmark requires: " + ", ".join(sorted(missing))
        )
    work = frame.copy()
    work["ts_code"] = work["ts_code"].astype(str)
    work["trade_date"] = work["trade_date"].astype(str).str.replace(
        "-", "", regex=False
    )
    price, price_basis, return_basis = _benchmark_price_series(
        work,
        primary_price="open",
    )
    work["_benchmark_price"] = price
    work = work.sort_values(
        ["ts_code", "trade_date"], kind="mergesort"
    ).reset_index(drop=True)
    price = pd.to_numeric(work["_benchmark_price"], errors="coerce").replace(
        0.0, np.nan
    )
    forward = price.groupby(work["ts_code"], sort=False).shift(-1) / price - 1.0
    benchmark = (
        pd.DataFrame({
            "trade_date": work["trade_date"].astype(str),
            "ret": forward,
        })
        .groupby("trade_date", sort=True)["ret"]
        .mean()
    )
    work["benchmark_ret"] = work["trade_date"].map(benchmark)
    work = work.drop(columns=["_benchmark_price"], errors="ignore")
    work = _mark_benchmark_identity(
        work,
        return_basis=return_basis,
        price_basis=price_basis,
        grade="EQUAL_WEIGHT_PROXY",
    )
    return work, {
        "status": "ATTACHED",
        "kind": "equal-weight-universe-total-return-proxy",
        "returnBasis": return_basis,
        "priceColumn": price_basis,
        "benchmarkGrade": "EQUAL_WEIGHT_PROXY",
        "labelVersion": (
            LABEL_VERSION
            if return_basis == "next-open"
            else "close-to-close-proxy/1"
        ),
        "dates": int(benchmark.index.size),
        "finiteDates": int(np.isfinite(benchmark.to_numpy(dtype=float)).sum()),
        "note": (
            "A return series computed from this panel without splicing price "
            "bases. It is not a third-party index and cannot by itself qualify "
            "a publication-grade proof."
        ),
    }


def attach_market_benchmark(
    frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame,
    *,
    symbol: str | None = None,
    source: str = "caller-provided market series",
    minimum_coverage: float = 0.9,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach a forward market return series on one consistent price basis.

    This is intentionally separate from the equal-weight diagnostic proxy.
    Missing or misaligned benchmark observations fail instead of silently
    falling back to another definition.
    """
    required = {"trade_date"}
    missing = required.difference(benchmark_frame.columns)
    if missing:
        raise ValueError(
            "market benchmark requires: " + ", ".join(sorted(missing))
        )
    if not 0.5 <= float(minimum_coverage) <= 1.0:
        raise ValueError("minimumCoverage must be between 0.5 and 1.0")
    bench = benchmark_frame.copy()
    bench["trade_date"] = (
        bench["trade_date"].astype(str).str.replace("-", "", regex=False)
    )
    selected_symbol = str(symbol or "").strip()
    if "ts_code" in bench.columns:
        bench["ts_code"] = bench["ts_code"].astype(str)
        available = sorted(bench["ts_code"].dropna().unique().tolist())
        if selected_symbol:
            bench = bench[bench["ts_code"] == selected_symbol]
        elif len(available) == 1:
            selected_symbol = str(available[0])
        else:
            raise ValueError(
                "market benchmark has multiple symbols; specify benchmarkSymbol"
            )
    if selected_symbol and "ts_code" not in bench.columns:
        bench["ts_code"] = selected_symbol
    if bench.empty:
        raise ValueError("market benchmark has no rows for the selected symbol")
    price, price_basis, return_basis = _benchmark_price_series(
        bench,
        primary_price="open",
    )
    bench["_benchmark_price"] = price
    bench = bench[
        bench["trade_date"].str.fullmatch(r"\d{8}", na=False)
        & bench["_benchmark_price"].notna()
        & (bench["_benchmark_price"] > 0.0)
    ].copy()
    if len(bench) < 2:
        raise ValueError("market benchmark has fewer than two valid closes")
    bench = bench.sort_values("trade_date", kind="mergesort")
    bench = bench.drop_duplicates("trade_date", keep="last")
    forward = (
        bench["_benchmark_price"].shift(-1)
        / bench["_benchmark_price"]
        - 1.0
    )
    returns = pd.Series(
        forward.to_numpy(dtype=float),
        index=bench["trade_date"].to_numpy(),
    )
    work = frame.copy()
    panel_dates = (
        work["trade_date"].astype(str).str.replace("-", "", regex=False)
    )
    work["benchmark_ret"] = panel_dates.map(returns)
    unique_dates = int(panel_dates.nunique())
    finite_dates = int(
        work.loc[
            work["benchmark_ret"].notna(), "trade_date"
        ].astype(str).str.replace("-", "", regex=False).nunique()
    )
    # The final panel date has no in-window forward observation by design.
    coverage = finite_dates / max(unique_dates - 1, 1)
    if coverage < float(minimum_coverage):
        raise ValueError(
            "market benchmark coverage is too low: "
            f"{coverage:.3f} < {float(minimum_coverage):.3f}"
        )
    work = _mark_benchmark_identity(
        work,
        return_basis=return_basis,
        price_basis=price_basis,
        grade="MARKET" if return_basis == "next-open" else "CLOSE_PROXY",
    )
    return work, {
        "status": "ATTACHED",
        "kind": "market-index-or-etf-forward-return",
        "source": str(source),
        "symbol": selected_symbol or None,
        "returnBasis": return_basis,
        "priceColumn": price_basis,
        "benchmarkGrade": (
            "MARKET" if return_basis == "next-open" else "CLOSE_PROXY"
        ),
        "labelVersion": (
            LABEL_VERSION
            if return_basis == "next-open"
            else "close-to-close-proxy/1"
        ),
        "sourceRows": len(bench),
        "sourceStart": str(bench["trade_date"].iloc[0]),
        "sourceEnd": str(bench["trade_date"].iloc[-1]),
        "panelDates": unique_dates,
        "finiteDates": finite_dates,
        "coverage": round(coverage, 6),
        "minimumCoverage": float(minimum_coverage),
        "note": (
            "Forward returns from an independently supplied market index or "
            "ETF series on one consistent price basis; this is not an "
            "equal-weight proxy."
        ),
    }


def _values_frame(
    frame: pd.DataFrame,
    values: Sequence[float],
) -> pd.DataFrame:
    if len(values) != len(frame):
        raise ValueError("factor output length does not match the panel")
    return pd.DataFrame({
        "ts_code": frame["ts_code"].astype(str).to_numpy(),
        "trade_date": (
            frame["trade_date"].astype(str).str.replace("-", "", regex=False)
        ).to_numpy(),
        "value": np.asarray(values, dtype=float),
    })


def _evaluate_values(
    *,
    factor_name: str,
    source: str,
    frame: pd.DataFrame,
) -> np.ndarray:
    if factor_name:
        from quant_sdk.catalog_factors import compute_named_factor

        return compute_named_factor(factor_name, frame).to_numpy(dtype=float)
    if source.strip():
        from quant_sdk.research import _compute_factor_series

        return np.asarray(_compute_factor_series(source, frame), dtype=float)
    raise ValueError("evidence.audit requires factorName or source")


def _static_source_audit(source: str) -> dict[str, Any]:
    if not source.strip():
        return {
            "status": "NOT_APPLICABLE",
            "reason": "catalog factor is executed through the trusted graph runtime",
        }
    from quant_core.audit import audit_factor_code

    audit = audit_factor_code(source)
    criticals = [
        {
            "line": int(item.line),
            "explanation": str(item.explanation),
        }
        for item in getattr(audit, "criticals", ())
    ]
    warnings = [
        {
            "line": int(item.line),
            "explanation": str(item.explanation),
        }
        for item in getattr(audit, "warnings", ())
    ]
    return {
        "status": "PASSED" if audit.passed else "FAILED",
        "codeHash": str(getattr(audit, "code_hash", "")),
        "criticals": criticals,
        "warnings": warnings,
        "rejections": [str(item) for item in getattr(audit, "rejections", ())],
    }


def _causality_probes(
    *,
    frame: pd.DataFrame,
    values: np.ndarray,
    factor_name: str,
    source: str,
    cut_quantiles: Sequence[float],
    tolerance: float,
) -> dict[str, Any]:
    dates = sorted(frame["trade_date"].astype(str).str.replace(
        "-", "", regex=False
    ).unique())
    full = _values_frame(frame, values)
    probes: list[dict[str, Any]] = []
    for quantile in cut_quantiles:
        q = float(quantile)
        if not 0.1 <= q <= 0.9:
            raise ValueError("cut quantiles must be between 0.1 and 0.9")
        index = min(len(dates) - 2, max(1, int(len(dates) * q)))
        cut_date = dates[index]
        prefix = frame[
            frame["trade_date"].astype(str).str.replace(
                "-", "", regex=False
            ) <= cut_date
        ].reset_index(drop=True)
        try:
            candidate_values = _evaluate_values(
                factor_name=factor_name,
                source=source,
                frame=prefix,
            )
            candidate = _values_frame(prefix, candidate_values)
            paired = full[full["trade_date"] <= cut_date].merge(
                candidate,
                on=["ts_code", "trade_date"],
                how="inner",
                suffixes=("_full", "_truncated"),
            )
            full_values = paired["value_full"].to_numpy(dtype=float)
            truncated_values = paired["value_truncated"].to_numpy(dtype=float)
            full_finite = np.isfinite(full_values)
            truncated_finite = np.isfinite(truncated_values)
            changed_finiteness = full_finite != truncated_finite
            both_finite = full_finite & truncated_finite
            differences = np.abs(full_values - truncated_values)
            changed_value = both_finite & (
                differences > float(tolerance)
            )
            changed = changed_finiteness | changed_value
            max_difference = (
                float(np.max(differences[both_finite]))
                if both_finite.any()
                else 0.0
            )
            samples = paired.loc[
                changed, ["ts_code", "trade_date", "value_full", "value_truncated"]
            ].head(5).to_dict(orient="records")
            probes.append({
                "cutDate": cut_date,
                "comparedRows": len(paired),
                "changedRows": int(changed.sum()),
                "changedFinitenessRows": int(changed_finiteness.sum()),
                "maxAbsDifference": max_difference,
                "passed": bool(not changed.any()),
                "samples": samples,
            })
        except Exception as error:  # noqa: BLE001 - report one failed probe
            probes.append({
                "cutDate": cut_date,
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            })
    passed = bool(probes) and all(probe.get("passed") for probe in probes)
    return {
        "status": "PASSED" if passed else "FAILED",
        "method": "truncate-future-dates-and-recompute-prefix",
        "labelsRemovedBeforeExecution": True,
        "tolerance": float(tolerance),
        "probes": probes,
        "meaning": (
            "A causal factor's stored prefix must not change when all later "
            "observations are removed."
        ),
    }


def _annualized_curve(curve: Sequence[float] | None) -> float | None:
    values = np.asarray(curve or [], dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2 or values[0] <= 0.0:
        return None
    return float(
        (values[-1] / values[0]) ** (252.0 / max(values.size - 1, 1)) - 1.0
    )


def _engine_case(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    *,
    n_groups: int,
    costs: bool,
    allow_short: bool,
    n_trials: int,
    commission_rate: float | None = None,
    impact_coeff: float | None = None,
) -> dict[str, Any]:
    from quant_sdk.local_backtest import run_local_backtest

    result = run_local_backtest(
        "",
        frame,
        factor_matrix=matrix,
        n_groups=int(n_groups),
        costs=bool(costs),
        allow_short=bool(allow_short),
        commission_rate=commission_rate,
        impact_coeff=impact_coeff,
        n_trials=int(n_trials),
    )
    curves = result.get("curves") or {}
    strategy_annualized = _annualized_curve(curves.get("ls"))
    benchmark_annualized = _annualized_curve(curves.get("benchmark"))
    benchmark_is_fallback = any(
        "benchmark_ret=0" in str(item)
        for item in result.get("approximations") or []
    )
    benchmark_identity = result.get("benchmark") or {}
    benchmark_basis = str(benchmark_identity.get("basis") or "")
    benchmark_grade = str(benchmark_identity.get("grade") or "")
    if benchmark_is_fallback:
        benchmark_status = "FALLBACK_ZERO_NOT_EVALUATED"
    elif benchmark_grade == "MARKET" and benchmark_basis.startswith("next-open:"):
        benchmark_status = "OBSERVED"
    elif benchmark_grade == "EQUAL_WEIGHT_PROXY":
        benchmark_status = "OBSERVED_PROXY"
    elif benchmark_basis.startswith("close-to-close:"):
        benchmark_status = "OBSERVED_CLOSE_PROXY"
    else:
        benchmark_status = "OBSERVED_UNVERIFIED_BASIS"
    return {
        "ok": True,
        "metrics": result.get("metrics"),
        "icMean": result.get("ic_mean"),
        "turnoverMean": result.get("turnover_mean"),
        "friction": result.get("friction_breakdown"),
        "blockedOrders": result.get("blocked_orders"),
        "benchmark": {
            "status": benchmark_status,
            "basis": benchmark_basis or None,
            "grade": benchmark_grade or None,
            "strategyAnnualized": strategy_annualized,
            "benchmarkAnnualized": benchmark_annualized,
            "excessAnnualized": (
                strategy_annualized - benchmark_annualized
                if strategy_annualized is not None
                and benchmark_annualized is not None
                else None
            ),
        },
        "capacity": result.get("capacity"),
        "returns": {
            "dates": len(curves.get("dates") or []),
            "strategyFinal": (
                curves.get("ls")[-1] if curves.get("ls") else None
            ),
            "benchmarkFinal": (
                curves.get("benchmark")[-1]
                if curves.get("benchmark") else None
            ),
        },
    }


def audit_factor(
    *,
    data_path: str | Path,
    start: str,
    end: str,
    factor_name: str | None = None,
    source: str = "",
    name: str = "factor",
    selection_ratio: float = 0.7,
    purge_days: int = 1,
    embargo_days: int = 1,
    horizon: int = 1,
    min_names: int = 5,
    n_groups: int = 5,
    costs: bool = True,
    allow_short: bool = False,
    n_trials: int = 1,
    direction: int = 1,
    benchmark_code: str = "000300.SH",
    cut_quantiles: Sequence[float] = (0.55, 0.7, 0.85),
    tolerance: float = 1e-12,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Audit one local factor and return a portable evidence summary."""
    from quant_sdk.research_workflow import (
        _compact_date,
        _factor_matrix,
        _ic_summary,
        _load_frame,
        _rank_ic_daily,
        _window,
    )

    start = _compact_date(start, "start")
    end = _compact_date(end, "end")
    factors = str(factor_name or "").strip()
    source_text = str(source or "")
    try:
        factor_direction = int(direction)
    except (TypeError, ValueError) as error:
        raise ValueError("direction must be -1 or 1") from error
    if factor_direction not in {-1, 1}:
        raise ValueError("direction must be -1 or 1")
    if not factors and not source_text.strip():
        raise ValueError("audit requires factorName or source")
    static = _static_source_audit(source_text)
    if static["status"] == "FAILED":
        result = {
            "format": AUDIT_VERSION,
            "status": "REJECTED_LEAKAGE",
            "validated": False,
            "factor": {"name": str(name), "factorName": factors or None},
            "staticAudit": static,
            "runtimeCausality": {
                "status": "NOT_RUN",
                "reason": "source was rejected before execution",
            },
            "checks": {"leakagePassed": False},
            "submitted": False,
            "published": False,
            "note": "Static source audit failed; the factor was not executed.",
        }
        if output:
            result["output"] = str(_write_json(output, result))
        return _jsonable(result)

    frame = _load_frame(
        data_path, start, end, benchmark_code=str(benchmark_code)
    )
    split = research_split(
        frame["trade_date"].astype(str).tolist(),
        selection_ratio=selection_ratio,
        purge_days=purge_days,
        embargo_days=embargo_days,
    )
    values = _evaluate_values(
        factor_name=factors,
        source=source_text,
        frame=frame,
    )
    runtime = _causality_probes(
        frame=frame,
        values=values,
        factor_name=factors,
        source=source_text,
        cut_quantiles=cut_quantiles,
        tolerance=tolerance,
    )
    if runtime["status"] != "PASSED":
        result = {
            "format": AUDIT_VERSION,
            "status": "REJECTED_LEAKAGE",
            "validated": False,
            "factor": {"name": str(name), "factorName": factors or None},
            "window": {"start": start, "end": end},
            "split": split,
            "staticAudit": static,
            "runtimeCausality": runtime,
            "checks": {"leakagePassed": False},
            "submitted": False,
            "published": False,
            "note": (
                "Prefix recomputation changed stored factor values; the factor "
                "depends on future observations and was not promoted."
            ),
        }
        if output:
            result["output"] = str(_write_json(output, result))
        return _jsonable(result)

    oriented_values = np.asarray(values, dtype=float) * factor_direction
    matrix = _factor_matrix(frame, oriented_values)
    selection_frame, selection_matrix = _window(
        frame, matrix, split["selectionStart"], split["selectionEnd"]
    )
    holdout_frame, holdout_matrix = _window(
        frame, matrix, split["holdoutStart"], split["holdoutEnd"]
    )

    selection_ic = _ic_summary(_rank_ic_daily(
        selection_frame,
        _values_for_frame(selection_frame, frame, oriented_values),
        horizon=int(horizon),
        min_names=int(min_names),
    ))
    holdout_ic = _ic_summary(_rank_ic_daily(
        holdout_frame,
        _values_for_frame(holdout_frame, frame, oriented_values),
        horizon=int(horizon),
        min_names=int(min_names),
    ))

    selection_case = _engine_case(
        selection_frame,
        selection_matrix,
        n_groups=n_groups,
        costs=costs,
        allow_short=allow_short,
        n_trials=n_trials,
    )
    holdout_case = _engine_case(
        holdout_frame,
        holdout_matrix,
        n_groups=n_groups,
        costs=costs,
        allow_short=allow_short,
        n_trials=n_trials,
    )
    cost_sensitivity = []
    for label, enabled, commission, impact in (
        ("zero", False, None, None),
        ("normal", True, None, None),
        ("high", True, 0.0008, 0.25),
        ("severe", True, 0.0015, 0.50),
    ):
        case = _engine_case(
            holdout_frame,
            holdout_matrix,
            n_groups=n_groups,
            costs=enabled,
            allow_short=allow_short,
            n_trials=n_trials,
            commission_rate=commission,
            impact_coeff=impact,
        )
        cost_sensitivity.append({"name": label, **case})

    selection_mean = selection_ic.get("meanIc")
    holdout_mean = holdout_ic.get("meanIc")
    holdout_annualized = (
        holdout_case.get("metrics") or {}
    ).get("annualized")
    holdout_excess = (holdout_case.get("benchmark") or {}).get(
        "excessAnnualized"
    )
    benchmark_status = (holdout_case.get("benchmark") or {}).get("status")
    capacity_status = (holdout_case.get("capacity") or {}).get("status")
    checks = {
        "staticSourcePassed": static["status"] in {
            "PASSED", "NOT_APPLICABLE"
        },
        "runtimeCausalityPassed": runtime["status"] == "PASSED",
        "selectionObserved": bool(
            selection_mean is not None
            and int(selection_ic.get("days") or 0) >= 10
        ),
        "holdoutObserved": bool(
            holdout_mean is not None and int(holdout_ic.get("days") or 0) >= 10
        ),
        "holdoutSameDirection": bool(
            selection_mean is not None
            and holdout_mean is not None
            and float(selection_mean) != 0.0
            and float(holdout_mean) != 0.0
            and np.sign(float(selection_mean)) == np.sign(float(holdout_mean))
        ),
        "holdoutAbsoluteAfterCostPositive": bool(
            holdout_annualized is not None and float(holdout_annualized) > 0.0
        ),
        "holdoutExcessAfterCostPositive": bool(
            holdout_excess is not None and float(holdout_excess) > 0.0
        ),
        "benchmarkObserved": benchmark_status == "OBSERVED",
        "capacityObserved": capacity_status == "OK",
    }
    checks["leakagePassed"] = bool(
        checks["staticSourcePassed"] and checks["runtimeCausalityPassed"]
    )
    validation_checks = {
        key: value
        for key, value in checks.items()
        if key != "holdoutAbsoluteAfterCostPositive"
    }
    validated = all(validation_checks.values())
    result = {
        "format": AUDIT_VERSION,
        "status": "LOCAL_PROOF_OF_ALPHA" if validated else "RESEARCH_ONLY",
        "validated": validated,
        "executionBoundary": "local",
        "factor": {
            "name": str(name),
            "factorName": factors or None,
            "sourceProvided": bool(source_text.strip()),
            "direction": int(factor_direction),
        },
        "window": {"start": start, "end": end},
        "split": split,
        "configuration": {
            "horizon": int(horizon),
            "label": label_contract(frame, int(horizon)),
            "minNames": int(min_names),
            "nGroups": int(n_groups),
            "costs": bool(costs),
            "allowShort": bool(allow_short),
            "nTrials": int(n_trials),
            "direction": int(factor_direction),
            "benchmarkCode": (
                (frame.attrs.get("benchmarkResolution") or {}).get("code")
                or str(benchmark_code)
            ),
            "benchmarkResolution": frame.attrs.get("benchmarkResolution"),
        },
        "pointInTime": {
            "status": "OBSERVED",
            "signalTime": "session-close:t",
            "executionTime": "open:t+1",
            "exitTime": f"open:t+1+{int(horizon)}",
            "label": label_contract(frame, int(horizon)),
            "dataWindow": {"start": start, "end": end},
        },
        "staticAudit": static,
        "runtimeCausality": runtime,
        "outOfSample": {
            "selection": selection_ic,
            "holdout": holdout_ic,
            "selectionCase": selection_case,
            "holdoutCase": holdout_case,
            "protocol": (
                "Chronological split; selection ends before purge buffer and "
                "holdout begins after embargo buffer."
            ),
        },
        "costSensitivity": cost_sensitivity,
        "capacity": holdout_case.get("capacity"),
        "benchmark": holdout_case.get("benchmark"),
        "checks": checks,
        "validationChecks": validation_checks,
        "submitted": False,
        "published": False,
        "disclosure": {
            "sourceIncludedInAudit": bool(source_text.strip()),
            "dataUploaded": False,
            "automaticSubmission": False,
        },
        "note": (
            "LOCAL_PROOF_OF_ALPHA means this candidate passed the declared "
            "causality, purged/embargoed holdout, cost, benchmark and capacity "
            "checks on this data slice. It is not an official ranking or an "
            "investment recommendation."
        ),
    }
    if output:
        result["output"] = str(_write_json(output, result))
    return _jsonable(result)


def _values_for_frame(
    subset: pd.DataFrame,
    full: pd.DataFrame,
    full_values: np.ndarray,
) -> np.ndarray:
    keyed = _values_frame(full, full_values)
    requested = subset[["ts_code", "trade_date"]].copy()
    requested["ts_code"] = requested["ts_code"].astype(str)
    requested["trade_date"] = requested["trade_date"].astype(str).str.replace(
        "-", "", regex=False
    )
    requested["_order"] = np.arange(len(requested))
    merged = requested.merge(
        keyed,
        on=["ts_code", "trade_date"],
        how="left",
        validate="one_to_one",
    ).sort_values("_order", kind="mergesort")
    return merged["value"].to_numpy(dtype=float)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a local factor candidate")
    parser.add_argument("--data", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--factor-name")
    parser.add_argument("--source-file")
    parser.add_argument("--source", default="")
    parser.add_argument("--name", default="factor")
    parser.add_argument("--selection-ratio", type=float, default=0.7)
    parser.add_argument("--purge-days", type=int, default=1)
    parser.add_argument("--embargo-days", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--min-names", type=int, default=5)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--no-costs", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--benchmark-code", default="000300.SH")
    parser.add_argument("--cut-quantiles", default="0.55,0.7,0.85")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    source = args.source
    if args.source_file:
        source = Path(args.source_file).expanduser().read_text(encoding="utf-8")
    quantiles = [
        float(item.strip())
        for item in str(args.cut_quantiles).split(",")
        if item.strip()
    ]
    result = audit_factor(
        data_path=args.data,
        start=args.start,
        end=args.end,
        factor_name=args.factor_name,
        source=source,
        name=args.name,
        selection_ratio=args.selection_ratio,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
        horizon=args.horizon,
        min_names=args.min_names,
        n_groups=args.groups,
        costs=not args.no_costs,
        allow_short=bool(args.allow_short),
        n_trials=args.trials,
        direction=args.direction,
        benchmark_code=args.benchmark_code,
        cut_quantiles=quantiles,
        tolerance=args.tolerance,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] != "REJECTED_LEAKAGE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
