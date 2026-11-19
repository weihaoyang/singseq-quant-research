"""Local factor-research workflow helpers.

These helpers provide the missing product-level workflow around the canonical
engine.  Every operation reads researcher-owned data on this machine and writes
results beside the local job workspace.  Nothing here uploads, submits, ranks a
competition entry, or mutates a production service.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_sdk.labels import canonical_forward_returns

WORKFLOW_VERSION = "local-research-workflow-1"
REQUIRED_COLUMNS = ("ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount")
OPTIONAL_COLUMNS = (
    "adj_close", "adj_factor", "benchmark_ret", "up_limit", "down_limit", "st",
    "list_date", "industry_id", "circ_mv", "pool_membership", "shortable",
    "borrow_rate", "cash_dividend", "turnover_rate", "pb",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    return target


def _compact_date(value: str, label: str) -> str:
    text = str(value or "").strip().replace("-", "")
    if not re.fullmatch(r"\d{8}", text):
        raise ValueError(f"{label} must use YYYYMMDD or YYYY-MM-DD")
    return text


def _load_frame(data_path: str | Path, start: str, end: str, *,
                benchmark_code: str = "000300.SH") -> pd.DataFrame:
    path = Path(data_path).expanduser()
    if not path.is_file():
        raise ValueError(f"research data does not exist: {path}")
    from quant_sdk.benchmark_resolver import resolve_benchmark, sample_symbols

    benchmark = resolve_benchmark(
        sample_symbols(path, start=start, end=end),
        explicit=benchmark_code,
    )
    effective_benchmark = str(benchmark.get("code") or "")
    if path.suffix.lower() in {".duckdb", ".db"}:
        from quant_sdk.local_backtest import load_panel

        frame = load_panel(
            path, start, end, benchmark_code=effective_benchmark,
        )
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("research data must be CSV, Parquet, DuckDB, or DB")
    if frame.empty:
        raise ValueError("research data is empty for the requested window")
    for column in ("ts_code", "trade_date"):
        if column not in frame.columns:
            raise ValueError(f"research data is missing identity column: {column}")
    work = frame.copy()
    work["ts_code"] = work["ts_code"].astype(str)
    work["trade_date"] = work["trade_date"].astype(str).str.replace("-", "", regex=False)
    work = work[work["trade_date"].between(str(start), str(end))]
    if work.empty:
        raise ValueError("research data has no rows in the requested window")
    work = work.sort_values(
        ["ts_code", "trade_date"], kind="mergesort",
    ).reset_index(drop=True)
    work.attrs["benchmarkResolution"] = benchmark
    return work


def _relation_for_duckdb(connection, path: Path) -> tuple[str, list[str]]:
    tables = [str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()]
    if path.suffix.lower() not in {".duckdb", ".db"}:
        if path.suffix.lower() == ".parquet":
            escaped = str(path).replace("'", "''")
            return f"read_parquet('{escaped}')", tables
        if path.suffix.lower() == ".csv":
            escaped = str(path).replace("'", "''")
            return f"read_csv_auto('{escaped}', sample_size=20000)", tables
    if not tables:
        raise ValueError("database contains no tables")
    if "daily_raw" in tables:
        return '"daily_raw"', tables
    for table in tables:
        quoted = '"' + table.replace('"', '""') + '"'
        columns = {str(row[0]) for row in connection.execute(f"DESCRIBE {quoted}").fetchall()}
        if {"ts_code", "trade_date"}.issubset(columns):
            return quoted, tables
    quoted = '"' + tables[0].replace('"', '""') + '"'
    return quoted, tables


def _table_coverage(connection, tables: Sequence[str], start: str, end: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for table in list(tables)[:40]:
        quoted = '"' + str(table).replace('"', '""') + '"'
        try:
            columns = {str(row[0]): str(row[1]) for row in connection.execute(f"DESCRIBE {quoted}").fetchall()}
            row: dict[str, Any] = {"name": str(table), "columns": len(columns)}
            if "trade_date" in columns:
                row["rows"], row["min_date"], row["max_date"] = connection.execute(
                    f"SELECT count(*), min(replace(CAST(trade_date AS VARCHAR), '-', '')), "
                    f"max(replace(CAST(trade_date AS VARCHAR), '-', '')) FROM {quoted} "
                    "WHERE replace(CAST(trade_date AS VARCHAR), '-', '') BETWEEN ? AND ?",
                    [start, end],
                ).fetchone()
            else:
                row["rows"] = int(connection.execute(f"SELECT count(*) FROM {quoted}").fetchone()[0])
            result.append(row)
        except Exception as error:  # pragma: no cover - diagnostic table may be locked/unreadable
            result.append({"name": str(table), "error": f"{type(error).__name__}: {error}"})
    return result


def inspect_data(data_path: str | Path, *, start: str, end: str,
                 benchmark_code: str = "000300.SH") -> dict[str, Any]:
    """Summarise a local panel without materialising its full contents."""
    import duckdb

    path = Path(data_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"research data does not exist: {path}")
    start = _compact_date(start, "start")
    end = _compact_date(end, "end")
    if start > end:
        raise ValueError("start must not be after end")

    connection = duckdb.connect(str(path), read_only=True) if path.suffix.lower() in {".duckdb", ".db"} else duckdb.connect()
    try:
        relation, tables = _relation_for_duckdb(connection, path)
        date_text = "replace(CAST(trade_date AS VARCHAR), '-', '')"
        try:
            symbols = [
                str(row[0])
                for row in connection.execute(
                    f"SELECT DISTINCT ts_code FROM {relation} "
                    f"WHERE {date_text} BETWEEN ? AND ? "
                    "ORDER BY ts_code LIMIT 200",
                    [start, end],
                ).fetchall()
                if str(row[0]).strip()
            ]
        except Exception:
            symbols = []
        from quant_sdk.benchmark_resolver import resolve_benchmark

        benchmark = resolve_benchmark(symbols, explicit=benchmark_code)
        described = connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
        columns = [{"name": str(row[0]), "type": str(row[1])} for row in described]
        names = {item["name"] for item in columns}
        result: dict[str, Any] = {
            "version": WORKFLOW_VERSION,
            "status": "INSPECTED",
            "dataPath": str(path),
            "window": {"start": start, "end": end},
            "tables": _table_coverage(connection, tables, start, end) if tables else [],
            "columns": columns,
            "requiredColumns": list(REQUIRED_COLUMNS),
            "missingRequiredColumns": sorted(set(REQUIRED_COLUMNS) - names),
            "optionalColumnsPresent": [name for name in OPTIONAL_COLUMNS if name in names],
            "benchmarkCode": benchmark.get("code"),
            "benchmarkResolution": benchmark,
        }
        if not {"ts_code", "trade_date"}.issubset(names):
            result.update({
                "status": "NOT_A_PANEL",
                "readyForBacktest": False,
                "warnings": ["数据缺少 ts_code 或 trade_date，不能作为本地因子研究面板。"],
            })
            return _jsonable(result)

        where = f"WHERE {date_text} BETWEEN ? AND ?"
        rows, codes, dates, min_date, max_date = connection.execute(
            f"SELECT count(*), count(DISTINCT ts_code), count(DISTINCT CAST(trade_date AS VARCHAR)), "
            f"min({date_text}), max({date_text}) "
            f"FROM {relation} {where}",
            [start, end],
        ).fetchone()
        result["coverage"] = {
            "rows": int(rows or 0),
            "stocks": int(codes or 0),
            "dates": int(dates or 0),
            "minDate": str(min_date) if min_date is not None else None,
            "maxDate": str(max_date) if max_date is not None else None,
        }

        missing: dict[str, float] = {}
        for column in [name for name in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS) if name in names]:
            quoted = '"' + column.replace('"', '""') + '"'
            if column in {"ts_code", "trade_date", "industry_id", "list_date", "st", "pool_membership", "shortable"}:
                numerator = f"count(*) FILTER (WHERE {quoted} IS NULL)"
            else:
                numerator = (
                    f"count(*) FILTER (WHERE {quoted} IS NULL OR "
                    f"TRY_CAST({quoted} AS DOUBLE) IS NULL OR "
                    f"NOT isfinite(TRY_CAST({quoted} AS DOUBLE)))"
                )
            value = connection.execute(
                f"SELECT {numerator}::DOUBLE / greatest(count(*), 1) FROM {relation} {where}",
                [start, end],
            ).fetchone()[0]
            missing[column] = round(float(value or 0.0), 6)
        result["missingRatio"] = missing

        can_engine = set(REQUIRED_COLUMNS).issubset(names) and int(dates or 0) >= 3 and int(codes or 0) >= 10
        adjusted = (
            ("adj_close" in names and missing.get("adj_close", 1.0) < 1.0)
            or ("adj_factor" in names and missing.get("adj_factor", 1.0) < 1.0)
        )
        benchmark = (
            ("benchmark_ret" in names and missing.get("benchmark_ret", 1.0) < 1.0)
            or "index_daily" in tables
        )
        neutralization = {"industry_id", "circ_mv"}.issubset(names)
        limits = {"up_limit", "down_limit"}.issubset(names)
        result["capabilities"] = {
            "canonicalBacktest": bool(can_engine),
            "adjustedPriceReference": bool(adjusted),
            "benchmark": bool(benchmark),
            "industryAndSizeNeutralization": bool(neutralization),
            "historicalPriceLimits": bool(limits),
            "pointInTimeSt": "st" in names,
            "shortingInventory": "shortable" in names,
            "marketCap": "circ_mv" in names,
        }
        result["readyForBacktest"] = bool(can_engine)
        warnings: list[str] = []
        if not can_engine:
            warnings.append("回测至少需要完整 OHLCV/amount、3 个交易日和 10 只股票。")
        if not adjusted:
            warnings.append("没有可用的复权价格参考；结果会明确记录 close 近似。")
        if not benchmark:
            warnings.append("没有 benchmark_ret 或 index_daily；本地回测会使用标记为 0 收益的基准代理。")
        if not neutralization:
            warnings.append("没有 industry_id + circ_mv，无法执行行业与市值中性化。")
        if not limits:
            warnings.append("没有历史涨跌停列，引擎会按收盘价近似。")
        result["warnings"] = warnings
        result["status"] = "READY" if can_engine else "INCOMPLETE"
        return _jsonable(result)
    finally:
        connection.close()


def _normalize_candidates(
    candidates: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            raise ValueError(f"candidate {index + 1} must be an object")
        name = str(item.get("name") or item.get("candidate") or "").strip()
        source = str(item.get("source") or "").strip()
        factor_name = str(item.get("factorName") or item.get("factor_name") or "").strip()
        try:
            direction = int(item.get("direction", 1))
        except (TypeError, ValueError) as error:
            raise ValueError(f"candidate {name} direction must be -1 or 1") from error
        if direction not in {-1, 1}:
            raise ValueError(f"candidate {name} direction must be -1 or 1")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,79}", name):
            raise ValueError(f"candidate {index + 1} has an invalid name")
        if name in seen:
            raise ValueError(f"duplicate candidate name: {name}")
        if not source and not factor_name:
            raise ValueError(f"candidate {name} is missing source or factorName")
        seen.add(name)
        result.append({
            "name": name,
            "source": source,
            "direction": direction,
            **({"factorName": factor_name} if factor_name else {}),
        })
    if not 1 <= len(result) <= 30:
        raise ValueError("batch must contain 1 to 30 candidates")
    return result


def _factor_matrix(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    dates = sorted(frame["trade_date"].astype(str).unique())
    codes = sorted(frame["ts_code"].astype(str).unique())
    rows = frame[["trade_date", "ts_code"]].copy()
    rows["trade_date"] = rows["trade_date"].astype(str)
    rows["ts_code"] = rows["ts_code"].astype(str)
    rows["factor_value"] = np.asarray(values, dtype=float)
    return (
        rows.pivot(index="trade_date", columns="ts_code", values="factor_value")
        .reindex(index=dates, columns=codes)
        .to_numpy(dtype=float)
    )


def _compact_backtest(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status", "dates", "codes", "metrics", "ic_mean", "turnover_mean",
        "friction_breakdown", "blocked_orders", "approximations", "novelty",
        "scoring", "tearsheet_metrics", "style_exposure", "attribution",
        "capacity",
    )
    return {key: result.get(key) for key in keys if key in result}


def batch_evaluate(*, data_path: str, start: str, end: str,
                   candidates: Iterable[dict[str, Any]],
                   out: str = "batch.json", n_groups: int = 5,
                   costs: bool = True, allow_short: bool = False,
                   neutralize: bool = False, novelty: bool = False,
                   n_trials: int | None = None, reports: bool = False,
                   benchmark_code: str = "000300.SH",
                   backend: str = "auto") -> dict[str, Any]:
    """Evaluate multiple candidate sources against one local panel."""
    from quant_sdk.local_backtest import run_local_backtest
    from quant_sdk.research import _compute_factor_series

    start = _compact_date(start, "start")
    end = _compact_date(end, "end")
    items = _normalize_candidates(candidates)
    frame = _load_frame(data_path, start, end, benchmark_code=benchmark_code)
    factor_dir = Path(out).parent / "factors"
    factor_dir.mkdir(parents=True, exist_ok=True)
    trials = int(n_trials if n_trials is not None else len(items))
    rows: list[dict[str, Any]] = []
    for item in items:
        try:
            if item.get("factorName"):
                from quant_sdk.catalog_factors import compute_named_factor
                values = compute_named_factor(item["factorName"], frame).to_numpy(dtype=float)
            else:
                values = _compute_factor_series(item["source"], frame)
            values = np.asarray(values, dtype=float) * int(item["direction"])
            matrix = _factor_matrix(frame, values)
            report_path = factor_dir / f"{item['name']}.html" if reports else None
            result = run_local_backtest(
                item["source"], frame, factor_matrix=matrix,
                n_groups=n_groups, costs=costs, allow_short=allow_short,
                neutralize=neutralize, with_novelty=novelty,
                n_trials=trials, backend=backend,
                tearsheet_path=str(report_path) if report_path else None,
            )
            csv_path = factor_dir / f"{item['name']}.csv"
            factor_rows = frame[["ts_code", "trade_date"]].copy()
            factor_rows["factor_value"] = values
            factor_rows.to_csv(csv_path, index=False)
            summary = _compact_backtest(result)
            rank_score = (
                result.get("scoring", {}).get("final_score")
                if result.get("scoring") else abs(result.get("ic_mean") or 0.0)
            )
            rows.append({
                "name": item["name"],
                "source": item.get("source", ""),
                "factorName": item.get("factorName"),
                "direction": int(item["direction"]),
                "ok": True,
                "rankScore": rank_score,
                "factorRows": int(np.isfinite(values).sum()),
                "factorCsv": str(csv_path),
                "report": str(report_path) if report_path else None,
                "summary": summary,
            })
        except Exception as error:
            rows.append({
                "name": item["name"], "source": item.get("source", ""), "factorName": item.get("factorName"), "ok": False,
                "error": f"{type(error).__name__}: {error}",
            })
    rows.sort(key=lambda row: (
        not row.get("ok"),
        -(float(row["rankScore"]) if row.get("rankScore") is not None else -math.inf),
        row["name"],
    ))
    result = {
        "version": WORKFLOW_VERSION,
        "status": "LOCAL_UNVERIFIED",
        "dataPath": str(Path(data_path).expanduser().resolve()),
        "window": {"start": start, "end": end},
        "panel": {"rows": len(frame), "dates": int(frame["trade_date"].nunique()),
                  "stocks": int(frame["ts_code"].nunique())},
        "configuration": {
            "nGroups": n_groups, "costs": costs, "allowShort": allow_short,
            "neutralize": neutralize, "novelty": novelty, "nTrials": trials,
            "backend": backend,
        },
        "count": len(rows),
        "succeeded": sum(bool(row.get("ok")) for row in rows),
        "candidates": rows,
        "note": "本地批量筛选；入选候选仍需单独看报告，不自动保存、提交或发布。",
    }
    _write_json(out, result)
    return result


def factor_names(*, data_path: str, start: str, end: str, query: str = "",
                 prefix: str = "", limit: int = 200,
                 benchmark_code: str = "000300.SH") -> dict[str, Any]:
    """List implementation-backed factor names for one local data panel."""
    from collections import Counter

    from quant_sdk.catalog_factors import runnable_factor_names, runnable_factor_specs

    start = _compact_date(start, "start")
    end = _compact_date(end, "end")
    limit = int(limit)
    if limit < 0 or limit > 5000:
        raise ValueError("limit must be between 0 and 5000")
    frame = _load_frame(data_path, start, end, benchmark_code=benchmark_code)
    all_names = runnable_factor_names(frame)
    needle = str(query or "").strip().lower()
    prefix_key = str(prefix or "").strip().lower()
    names = [
        name for name in all_names
        if (not needle or needle in name.lower())
        and (not prefix_key or name.lower().startswith(prefix_key))
    ]
    selected = names if limit == 0 else names[:limit]
    specs = runnable_factor_specs(frame, names=selected)
    by_name = {item["name"]: item for item in specs}
    manifest_path = Path(__file__).with_name("factor_capability_manifest.json")
    try:
        raw_manifest = manifest_path.read_bytes()
        manifest = json.loads(raw_manifest.decode("utf-8"))
        catalog = {
            "catalogId": manifest.get("catalogId"),
            "catalogSha256": hashlib.sha256(raw_manifest).hexdigest(),
            "registeredCount": int(manifest.get("count") or 0),
            "executionValidatedCount": sum(
                1 for item in manifest.get("factors") or []
                if item.get("executionValidated") is not False
            ),
        }
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        catalog = {
            "catalogId": None,
            "catalogSha256": None,
            "registeredCount": None,
            "executionValidatedCount": None,
        }

    def family(name: str) -> str:
        lower = name.lower()
        for value in (
            "alpha101", "alpha191", "alpha360", "talib", "mytt",
            "pta", "qtl", "ta", "talipp", "pyti",
        ):
            if lower.startswith(value + "_"):
                return value
        if lower.startswith(("alpha158_", "k_", "roc_", "ma_", "std_", "corr_")):
            return "qlib_alpha158"
        return name.split("_", 1)[0].lower()

    return {
        "status": "LOCAL_READY",
        "catalog": catalog,
        "dataPath": str(Path(data_path).expanduser().resolve()),
        "window": {"start": start, "end": end},
        "panel": {
            "rows": len(frame),
            "dates": int(frame["trade_date"].nunique()),
            "stocks": int(frame["ts_code"].nunique()),
        },
        "query": str(query or ""),
        "prefix": str(prefix or ""),
        "total": len(all_names),
        "matched": len(names),
        "returned": len(selected),
        "families": dict(sorted(Counter(family(name) for name in names).items())),
        "names": selected,
        "factors": [by_name[name] for name in selected if name in by_name],
        "note": "factors 给出每个入口的输入字段、函数来源和输出；均为用户本机计算。",
    }


def _rank_ic_daily(frame: pd.DataFrame, values: Sequence[float], *,
                   horizon: int, min_names: int) -> pd.DataFrame:
    work = frame.copy()
    work["factor_value"] = np.asarray(values, dtype=float)
    work["target_ret"] = canonical_forward_returns(work, int(horizon))
    work["stock_ret"] = work.groupby("ts_code", sort=False)["close"].pct_change(fill_method=None)
    market_return = work.groupby("trade_date", sort=False)["stock_ret"].mean()
    frame_ic = work[["trade_date", "factor_value", "target_ret"]].dropna().copy()
    if frame_ic.empty:
        return pd.DataFrame(columns=["trade_date", "rank_ic", "names", "market_return", "state"])
    frame_ic["xr"] = frame_ic.groupby("trade_date", sort=False)["factor_value"].rank(pct=True)
    frame_ic["yr"] = frame_ic.groupby("trade_date", sort=False)["target_ret"].rank(pct=True)
    frame_ic["xy"] = frame_ic["xr"] * frame_ic["yr"]
    frame_ic["x2"] = frame_ic["xr"] ** 2
    frame_ic["y2"] = frame_ic["yr"] ** 2
    grouped = frame_ic.groupby("trade_date", sort=False)
    means = grouped[["xr", "yr", "xy", "x2", "y2"]].mean()
    counts = grouped.size()
    covariance = means["xy"] - means["xr"] * means["yr"]
    variance_x = (means["x2"] - means["xr"] ** 2).clip(lower=0.0)
    variance_y = (means["y2"] - means["yr"] ** 2).clip(lower=0.0)
    ic = (covariance / np.sqrt(variance_x * variance_y).replace(0.0, np.nan)).dropna()
    ic = ic[counts.reindex(ic.index).fillna(0) >= min_names]
    output = pd.DataFrame({"trade_date": ic.index.astype(str), "rank_ic": ic.to_numpy()})
    output["names"] = counts.reindex(ic.index).to_numpy(dtype=int)
    output["market_return"] = output["trade_date"].map(market_return)
    output["state"] = np.select(
        [output["market_return"] > 0.001, output["market_return"] < -0.001],
        ["UP", "DOWN"], default="FLAT",
    )
    return output.sort_values("trade_date").reset_index(drop=True)


def _ic_summary(daily: pd.DataFrame) -> dict[str, Any]:
    values = pd.to_numeric(daily.get("rank_ic"), errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values):
        return {"days": 0, "meanIc": None, "icir": None, "positiveShare": None}
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "days": len(values),
        "meanIc": float(values.mean()),
        "icir": float(values.mean() / std) if std > 0 else None,
        "positiveShare": float((values > 0).mean()),
        "tStat": float(values.mean() / std * math.sqrt(len(values))) if std > 0 else None,
    }


def _window(frame: pd.DataFrame, matrix: np.ndarray, start: str | None,
            end: str | None) -> tuple[pd.DataFrame, np.ndarray]:
    dates = sorted(frame["trade_date"].astype(str).unique())
    codes = sorted(frame["ts_code"].astype(str).unique())
    selected = [index for index, date in enumerate(dates)
                if (start is None or date >= start) and (end is None or date <= end)]
    if not selected:
        raise ValueError("requested robustness window is empty")
    selected_dates = {dates[index] for index in selected}
    subset = frame[frame["trade_date"].astype(str).isin(selected_dates)].reset_index(drop=True)
    subset_codes = sorted(subset["ts_code"].astype(str).unique())
    code_positions = [codes.index(code) for code in subset_codes]
    return subset, matrix[np.ix_(np.asarray(selected), np.asarray(code_positions))]


def _engine_case(label: str, frame: pd.DataFrame, matrix: np.ndarray, *,
                 costs: bool, commission_rate: float | None = None,
                 impact_coeff: float | None = None, neutralize: bool = False,
                 n_groups: int = 5, allow_short: bool = False,
                 n_trials: int = 1, novelty: bool = False,
                 backend: str = "auto") -> dict[str, Any]:
    from quant_sdk.local_backtest import run_local_backtest

    try:
        result = run_local_backtest(
            "", frame, factor_matrix=matrix, n_groups=n_groups, costs=costs,
            commission_rate=commission_rate, impact_coeff=impact_coeff,
            neutralize=neutralize, allow_short=allow_short, n_trials=n_trials,
            with_novelty=novelty, backend=backend,
        )
        return {"name": label, "ok": True, "metrics": result.get("metrics"),
                "icMean": result.get("ic_mean"), "turnoverMean": result.get("turnover_mean"),
                "friction": result.get("friction_breakdown"),
                "blockedOrders": result.get("blocked_orders"),
                "novelty": result.get("novelty"), "scoring": result.get("scoring"),
                "accelerationMode": result.get("acceleration_mode")}
    except Exception as error:
        return {"name": label, "ok": False, "error": f"{type(error).__name__}: {error}"}


def robustness(*, data_path: str, start: str, end: str, source: str = "",
               factor_name: str | None = None,
               name: str = "factor", out: str = "robustness.json",
               horizon: int = 1, split_ratio: float = 0.7,
               min_names: int = 20, n_groups: int = 5,
               allow_short: bool = False, costs: bool = True,
               neutralize: bool = False, n_trials: int = 1,
               benchmark_code: str = "000300.SH",
               backend: str = "auto") -> dict[str, Any]:
    """Run local OOS, cost, regime and yearly robustness diagnostics."""
    from quant_sdk.research import _compute_factor_series

    start = _compact_date(start, "start")
    end = _compact_date(end, "end")
    if not 0.1 <= float(split_ratio) <= 0.9:
        raise ValueError("split_ratio must be between 0.1 and 0.9")
    frame = _load_frame(data_path, start, end, benchmark_code=benchmark_code)
    named = str(factor_name or "").strip()
    if named:
        from quant_sdk.catalog_factors import compute_named_factor

        values = compute_named_factor(named, frame).to_numpy(dtype=float)
    elif str(source).strip():
        values = _compute_factor_series(str(source), frame)
    else:
        raise ValueError("robustness requires source or factor_name")
    matrix = _factor_matrix(frame, values)
    dates = sorted(frame["trade_date"].astype(str).unique())
    split_index = min(len(dates) - 1, max(1, int(len(dates) * float(split_ratio))))
    split_date = dates[split_index]
    train_frame, train_matrix = _window(frame, matrix, None, dates[split_index - 1])
    oos_frame, oos_matrix = _window(frame, matrix, split_date, None)

    daily_ic = _rank_ic_daily(frame, values, horizon=int(horizon), min_names=int(min_names))
    yearly = {
        str(year): _ic_summary(group)
        for year, group in daily_ic.assign(year=daily_ic["trade_date"].str[:4]).groupby("year")
    } if not daily_ic.empty else {}
    regimes = {
        str(state): _ic_summary(group)
        for state, group in daily_ic.groupby("state") if not group.empty
    } if not daily_ic.empty else {}

    engine_cases = [
        _engine_case("full", frame, matrix, costs=costs, neutralize=neutralize,
                     n_groups=n_groups, allow_short=allow_short, n_trials=n_trials,
                     novelty=True, backend=backend),
        _engine_case("selection", train_frame, train_matrix, costs=costs,
                     neutralize=neutralize, n_groups=n_groups,
                     allow_short=allow_short, n_trials=n_trials,
                     backend=backend),
        _engine_case("out_of_sample", oos_frame, oos_matrix, costs=costs,
                     neutralize=neutralize, n_groups=n_groups,
                     allow_short=allow_short, n_trials=n_trials,
                     backend=backend),
    ]
    cost_sensitivity = []
    for label, enabled, commission, impact in (
        ("zero", False, None, None),
        ("normal", True, None, None),
        ("high", True, 0.0008, 0.25),
        ("severe", True, 0.0015, 0.50),
    ):
        case = _engine_case(
            label, frame, matrix, costs=enabled, commission_rate=commission,
            impact_coeff=impact, neutralize=neutralize, n_groups=n_groups,
            allow_short=allow_short, n_trials=n_trials, backend=backend,
        )
        cost_sensitivity.append(case)

    if {"industry_id", "circ_mv"}.issubset(frame.columns):
        neutralization = _engine_case(
            "industry_size_neutralized", frame, matrix, costs=costs,
            neutralize=True, n_groups=n_groups, allow_short=allow_short,
            n_trials=n_trials, backend=backend,
        )
    else:
        neutralization = {
            "name": "industry_size_neutralized", "ok": False,
            "error": "industry_id and circ_mv are required",
        }

    result = {
        "version": WORKFLOW_VERSION,
        "status": "LOCAL_UNVERIFIED",
        "factor": str(name),
        "source": str(source),
        "factorName": named or None,
        "dataPath": str(Path(data_path).expanduser().resolve()),
        "window": {"start": start, "end": end},
        "split": {"ratio": float(split_ratio), "date": split_date},
        "configuration": {"backend": backend},
        "icStability": {
            "overall": _ic_summary(daily_ic),
            "yearly": yearly,
            "marketState": regimes,
            "stateDefinition": "same-day equal-weight close-to-close market return; UP > 10bp, DOWN < -10bp",
        },
        "engineCases": engine_cases,
        "costSensitivity": cost_sensitivity,
        "neutralization": neutralization,
        "seedStability": {
            "status": "NOT_APPLICABLE",
            "reason": "canonical local backtest is deterministic; randomness only applies to model training",
        },
        "note": "直接诊断结果，不是强制门禁文件；本地结果仍为 LOCAL_UNVERIFIED。",
    }
    _write_json(out, result)
    csv = Path(out).parent / f"{name}.csv"
    output = frame[["ts_code", "trade_date"]].copy()
    output["factor_value"] = values
    output.to_csv(csv, index=False)
    result["factorCsv"] = str(csv)
    return _jsonable(result)


def local_portfolio(*, data_path: str, start: str, end: str,
                    factors: Iterable[dict[str, Any]], out: str = "portfolio.json",
                    objective: str = "max_sharpe", long_only: bool = False,
                    max_weight: float = 0.5, max_gross: float = 1.0,
                    horizon: int = 1, min_names: int = 20,
                    participation: float = 0.1,
                    n_groups: int = 5, costs: bool = True,
                    allow_short: bool = False, neutralize: bool = False,
                    benchmark_code: str = "000300.SH",
                    backend: str = "auto") -> dict[str, Any]:
    """Optimise weights over local factor sources and replay the combined score."""
    from evaluation import portfolio
    from quant_sdk.local_backtest import run_local_backtest
    from quant_sdk.research import _compute_factor_series

    start = _compact_date(start, "start")
    end = _compact_date(end, "end")
    items = _normalize_candidates(factors)
    if len(items) < 2:
        raise ValueError("local portfolio requires at least two factors")
    frame = _load_frame(data_path, start, end, benchmark_code=benchmark_code)
    values = {}
    for item in items:
        if item.get("factorName"):
            from quant_sdk.catalog_factors import compute_named_factor

            values[item["name"]] = compute_named_factor(item["factorName"], frame).to_numpy(
                dtype=float
            )
        else:
            values[item["name"]] = _compute_factor_series(item["source"], frame)
    factor_returns = portfolio.factor_return_series(
        frame, values, horizon=int(horizon), min_names=int(min_names),
    )
    optimized = portfolio.optimize(
        factor_returns, objective=str(objective), long_only=bool(long_only),
        max_weight=float(max_weight), max_gross=float(max_gross),
    )
    weights = optimized["weights"]
    covariance = portfolio.risk_covariance(factor_returns)
    diversification = portfolio.diversification_ratio(weights, covariance)
    capacity = portfolio.capacity(
        frame, weights, participation=float(participation), turnover=1.0,
    )
    normalized = {}
    for name, raw in values.items():
        series = pd.Series(raw, index=frame.index, dtype=float)
        grouped = series.groupby(frame["trade_date"])
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0.0, np.nan)
        normalized[name] = ((series - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    combined = sum(normalized[name] * float(weights.get(name, 0.0)) for name in normalized)
    matrix = _factor_matrix(frame, combined.to_numpy(dtype=float))
    backtest = run_local_backtest(
        "", frame, factor_matrix=matrix, n_groups=n_groups, costs=costs,
        allow_short=allow_short, neutralize=neutralize, with_novelty=True,
        n_trials=len(items), backend=backend,
    )
    result = {
        "version": WORKFLOW_VERSION,
        "status": "LOCAL_UNVERIFIED",
        "dataPath": str(Path(data_path).expanduser().resolve()),
        "window": {"start": start, "end": end},
        "configuration": {"backend": backend},
        "factors": [
            {
                "name": item["name"],
                "source": item["source"],
                **({"factorName": item["factorName"]} if item.get("factorName") else {}),
            }
            for item in items
        ],
        "optimization": optimized,
        "factorReturns": {
            "dates": len(factor_returns),
            "annualizedVolatility": {
                name: float(np.sqrt(max(covariance.loc[name, name], 0.0)))
                for name in covariance.index
            },
        },
        "diversificationRatio": diversification,
        "capacity": capacity,
        "combinedBacktest": _compact_backtest(backtest),
        "note": "组合权重来自本地因子收益代理；combinedBacktest 为 canonical 本地回放，仍非官方结果。",
    }
    _write_json(out, result)
    return _jsonable(result)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="quant-sdk workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect")
    inspect.add_argument("--data", required=True)
    inspect.add_argument("--start", required=True)
    inspect.add_argument("--end", required=True)
    inspect.add_argument("--benchmark-code", default="000300.SH")
    inspect.add_argument("--out")

    args = parser.parse_args(argv)
    if args.command == "inspect":
        result = inspect_data(args.data, start=args.start, end=args.end,
                              benchmark_code=args.benchmark_code)
        if args.out:
            _write_json(args.out, result)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
