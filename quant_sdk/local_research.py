"""Small, reproducible local factor runner shipped in the research toolkit.

It reads a researcher-owned CSV/Parquet panel, calls ``compute_factor(df)``,
and writes the canonical factor-result CSV. It never uploads data or results.
"""
from __future__ import annotations

import argparse
import importlib.util
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd


def load_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path).expanduser()
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    return pd.read_csv(source)


def load_factor(path: str | Path | Callable[[pd.DataFrame], pd.Series]):
    from quant_sdk.local import audit_factor_source

    if callable(path):
        return path
    if isinstance(path, str) and ("\n" in path or "\r" in path) and "compute_factor" in path:
        audit_factor_source(path)
        module = types.ModuleType("quant_local_factor_source")
        exec(compile(path, "<local_factor>", "exec"), module.__dict__)  # noqa: S102
        fn = getattr(module, "compute_factor", None)
        if not callable(fn):
            raise ValueError("因子源码必须定义 compute_factor(df) -> pandas.Series")
        return fn
    source = Path(path).expanduser()
    audit_factor_source(source.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("quant_local_factor", source)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法载入因子文件: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "compute_factor", None)
    if not callable(fn):
        raise ValueError("因子文件必须定义 compute_factor(df) -> pandas.Series")
    return fn


def preflight(data: str | Path, factor: str | Path | Callable[[pd.DataFrame], pd.Series]) -> dict[str, Any]:
    """Run an offline contract check against researcher-owned real data.

    This deliberately has no network or server fallback.  The returned state is
    ``LOCAL`` and is evidence that the code ran against the selected panel, not
    an official performance result.
    """
    started = time.perf_counter()
    try:
        frame = load_frame(data)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False, "status": "LOCAL_PREFLIGHT_FAILED", "panel_source": "researcher_owned",
            "sample_rows": 0, "columns": [], "null_ratio": 0.0,
            "errors": [f"DATA_LOAD_ERROR[{type(exc).__name__}]: {exc}"],
            "duration_ms": int((time.perf_counter() - started) * 1000),
        }

    required = {"ts_code", "trade_date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        return {
            "ok": False, "status": "LOCAL_PREFLIGHT_FAILED", "panel_source": "researcher_owned",
            "sample_rows": len(frame), "columns": [str(c) for c in frame.columns],
            "null_ratio": 0.0,
            "errors": [f"MISSING_IDENTITY_COLUMNS: {', '.join(missing)}"],
            "duration_ms": int((time.perf_counter() - started) * 1000),
        }

    frame = frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    try:
        values = load_factor(factor)(frame)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False, "status": "LOCAL_PREFLIGHT_FAILED", "panel_source": "researcher_owned",
            "sample_rows": len(frame), "columns": [str(c) for c in frame.columns],
            "null_ratio": 0.0,
            "errors": [f"FACTOR_EXECUTION_ERROR[{type(exc).__name__}]: {exc}"],
            "duration_ms": int((time.perf_counter() - started) * 1000),
        }

    errors: list[str] = []
    if not isinstance(values, pd.Series):
        errors.append(f"INVALID_RETURN: compute_factor must return pandas.Series, got {type(values).__name__}")
        null_ratio = 0.0
    else:
        if len(values) != len(frame):
            errors.append(f"LENGTH_MISMATCH: output has {len(values)} rows, input has {len(frame)}")
        if not pd.api.types.is_numeric_dtype(values.dtype):
            errors.append(f"NON_NUMERIC: output dtype is {values.dtype}")
        null_ratio = round(float(values.isna().mean()), 6) if len(values) else 0.0
        if len(values) and values.isna().all():
            errors.append("ALL_NULL: factor output is entirely null")

    return {
        "ok": not errors,
        "status": "LOCAL_PREFLIGHT_PASSED" if not errors else "LOCAL_PREFLIGHT_FAILED",
        "verification": "LOCAL",
        "panel_source": "researcher_owned",
        "sample_rows": len(frame),
        "columns": [str(c) for c in frame.columns],
        "null_ratio": null_ratio,
        "errors": errors,
        "duration_ms": int((time.perf_counter() - started) * 1000),
    }


def run(data: str | Path, factor: str | Path, output: str | Path) -> dict[str, Any]:
    frame = load_frame(data)
    required = {"ts_code", "trade_date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"数据缺少身份列: {', '.join(missing)}")
    frame = frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    values = load_factor(factor)(frame)
    if not isinstance(values, pd.Series) or len(values) != len(frame):
        raise ValueError("compute_factor 必须返回与输入等长的 pandas.Series")
    result = frame[["ts_code", "trade_date"]].copy()
    numeric = pd.to_numeric(values, errors="coerce")
    if len(numeric) and int(numeric.notna().sum()) == 0:
        # Silently writing an all-null factor_results.csv used to look like success.
        raise ValueError(
            "compute_factor 的输出无法转换为数值（全为文本/对象），拒绝写出全空结果")
    result["factor_value"] = numeric.to_numpy()
    result.to_csv(Path(output).expanduser(), index=False)
    return {"status": "LOCAL", "rows": len(result), "output": str(Path(output).expanduser()),
            "non_null_factor_rows": int(result["factor_value"].notna().sum())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local factor against researcher-owned data")
    parser.add_argument("--data", required=True, help="CSV or Parquet panel")
    parser.add_argument("--factor", required=True, help="Python file defining compute_factor(df)")
    parser.add_argument("--output", default="factor_results.csv", help="Canonical result CSV")
    args = parser.parse_args()
    print(run(args.data, args.factor, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
