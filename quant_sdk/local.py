"""Local factor mining: fetch a data slice with the caller's own token and
compute factor values + Rank IC on the caller's own machine.

Computation never leaves the machine: the factor source is executed locally,
only data rows are pulled through the metered ``/api/v1/data/query`` endpoint
using the caller's own API token. The server is never asked to run anything.

Output is explicitly ``LOCAL_UNVERIFIED``: nothing here is an official result
and nothing is uploaded automatically.
"""
from __future__ import annotations

import builtins
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["SAFE_IMPORT_ROOTS", "LocalFactorRunner", "run_local_factor"]

SAFE_IMPORT_ROOTS = frozenset((
    "__future__", "math", "cmath", "statistics", "datetime", "decimal",
    "fractions", "itertools", "functools", "collections", "random", "re",
    "string", "typing", "dataclasses", "abc", "enum", "numbers", "operator",
    "copy", "warnings", "contextlib", "textwrap", "pprint",
    "numpy", "pandas", "polars", "scipy",
))
_BLOCKED = frozenset((
    "open", "eval", "exec", "compile", "input", "breakpoint", "exit", "quit",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
    "help", "memoryview",
))
_REAL_IMPORT = builtins.__import__


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name).split(".", 1)[0]
    trusted_sdk_module = (
        str(name) == "quant_sdk.operators"
        or str(name).startswith("quant_sdk.operators.")
        or str(name) == "quant_sdk.catalog_factors"
    )
    if level != 0 or (root not in SAFE_IMPORT_ROOTS and not trusted_sdk_module):
        raise ImportError(f"本地运行禁止导入模块 {name!r}")
    return _REAL_IMPORT(name, globals, locals, fromlist, level)


def _safe_builtins() -> dict[str, Any]:
    out = {n: getattr(builtins, n) for n in dir(builtins)}
    for n in _BLOCKED:
        out.pop(n, None)
    out["__import__"] = _safe_import
    return out


def audit_factor_source(source_code: str) -> None:
    """Apply the platform's static factor contract before local execution."""
    from quant_core.audit import audit_factor_code

    audit = audit_factor_code(source_code)
    if audit.passed:
        return
    details = [str(item) for item in getattr(audit, "rejections", ())]
    criticals = [
        f"line {item.line}: {item.explanation}"
        for item in getattr(audit, "criticals", ())
    ]
    reason = "; ".join([*details, *criticals]) or "unknown static-audit rejection"
    raise RuntimeError(f"因子源码未通过平台静态检查：{reason}")


class LocalFactorRunner:
    """Run a submitted ``compute_factor`` locally against the caller's data."""

    def __init__(self, client: Any):
        self.client = client

    def fetch_panel(self, ts_codes: Sequence[str], start: str, end: str,
                    *, api: str = "daily", page: int = 1000) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for code in ts_codes:
            offset = 0
            while True:
                df = self._fetch_page(api, code, start, end, page, offset)
                if df is None or df.empty:
                    break
                frames.append(df)
                if len(df) < page:
                    break
                offset += page
        if not frames:
            raise RuntimeError("未取到任何数据行；请检查 ts_code / 区间 / token 配额")
        panel = pd.concat(frames, ignore_index=True)
        panel["trade_date"] = panel["trade_date"].astype(str)
        return panel

    def _fetch_page(self, api: str, code: str, start: str, end: str,
                    limit: int, offset: int) -> pd.DataFrame:
        # A contestant runs the client in remote mode with their own token; the
        # canonical data endpoint takes a top-level ``offset`` (not a param), so
        # page through it directly. Local DuckDB mode returns a single page.
        if getattr(self.client, "mode", "remote") == "local":
            query = getattr(self.client, "query", None)
            if callable(query):
                return query(api, ts_code=code, start_date=start,
                             end_date=end, limit=limit)
        session = getattr(self.client, "session", None)
        base_url = getattr(self.client, "base_url", "")
        if session is None:
            raise RuntimeError("client 缺少可用的查询接口")
        res = session.post(
            f"{base_url}/api/v1/data/query",
            json={"api": api,
                  "params": {"ts_code": code, "start_date": start,
                             "end_date": end},
                  "limit": limit, "offset": offset},
            timeout=60,
        )
        res.raise_for_status()
        body = res.json()
        data = body.get("data", body)
        return pd.DataFrame(data.get("rows", []),
                            columns=data.get("columns", []))

    def compute(self, source_code: str, panel: pd.DataFrame,
                *, price_col: str = "open") -> dict[str, Any]:
        if price_col not in panel.columns:
            raise RuntimeError(f"面板缺少 {price_col} 列")
        audit_factor_source(source_code)
        work = panel.sort_values(["ts_code", "trade_date"]).copy()
        # The factor must see exactly the frame its values are attached to (a differently
        # ordered frame silently swaps values between stocks), and must NOT see the
        # forward return: `_fwd` is added after the call.
        env: dict[str, Any] = {"__builtins__": _safe_builtins(),
                               "__name__": "__local_factor__",
                               "pd": pd, "pandas": pd, "np": np, "numpy": np}
        exec(compile(source_code, "<local_factor>", "exec"), env)
        fn = env.get("compute_factor")
        if not callable(fn):
            raise RuntimeError("未找到 def compute_factor(df) 入口函数")
        values = fn(work.copy())
        with np.errstate(invalid="ignore", divide="ignore"):
            work["_fwd"] = (work.groupby("ts_code")[price_col].shift(-1)
                            / work[price_col] - 1.0)
        if isinstance(values, pd.Series):
            series = values.reset_index(drop=True)
        elif isinstance(values, pd.DataFrame) and "factor_value" in values.columns:
            series = values["factor_value"].reset_index(drop=True)
        else:
            raise RuntimeError("compute_factor 必须返回 pandas.Series")
        if len(series) != len(work):
            raise RuntimeError(f"输出长度 {len(series)} 与面板 {len(work)} 不一致")
        work["factor_value"] = series.to_numpy()
        self.last_work = work
        return self._metrics(work)

    @staticmethod
    def _metrics(work: pd.DataFrame) -> dict[str, Any]:
        finite = work[np.isfinite(work["factor_value"]) & np.isfinite(work["_fwd"])]
        ic_series: list[float] = []
        for _, group in finite.groupby("trade_date"):
            if len(group) < 5:
                continue
            ic = group["factor_value"].corr(group["_fwd"], method="spearman")
            if ic is not None and np.isfinite(ic):
                ic_series.append(float(ic))
        ic_arr = np.asarray(ic_series, dtype=float)
        mean_ic = float(ic_arr.mean()) if ic_arr.size else None
        # ddof=1 matches the platform's `icir_daily` (a population std would differ by
        # sqrt(n/(n-1)) and make the local IC_IR disagree with the platform report).
        ic_std = float(ic_arr.std(ddof=1)) if ic_arr.size > 1 else 0.0
        icir = (float(ic_arr.mean() / ic_std) if ic_arr.size > 1 and ic_std > 0
                else None)
        return {
            "status": "LOCAL_UNVERIFIED",
            "data_mode": "local",
            "rows": len(work),
            "scored_rows": len(finite),
            "coverage": round(len(finite) / len(work), 6) if len(work) else 0.0,
            "ic_days": int(ic_arr.size),
            "mean_rank_ic": None if mean_ic is None else round(mean_ic, 6),
            "ic_ir": None if icir is None else round(icir, 6),
            "note": "本地计算，仅供研究；不是官方结果，未上传服务器。",
        }


def run_local_factor(source_code: str, *, ts_codes: Iterable[str], start: str,
                     end: str, token: str | None = None,
                     base_url: str | None = None,
                     api: str = "daily", price_col: str = "open",
                     result_csv: str | Path | None = None,
                     client: Any = None) -> dict[str, Any]:
    """End-to-end local run: fetch slice -> compute -> metrics (optional CSV)."""
    if client is None:
        from quant_sdk.provider import QuantClient
        client = QuantClient(base_url=base_url, token=token)
    codes = [c for c in ts_codes if str(c).strip()]
    if not codes:
        raise RuntimeError("至少需要一个 ts_code")
    runner = LocalFactorRunner(client)
    panel = runner.fetch_panel(codes, start, end, api=api)
    metrics = runner.compute(source_code, panel, price_col=price_col)
    if result_csv:
        out = Path(result_csv).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        work = getattr(runner, "last_work", None)
        if work is None:
            raise RuntimeError("内部错误：缺少本地计算结果")
        pd.DataFrame({
            "ts_code": work["ts_code"].to_numpy(),
            "trade_date": work["trade_date"].to_numpy(),
            "factor_value": work["factor_value"].to_numpy(),
        }).to_csv(out, index=False, encoding="utf-8")
        metrics["result_csv"] = str(out)
    return metrics


if __name__ == "__main__":  # pragma: no cover - manual smoke
    import argparse

    parser = argparse.ArgumentParser(description="Local factor runner")
    parser.add_argument("--file", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--url", default=None)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()
    print(json.dumps(run_local_factor(
        Path(args.file).read_text(encoding="utf-8"),
        ts_codes=args.codes.split(","), start=args.start, end=args.end,
        token=args.token, base_url=args.url), ensure_ascii=False, indent=2))
