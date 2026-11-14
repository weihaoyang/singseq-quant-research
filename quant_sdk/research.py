"""Token-to-factor research runner and factor scaffold.

Two entry points:

* ``scaffold`` — emit a ``compute_factor(df) -> pandas.Series`` template.
* ``run`` — take an (optional) gateway token + an already-downloaded panel,
  run the canonical local backtest, write the factor-value CSV, an institutional
  tearsheet and (optionally) the novelty/anti-crowding score.

The gateway token is a data-plane credential: when provided it is verified
against the research gateway before the run, so a revoked/expired token fails
closed instead of producing a result on stale data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

__all__ = ["scaffold_factor", "run_research", "run_named_factor", "main"]

_TEMPLATE = '''"""Factor: {name} ({family}, window={window}).

Contract: ``compute_factor(df) -> pandas.Series`` aligned to ``df`` rows.
``df`` has ts_code, trade_date, open, high, low, close, vol, amount.
Returns a per-row factor value; the platform standardises cross-sectionally.
"""
import numpy as np
import pandas as pd


def compute_factor(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"], kind="mergesort")
    value = {expr}
    out = pd.Series(value, index=df.index, dtype=float)
    return out.replace([np.inf, -np.inf], np.nan)
'''

_EXPR = {
    "momentum": "df.groupby('ts_code')['close'].pct_change({window})",
    "reversal": "-df.groupby('ts_code')['close'].pct_change({window})",
    "volatility": ("df.groupby('ts_code')['close'].pct_change()"
                   ".groupby(df['ts_code']).rolling({window}).std().reset_index(level=0, drop=True)"),
    "volume_ratio": ("df.groupby('ts_code')['vol'].rolling(5).mean().reset_index(level=0, drop=True) / "
                     "df.groupby('ts_code')['vol'].rolling({window}).mean().reset_index(level=0, drop=True) - 1.0"),
}


def scaffold_factor(name: str = "my_factor", family: str = "momentum",
                    window: int = 20) -> str:
    expr = _EXPR.get(family, _EXPR["momentum"]).format(window=window)
    return _TEMPLATE.format(name=name, family=family, window=window, expr=expr)


def _compute_factor_series(source: str, df: pd.DataFrame) -> np.ndarray:
    from quant_sdk.local import _safe_builtins

    env: Dict[str, object] = {"__builtins__": _safe_builtins(), "__name__": "__factor__",
                              "pd": pd, "pandas": pd, "np": np, "numpy": np}
    exec(compile(source, "<factor>", "exec"), env)  # noqa: S102 - researcher's own code, sandboxed builtins
    fn = env.get("compute_factor")
    if not callable(fn):
        raise RuntimeError("因子源码必须定义 def compute_factor(df)")
    values = fn(df.copy())
    if isinstance(values, pd.DataFrame) and "factor_value" in values.columns:
        values = values["factor_value"]
    if not isinstance(values, pd.Series):
        raise RuntimeError("compute_factor 必须返回 pandas.Series")
    if len(values) != len(df) or not values.index.is_unique or set(values.index) != set(df.index):
        raise RuntimeError("compute_factor 输出必须与输入行索引一一对应")
    return pd.to_numeric(values.reindex(df.index), errors="coerce").to_numpy(dtype=float)


def _write_result(output, filename, frame):
    output = output.loc[np.isfinite(output["factor_value"])].copy()
    target = Path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(target, index=False)
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = {"csv_sha256": digest, "rows": len(output),
                  "package_version": frame.attrs.get("package_version"),
                  "package_hash": frame.attrs.get("package_hash"),
                  "data_warnings": dict(frame.attrs)}
    Path(str(target) + ".provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def _verify_token(token: str, gateway: Optional[str]) -> str:
    import urllib.request

    base = (gateway or "https://quant.singseq.com").rstrip("/")
    req = urllib.request.Request(f"{base}/api/v1/research/mesh/status",
                                 headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"网关授权校验失败：{exc}") from exc
    state = str((body.get("data") or {}).get("access_state", "UNKNOWN"))
    if state not in {"AUTHORIZED"}:
        if state == "DISQUALIFIED":
            raise SystemExit(
                "网关未授权（DISQUALIFIED）：本赛季参赛资格已被取消，自助报名不会恢复；"
                "如需申诉请联系主办方。")
        raise SystemExit(f"网关未授权（{state}）：请先在 Portal 报名赛季或刷新授权。")
    return state


def run_research(*, factor_source: str, data_path: str, start: str, end: str,
                 out_csv: Optional[str] = None, tearsheet_path: Optional[str] = None,
                 with_novelty: bool = False, token: Optional[str] = None,
                 gateway: Optional[str] = None, groups: int = 5,
                 benchmark_code: str = "000300.SH") -> Dict[str, object]:
    from quant_sdk.local_backtest import load_panel, run_local_backtest

    auth = _verify_token(token, gateway) if token else "unverified"
    path = Path(data_path)
    if path.suffix.lower() in {".duckdb", ".db"}:
        df = load_panel(path, start, end, benchmark_code=benchmark_code)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if df.empty:
        raise SystemExit("输入面板为空，请核对数据路径与日期覆盖。")

    df["trade_date"] = df["trade_date"].astype(str)
    df = df.loc[df.trade_date.between(str(start), str(end))].copy()
    df = df.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    factor_values = _compute_factor_series(factor_source, df)
    if out_csv:
        out = df[["ts_code", "trade_date"]].copy()
        out["factor_value"] = factor_values
        _write_result(out, out_csv, df)

    result = run_local_backtest(factor_source, df, n_groups=groups,
                                with_novelty=bool(with_novelty),
                                tearsheet_path=tearsheet_path)
    result["gateway_auth"] = auth
    result["panel_rows"] = int(len(factor_values))
    result["factor_rows"] = int(np.isfinite(factor_values).sum())
    result["nonfinite_factor_rows"] = int((~np.isfinite(factor_values)).sum())
    result["data_warnings"] = dict(df.attrs)
    return result


def run_named_factor(*, factor_name: str, data_path: str, start: str, end: str,
                     out_csv: Optional[str] = None,
                     tearsheet_path: Optional[str] = None,
                     with_novelty: bool = False, groups: int = 5,
                     benchmark_code: str = "000300.SH") -> Dict[str, object]:
    """Run an implementation-backed catalog factor through the local engine.

    The adapter is deliberately local-only.  It reuses ``run_local_backtest``
    after computing the named factor, so the result keeps the same
    ``LOCAL_UNVERIFIED`` boundary as source-based research.
    """
    from quant_sdk.catalog_factors import compute_named_factor
    from quant_sdk.local_backtest import load_panel, run_local_backtest

    path = Path(data_path)
    if path.suffix.lower() in {".duckdb", ".db"}:
        frame = load_panel(path, start, end, benchmark_code=benchmark_code)
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    if frame.empty:
        raise SystemExit("数据为空：请确认研究面板已准备好。")

    work = frame.copy()
    work["ts_code"] = work["ts_code"].astype(str)
    work["trade_date"] = work["trade_date"].astype(str)
    work = work.loc[work.trade_date.between(str(start), str(end))].copy()
    work = work.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    values = compute_named_factor(factor_name, work)
    factor_rows = work[["trade_date", "ts_code"]].copy()
    factor_rows["factor_value"] = values.to_numpy(dtype=float)
    dates = sorted(work["trade_date"].astype(str).unique())
    codes = sorted(work["ts_code"].astype(str).unique())
    factor_matrix = (factor_rows.pivot_table(
        index="trade_date", columns="ts_code", values="factor_value", aggfunc="last")
        .reindex(index=dates, columns=codes).to_numpy(dtype=float))
    source = (
        "def compute_factor(df):\n"
        "    # Values are supplied by the trusted named-factor adapter.\n"
        "    raise RuntimeError('named factor values are attached by the runner')\n"
    )
    result = run_local_backtest(
        source, work, factor_matrix=factor_matrix,
        n_groups=groups, with_novelty=bool(with_novelty),
        tearsheet_path=tearsheet_path,
    )
    result.update({"factor_name": str(factor_name),
                   "runnable": True,
                   "execution_boundary": "catalog_implementation_local",
                   "panel_rows": len(work), "factor_rows": int(np.isfinite(values).sum()),
                   "nonfinite_factor_rows": int((~np.isfinite(values)).sum()),
                   "data_warnings": dict(work.attrs)})
    if out_csv:
        output = work[["ts_code", "trade_date"]].copy()
        output["factor_value"] = values.to_numpy(dtype=float)
        _write_result(output, out_csv, work)
    return result


def build_staging(package_dir: str, db_path: Optional[str] = None, *,
                  include_adjusted: bool = True,
                  limit_files: Optional[int] = None) -> Dict[str, object]:
    """One-command bridge: downloaded research package -> staging DuckDB.

    Reads the gateway package layout (``equity_daily_raw.zip`` + ``stock_list.csv``
    + optional ``delisted_list.csv`` / ``trade_calendar.csv`` / ``index_daily.zip``)
    and builds an isolated, resumable staging database the backtest can load.
    """
    from research.import_staging_daily import import_staging

    pkg = Path(package_dir)
    if not pkg.is_dir():
        raise SystemExit(f"研究包目录不存在：{pkg}（请先用客户端“同步研究包”下载）")
    db = Path(db_path) if db_path else pkg.parent / f"{pkg.name}.staging.duckdb"
    result = import_staging(pkg, db, limit_files, include_adjusted)
    if result.get("failures"):
        raise RuntimeError(f"研究包导入未完成，保留断点后可重试：{result['failures'][:3]}")
    return {"db": str(db), "status": result.get("status"),
            "daily_rows": (result.get("table_coverage", {}).get("daily_raw") or {}).get("rows", 0),
            "table_coverage": result.get("table_coverage", {}),
            "identity_quarantine": result.get("identity_quarantine", {}),
            "scope_exclusions": result.get("scope_exclusions", {}),
            "failures": result.get("failures", [])}


def auto(*, package_dir: str, start: str, end: str,
         factor_source: Optional[str] = None,
         scaffold: Optional[tuple] = None,
         db_path: Optional[str] = None, out_csv: Optional[str] = None,
         tearsheet_path: Optional[str] = None, with_novelty: bool = True,
         token: Optional[str] = None, gateway: Optional[str] = None,
         include_adjusted: bool = True, groups: int = 5) -> Dict[str, object]:
    """Fully headless E2E: package -> staging -> factor -> CSV + tearsheet + score.

    One command lets any (AI) tool go from a downloaded research package to a
    scored factor result without touching the GUI.
    """
    bridge = build_staging(package_dir, db_path, include_adjusted=include_adjusted)
    if factor_source is None:
        name, family, window = scaffold or ("auto_momentum", "momentum", 20)
        factor_source = scaffold_factor(name, family, window)
    result = run_research(factor_source=factor_source, data_path=str(bridge["db"]),
                          start=start, end=end, out_csv=out_csv,
                          tearsheet_path=tearsheet_path, with_novelty=with_novelty,
                          token=token, gateway=gateway, groups=groups)
    result["staging"] = bridge
    return result


def capabilities() -> Dict[str, object]:
    """Machine-readable entrypoints so any AI tool can discover how to start."""
    from evaluation import alpha_zoo, novelty_scoring

    return {
        "package": "quant-research-toolkit",
        "entrypoints": {
            "scaffold": "python -m quant_sdk.research scaffold --name my_factor --family momentum --window 20 --out factor.py",
            "bridge": "python -m quant_sdk.research bridge --package-dir <downloaded-package> --db staging.duckdb",
            "run": "python -m quant_sdk.research run --file factor.py --data staging.duckdb --start 20210101 --end 20241231 --out factor.csv --novelty --tearsheet report.html",
            "auto": "python -m quant_sdk.research auto --package-dir <downloaded-package> --start 20210101 --end 20241231 --out factor.csv --tearsheet report.html",
            "capabilities": "python -m quant_sdk.research capabilities",
            "named": "python -m quant_sdk.research named --factor-name <catalog-slug> --data <panel> --start YYYYMMDD --end YYYYMMDD",
        },
        "factor_contract": {
            "python_entry": "compute_factor(df) -> pandas.Series",
            "required_identity": ["ts_code", "trade_date"],
            "result_file": "ts_code,trade_date,factor_value",
        },
        "zoo_version": alpha_zoo.ZOO_VERSION,
        "scoring_version": novelty_scoring.SCORING_VERSION,
        "data": {"delivery": "researcher-owned local files",
                 "formats": ["CSV", "Parquet", "DuckDB"],
                 "offline": True},
        "requirements": ["Python >= 3.11", "pandas >= 2.1", "numpy >= 1.26",
                         "duckdb >= 1.1", "filelock >= 3.12", "scipy >= 1.11"],
    }


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", default=None, help="因子值 CSV 输出路径")
    p.add_argument("--tearsheet", default=None, help="HTML tearsheet 输出路径")
    p.add_argument("--novelty", action="store_true", help="计算与主流因子库的差异度与加权评分")
    p.add_argument("--token", default=None, help="网关设备 token（可选，运行前授权校验）")
    p.add_argument("--gateway", default=None)
    p.add_argument("--groups", type=int, default=5)
    p.add_argument("--benchmark-code", default="000300.SH")


def _summary(result: Dict[str, object], out_csv: Optional[str]) -> Dict[str, object]:
    return {
        "status": result.get("status"),
        "gateway_auth": result.get("gateway_auth"),
        "factor_rows": result.get("factor_rows"),
          "panel_rows": result.get("panel_rows"),
          "nonfinite_factor_rows": result.get("nonfinite_factor_rows"),
          "data_warnings": result.get("data_warnings"),
        "ic_mean": result.get("ic_mean"),
        "curves": result.get("curves"),
        "tearsheet_metrics": result.get("tearsheet_metrics"),
        "novelty": result.get("novelty"),
        "scoring": result.get("scoring"),
        "tearsheet_path": result.get("tearsheet_path"),
        "out_csv": out_csv,
        "staging": result.get("staging"),
    }


def mine_research(*, data_path: str, start: str, end: str, top_k: int = 20,
                  out: Optional[str] = None,
                  benchmark_code: str = "000300.SH") -> Dict[str, object]:
    """Rank a bounded grammar of candidates by the novelty-weighted score."""
    from evaluation import mining
    from quant_sdk.local_backtest import load_panel

    path = Path(data_path)
    if path.suffix.lower() in {".duckdb", ".db"}:
        df = load_panel(path, start, end, benchmark_code=benchmark_code)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if df.empty:
        raise SystemExit("数据为空：请先下载研究包并 bridge 到 staging DuckDB。")
    rows = mining.mine(df, top_k=int(top_k))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"version": mining.MINE_VERSION, "count": len(rows), "top": rows[:10], "out": out}


def main() -> int:
    parser = argparse.ArgumentParser(prog="quant-research",
                                     description="Local factor research and execution-aware backtesting")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scaffold", help="生成因子 compute_factor 模板")
    s.add_argument("--name", default="my_factor")
    s.add_argument("--family", default="momentum",
                   choices=sorted(_EXPR))
    s.add_argument("--window", type=int, default=20)
    s.add_argument("--out", default="factor.py")

    r = sub.add_parser("run", help="运行因子：数据 -> 因子值 CSV + tearsheet + 差异度评分")
    r.add_argument("--file", required=True, help="因子源码（compute_factor）")
    r.add_argument("--data", required=True, help="面板 CSV/Parquet 或 staging DuckDB")
    r.add_argument("--start", required=True)
    r.add_argument("--end", required=True)
    _add_run_args(r)

    n = sub.add_parser("named", help="运行已有实现的目录因子；结果为 LOCAL_UNVERIFIED")
    n.add_argument("--factor-name", required=True)
    n.add_argument("--data", required=True, help="面板 CSV/Parquet 或 staging DuckDB")
    n.add_argument("--start", required=True)
    n.add_argument("--end", required=True)
    _add_run_args(n)

    b = sub.add_parser("bridge", help="一键桥接：研究包 -> staging DuckDB")
    b.add_argument("--package-dir", required=True, help="已下载的研究包目录")
    b.add_argument("--db", default=None, help="输出 staging DuckDB 路径")
    b.add_argument("--no-adjusted", action="store_true", help="不导入复权价表")
    b.add_argument("--limit-files", type=int, default=None)

    a = sub.add_parser("auto", help="无头 E2E：研究包 -> staging -> 因子 -> CSV + 报告 + 差异度评分")
    a.add_argument("--package-dir", required=True)
    a.add_argument("--start", required=True)
    a.add_argument("--end", required=True)
    a.add_argument("--file", default=None, help="因子源码（缺省用内置 momentum 模板）")
    a.add_argument("--scaffold", default="auto_momentum,momentum,20", help="name,family,window")
    a.add_argument("--db", default=None)
    a.add_argument("--no-adjusted", action="store_true")
    _add_run_args(a)

    sub.add_parser("capabilities", help="输出机器可读入口与契约（供 AI 工具发现）")

    m = sub.add_parser("mine", help="自动挖掘：候选因子按 IC + 差异度评分排序")
    m.add_argument("--data", required=True, help="面板 CSV/Parquet 或 staging DuckDB")
    m.add_argument("--start", required=True)
    m.add_argument("--end", required=True)
    m.add_argument("--top", type=int, default=20)
    m.add_argument("--out", default=None)
    m.add_argument("--benchmark-code", default="000300.SH")

    lint = sub.add_parser("lint", help="提交前静态检查：用平台同一份规则检查因子源码")
    lint.add_argument("--file", required=True, help="因子源码（compute_factor）")

    args = parser.parse_args()

    if args.command == "lint":
        # The exact rule set the platform applies at submission time, shipped with the
        # toolkit so a researcher can find out before submitting instead of after.
        from server.sandbox.ast_audit import audit_factor_code

        source = Path(args.file).read_text(encoding="utf-8")
        result = audit_factor_code(source)
        print(json.dumps({
            "passed": bool(result.passed),
            "rejections": list(result.rejections),
            "criticals": [item.to_dict() for item in result.criticals],
            "factor_classes": list(result.factor_classes),
            "model_classes": list(result.model_classes),
        }, ensure_ascii=False, indent=2))
        return 0

    if args.command == "capabilities":
        print(json.dumps(capabilities(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "scaffold":
        Path(args.out).write_text(scaffold_factor(args.name, args.family, args.window),
                                  encoding="utf-8")
        print(f"已生成因子模板：{args.out}")
        return 0
    if args.command == "bridge":
        print(json.dumps(build_staging(args.package_dir, args.db,
                                       include_adjusted=not args.no_adjusted,
                                       limit_files=args.limit_files),
                         ensure_ascii=False, indent=2))
        return 0
    if args.command == "auto":
        source = Path(args.file).read_text(encoding="utf-8") if args.file else None
        spec = tuple(args.scaffold.split(",")) if not source else None
        result = auto(package_dir=args.package_dir, start=args.start, end=args.end,
                      factor_source=source, scaffold=spec, db_path=args.db,
                      out_csv=args.out, tearsheet_path=args.tearsheet,
                      with_novelty=True, token=args.token, gateway=args.gateway,
                      include_adjusted=not args.no_adjusted, groups=args.groups)
        print(json.dumps(_summary(result, args.out), ensure_ascii=False, indent=2))
        return 0

    if args.command == "named":
        result = run_named_factor(
            factor_name=args.factor_name, data_path=args.data,
            start=args.start, end=args.end, out_csv=args.out,
            tearsheet_path=args.tearsheet, with_novelty=bool(args.novelty),
            groups=args.groups, benchmark_code=args.benchmark_code)
        print(json.dumps(_summary(result, args.out) | {
            "factor_name": result.get("factor_name"),
            "runnable": result.get("runnable"),
            "execution_boundary": result.get("execution_boundary"),
        }, ensure_ascii=False, indent=2))
        return 0

    if args.command == "mine":
        print(json.dumps(mine_research(data_path=args.data, start=args.start, end=args.end,
                                       top_k=args.top, out=args.out,
                                       benchmark_code=args.benchmark_code),
                         ensure_ascii=False, indent=2))
        return 0

    result = run_research(
        factor_source=Path(args.file).read_text(encoding="utf-8"),
        data_path=args.data, start=args.start, end=args.end, out_csv=args.out,
        tearsheet_path=args.tearsheet, with_novelty=bool(args.novelty),
        token=args.token, gateway=args.gateway, groups=args.groups,
        benchmark_code=args.benchmark_code)
    print(json.dumps(_summary(result, args.out), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
