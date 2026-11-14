"""Local intraday/minute backtest built on the canonical intraday engine.

Loads minute bars from a local file/dir (the researcher's own data), runs the
submitted ``compute_factor`` to form a per-bar signal, and executes the
canonical intraday engine. Nothing is uploaded; the result is
``LOCAL_UNVERIFIED``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from quant_sdk.local import _safe_builtins

__all__ = ["load_minute_bars", "load_minute_universe", "run_local_intraday",
           "run_local_intraday_cross_section", "index_benchmark_returns"]

_AGG_FUNCS = {"last": "last", "mean": "mean", "first": "first",
              "max": "max", "min": "min", "median": "median"}

_RENAME = {
    "日期": "trade_date", "时间": "trade_time", "股票代码": "ts_code", "代码": "ts_code",
    "开盘": "open", "最高": "high", "最低": "low", "收盘": "close",
    "开盘价": "open", "最高价": "high", "最低价": "low", "收盘价": "close",
    "成交量": "vol", "成交额": "amount",
}
_REQUIRED = ("trade_time", "open", "high", "low", "close", "vol", "amount")
_RAW_PARQUET_COLUMNS = {"code", "date", "pre_close", "change", "pct_chg", *_REQUIRED}


def _canonical_codes(values: pd.Series) -> pd.Series:
    return (values.astype("string").str.strip().str.upper()
            .str.replace(r"^(SH|SZ|BJ)([0-9]{6})$", r"\2.\1", regex=True))


def _read_minute_file(source: Path, *, code: Optional[str] = None) -> pd.DataFrame:
    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source)
        if "ts_code" not in frame.columns and _RAW_PARQUET_COLUMNS <= set(frame.columns):
            # The daily vendor archive is in shares, while normalized releases
            # and the annual stock CSVs already report lots.
            frame = frame.rename(columns={"code": "ts_code", "date": "trade_date"})
            frame["vol"] = pd.to_numeric(frame["vol"]) / 100
    else:
        frame = pd.read_csv(source, dtype={key: str for key in ("代码", "股票代码", "ts_code")})
        if {"时间", "代码", "名称", "开盘价", "成交额"} <= set(frame.columns) and "成交量" not in frame:
            # Archive index bars carry prices and amount, without volume.
            frame["vol"] = np.nan
    return _normalize_bars(frame, code=code, stem=source.stem)


def _normalize_bars(frame: pd.DataFrame, *, code: Optional[str] = None,
                    stem: Optional[str] = None) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = frame.rename(columns=_RENAME)
    if "trade_time" not in frame.columns and {"trade_date", "time"} <= set(frame.columns):
        frame["trade_time"] = (frame["trade_date"].astype(str) + " "
                               + frame["time"].astype(str))
    missing = [c for c in _REQUIRED if c not in frame.columns]
    if missing:
        raise RuntimeError(f"分钟数据缺少列: {sorted(missing)}")
    if "ts_code" not in frame.columns:
        frame["ts_code"] = code or stem
    frame["ts_code"] = _canonical_codes(frame["ts_code"])
    if code is not None:
        requested = _canonical_codes(pd.Series([code])).iloc[0]
        selected = frame["ts_code"].eq(requested)
        if "." in requested:
            selected |= frame["ts_code"].eq(requested.split(".", 1)[0])
        frame = frame.loc[selected].copy()
        if frame.empty:
            raise RuntimeError(f"分钟文件中没有证券 {requested}")
        frame["ts_code"] = requested
    frame["trade_time"] = frame["trade_time"].astype(str)
    if "trade_date" not in frame.columns:
        frame["trade_date"] = (pd.to_datetime(frame["trade_time"], format="mixed", errors="coerce")
                               .dt.strftime("%Y%m%d"))
    return frame


def load_minute_bars(path: str | Path, *, code: Optional[str] = None) -> pd.DataFrame:
    """Load a minute-bar file (or the first file in a directory)."""
    source = Path(path).expanduser()
    if source.is_dir():
        files = sorted(source.rglob("*.parquet")) + sorted(source.rglob("*.csv"))
        if not files:
            raise RuntimeError("目录下没有 parquet/csv 分钟文件")
        source = files[0]
    return _read_minute_file(source, code=code)


def load_minute_universe(path: str | Path, *, limit: Optional[int] = None,
                         pattern: str = "") -> pd.DataFrame:
    """Load every minute file under a directory (a downloaded minute release).

    A materialized minute release stores one file per code; the cross-section
    runner needs all of them. ``limit`` bounds the file count for a smoke test.
    """
    root = Path(path).expanduser()
    if not root.is_dir():
        raise RuntimeError(f"不是目录: {root}")
    files = sorted(root.rglob("*.parquet")) + sorted(root.rglob("*.csv"))
    if pattern:
        import fnmatch
        files = [f for f in files if fnmatch.fnmatch(f.name, pattern)]
    if limit is not None:
        files = files[:max(0, int(limit))]
    if not files:
        raise RuntimeError("目录下没有 parquet/csv 分钟文件")
    return pd.concat([_read_minute_file(source) for source in files], ignore_index=True)


def _signal(source_code: str, frame: pd.DataFrame) -> np.ndarray:
    env: Dict[str, Any] = {"__builtins__": _safe_builtins(),
                           "__name__": "__intraday__",
                           "pd": pd, "pandas": pd, "np": np, "numpy": np}
    exec(compile(source_code, "<intraday>", "exec"), env)
    fn = env.get("compute_factor")
    if not callable(fn):
        raise RuntimeError("未找到 def compute_factor(df) 入口函数")
    out = fn(frame.copy())
    if isinstance(out, pd.Series):
        series = out.reset_index(drop=True)
    elif isinstance(out, pd.DataFrame) and "factor_value" in out.columns:
        series = out["factor_value"].reset_index(drop=True)
    else:
        raise RuntimeError("compute_factor 必须返回 pandas.Series")
    if len(series) != len(frame):
        raise RuntimeError(f"信号长度 {len(series)} 与 bar 数 {len(frame)} 不一致")
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def run_local_intraday(source_code: str, bars: pd.DataFrame, *,
                       buy_threshold: float = 0.0, sell_threshold: float = 0.0,
                       order_shares: int = 10_000, base_shares: int = 0,
                       allow_short: bool = False) -> Dict[str, Any]:
    from server.engine.intraday import run_intraday


    if "ts_code" in bars.columns and bars["ts_code"].nunique() > 1:
        raise RuntimeError("单标的回测需要用 --code 选定一只证券，或改用截面回测")
    if bars["vol"].isna().any():
        raise RuntimeError("分钟数据缺少成交量，无法进行撮合")
    if bars["trade_time"].duplicated().any():
        raise RuntimeError("分钟数据存在重复时间，请选择互不重叠的单一频率行情")
    frame = bars.sort_values("trade_time").reset_index(drop=True)
    frame["trade_time"] = frame["trade_time"].astype(str)
    signal = _signal(source_code, frame)
    res = run_intraday(frame, signal, buy_threshold=buy_threshold,
                       sell_threshold=sell_threshold, order_shares=order_shares,
                       base_shares=base_shares, allow_short=allow_short)
    return {
        "status": "LOCAL_UNVERIFIED",
        "data_mode": "local_intraday",
        "engine": "canonical_intraday (server.engine.intraday)",
        "matching": "next-bar-open",
        "bars": int(len(frame)),
        "allow_short": allow_short,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "trade_count": res.trade_count,
        "round_trips": res.round_trips,
        "turnover": round(float(res.turnover), 6),
        "costs": res.costs,
        "blocked": res.blocked,
        "metrics": res.metrics,
        "note": "日内本地回测（canonical intraday 引擎）；非官方结果，也不会自动上传。",
    }


def index_benchmark_returns(index_code: str, dates: Sequence[str], *,
                            db_path: str | Path | None = None) -> np.ndarray:
    """Real index close-to-close returns from the local ``index_daily`` table.

    Returns an array aligned to ``dates`` (0.0 on days without an index row and
    on the first day). Raises when the index is absent or does not overlap the
    panel, so callers can fall back to the labelled equal-weight proxy instead
    of silently benchmarking against a wrong or empty series.
    """
    import duckdb

    target = (Path(db_path) if db_path
              else Path(__file__).resolve().parents[1] / "tushare_data.duckdb")
    if not target.is_file():
        raise RuntimeError("index benchmark database not found")
    ordered = [str(day) for day in dates]
    if not ordered:
        raise RuntimeError("no dates to benchmark")
    con = duckdb.connect(str(target), read_only=True)
    try:
        rows = con.execute(
            "SELECT CAST(trade_date AS VARCHAR), close FROM index_daily "
            "WHERE ts_code = ? AND CAST(trade_date AS VARCHAR) BETWEEN ? AND ? "
            "ORDER BY CAST(trade_date AS VARCHAR)",
            [str(index_code), ordered[0], ordered[-1]],
        ).fetchall()
    finally:
        con.close()
    if not rows:
        raise RuntimeError(f"index benchmark has no rows: {index_code}")
    series = pd.Series({str(row[0]): float(row[1]) for row in rows}).sort_index()
    rets = series.pct_change().reindex(ordered)
    if int(rets.notna().sum()) == 0:
        raise RuntimeError(f"index benchmark {index_code} does not overlap the panel")
    return rets.fillna(0.0).to_numpy(dtype=np.float64)


def run_local_intraday_cross_section(source_code: str, bars: pd.DataFrame, *,
                                     agg: str = "last", n_groups: int = 5,
                                     costs: bool = True, neutralize: bool = False,
                                     allow_short: bool = False,
                                     up_limit_pct: float = 0.10,
                                     benchmark_ret: Optional[np.ndarray] = None,
                                     benchmark_index_code: Optional[str] = None
                                     ) -> Dict[str, Any]:
    """Multi-name intraday cross-section: intraday signal -> daily portfolio.

    A-share T+1 means a portfolio cannot round-trip intraday, so the correct
    construction is to derive a per-``(ts_code, trade_date)`` factor from the
    minute bars (aggregated with ``agg``) and then run the **same canonical
    daily engine** the platform uses (next-day-open matching, T+1, price limits,
    costs).  The result is ``LOCAL_UNVERIFIED``.
    """
    from quant_sdk.local_backtest import run_local_backtest


    frame = bars.copy()
    if frame["vol"].isna().any():
        raise RuntimeError("截面回测需要有成交量的股票分钟行情")
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["trade_time"] = frame["trade_time"].astype(str)
    if frame.duplicated(["ts_code", "trade_time"]).any():
        raise RuntimeError("分钟数据存在重复的证券和时间键，请按单一频率选择互不重叠的文件")
    frame = frame.sort_values(["trade_date", "trade_time"]).reset_index(drop=True)
    frame["factor_value"] = _signal(source_code, frame)

    agg_key = _AGG_FUNCS.get(str(agg).lower(), "last")
    grouped = frame.groupby(["ts_code", "trade_date"], as_index=False)
    daily = grouped.agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), vol=("vol", "sum"), amount=("amount", "sum"),
        factor_value=("factor_value", agg_key))
    for column in ("up_limit", "down_limit"):
        if column in frame.columns:
            daily[column] = grouped[column].last().to_numpy()

    dates = sorted(daily["trade_date"].unique())
    codes = sorted(daily["ts_code"].astype(str).unique())
    matrix = (daily.pivot_table(index="trade_date", columns="ts_code",
                                values="factor_value", aggfunc=agg_key)
              .reindex(index=dates, columns=codes).to_numpy(dtype=np.float64))
    benchmark_proxy = False
    benchmark_source = "provided"
    if benchmark_ret is None and benchmark_index_code:
        # Prefer a real index series from the local `index_daily` so a minute
        # cross-section is benchmarked against an actual index, not a proxy.
        try:
            benchmark_ret = index_benchmark_returns(benchmark_index_code, dates)
            benchmark_source = f"index_daily:{benchmark_index_code}"
        except Exception:  # noqa: BLE001 - fall back to the labelled proxy
            benchmark_ret = None
    if benchmark_ret is None:
        # No real index supplied/available: use the equal-weight cross-section of
        # the panel as an explicit market proxy so the result is never presented
        # as benchmarked against a real index.
        close_pivot = (daily.pivot_table(index="trade_date", columns="ts_code",
                                         values="close", aggfunc="last")
                       .reindex(index=dates, columns=codes))
        benchmark_ret = (close_pivot.pct_change().mean(axis=1, skipna=True)
                         .fillna(0.0).to_numpy(dtype=np.float64))
        benchmark_proxy = True
        benchmark_source = "equal_weight_proxy"
    result = run_local_backtest(
        source_code, daily, factor_matrix=matrix, benchmark_ret=benchmark_ret,
        allow_short=allow_short, n_groups=n_groups, costs=costs,
        neutralize=neutralize, up_limit_pct=up_limit_pct)
    result["data_mode"] = "local_intraday_cross_section"
    result["intraday_aggregation"] = agg_key
    result["bars"] = int(len(frame))
    result["benchmark_source"] = benchmark_source
    if benchmark_proxy:
        approximations = result.setdefault("approximations", [])
        if isinstance(approximations, list):
            approximations.append("benchmark=等权全市场收益代理（非官方指数）")
    result["intraday_note"] = (
        "日内信号按 (ts_code, trade_date) 聚合为日频因子后，复用与服务端相同的 "
        "canonical 日频引擎（次日开盘成交、T+1、涨跌停、成本）；本地结果 LOCAL_UNVERIFIED。")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Local intraday backtest")
    parser.add_argument("--file", required=True, help="因子源码（compute_factor）")
    parser.add_argument("--data", required=True, help="分钟 parquet/csv 文件或目录")
    parser.add_argument("--code", default=None, help="选择回测证券，或为无代码数据指定证券")
    parser.add_argument("--buy-threshold", type=float, default=0.0)
    parser.add_argument("--sell-threshold", type=float, default=0.0)
    parser.add_argument("--order-shares", type=int, default=10_000)
    parser.add_argument("--base-shares", type=int, default=0)
    parser.add_argument("--cross-sectional", action="store_true",
                        help="多标的日内截面（日内信号聚合为日频因子）")
    parser.add_argument("--agg", default="last",
                        choices=sorted(_AGG_FUNCS), help="日内聚合方式")
    parser.add_argument("--groups", type=int, default=5, help="分组数")
    parser.add_argument("--no-costs", action="store_true", help="关闭成本")
    parser.add_argument("--neutralize", action="store_true",
                        help="按行业 + log 市值中性化")
    args = parser.parse_args()
    source = Path(args.file).read_text(encoding="utf-8")
    if args.cross_sectional:
        bars = load_minute_universe(args.data)
        result = run_local_intraday_cross_section(
            source, bars, agg=args.agg, n_groups=args.groups,
            costs=not args.no_costs, neutralize=bool(args.neutralize))
    else:
        bars = load_minute_bars(args.data, code=args.code)
        result = run_local_intraday(
            source, bars, buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
            order_shares=args.order_shares, base_shares=args.base_shares)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
