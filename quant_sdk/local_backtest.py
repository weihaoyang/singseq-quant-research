"""Local canonical backtest.

Runs the **same** replay engine the server uses (``server.engine.backtest``)
against a researcher-owned panel on the researcher's own machine. The factor
source is executed locally; nothing is uploaded.

Parity note: the server's official replay uses point-in-time universes, ST
flags and borrow inventory that the public research package does not contain.
When those columns are absent this module substitutes documented
approximations and records them under ``approximations`` so a local result is
never mistaken for an official one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from quant_sdk.local import _safe_builtins
from quant_sdk.archive_scope import SZSE_B_SHARE_SQL, SZSE_B_SHARE_SYMBOL_SQL

__all__ = ["panel_from_frame", "run_local_backtest", "load_panel"]

_REQUIRED = ("ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount")


def _matrix(work: pd.DataFrame, column: str, dates: Sequence[str],
            codes: Sequence[str], *, dtype=float) -> np.ndarray:
    table = work.pivot_table(index="trade_date", columns="ts_code",
                             values=column, aggfunc="last")
    return table.reindex(index=list(dates), columns=list(codes)).to_numpy(dtype=dtype)


def _benchmark_series(work: pd.DataFrame, dates: Sequence[str],
                      benchmark_ret: Optional[np.ndarray]) -> np.ndarray:
    if benchmark_ret is not None:
        # An explicit copy: pandas/numpy may hand back a read-only array, and the
        # engine must be free to use this buffer (Copy-on-Write made read-only
        # views the default for to_numpy in pandas 3).
        arr = np.array(benchmark_ret, dtype=float)
        if arr.shape[0] != len(dates):
            raise ValueError("benchmark_ret length must equal the number of dates")
        return arr
    if "benchmark_ret" in work.columns:
        per_date = (work.dropna(subset=["benchmark_ret"])
                    .groupby("trade_date")["benchmark_ret"].last())
        # Same reason: copy before filling the trailing value, otherwise
        # ``arr[-1] = 0.0`` raises "assignment destination is read-only" on
        # pandas 3 / numpy 2.
        arr = np.array(per_date.reindex(list(dates)).to_numpy(dtype=float), dtype=float)
        if np.isfinite(arr[:-1]).all():
            if not np.isfinite(arr[-1]):
                arr[-1] = 0.0
            return arr
    raise ValueError(
        "缺少基准收益：请在面板提供 benchmark_ret 列（每交易日一行）"
        "或通过 benchmark_ret 参数传入指数日收益序列")


def panel_from_frame(df: pd.DataFrame, *, benchmark_ret: Optional[np.ndarray] = None,
                     up_limit_pct: float = 0.10):
    """Build a dense canonical ``Panel`` from a long researcher panel."""
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"面板缺少必需列: {', '.join(missing)}")
    work = df.copy()
    work["trade_date"] = work["trade_date"].astype(str)
    work["ts_code"] = work["ts_code"].astype(str)
    dates = sorted(work["trade_date"].unique())
    codes = sorted(work["ts_code"].unique())

    close = _matrix(work, "close", dates, codes)
    adj_close = (_matrix(work, "adj_close", dates, codes)
                 if "adj_close" in work.columns else close.copy())
    up_limit = (_matrix(work, "up_limit", dates, codes)
                if "up_limit" in work.columns
                else np.round(close * (1.0 + up_limit_pct), 2))
    down_limit = (_matrix(work, "down_limit", dates, codes)
                  if "down_limit" in work.columns
                  else np.round(close * (1.0 - up_limit_pct), 2))
    st = (_matrix(work, "st", dates, codes)
          if "st" in work.columns else np.zeros_like(close, dtype=bool))
    st = np.nan_to_num(st, nan=0.0).astype(bool)

    if "list_date" in work.columns:
        first = (work.dropna(subset=["list_date"])
                 .groupby("ts_code")["list_date"].min())
    else:
        first = work.groupby("ts_code")["trade_date"].min()
    first = first.reindex(list(codes))
    list_dates = np.asarray(
        [int(str(x).replace("-", "")[:8]) if str(x).strip() not in ("", "nan", "None", "NaT")
         else 0 for x in first.to_numpy()], dtype=np.int64)

    bench = _benchmark_series(work, dates, benchmark_ret)

    from server.engine.backtest import Panel

    T, N = len(dates), len(codes)
    optional = {}
    for column in ("circ_mv", "pb", "turnover_rate", "industry_id",
                   "pool_membership", "cash_dividend", "shortable", "borrow_rate"):
        if column not in work.columns:
            continue
        values = _matrix(work, column, dates, codes,
                         dtype=object if column == "industry_id" else float)
        if column in ("pool_membership", "shortable"):
            values = np.nan_to_num(values, nan=0.0).astype(bool)
        optional[column] = values
    return Panel(
        dates=np.asarray(dates, dtype=np.int64).astype("int64"),
        codes=np.asarray(codes, dtype=object),
        factor=np.full((T, N), np.nan, dtype=float),
        open=_matrix(work, "open", dates, codes),
        high=_matrix(work, "high", dates, codes),
        low=_matrix(work, "low", dates, codes),
        close=close,
        adj_close=adj_close,
        up_limit=up_limit,
        down_limit=down_limit,
        vol=np.nan_to_num(_matrix(work, "vol", dates, codes)),
        amount=np.nan_to_num(_matrix(work, "amount", dates, codes)),
        list_date=list_dates,
        st=st,
        benchmark_ret=bench,
        meta={"source": "local_research_panel"},
        **optional,
    )


#: Forward-looking label columns the engine derives for scoring.  They must never
#: reach a factor (the server's worker rejects ``target_ret`` for the same reason):
#: ``benchmark_ret`` is ``close.shift(-1)``, i.e. tomorrow's value.
_LABEL_COLUMNS = ("benchmark_ret", "target_ret", "fwd_ret", "forward_return")


def _factor_matrix(source_code: str, work: pd.DataFrame, dates: Sequence[str],
                   codes: Sequence[str]) -> np.ndarray:
    env: Dict[str, Any] = {"__builtins__": _safe_builtins(),
                           "__name__": "__local_backtest__",
                           "pd": pd, "pandas": pd, "np": np, "numpy": np}
    exec(compile(source_code, "<local_backtest>", "exec"), env)
    fn = env.get("compute_factor")
    if not callable(fn):
        raise RuntimeError("未找到 def compute_factor(df) 入口函数")
    frame = work.drop(columns=[name for name in _LABEL_COLUMNS if name in work.columns],
                      errors="ignore")
    values = fn(frame.copy())
    if isinstance(values, pd.DataFrame) and "factor_value" in values.columns:
        series = values["factor_value"]
    elif isinstance(values, pd.Series):
        series = values
    else:
        raise RuntimeError("compute_factor 必须返回 pandas.Series")
    if len(series) != len(work):
        raise RuntimeError(f"输出长度 {len(series)} 与面板 {len(work)} 不一致")
    if not series.index.is_unique or set(series.index) != set(work.index):
        raise RuntimeError("compute_factor 输出必须与输入行索引一一对应")
    series = series.reindex(work.index)
    date_idx = {d: i for i, d in enumerate(dates)}
    code_idx = {c: i for i, c in enumerate(codes)}
    matrix = np.full((len(dates), len(codes)), np.nan, dtype=float)
    rows = work[["trade_date", "ts_code"]].to_numpy()
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    for k in range(len(work)):
        matrix[date_idx[str(rows[k][0])], code_idx[str(rows[k][1])]] = vals[k]
    return matrix


def _scoring_inputs(result: Any) -> Dict[str, float]:
    """Derive the competition scoring inputs the local engine can measure."""
    ic = np.asarray(getattr(result, "ic_series", np.asarray([])), dtype=float)
    ic = ic[np.isfinite(ic)]
    ic_mean = float(np.mean(ic)) if ic.size else 0.0
    icir = float(ic_mean / (np.std(ic) + 1e-12)) if ic.size else 0.0
    half = ic.size // 2
    ic_in = float(np.mean(np.abs(ic[:half]))) if half else 0.0
    ic_out = float(np.mean(np.abs(ic[half:]))) if half else 0.0
    retention = ic_out / ic_in if ic_in > 0 else 1.0
    monotonicity = 0.0
    try:
        curves = np.asarray(getattr(result, "group_curves", None), dtype=float)
        if curves.ndim == 2 and curves.shape[1] > 1:
            start = np.where(curves[:, 0] == 0, np.nan, curves[:, 0])
            totals = curves[:, -1] / start - 1.0
            mask = np.isfinite(totals)
            if mask.sum() > 2:
                from scipy.stats import spearmanr
                rho = spearmanr(np.arange(len(totals))[mask], totals[mask]).correlation
                monotonicity = float(abs(rho)) if np.isfinite(rho) else 0.0
    except (TypeError, ValueError):
        monotonicity = 0.0
    return {"ic_mean": ic_mean, "icir": icir,
            "oos_retention": retention, "monotonicity": monotonicity}


def run_local_backtest(source_code: str, df: pd.DataFrame, *,
                       benchmark_ret: Optional[np.ndarray] = None,
                       allow_short: bool = False, n_groups: int = 5,
                       costs: bool = True, up_limit_pct: float = 0.10,
                       neutralize: bool = False,
                       start_date: Optional[int] = None,
                       end_date: Optional[int] = None,
                       factor_matrix: Optional[np.ndarray] = None,
                       with_novelty: bool = False,
                       n_trials: int = 1,
                       tearsheet_path: Optional[str] = None
                       ) -> Dict[str, Any]:
    """Run the canonical engine locally and return labelled metrics.

    ``factor_matrix`` lets a caller supply an already-built factor aligned to the
    sorted ``(trade_date, ts_code)`` order (used by the intraday cross-section
    runner); when omitted the factor is computed from ``source_code``.
    """
    work = df.copy()
    work["trade_date"] = work["trade_date"].astype(str)
    work = work.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    dates = sorted(work["trade_date"].unique())
    codes = sorted(work["ts_code"].astype(str).unique())
    factor = (np.asarray(factor_matrix, dtype=float)
              if factor_matrix is not None
              else _factor_matrix(source_code, work, dates, codes))

    benchmark_fallback = False
    if benchmark_ret is None and "benchmark_ret" not in work.columns:
        # A staging DB built from the daily package may not include index_daily.
        # Fall back to a zero benchmark (labelled), so a run never silently
        # mis-uses a real index it does not have.
        benchmark_ret = np.zeros(len(dates), dtype=float)
        benchmark_fallback = True
    panel = panel_from_frame(work, benchmark_ret=benchmark_ret,
                             up_limit_pct=up_limit_pct)
    if "adj_close" in work and not np.isfinite(pd.to_numeric(work.adj_close, errors="coerce")).any():
        raise ValueError("请求窗口没有可验证的复权收盘参考，不能生成回放结果；请使用已覆盖区间或导入该窗口的真实复权因子。")
    panel.factor = factor

    from server.engine.backtest import BacktestConfig, run_backtest
    from server.engine.friction import FrictionConfig

    if neutralize and not {"industry_id", "circ_mv"}.issubset(work.columns):
        # ``BacktestConfig.neutralize`` is a silent no-op without these columns, so
        # a factor would look "neutralized" locally while the server replay always
        # neutralizes industry and residualizes against the baseline risk factors.
        raise RuntimeError(
            "neutralize 需要 PIT 行业与流通市值列（industry_id / circ_mv），"
            "当前面板没有这两个列；静默跳过会让本地结果与服务端口径不一致。"
            "请去掉 --neutralize，或改用带这两列的 DuckDB 数据源。")
    if allow_short and "shortable" not in work.columns:
        raise RuntimeError("做空回测需要真实的 shortable 借券标记列，请提供借券数据或关闭做空。")
    cfg = BacktestConfig(
        n_groups=n_groups, start_date=start_date, end_date=end_date,
        costs_enabled=costs, neutralize=bool(neutralize),
        friction=FrictionConfig(allow_short=allow_short, max_participation=0.10),
    )
    result = run_backtest(panel, cfg, backend="cpu")

    date_index = {d: i for i, d in enumerate(dates)}
    code_index = {c: i for i, c in enumerate(codes)}
    factor_long = pd.Series(
        [factor[date_index[str(r[0])], code_index[str(r[1])]]
         for r in work[["trade_date", "ts_code"]].to_numpy(dtype=object)],
        index=work.index, dtype=float)

    report: Optional[Dict[str, Any]] = None
    if with_novelty or tearsheet_path is not None:
        from evaluation import novelty as _novelty
        from evaluation import novelty_scoring as _scoring
        from evaluation import tearsheet as _tearsheet

        report = _tearsheet.build(result, work, factor_long.to_numpy(),
                                  n_groups=n_groups, n_trials=int(n_trials))
        if with_novelty:
            try:
                report["novelty"] = _novelty.novelty(work, factor_long).as_dict()
            except ValueError as exc:
                # Novelty is an optional comparison layer for local research. A
                # small or sparse panel may have no same-date Zoo comparison,
                # but that must not discard an otherwise valid local backtest.
                # Keep the reason explicit and leave scoring unavailable; this
                # remains LOCAL_UNVERIFIED and cannot qualify for Official.
                if not str(exc).startswith("novelty:"):
                    raise
                report["novelty"] = {
                    "status": "UNAVAILABLE",
                    "reason": str(exc),
                    "novelty": None,
                    "max_corr": None,
                    "most_correlated": None,
                }
            if report["novelty"].get("status") != "UNAVAILABLE":
                scoring_inputs = _scoring_inputs(result)
                dsr = (report.get("metrics") or {}).get("dsr")
                if dsr is not None:
                    scoring_inputs["dsr"] = dsr
                report["scoring"] = _scoring.final_score(
                    scoring_inputs, report["novelty"]["novelty"],
                    zoo_version=report["novelty"]["zoo_version"])

    approximations: List[str] = []
    if "up_limit" not in work.columns:
        approximations.append(f"涨跌停由收盘价 ±{up_limit_pct:.0%} 近似（包内无 stk_limit）")
    if "st" not in work.columns:
        approximations.append("无 PIT ST 标记：全部按非 ST 处理")
    if "adj_close" not in work.columns:
        approximations.append("无复权价：adj_close = close")
    if "pool_membership" not in work.columns:
        approximations.append("无 PIT 指数成分：universe 视为全市场")
    if "shortable" not in work.columns:
        approximations.append("无借券库存：long-only 不受影响，short 不受支持")
    approximations.append(
        "行业与市值中性化已按配置执行；未执行官方 7 个基线风险因子正交化"
        if neutralize else
        "未启用行业与市值中性化，未执行官方 7 个基线风险因子正交化")
    if benchmark_fallback:
        approximations.append("无基准指数：benchmark_ret=0（本地缺 index_daily，以 0 收益为基准代理）")

    if report is not None:
        report["configuration"] = {
            "n_groups": n_groups, "costs_enabled": costs, "allow_short": allow_short,
            "neutralize": bool(neutralize), "up_limit_pct_fallback": up_limit_pct,
            "matching": "next-day-open", "rebalance": "daily", "n_trials": int(n_trials),
        }
        report["approximations"] = approximations
        if tearsheet_path is not None:
            _tearsheet.write_report(report, tearsheet_path)

    ic = np.asarray(result.ic_series, dtype=float)
    turnover = np.asarray(result.turnover_series, dtype=float)
    out: Dict[str, Any] = {
        "status": "LOCAL_UNVERIFIED",
        "data_mode": "local",
        "engine": "canonical_next_day_open (server.engine.backtest)",
        "matching": "next-day-open",
        "allow_short": allow_short,
        "costs_enabled": costs,
        "neutralize": bool(neutralize),
        "dates": int(len(panel.dates)),
        "codes": int(len(panel.codes)),
        "metrics": {k: (None if v is None or not np.isfinite(v) else round(float(v), 6))
                    for k, v in result.metrics.items()},
        "ic_mean": None if not np.isfinite(ic).any() else round(float(np.nanmean(ic)), 6),
        "turnover_mean": None if not np.isfinite(turnover).any()
        else round(float(np.nanmean(turnover)), 6),
        "friction_breakdown": {k: round(float(v), 6)
                               for k, v in result.friction_breakdown.items()},
        "blocked_orders": dict(result.blocked_orders),
        "approximations": approximations,
        "note": "本地回测：与服务器同一 canonical 引擎，但本地数据可能缺少 PIT 池/ST/借券；"
                "不是官方结果，也不会自动上传。",
    }
    if report is not None:
        out["novelty"] = report.get("novelty")
        out["scoring"] = report.get("scoring")
        out["curves"] = report.get("curves")
        out["tearsheet_metrics"] = report.get("metrics")
        out["style_exposure"] = report.get("style_exposure")
        out["attribution"] = report.get("attribution")
        if tearsheet_path is not None:
            out["tearsheet_path"] = str(tearsheet_path)
    return out


def load_panel(db_path: str | Path, start: str, end: str, *,
               codes: Optional[Iterable[str]] = None,
               adj_table: str = "daily_qfq",
               benchmark_code: str = "000300.SH") -> pd.DataFrame:
    """Load a long panel from a locally imported staging DuckDB.

    When the staging DB also imported ``index_daily`` (the gateway
    ``index_daily.zip``), the benchmark return series for ``benchmark_code``
    is attached as a ``benchmark_ret`` column so a local canonical backtest
    runs out of the box.
    """
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "daily_raw" not in tables:
            raise RuntimeError(f"{db_path} 缺少 daily_raw 表；请先运行 research.import_staging_daily")
        where = "trade_date >= ? AND trade_date <= ?"
        params: list[Any] = [str(start), str(end)]
        code_list = [str(c) for c in (codes or []) if str(c).strip()]
        if code_list:
            where += " AND ts_code IN (" + ",".join(["?"] * len(code_list)) + ")"
            params.extend(code_list)
        raw_columns = {r[0] for r in con.execute("DESCRIBE daily_raw").fetchall()}
        extra_columns = [column for column in (
            "up_limit", "down_limit", "st", "circ_mv", "pb", "turnover_rate",
            "industry_id", "pool_membership", "cash_dividend", "shortable", "borrow_rate")
            if column in raw_columns]
        selection = ", ".join([*_REQUIRED, *extra_columns])
        raw = con.execute(
            f"SELECT {selection} "
            f"FROM daily_raw WHERE {where} AND ts_code NOT IN ({SZSE_B_SHARE_SYMBOL_SQL}) "
            "AND ts_code NOT LIKE '%.UNKNOWN'", params).df()
        listing_fallback_rows = 0
        if "stock_basic_snapshot" in tables:
            listing_columns = {r[0] for r in con.execute("DESCRIBE stock_basic_snapshot").fetchall()}
            if "list_date" in listing_columns:
                listing = con.execute("SELECT ts_code,list_date FROM stock_basic_snapshot").df()
                raw = raw.merge(listing, on="ts_code", how="left", validate="many_to_one")
        if "list_date" not in raw:
            raw["list_date"] = None
        listing_text = raw.list_date.astype(str).str.replace(r"[^0-9]", "", regex=True).str[:8]
        raw["list_date"] = pd.to_datetime(listing_text, format="%Y%m%d", errors="coerce").dt.strftime("%Y%m%d")
        missing_listing = raw.list_date.isna()
        if missing_listing.any():
            # A conservative first-observed history bound, never the requested
            # window's first date (which falsely made every stock a fresh IPO).
            first_seen = con.execute("SELECT ts_code,min(trade_date) first_seen FROM daily_raw GROUP BY ts_code").df().set_index("ts_code").first_seen
            raw.loc[missing_listing, "list_date"] = raw.loc[missing_listing, "ts_code"].map(first_seen)
            listing_fallback_rows = int(missing_listing.sum())
        excluded_identity_rows, excluded_b_share_rows = con.execute(f"""
            SELECT count(*) FILTER (WHERE left(ts_code, 6) NOT IN ({SZSE_B_SHARE_SQL})),
                   count(*) FILTER (WHERE left(ts_code, 6) IN ({SZSE_B_SHARE_SQL}))
            FROM daily_raw WHERE {where}
            AND (ts_code IN ({SZSE_B_SHARE_SYMBOL_SQL}) OR ts_code LIKE '%.UNKNOWN')""",
            params).fetchall()[0]
        if "unresolved_daily_raw" in tables:
            excluded_where = "trade_date >= ? AND trade_date <= ?"
            excluded_params = [str(start), str(end)]
            if code_list:
                excluded_where += " AND left(ts_code, 6) IN (" + ",".join("?" for _ in code_list) + ")"
                excluded_params.extend(code[:6] for code in code_list)
            identity_rows, b_share_rows = con.execute(f"""
                SELECT count(*) FILTER (WHERE left(ts_code, 6) NOT IN ({SZSE_B_SHARE_SQL})),
                       count(*) FILTER (WHERE left(ts_code, 6) IN ({SZSE_B_SHARE_SQL}))
                FROM unresolved_daily_raw WHERE {excluded_where}""", excluded_params).fetchall()[0]
            excluded_identity_rows += identity_rows
            excluded_b_share_rows += b_share_rows
        if adj_table in tables:
            adj = con.execute(
                f"SELECT ts_code, trade_date, close AS adj_close FROM {adj_table} "
                f"WHERE {where}", params).df()
            raw = raw.merge(adj, on=["ts_code", "trade_date"], how="left")
        factor_adjusted_rows = 0
        if "adj_factor" in tables and ("adj_close" not in raw or raw.adj_close.isna().any()):
            factors = con.execute(
                f"SELECT ts_code, trade_date, adj_factor FROM adj_factor WHERE {where}", params).df()
            raw = raw.merge(factors, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
            values = pd.to_numeric(raw.adj_factor, errors="coerce")
            valid = np.isfinite(values) & (values > 0) & np.isfinite(raw.close)
            complete_stock_window = valid.groupby(raw.ts_code).transform("all")
            if "adj_close" not in raw:
                raw["adj_close"] = np.nan
            needs_reference = raw.adj_close.isna().groupby(raw.ts_code).transform("any")
            chosen = complete_stock_window & needs_reference
            # Provider-documented HFQ reference. Never splice it into an archive
            # price scale within one stock, and never persist a manufactured bar.
            raw.loc[chosen, "adj_close"] = raw.loc[chosen, "close"] * values.loc[chosen]
            factor_adjusted_rows = int(chosen.sum())
            raw = raw.drop(columns="adj_factor")
        if "index_daily" in tables:
            bench = con.execute(
                "SELECT trade_date, close FROM index_daily WHERE ts_code=? "
                "AND trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
                [str(benchmark_code), str(start), str(end)]).df()
            if not bench.empty:
                bench = bench.sort_values("trade_date").reset_index(drop=True)
                bench["benchmark_ret"] = bench["close"].shift(-1) / bench["close"] - 1.0
                raw = raw.merge(bench[["trade_date", "benchmark_ret"]],
                                on="trade_date", how="left")
        raw.attrs["unresolved_identity_rows_excluded"] = excluded_identity_rows
        raw.attrs["out_of_scope_b_share_rows_excluded"] = excluded_b_share_rows
        raw.attrs["listing_date_first_observation_rows"] = listing_fallback_rows
        raw.attrs["factor_adjusted_reference_rows"] = factor_adjusted_rows
        if factor_adjusted_rows:
            raw.attrs["adjustment_reference"] = "raw_close * delivered_adj_factor; complete per-stock requested window; no price-table writes"
        if "research_package_metadata" in tables:
            metadata = dict(con.execute("SELECT key,value FROM research_package_metadata").fetchall())
            raw.attrs.update(metadata)
        if "adj_close" in raw:
            raw.attrs["missing_adjusted_close_rows"] = int(raw.adj_close.isna().sum())
        return raw
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local canonical backtest")
    parser.add_argument("--file", required=True, help="factor source (compute_factor)")
    parser.add_argument("--data", required=True, help="panel CSV/Parquet OR staging DuckDB")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--no-costs", action="store_true")
    parser.add_argument("--neutralize", action="store_true",
                        help="按 PIT 行业 + log 市值做截面中性化")
    parser.add_argument("--benchmark-code", default="000300.SH")
    parser.add_argument("--tearsheet", default=None,
                        help="写入 HTML tearsheet 报告到该路径")
    parser.add_argument("--novelty", action="store_true",
                        help="计算与主流因子库的差异度并给出差异度加权评分")
    parser.add_argument("--trials", type=int, default=1,
                        help="为 Deflated Sharpe 声明尝试过的因子数（防过拟合）")
    args = parser.parse_args()

    path = Path(args.data)
    if path.suffix.lower() in {".duckdb", ".db"}:
        df = load_panel(path, args.start, args.end,
                        benchmark_code=args.benchmark_code)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    result = run_local_backtest(
        Path(args.file).read_text(encoding="utf-8"), df,
        allow_short=args.allow_short, n_groups=args.groups, costs=not args.no_costs,
        neutralize=bool(getattr(args, "neutralize", False)),
        with_novelty=bool(args.novelty), n_trials=int(args.trials),
        tearsheet_path=args.tearsheet)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
