"""Check staging factor signals against API adjusted-price labels on common keys."""
from __future__ import annotations

import argparse
import json
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from research.run_staging_factors import _metadata_universe

FACTORS = ["reversal_5", "momentum_20", "volatility_20", "volume_ratio"]

def _summary(frame: pd.DataFrame, factor: str) -> dict:
    rows = []
    for day, group in frame.groupby("trade_date", sort=True):
        good = group[[factor, "target_ret"]].replace([np.inf, -np.inf], np.nan).dropna()
        value = np.nan
        if len(good) >= 50 and good[factor].nunique() > 1 and good.target_ret.nunique() > 1:
            value = good[factor].corr(good.target_ret, method="spearman")
        rows.append({"trade_date": day, "n": len(good), "rank_ic": value})
    daily = pd.DataFrame(rows)
    values = daily.rank_ic.dropna().to_numpy()
    sd = values.std(ddof=1) if len(values) > 1 else 0.0
    return {"days": int(len(values)), "mean_rank_ic": float(values.mean()) if len(values) else None,
            "icir_daily": float(values.mean() / sd) if len(values) and sd > 0 else None,
            "positive_share": float((values > 0).mean()) if len(values) else None,
            "daily": daily}

def run(staging_db: Path, api_db: Path, start: str, end: str, max_stocks: int, output: Path, anchor_date: str | None = None) -> dict:
    stage = duckdb.connect(str(staging_db), read_only=True)
    try:
        anchor = anchor_date or str(stage.execute("SELECT MIN(trade_date) FROM daily_raw WHERE trade_date >= ?", [start]).fetchone()[0])
        codes, universe = _metadata_universe(stage, anchor, max_stocks)
        if len(codes) < 50: raise RuntimeError(f"need at least 50 staging stocks, found {len(codes)}")
        stage.register("_codes", pd.DataFrame({"ts_code": codes}))
        warmup = (pd.Timestamp(start) - pd.Timedelta(days=90)).strftime("%Y%m%d")
        stage_frame = stage.execute(
            """SELECT d.ts_code,d.trade_date,d.open,d.close,d.vol,d.amount
               FROM daily_raw d JOIN _codes USING(ts_code)
               WHERE d.trade_date BETWEEN ? AND ? ORDER BY d.ts_code,d.trade_date""",
            [warmup, end]).df()
    finally:
        stage.close()

    api = duckdb.connect(str(api_db), read_only=True)
    api_end = (pd.Timestamp(end) + pd.Timedelta(days=45)).strftime("%Y%m%d")
    api_parts = []
    try:
        # Bound the largest intermediate join. A single query over all symbols
        # can materialize several hundred thousand rows per auxiliary table.
        for offset in range(0, len(codes), 64):
            chunk = codes[offset:offset + 64]
            marks = ",".join("?" for _ in chunk)
            api_parts.append(api.execute(
                f"""SELECT CAST(d.ts_code AS VARCHAR) ts_code, CAST(d.trade_date AS VARCHAR) trade_date,
                          d.open,d.close,a.adj_factor,b.circ_mv,l.up_limit,l.down_limit
                   FROM daily d
                   LEFT JOIN adj_factor a USING(ts_code,trade_date)
                   LEFT JOIN daily_basic b USING(ts_code,trade_date)
                   LEFT JOIN stk_limit l USING(ts_code,trade_date)
                   WHERE d.ts_code IN ({marks}) AND CAST(d.trade_date AS VARCHAR) BETWEEN ? AND ?
                   ORDER BY d.ts_code,d.trade_date""", [*chunk, warmup, api_end]).df())
            gc.collect()
        api_frame = pd.concat(api_parts, ignore_index=True) if api_parts else pd.DataFrame()
    finally:
        api.close()
        del api_parts
        gc.collect()

    stage_frame.trade_date = stage_frame.trade_date.astype(str).str.replace("-", "", regex=False)
    api_frame.trade_date = api_frame.trade_date.astype(str).str.replace("-", "", regex=False)
    stage_frame = stage_frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    sg = stage_frame.groupby("ts_code", sort=False)
    returns = sg.close.pct_change(fill_method=None)
    stage_frame["reversal_5"] = -sg.close.pct_change(5, fill_method=None)
    stage_frame["momentum_20"] = sg.close.pct_change(20, fill_method=None)
    stage_frame["volatility_20"] = returns.groupby(stage_frame.ts_code).transform(lambda x: x.rolling(20, min_periods=20).std())
    stage_frame["volume_ratio"] = sg.vol.transform(lambda x: x.rolling(5, min_periods=5).mean() / x.rolling(20, min_periods=20).mean())
    merged = stage_frame.merge(api_frame, on=["ts_code", "trade_date"], how="inner", suffixes=("_stage", "_api"))
    merged["adj_open"] = merged.open_api * merged.adj_factor
    merged = merged.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    sessions = pd.Index(sorted(merged.trade_date.unique()))
    merged["session_id"] = sessions.get_indexer(merged.trade_date)
    g = merged.groupby("ts_code", sort=False)
    entry = g.adj_open.shift(-1); exit_price = g.adj_open.shift(-6)
    exit_session = g.session_id.shift(-6)
    valid = exit_session.eq(merged.session_id + 6) & entry.gt(0) & exit_price.gt(0)
    merged["target_ret"] = (exit_price / entry - 1).where(valid)
    sample = merged[merged.trade_date.between(start, end)].copy()
    reports = {}
    output.mkdir(parents=True, exist_ok=True)
    for factor in FACTORS:
        result = _summary(sample, factor)
        result["daily"].to_csv(output / f"{factor}_daily_ic.csv", index=False)
        result.pop("daily")
        reports[factor] = result
    result = {
        "status": "UNVERIFIED", "promotion": "BLOCKED", "staging_db": str(staging_db), "api_db": str(api_db),
        "start": start, "end": end, "candidate_stocks": len(codes), "staging_rows": len(stage_frame), "universe": universe,
        "api_rows": len(api_frame), "matched_rows": len(merged), "label": "API adj_factor adjusted next-open to six-session-open return",
        "coverage": {"adj_factor_positive": float((sample.adj_factor > 0).mean()), "circ_mv_positive": float((sample.circ_mv > 0).mean()), "limits_present": float((sample.up_limit.notna() & sample.down_limit.notna()).mean())},
        "factors": reports,
        "limitations": ["API database is not full-audit complete", "common-key diagnostic is not portfolio replay", "PIT universe and amount units remain unresolved"],
    }
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--staging-db", default="daily_staging.duckdb"); ap.add_argument("--api-db", default=str(Path(__file__).resolve().parents[1] / "tushare_data.duckdb")); ap.add_argument("--start", default="20210101"); ap.add_argument("--end", default="20241231"); ap.add_argument("--max-stocks", type=int, default=512); ap.add_argument("--anchor-date", help="PIT universe anchor, defaults to first session on/after --start"); ap.add_argument("--output", default="research_runs/common_key_factor_check_v1")
    args = ap.parse_args()
    print(json.dumps(run(Path(args.staging_db), Path(args.api_db), args.start, args.end, args.max_stocks, Path(args.output), args.anchor_date), ensure_ascii=False, indent=2, allow_nan=False))
    return 0
if __name__ == "__main__": raise SystemExit(main())
