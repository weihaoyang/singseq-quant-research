"""Run an auditable raw-price factor screen on the isolated staging database.

This is deliberately an IC-only research gate. The source has no point-in-time
adjustment factors, limits or market-cap history, so the output is UNVERIFIED
and cannot be promoted to a trading or portfolio result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

FACTORS = ["reversal_5", "momentum_20", "volatility_20", "volume_ratio"]

def _metadata_universe(conn, anchor: str, max_stocks: int) -> tuple[list[str], dict]:
    """Select the universe using only metadata visible at the anchor date."""
    tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    if "stock_basic_snapshot" not in tables or "delisted_basic_snapshot" not in tables:
        raise RuntimeError("staging metadata snapshots are required for PIT universe selection")
    active = conn.execute("""
        SELECT ts_code, list_date, delist_date FROM stock_basic_snapshot
        WHERE ts_code IS NOT NULL AND ts_code <> ''
    """).df()
    delisted = conn.execute("""
        SELECT "TS代码" AS ts_code, "上市日期" AS list_date, "退市日期" AS delist_date
        FROM delisted_basic_snapshot WHERE "TS代码" IS NOT NULL AND "TS代码" <> ''
    """).df()
    meta = pd.concat([active, delisted], ignore_index=True).drop_duplicates("ts_code", keep="first")
    for col in ("list_date", "delist_date"):
        meta[col] = meta[col].astype(str).str.replace("-", "", regex=False).replace({"": np.nan, "nan": np.nan, "NaT": np.nan})
    meta["list_date"] = meta["list_date"].fillna("99999999")
    meta["delist_date"] = meta["delist_date"].fillna("99999999")
    anchor_codes = conn.execute("SELECT DISTINCT ts_code FROM daily_raw WHERE trade_date = ?", [anchor]).df()
    eligible = anchor_codes.merge(meta, on="ts_code", how="left")
    eligible = eligible[(eligible.list_date != "99999999") & (eligible.list_date <= anchor)]
    eligible = eligible[eligible.delist_date >= anchor]
    eligible = eligible.sort_values("ts_code")
    codes = eligible.ts_code.head(max_stocks).tolist()
    return codes, {"anchor_date": anchor, "anchor_candidates": int(len(anchor_codes)),
                   "metadata_matched": int(eligible.shape[0]), "selected": len(codes),
                   "metadata_source": "stock_basic_snapshot+delisted_basic_snapshot"}

def _calendar_sessions(conn, start: str, end: str) -> pd.Index:
    tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    if "trade_cal_snapshot" not in tables:
        raise RuntimeError("trade_cal_snapshot is required for calendar-safe labels")
    cal = conn.execute(
        "SELECT \"日期\" AS cal_date FROM trade_cal_snapshot WHERE \"是否交易\" = '交易'"
    ).df()
    sessions = pd.to_datetime(cal.cal_date, errors="coerce").dt.strftime("%Y%m%d")
    sessions = pd.Index(sorted(sessions.dropna().unique()))
    sessions = sessions[(sessions >= start) & (sessions <= end)]
    if len(sessions) < 50:
        raise RuntimeError("staging trading calendar has too few sessions in research range")
    return sessions

def _rank_ic(frame: pd.DataFrame, factor: str, min_stocks: int = 50) -> pd.DataFrame:
    rows = []
    for day, group in frame.groupby("trade_date", sort=True):
        good = group[[factor, "target_ret"]].replace([np.inf, -np.inf], np.nan).dropna()
        ic = np.nan
        if len(good) >= min_stocks and good[factor].nunique() > 1 and good.target_ret.nunique() > 1:
            ic = good[factor].corr(good.target_ret, method="spearman")
        rows.append({"trade_date": day, "n": len(good), "rank_ic": ic})
    return pd.DataFrame(rows)

def _summary(ic: pd.DataFrame) -> dict:
    values = ic.rank_ic.dropna().to_numpy()
    if len(values) == 0:
        return {"days": 0, "mean_rank_ic": None, "icir_daily": None, "positive_share": None}
    sd = values.std(ddof=1) if len(values) > 1 else 0.0
    return {"days": int(len(values)), "mean_rank_ic": float(values.mean()),
            "icir_daily": float(values.mean() / sd) if sd > 0 else None,
            "positive_share": float((values > 0).mean())}


def _quantile_portfolios(frame: pd.DataFrame, factor: str, groups: int = 5,
                         min_stocks: int = 50) -> pd.DataFrame:
    """Build descriptive, raw-label quantile portfolios from the staging panel.

    This is deliberately a research diagnostic, not an execution replay:
    target_ret is the real raw open-to-open label, while limits, adjustment
    factors and point-in-time membership are unavailable in staging.  Missing
    observations stay missing and memberships are deterministic by code.
    """
    if groups < 2:
        raise ValueError("groups must be at least 2")
    required = {"trade_date", "ts_code", factor, "target_ret"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing portfolio columns: {sorted(required - set(frame.columns))}")
    previous: dict[str, float] = {}
    rows: list[dict] = []
    for day, raw in frame.groupby("trade_date", sort=True):
        good = raw[["ts_code", factor, "target_ret"]].replace(
            [np.inf, -np.inf], np.nan).dropna()
        if len(good) < min_stocks or good[factor].nunique() < groups:
            rows.append({"trade_date": str(day), "n": int(len(good)),
                         "coverage": float(len(good) / max(len(raw), 1)),
                         **{f"G{i + 1}": None for i in range(groups)},
                         "long_short": None, "turnover": None})
            continue
        # Stable code ordering resolves ties without using future returns.
        good = good.sort_values([factor, "ts_code"], kind="stable").reset_index(drop=True)
        good["group"] = np.minimum(
            groups - 1, (np.arange(len(good)) * groups // len(good)).astype(int))
        group_returns = good.groupby("group", sort=False)["target_ret"].mean()
        weights: dict[str, float] = {}
        top = good.loc[good.group == groups - 1, "ts_code"].tolist()
        bottom = good.loc[good.group == 0, "ts_code"].tolist()
        for code in top:
            weights[str(code)] = 0.5 / len(top)
        for code in bottom:
            weights[str(code)] = -0.5 / len(bottom)
        if previous:
            names = set(previous) | set(weights)
            turnover = 0.5 * sum(abs(weights.get(name, 0.0) - previous.get(name, 0.0))
                                for name in names)
        else:
            turnover = 1.0
        previous = weights
        values = {f"G{i + 1}": _finite_or_none(group_returns.get(i))
                  for i in range(groups)}
        values.update({
            "trade_date": str(day), "n": int(len(good)),
            "coverage": float(len(good) / max(len(raw), 1)),
            "long_short": _finite_or_none(
                group_returns.get(groups - 1) - group_returns.get(0)),
            "turnover": float(turnover),
        })
        rows.append(values)
    return pd.DataFrame(rows)


def _finite_or_none(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _portfolio_summary(daily: pd.DataFrame, groups: int) -> dict:
    if daily.empty:
        return {"days": 0, "status": "UNAVAILABLE"}
    result = {"days": int(daily["long_short"].notna().sum()),
              "coverage_mean": _finite_or_none(daily.coverage.mean()),
              "turnover_mean": _finite_or_none(daily.turnover.mean())}
    for name in [f"G{i + 1}" for i in range(groups)] + ["long_short"]:
        values = pd.to_numeric(daily[name], errors="coerce").dropna().to_numpy()
        if not len(values):
            result[name] = {"days": 0, "total_return": None,
                            "annual_return": None}
            continue
        total = float(np.prod(1.0 + values) - 1.0)
        result[name] = {
            "days": int(len(values)), "total_return": total,
            "annual_return": float((1.0 + total) ** (252.0 / len(values)) - 1.0)
            if 1.0 + total > 0 else None,
        }
    return result

def run(db: Path, start: str, end: str, max_stocks: int, output: Path, anchor_date: str | None = None, price_table: str | None = None) -> dict:
    conn = duckdb.connect(str(db), read_only=True)
    try:
        tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
        if "daily_raw" not in tables:
            raise RuntimeError("staging database is missing daily_raw")
        selected_price_table = price_table or ("daily_hfq" if "daily_hfq" in tables else "daily_raw")
        if selected_price_table not in {"daily_raw", "daily_hfq", "daily_qfq"} or selected_price_table not in tables:
            raise RuntimeError(f"staging database is missing price table: {selected_price_table}")
        anchor = anchor_date or str(conn.execute("SELECT MIN(trade_date) FROM daily_raw WHERE trade_date >= ?", [start]).fetchone()[0])
        codes, universe = _metadata_universe(conn, anchor, max_stocks)
        sessions = _calendar_sessions(conn, start, end)
        if len(codes) < 50:
            raise RuntimeError(f"need at least 50 stocks, found {len(codes)}")
        placeholders = ",".join("?" for _ in codes)
        panel = conn.execute(
            f"""SELECT r.ts_code, r.trade_date, p.open, p.close, r.vol, r.amount
                FROM daily_raw r JOIN {selected_price_table} p USING (ts_code, trade_date)
                WHERE r.trade_date BETWEEN ? AND ? AND r.ts_code IN ({placeholders})
                ORDER BY r.ts_code, r.trade_date""", [start, end, *codes]).df()
    finally:
        conn.close()
    panel.trade_date = panel.trade_date.astype(str)
    panel["session_id"] = sessions.get_indexer(panel.trade_date)
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    g = panel.groupby("ts_code", sort=False)
    returns = g.close.pct_change(fill_method=None)
    panel["reversal_5"] = -g.close.pct_change(5, fill_method=None)
    panel["momentum_20"] = g.close.pct_change(20, fill_method=None)
    panel["volatility_20"] = returns.groupby(panel.ts_code).transform(lambda x: x.rolling(20, min_periods=20).std())
    panel["volume_ratio"] = g.vol.transform(lambda x: x.rolling(5, min_periods=5).mean() / x.rolling(20, min_periods=20).mean())
    next_open = g.open.shift(-1)
    exit_open = g.open.shift(-6)
    end_session = g.session_id.shift(-6)
    valid = end_session.eq(panel.session_id + 6) & next_open.gt(0) & exit_open.gt(0) & g.vol.shift(-1).gt(0) & g.vol.shift(-6).gt(0)
    panel["target_ret"] = (exit_open / next_open - 1).where(valid)
    reports = {}
    for factor in FACTORS:
        factor_dir = output / factor
        factor_dir.mkdir(parents=True, exist_ok=True)
        daily = _rank_ic(panel, factor)
        daily.to_csv(factor_dir / "daily_ic.csv", index=False)
        portfolio_reports = {}
        for groups in (5, 10):
            portfolio_daily = _quantile_portfolios(panel, factor, groups=groups)
            portfolio_daily.to_csv(factor_dir / f"quantile_{groups}.csv", index=False)
            portfolio_reports[str(groups)] = {
                "summary": _portfolio_summary(portfolio_daily, groups),
                "status": "UNVERIFIED_RAW_LABEL_DIAGNOSTIC",
                "execution": {
                    "status": "BLOCKED",
                    "reason": "staging lacks authoritative adjustment factors, opening limits, PIT membership and ST/delist execution state",
                },
                "cost_sensitivity": {
                    "status": "BLOCKED",
                    "requested_bps_per_side": [20, 50, 100],
                    "reason": "cannot apply execution costs before fills and rejected orders are known",
                },
            }
        reports[factor] = {
            "all": _summary(daily),
            "2021_2022": _summary(daily[daily.trade_date.between("20210101", "20221231")]),
            "2023": _summary(daily[daily.trade_date.between("20230101", "20231231")]),
            "2024": _summary(daily[daily.trade_date.between("20240101", "20241231")]),
            "quantile_portfolios": portfolio_reports,
        }
    panel_hash = hashlib.sha256(pd.util.hash_pandas_object(panel[["ts_code", "trade_date", "open", "close", "vol", "amount", *FACTORS, "target_ret"]], index=False).values.tobytes()).hexdigest()
    result = {
        "status": "UNVERIFIED", "promotion": "BLOCKED", "db": str(db),
        "start": start, "end": end, "stocks": len(codes), "rows": len(panel),
        "price_table": selected_price_table,
        "panel_hash": panel_hash, "universe": universe, "calendar_sessions": len(sessions), "factors": reports,
        "label": f"{selected_price_table} close signal at t; {selected_price_table} next open entry; {selected_price_table} open exit after five sessions",
        "limitations": [
            "supplier adjusted price table is used; adjustment-factor provenance is not independently reconciled",
            "no point-in-time industry or market cap",
            "no up/down limits",
            "no costs or execution replay",
            "unknown historical-code mappings retained",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="daily_staging.duckdb")
    ap.add_argument("--start", default="20210101")
    ap.add_argument("--end", default="20241231")
    ap.add_argument("--max-stocks", type=int, default=128)
    ap.add_argument("--output", default="research_runs/staging_factor_screen_v1")
    ap.add_argument("--anchor-date", help="PIT universe anchor, defaults to first session on/after --start")
    ap.add_argument("--price-table", choices=["daily_raw", "daily_hfq", "daily_qfq"], help="price table; defaults to daily_hfq when available")
    args = ap.parse_args()
    if not 50 <= args.max_stocks <= 512: ap.error("--max-stocks must be between 50 and 512")
    print(json.dumps(run(Path(args.db), args.start, args.end, args.max_stocks, Path(args.output), args.anchor_date, args.price_table), ensure_ascii=False, indent=2, allow_nan=False))
    return 0

if __name__ == "__main__": raise SystemExit(main())
