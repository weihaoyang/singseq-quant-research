"""Run a bounded quantile-portfolio diagnostic on the isolated staging database.

The input is the same point-in-time stock pool and raw-price label used by
``run_staging_factors``.  The output is a diagnostic of cross-sectional
monotonicity, spread returns, membership turnover and abstract cost sensitivity.
It is deliberately not an execution replay: staging has no adjusted prices,
historical limits or complete execution metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from research.run_staging_factors import FACTORS, _calendar_sessions, _metadata_universe


DEFAULT_DB = Path("daily_staging.duckdb")
DEFAULT_COST_BPS = [20.0, 50.0, 100.0]


def _factor_panel(conn: duckdb.DuckDBPyConnection, start: str, end: str,
                  codes: list[str], sessions: pd.Index,
                  price_table: str = "daily_hfq") -> pd.DataFrame:
    placeholders = ",".join("?" for _ in codes)
    if price_table not in {"daily_raw", "daily_hfq", "daily_qfq"}:
        raise ValueError(f"unsupported price table: {price_table}")
    panel = conn.execute(
        f"""SELECT r.ts_code, r.trade_date, p.open, p.close, r.vol, r.amount
            FROM daily_raw r JOIN {price_table} p USING (ts_code, trade_date)
            WHERE r.trade_date BETWEEN ? AND ? AND r.ts_code IN ({placeholders})
            ORDER BY r.ts_code, r.trade_date""",
        [start, end, *codes],
    ).df()
    if panel.empty:
        raise RuntimeError("empty staging panel")
    panel["trade_date"] = panel["trade_date"].astype(str)
    panel["session_id"] = sessions.get_indexer(panel["trade_date"])
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    grouped = panel.groupby("ts_code", sort=False)
    returns = grouped["close"].pct_change(fill_method=None)
    panel["reversal_5"] = -grouped["close"].pct_change(5, fill_method=None)
    panel["momentum_20"] = grouped["close"].pct_change(20, fill_method=None)
    panel["volatility_20"] = returns.groupby(panel.ts_code).transform(
        lambda x: x.rolling(20, min_periods=20).std()
    )
    panel["volume_ratio"] = grouped["vol"].transform(
        lambda x: x.rolling(5, min_periods=5).mean()
        / x.rolling(20, min_periods=20).mean()
    )
    next_open = grouped["open"].shift(-1)
    exit_open = grouped["open"].shift(-6)
    end_session = grouped["session_id"].shift(-6)
    valid = (
        end_session.eq(panel["session_id"] + 6)
        & next_open.gt(0)
        & exit_open.gt(0)
        & grouped["vol"].shift(-1).gt(0)
        & grouped["vol"].shift(-6).gt(0)
    )
    panel["target_ret"] = (exit_open / next_open - 1).where(valid)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel


def _assign_groups(frame: pd.DataFrame, factor: str, quantiles: int) -> pd.DataFrame:
    """Assign cross-sectional groups using only the signal-date factor."""
    work = frame[["ts_code", "trade_date", factor, "target_ret"]].copy()
    work = work.dropna(subset=[factor, "target_ret"])
    if work.empty:
        work["group"] = pd.Series(dtype="Int64")
        return work
    # Rank first so repeated factor values do not make qcut edges unstable.
    work["rank_pct"] = work.groupby("trade_date")[factor].rank(method="first", pct=True)
    work["group"] = np.ceil(work["rank_pct"] * quantiles).astype(int).clip(1, quantiles)
    return work


def _membership_turnover(previous: set[str] | None, current: set[str]) -> float:
    if previous is None or not current:
        return np.nan
    # One-way turnover for an equal-weight group. A full replacement is 1.0.
    return float(1.0 - len(previous & current) / max(len(current), 1))


def _portfolio_series(grouped: pd.DataFrame, factor: str, quantiles: int,
                      rebalance_dates: set[str]) -> tuple[pd.DataFrame, dict]:
    grouped = grouped[grouped.trade_date.isin(rebalance_dates)].copy()
    rows: list[dict] = []
    previous: dict[int, set[str]] = {}
    for date, day in grouped.groupby("trade_date", sort=True):
        for group in range(1, quantiles + 1):
            members = day.loc[day.group == group, "ts_code"].astype(str)
            current = set(members)
            returns = day.loc[day.group == group, "target_ret"]
            rows.append({
                "trade_date": str(date),
                "factor": factor,
                "group": group,
                "n_stocks": int(len(returns)),
                "gross_return": float(returns.mean()) if len(returns) else np.nan,
                "turnover": _membership_turnover(previous.get(group), current),
            })
            previous[group] = current
    result = pd.DataFrame(rows)
    if result.empty:
        return result, {"sessions": 0, "mean_spread": None}
    wide = result.pivot(index="trade_date", columns="group", values="gross_return")
    high = wide.get(quantiles, pd.Series(index=wide.index, dtype=float))
    low = wide.get(1, pd.Series(index=wide.index, dtype=float))
    spread = (high - low).rename("spread_return")
    spread_frame = spread.reset_index()
    result = result.merge(spread_frame, on="trade_date", how="left")
    valid = spread.dropna()
    summary = {
        "sessions": int(len(valid)),
        "mean_spread": float(valid.mean()) if len(valid) else None,
        "positive_spread_share": float((valid > 0).mean()) if len(valid) else None,
        "mean_turnover": float(result.turnover.dropna().mean()) if result.turnover.notna().any() else None,
    }
    return result, summary


def _cost_summary(series: pd.DataFrame, quantiles: int, cost_bps: float) -> dict:
    rate = float(cost_bps) / 10000.0
    if series.empty:
        return {"cost_bps_per_side": float(cost_bps), "sessions": 0,
                "gross_mean": None, "net_mean": None, "cost_mean": None}
    high = series[series.group == quantiles].set_index("trade_date")
    low = series[series.group == 1].set_index("trade_date")
    pair = high[["gross_return", "turnover"]].join(
        low[["gross_return", "turnover"]], lsuffix="_high", rsuffix="_low", how="inner"
    ).dropna(subset=["gross_return_high", "gross_return_low"])
    gross = pair.gross_return_high - pair.gross_return_low
    turnover = pair.turnover_high.fillna(0.0) + pair.turnover_low.fillna(0.0)
    net = gross - turnover * rate
    return {
        "cost_bps_per_side": float(cost_bps),
        "sessions": int(len(pair)),
        "gross_mean": float(gross.mean()) if len(gross) else None,
        "net_mean": float(net.mean()) if len(net) else None,
        "cost_mean": float((turnover * rate).mean()) if len(net) else None,
        "gross_cumulative": float((1.0 + gross).prod() - 1.0) if len(gross) else None,
        "net_cumulative": float((1.0 + net).prod() - 1.0) if len(net) else None,
    }


def run(db: Path, start: str, end: str, max_stocks: int, output: Path,
        anchor_date: str | None = None, quantiles: int = 5,
        rebalance_every: int = 5, cost_bps: list[float] | None = None,
        price_table: str | None = None) -> dict:
    if quantiles < 2 or rebalance_every < 1:
        raise ValueError("quantiles must be >= 2 and rebalance_every must be >= 1")
    costs = DEFAULT_COST_BPS if cost_bps is None else [float(x) for x in cost_bps]
    if any(x < 0 or not np.isfinite(x) for x in costs):
        raise ValueError("cost_bps must be finite and non-negative")
    conn = duckdb.connect(str(db), read_only=True)
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        if "daily_raw" not in tables:
            raise RuntimeError("staging database is missing daily_raw")
        anchor = anchor_date or str(conn.execute(
            "SELECT MIN(trade_date) FROM daily_raw WHERE trade_date >= ?", [start]
        ).fetchone()[0])
        codes, universe = _metadata_universe(conn, anchor, max_stocks)
        sessions = _calendar_sessions(conn, start, end)
        if len(codes) < 50:
            raise RuntimeError(f"need at least 50 stocks, found {len(codes)}")
        selected_price_table = price_table or ("daily_hfq" if "daily_hfq" in tables else "daily_raw")
        if selected_price_table not in {"daily_raw", "daily_hfq", "daily_qfq"} or selected_price_table not in tables:
            raise RuntimeError(f"staging database is missing price table: {selected_price_table}")
        panel = _factor_panel(conn, start, end, codes, sessions, selected_price_table)
    finally:
        conn.close()

    signal_sessions = [str(day) for i, day in enumerate(sessions[:-6]) if i % rebalance_every == 0]
    grouped_reports: dict[str, dict] = {}
    all_rows: list[pd.DataFrame] = []
    for factor in FACTORS:
        assigned = _assign_groups(panel, factor, quantiles)
        series, summary = _portfolio_series(assigned, factor, quantiles, set(signal_sessions))
        grouped_reports[factor] = {"gross": summary,
                                   "cost_sensitivity": [_cost_summary(series, quantiles, cost) for cost in costs]}
        if not series.empty:
            all_rows.append(series)
        factor_dir = output / factor
        factor_dir.mkdir(parents=True, exist_ok=True)
        series.to_csv(factor_dir / "quantile_returns.csv", index=False)
        grouped_reports[factor]["annual"] = {
            str(year): _cost_summary(series[series.trade_date.str[:4] == str(year)], quantiles, costs[0])
            for year in sorted({str(x)[:4] for x in series.trade_date})
        } if not series.empty else {}

    panel_hash = hashlib.sha256(pd.util.hash_pandas_object(
        panel[["ts_code", "trade_date", "open", "close", "vol", "amount", *FACTORS, "target_ret"]],
        index=False,
    ).values.tobytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "UNVERIFIED",
        "promotion": "BLOCKED",
        "db": str(db),
        "start": start,
        "end": end,
        "stocks": len(codes),
        "rows": int(len(panel)),
        "panel_hash": panel_hash,
        "price_table": selected_price_table,
        "universe": universe,
        "calendar_sessions": int(len(sessions)),
        "signal_sessions": int(len(signal_sessions)),
        "quantiles": quantiles,
        "rebalance_every": rebalance_every,
        "cost_bps_per_side": costs,
        "label": f"{selected_price_table} close signal at t; {selected_price_table} next open entry; {selected_price_table} open exit after five sessions",
        "factors": grouped_reports,
        "limitations": [
            "supplier adjusted price table is used; adjustment-factor provenance is not independently reconciled",
            "no historical up/down limits, ST or delisting execution checks",
            "cost sensitivity is an abstract membership-turnover diagnostic, not broker fills",
            "no market-cap, industry or capacity neutralization",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(output / "all_quantile_returns.csv", index=False)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--start", default="20210101")
    parser.add_argument("--end", default="20241231")
    parser.add_argument("--max-stocks", type=int, default=512)
    parser.add_argument("--output", default="research_runs/staging_portfolios_v1")
    parser.add_argument("--anchor-date")
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--rebalance-every", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=DEFAULT_COST_BPS)
    parser.add_argument("--price-table", choices=["daily_raw", "daily_hfq", "daily_qfq"], help="price table; defaults to daily_hfq when available")
    args = parser.parse_args()
    if not 50 <= args.max_stocks <= 512:
        parser.error("--max-stocks must be between 50 and 512")
    result = run(Path(args.db), args.start, args.end, args.max_stocks, Path(args.output),
                 args.anchor_date, args.quantiles, args.rebalance_every, args.cost_bps,
                 args.price_table)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
