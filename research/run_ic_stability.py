"""Run a bounded, raw staging IC stability diagnostic from existing IC files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

from research.stability_diagnostics import (
    compare_neutralized, market_state_from_return, summarize_by_state, summarize_by_year,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _market_states(db: Path, start: str, end: str) -> pd.DataFrame:
    conn = duckdb.connect(str(db), read_only=True)
    try:
        return conn.execute("""
            WITH prices AS (
                SELECT trade_date, ts_code, close,
                       LAG(close) OVER (PARTITION BY ts_code ORDER BY trade_date) AS prev_close
                FROM daily_raw WHERE trade_date BETWEEN ? AND ?
            )
            SELECT trade_date, AVG(close / prev_close - 1.0) AS market_return,
                   COUNT(*) FILTER (WHERE prev_close > 0 AND close > 0) AS stocks
            FROM prices WHERE prev_close > 0 AND close > 0
            GROUP BY trade_date ORDER BY trade_date
        """, [start, end]).df()
    finally:
        conn.close()


def run(db: Path, source_dir: Path, start: str, end: str, output: Path,
        min_days: int = 20) -> dict:
    state = _market_states(db, start, end)
    state["trade_date"] = state.trade_date.astype(str).str.replace("-", "", regex=False)
    state["market_state"] = market_state_from_return(state.market_return)
    reports = {}
    files = sorted(source_dir.glob("*/daily_ic.csv"))
    if not files:
        raise RuntimeError(f"no daily_ic.csv files under {source_dir}")
    for path in files:
        daily = pd.read_csv(path, dtype={"trade_date": str})
        merged = daily.merge(state[["trade_date", "market_state", "market_return", "stocks"]],
                             on="trade_date", how="left", validate="one_to_one")
        reports[path.parent.name] = {
            "source": str(path), "source_sha256": _sha256(path),
            "days": int(len(merged)),
            "year": summarize_by_year(merged, min_days=min_days),
            "market_state": summarize_by_state(merged, min_days=min_days),
            "neutralization": compare_neutralized(merged, None, min_days=min_days),
            "state_definition": "same-day equal-weight close-to-close return; UP > 10bp, DOWN < -10bp, else FLAT",
        }
    result = {
        "status": "UNVERIFIED", "promotion": "BLOCKED", "data_mode": "staging",
        "db": str(db), "source_dir": str(source_dir), "start": start, "end": end,
        "market_state_days": int(len(state)), "market_state_source": "daily_raw close-to-close",
        "factors": reports,
        "limitations": ["raw staging labels only", "no PIT industry or market-cap neutralization",
                        "no adjustment, limits, costs or executable replay"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="daily_staging.duckdb")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--start", default="20210101")
    parser.add_argument("--end", default="20241231")
    parser.add_argument("--output", default="research_runs/ic_stability_20260930")
    parser.add_argument("--min-days", type=int, default=20)
    args = parser.parse_args()
    result = run(Path(args.db), Path(args.source_dir), args.start, args.end,
                 Path(args.output), args.min_days)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
