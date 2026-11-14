"""Read-only audit for the isolated daily staging database."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import duckdb

def audit(db: Path) -> dict:
    conn = duckdb.connect(str(db), read_only=True)
    try:
        tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
        required = {"daily_raw", "stock_basic_snapshot", "delisted_basic_snapshot", "trade_cal_snapshot", "import_manifest"}
        missing = sorted(required - tables)
        result = {"db": str(db), "status": "FAILED" if missing else "OK", "missing_tables": missing}
        if missing:
            return result
        result.update({
            "daily_rows": conn.execute("SELECT COUNT(*) FROM daily_raw").fetchone()[0],
            "daily_stocks": conn.execute("SELECT COUNT(DISTINCT ts_code) FROM daily_raw").fetchone()[0],
            "daily_min_date": conn.execute("SELECT MIN(trade_date) FROM daily_raw").fetchone()[0],
            "daily_max_date": conn.execute("SELECT MAX(trade_date) FROM daily_raw").fetchone()[0],
            "manifest_failed": conn.execute("SELECT COUNT(*) FROM import_manifest WHERE status='FAILED'").fetchone()[0],
            "empty_members": conn.execute("SELECT COUNT(*) FROM import_manifest WHERE status='OK' AND rows=0").fetchone()[0],
            "duplicate_keys": conn.execute("SELECT COUNT(*) FROM (SELECT ts_code, trade_date FROM daily_raw GROUP BY 1,2 HAVING COUNT(*) > 1)").fetchone()[0],
            "invalid_ohlcv": conn.execute("SELECT COUNT(*) FROM daily_raw WHERE open < 0 OR high < 0 OR low < 0 OR close < 0 OR vol < 0 OR amount < 0 OR high < low OR high < open OR high < close OR low > open OR low > close").fetchone()[0],
            "unknown_symbols": conn.execute("SELECT COUNT(DISTINCT ts_code) FROM daily_raw WHERE ts_code LIKE '%.UNKNOWN'").fetchone()[0],
            "calendar_rows": conn.execute("SELECT COUNT(*) FROM trade_cal_snapshot").fetchone()[0],
            "stock_list_rows": conn.execute("SELECT COUNT(*) FROM stock_basic_snapshot").fetchone()[0],
            "delisted_rows": conn.execute("SELECT COUNT(*) FROM delisted_basic_snapshot").fetchone()[0],
        })
        failures = []
        for key in ("manifest_failed", "duplicate_keys", "invalid_ohlcv"):
            if result[key]: failures.append(f"{key}={result[key]}")
        if result["empty_members"]:
            result["warnings"] = [f"empty_members={result['empty_members']}"]
        result["status"] = "FAILED" if failures else "PASSED"
        result["failures"] = failures
        return result
    finally:
        conn.close()

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--db", required=True); ap.add_argument("--output")
    args = ap.parse_args()
    result = audit(Path(args.db))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    return 0 if result["status"] in {"OK", "PASSED"} else 1
if __name__ == "__main__": raise SystemExit(main())
