"""Import downloaded A-share daily archives into an isolated, resumable DuckDB.

The importer never opens the live research database. Each zip member is a
checkpointed unit so an interrupted run can resume without duplicate keys.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
from filelock import FileLock, Timeout
from quant_sdk.archive_scope import SZSE_B_SHARE_CODES, SZSE_B_SHARE_SQL

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(r"E:\quant_staging\A股数据_zip")
DEFAULT_DB = Path("daily_staging.duckdb")
LIVE_DB = (ROOT / "tushare_data.duckdb").resolve()

RAW_COLUMNS = {
    "日期": "trade_date", "股票代码": "source_code", "开盘": "open",
    "收盘": "close", "最高": "high", "最低": "low",
    "成交量": "vol", "成交额": "amount", "振幅": "amplitude",
    "涨跌幅": "pct_chg", "涨跌额": "change", "换手率": "turnover_rate",
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()

def _assert_isolated(db: Path) -> None:
    if db.resolve() == LIVE_DB:
        raise RuntimeError(f"refusing to write live database: {db}")
    db.parent.mkdir(parents=True, exist_ok=True)

def _read_csv(path_or_bytes, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path_or_bytes, encoding="utf-8-sig", low_memory=False, **kwargs)


def _parse_yyyymmdd(values) -> "pd.Series":
    """Robustly normalise '20230103' / '2023-01-03' / int columns to YYYYMMDD.

    ``pd.to_datetime`` on a bare integer-like column interprets the number as
    nanoseconds-since-epoch and silently maps every row to 1970-01-01, which
    would corrupt an entire import. Parse the 8-digit text form explicitly.
    """
    text = values.astype(str).str.replace(r"[^0-9]", "", regex=True).str.slice(0, 8)
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce").dt.strftime("%Y%m%d")

def _load_symbol_map(path: Path) -> pd.DataFrame:
    frame = _read_csv(path, dtype=str).fillna("")
    frame.columns = [str(c).strip() for c in frame.columns]
    rename = {
        "TS代码": "ts_code", "ts_code": "ts_code", "symbol": "ts_code", "代码": "ts_code",
        "股票代码": "source_code", "source_code": "source_code",
        "股票名称": "name", "name": "name", "名称": "name",
        "所属行业": "industry", "industry": "industry", "行业": "industry",
        "上市状态": "list_status", "list_status": "list_status",
        "上市日期": "list_date", "list_date": "list_date",
        "退市日期": "delist_date", "delist_date": "delist_date",
        "交易所代码": "exchange", "exchange": "exchange", "交易所": "exchange",
    }
    frame = frame.rename(columns=rename)
    if "source_code" not in frame and "ts_code" in frame:
        frame["source_code"] = (frame["ts_code"].astype(str)
                                .str.extract(r"(\d+)", expand=False).str.zfill(6))
    if "source_code" not in frame:
        raise ValueError(f"{path.name} 缺少 ts_code/source_code 列")
    if "ts_code" not in frame:
        raise ValueError(f"{path.name} 缺少 ts_code 列")
    keep = [c for c in ["ts_code", "source_code", "name", "industry", "list_status", "list_date", "delist_date", "exchange"] if c in frame]
    return frame[keep].drop_duplicates("source_code")


# 网关发布的英文文件名 与 本地 staging 的中文文件名 两种布局都支持。
_ARCHIVE_ALIASES = {
    "daily": ("daily.zip", "equity_daily_raw.zip"),
    "qfq": ("daily_qfq.zip", "equity_daily_qfq.zip"),
    "hfq": ("daily_hfq.zip", "equity_daily_hfq.zip"),
}
_META_ALIASES = {
    "stock": ("股票列表.csv", "stock_list.csv"),
    "delisted": ("退市股票列表.csv", "delisted_list.csv"),
    "calendar": ("交易日历.csv", "trade_calendar.csv"),
}


def _resolve(source: Path, *names: str) -> Path | None:
    for name in names:
        candidate = source / name
        if candidate.exists():
            return candidate
    return None


_INDEX_ALIASES = {"index": ("index_daily.zip", "equity_index_daily.zip")}
_INDEX_COLUMNS = {
    "日期": "trade_date", "trade_date": "trade_date",
    "收盘": "close", "close": "close",
    "指数代码": "ts_code", "ts_code": "ts_code",
    "index_code": "ts_code", "代码": "ts_code",
}


def _ensure_index_table(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS index_daily (
          ts_code VARCHAR NOT NULL, trade_date VARCHAR NOT NULL, close DOUBLE,
          source_member VARCHAR, source_sha256 VARCHAR,
          PRIMARY KEY (ts_code, trade_date)
        )
    """)


def _index_member_frame(raw: bytes, member_name: str) -> pd.DataFrame:
    frame = _read_csv(io.BytesIO(raw), dtype=str).fillna("")
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = frame.rename(columns=_INDEX_COLUMNS)
    if "trade_date" not in frame.columns or "close" not in frame.columns:
        raise ValueError(f"index member missing date/close columns: {list(frame.columns)}")
    if "ts_code" not in frame.columns:
        frame["ts_code"] = Path(member_name).stem
    frame["trade_date"] = _parse_yyyymmdd(frame["trade_date"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "close", "ts_code"])
    return frame[["ts_code", "trade_date", "close"]]


def _import_index_archive(conn, archive: Path) -> dict:
    archive_hash = sha256(archive)
    members = _archive_members(archive, ".csv")
    done = 0
    rows = 0
    failures = []
    _ensure_index_table(conn)
    with zipfile.ZipFile(archive) as zf:
        for member in members:
            try:
                frame = _index_member_frame(zf.read(member), member)
                frame["source_member"] = member
                frame["source_sha256"] = archive_hash
                conn.register("_index_frame", frame)
                conn.execute("INSERT OR REPLACE INTO index_daily SELECT * FROM _index_frame")
                conn.unregister("_index_frame")
                done += 1
                rows += len(frame)
            except Exception as exc:  # noqa: BLE001
                failures.append({"member": member, "error": f"{type(exc).__name__}: {exc}"})
    conn.execute("CHECKPOINT")
    return {"archive": str(archive), "sha256": archive_hash, "members": len(members),
            "imported": done, "rows": rows, "failures": failures}

def _load_optional_table(path: Path, kind: str) -> pd.DataFrame:
    frame = _read_csv(path, dtype=str).fillna("")
    frame.columns = [str(c).strip() for c in frame.columns]
    frame.insert(0, "source_file", path.name)
    return frame

def _create_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS import_manifest (
          archive VARCHAR, member_name VARCHAR, member_crc UBIGINT, member_size BIGINT,
          member_sha256 VARCHAR, rows BIGINT, min_date VARCHAR, max_date VARCHAR,
          status VARCHAR, imported_at TIMESTAMP, error VARCHAR,
          PRIMARY KEY (archive, member_name)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_raw (
          ts_code VARCHAR, source_code VARCHAR, trade_date VARCHAR, open DOUBLE, high DOUBLE,
          low DOUBLE, close DOUBLE, vol DOUBLE, amount DOUBLE, amplitude DOUBLE, pct_chg DOUBLE,
          change DOUBLE, turnover_rate DOUBLE, source_member VARCHAR, source_sha256 VARCHAR,
          PRIMARY KEY (ts_code, trade_date)
        )
    """)
    conn.execute("CREATE TABLE IF NOT EXISTS stock_basic_snapshot AS SELECT * FROM (SELECT CAST(NULL AS VARCHAR) AS ts_code) WHERE FALSE")
    conn.execute("CREATE TABLE IF NOT EXISTS delisted_basic_snapshot AS SELECT * FROM (SELECT CAST(NULL AS VARCHAR) AS ts_code) WHERE FALSE")
    conn.execute("CREATE TABLE IF NOT EXISTS trade_cal_snapshot AS SELECT * FROM (SELECT CAST(NULL AS VARCHAR) AS cal_date) WHERE FALSE")


def _ensure_price_table(conn: duckdb.DuckDBPyConnection, table: str) -> None:
    """Create adjusted price tables with the same key contract as daily_raw.

    Older runs created these tables with ``CREATE TABLE AS`` and therefore no
    primary key; ``INSERT OR REPLACE`` then failed for every archive member.
    An empty legacy table can be repaired in place. A non-empty table without a
    key is rejected so an operator must review it before any data is replaced.
    """
    if table not in {"daily_qfq", "daily_hfq"}:
        raise ValueError(f"unsupported adjusted table: {table}")
    exists = conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_schema='main' AND table_name=?",
        [table],
    ).fetchone()[0]
    if exists:
        constraints = conn.execute(
            "SELECT constraint_type FROM duckdb_constraints() WHERE table_name=?",
            [table],
        ).fetchall()
        has_pk = any(row[0] == "PRIMARY KEY" for row in constraints)
        if has_pk:
            return
        rows = int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
        if rows:
            raise RuntimeError(f"adjusted table {table} has {rows} rows but no primary key")
        conn.execute(f"DROP TABLE {table}")
    conn.execute(f"""
        CREATE TABLE {table} (
          ts_code VARCHAR NOT NULL,
          source_code VARCHAR,
          trade_date VARCHAR NOT NULL,
          open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
          vol DOUBLE, amount DOUBLE, amplitude DOUBLE, pct_chg DOUBLE,
          change DOUBLE, turnover_rate DOUBLE, source_member VARCHAR,
          source_sha256 VARCHAR,
          PRIMARY KEY (ts_code, trade_date)
        )
    """)

def _append_or_replace_snapshot(conn, table: str, frame: pd.DataFrame) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.register("_snapshot_frame", frame)
    conn.execute(f"CREATE TABLE {table} AS SELECT * FROM _snapshot_frame")
    conn.unregister("_snapshot_frame")

def _archive_members(path: Path, suffix: str) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        return [n for n in zf.namelist() if n.lower().endswith(suffix) and not n.endswith("/")]

def _member_frame(raw: bytes, symbol_map: dict[str, str]) -> pd.DataFrame:
    frame = _read_csv(io.BytesIO(raw), dtype={"股票代码": str})
    frame.columns = [str(c).strip() for c in frame.columns]
    missing = set(RAW_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"daily member missing columns: {sorted(missing)}")
    frame = frame.rename(columns=RAW_COLUMNS)
    frame["source_code"] = frame["source_code"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
    frame["ts_code"] = frame["source_code"].map(symbol_map).fillna(frame["source_code"] + ".UNKNOWN")
    frame["trade_date"] = _parse_yyyymmdd(frame["trade_date"])
    for col in set(RAW_COLUMNS.values()) - {"trade_date", "source_code"}:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "ts_code"])
    return frame[["ts_code", "source_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "amplitude", "pct_chg", "change", "turnover_rate"]]

def _import_archive(conn, archive: Path, symbol_map: dict[str, str], table: str = "daily_raw", limit_files: int | None = None) -> dict:
    archive_hash = sha256(archive)
    members = _archive_members(archive, ".csv")
    if limit_files is not None:
        members = members[:limit_files]
    done = 0; skipped = 0; rows = 0; failures = []
    with zipfile.ZipFile(archive) as zf:
        for member in members:
            info = zf.getinfo(member)
            prior = conn.execute("SELECT member_sha256,status FROM import_manifest WHERE archive=? AND member_name=?", [str(archive), member]).fetchone()
            if prior and prior[0] == archive_hash and prior[1] == "OK":
                skipped += 1; continue
            try:
                raw = zf.read(member)
                frame = _member_frame(raw, symbol_map)
                if frame.duplicated(["ts_code", "trade_date"]).any():
                    raise ValueError("duplicate (ts_code, trade_date) keys inside archive member")
                frame["source_member"] = member
                frame["source_sha256"] = archive_hash
                conn.register("_daily_frame", frame)
                conn.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM _daily_frame")
                conn.unregister("_daily_frame")
                min_date = str(frame.trade_date.min()) if len(frame) else None
                max_date = str(frame.trade_date.max()) if len(frame) else None
                conn.execute("INSERT OR REPLACE INTO import_manifest VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OK', current_timestamp, NULL)", [str(archive), member, info.CRC, info.file_size, archive_hash, len(frame), min_date, max_date])
                # Auto-committed data and member receipts remain durable in the WAL.
                # Checkpoint in batches instead of rewriting the database per CSV.
                if (done + 1) % 100 == 0:
                    conn.execute("CHECKPOINT")
                done += 1; rows += len(frame)
            except Exception as exc:
                failures.append({"member": member, "error": f"{type(exc).__name__}: {exc}"})
                conn.execute("INSERT OR REPLACE INTO import_manifest VALUES (?, ?, ?, ?, ?, 0, NULL, NULL, 'FAILED', current_timestamp, ?)", [str(archive), member, info.CRC, info.file_size, archive_hash, failures[-1]["error"]])
    return {"archive": str(archive), "sha256": archive_hash, "members": len(members), "imported": done, "skipped": skipped, "rows": rows, "failures": failures}

def _import_supplemental(conn, source: Path) -> dict:
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    allowed = {"daily", "daily_basic", "adj_factor", "stk_limit", "moneyflow",
               "index_daily", "sw_daily", "limit_list_d", "top_list", "suspend_d"}
    result = {}
    for item in manifest.get("files", []):
        if item.get("dataset") != "canonical-incremental-tail":
            continue
        table = item.get("table")
        path = (source / item["stored_name"]).resolve()
        if table not in allowed or source.resolve() not in path.parents:
            raise ValueError("invalid supplemental table/path")
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"supplemental checksum mismatch: {table}")
        frame = pd.read_parquet(path)
        keys = item["primary_key"]
        # Provider event reasons can be unknown; retain them verbatim. Identity/date cannot be unknown.
        if len(frame) != item["rows"] or frame[["ts_code", "trade_date"]].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"supplemental key/row mismatch: {table}")
        frame["trade_date"] = _parse_yyyymmdd(frame["trade_date"])
        if frame["trade_date"].isna().any():
            raise ValueError(f"invalid supplemental dates: {table}")
        conn.register("_tail", frame)
        try:
            if table != "index_daily":
                conn.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM _tail')
            if table == "daily":
                # The offline archive stores shares/yuan; canonical tails use lots/thousand yuan.
                conn.execute("""INSERT OR REPLACE INTO daily_raw
                    SELECT ts_code, split_part(ts_code,'.',1), trade_date,
                    open, high, low, close, vol*100, amount*1000,
                    NULL, pct_chg, change, NULL, ?, ? FROM _tail""",
                    [path.name, item["sha256"]])
            elif table == "index_daily":
                _ensure_index_table(conn)
                conn.execute("INSERT OR REPLACE INTO index_daily SELECT ts_code, trade_date, close, ?, ? FROM _tail",
                             [path.name, item["sha256"]])
        finally:
            conn.unregister("_tail")
        result[table] = {"rows": len(frame), "min_date": frame.trade_date.min(),
                         "max_date": frame.trade_date.max(), "sha256": item["sha256"],
                         "nullable_key_rows": {key: int(frame[key].isna().sum()) for key in keys}}
    return result


def import_staging(source: Path, db: Path, limit_files: int | None = None, include_adjusted: bool = False,
                   *, memory_limit: str | None = None, threads: int | None = None) -> dict:
    _assert_isolated(db)
    source = source.resolve()
    raw_archive = _resolve(source, *_ARCHIVE_ALIASES["daily"])
    stock_csv = _resolve(source, *_META_ALIASES["stock"])
    if raw_archive is None:
        raise FileNotFoundError(f"缺少日线包，期望其一: {_ARCHIVE_ALIASES['daily']}")
    if stock_csv is None:
        raise FileNotFoundError(f"缺少股票列表，期望其一: {_META_ALIASES['stock']}")
    delisted_csv = _resolve(source, *_META_ALIASES["delisted"])
    calendar_csv = _resolve(source, *_META_ALIASES["calendar"])
    lock = FileLock(str(db.with_suffix(db.suffix + ".lock")), timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        raise RuntimeError(f"staging database is already being imported: {db}") from exc
    stopped = threading.Event()
    succeeded = False
    conn = None
    def progress(phase):
        payload = {"phase": phase, "db": str(db), "source": str(source),
                   "heartbeat": datetime.now(timezone.utc).isoformat(),
                   "bytes": db.stat().st_size if db.exists() else 0,
                   "resume": "repeat import with the same DB; import_manifest stores per-member hashes"}
        target = db.with_suffix(db.suffix + ".progress.json")
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        deadline = time.monotonic() + 5
        while True:
            try:
                os.replace(tmp, target)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
    def monitor():
        while not stopped.wait(10):
            progress("IMPORTING")
    progress("PREFLIGHT_OK")
    heartbeat = threading.Thread(target=monitor, daemon=True)
    heartbeat.start()
    try:
        configuration = {}
        if memory_limit is not None:
            configuration['memory_limit'] = memory_limit
        if threads is not None:
            configuration['threads'] = threads
        conn = duckdb.connect(str(db), config=configuration)
        conn.execute("CREATE TABLE IF NOT EXISTS research_package_metadata (key VARCHAR PRIMARY KEY,value VARCHAR)")
        conn.execute("DELETE FROM research_package_metadata")
        _create_schema(conn)
        symbol = _load_symbol_map(stock_csv)
        if delisted_csv is not None:
            historical = _load_symbol_map(delisted_csv)
            symbol = pd.concat([symbol, historical], ignore_index=True).drop_duplicates("source_code")
        symbol = symbol[~symbol.source_code.astype(str).isin(SZSE_B_SHARE_CODES)]
        _append_or_replace_snapshot(conn, "stock_basic_snapshot", symbol)
        if delisted_csv is not None:
            _append_or_replace_snapshot(
                conn, "delisted_basic_snapshot",
                _load_optional_table(delisted_csv, "delisted"))
        if calendar_csv is not None:
            _append_or_replace_snapshot(
                conn, "trade_cal_snapshot",
                _load_optional_table(calendar_csv, "calendar"))
        symbol_map = dict(zip(symbol.source_code.astype(str).str.zfill(6), symbol.ts_code.astype(str)))
        source_files = {
            p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in (raw_archive, stock_csv, delisted_csv, calendar_csv) if p is not None
        }
        result = {"status": "OK", "source": str(source), "db": str(db),
                  "source_files": source_files,
                  "runtime": {"memory_limit": conn.execute("SELECT current_setting('memory_limit')").fetchone()[0],
                              "threads": conn.execute("SELECT current_setting('threads')").fetchone()[0]}}
        result["daily"] = _import_archive(conn, raw_archive, symbol_map, limit_files=limit_files)
        if include_adjusted:
            for key, table in (("hfq", "daily_hfq"), ("qfq", "daily_qfq")):
                archive = _resolve(source, *_ARCHIVE_ALIASES[key])
                if archive is None:
                    continue
                _ensure_price_table(conn, table)
                result[table] = _import_archive(conn, archive, symbol_map,
                                                table=table, limit_files=limit_files)
        index_archive = _resolve(source, *_INDEX_ALIASES["index"])
        if index_archive is not None:
            result["index_daily"] = _import_index_archive(conn, index_archive)
            source_files[index_archive.name] = {
                "sha256": sha256(index_archive), "bytes": index_archive.stat().st_size}
            result["source_files"] = source_files
        result["supplemental"] = _import_supplemental(conn, source)
        conn.register("_identity_map", symbol[["source_code", "ts_code"]])
        result["identity_quarantine"] = {}
        result["scope_exclusions"] = {}
        try:
            tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
            for name in ("daily_raw", "daily_qfq", "daily_hfq"):
                if name not in tables:
                    continue
                conn.execute("BEGIN TRANSACTION")
                try:
                    quarantine = "unresolved_" + name
                    conn.execute(f"CREATE TABLE IF NOT EXISTS {quarantine} AS SELECT * FROM {name} WHERE FALSE")
                    conn.execute(f"""INSERT OR REPLACE INTO {name}
                        SELECT q.* REPLACE(m.ts_code AS ts_code) FROM {quarantine} q
                        JOIN _identity_map m ON q.source_code=m.source_code""")
                    conn.execute(f"DELETE FROM {quarantine} WHERE source_code IN (SELECT source_code FROM _identity_map)")
                    conn.execute(f"""INSERT OR REPLACE INTO {name}
                        SELECT d.* REPLACE(m.ts_code AS ts_code) FROM {name} d
                        JOIN _identity_map m ON d.source_code=m.source_code
                        WHERE d.ts_code LIKE '%.UNKNOWN'""")
                    conn.execute(f"""DELETE FROM {name} WHERE ts_code LIKE '%.UNKNOWN'
                        AND source_code IN (SELECT source_code FROM _identity_map)""")
                    conn.execute(f"""INSERT INTO {quarantine}
                        SELECT d.* REPLACE(
                            CASE WHEN d.source_code IN ({SZSE_B_SHARE_SQL})
                                 THEN d.source_code || '.UNKNOWN' ELSE d.ts_code END AS ts_code)
                        FROM {name} d
                        WHERE (d.ts_code LIKE '%.UNKNOWN' OR d.source_code IN ({SZSE_B_SHARE_SQL}))
                        AND NOT EXISTS (SELECT 1 FROM {quarantine} q
                                        WHERE q.source_code=d.source_code AND q.trade_date=d.trade_date)""")
                    conn.execute(f"""DELETE FROM {name}
                        WHERE ts_code LIKE '%.UNKNOWN' OR source_code IN ({SZSE_B_SHARE_SQL})""")
                    conn.execute("COMMIT")
                    # Exhaust the cursor before unregistering the map on DuckDB 2.
                    result["identity_quarantine"][name] = conn.execute(f"""
                        SELECT count(*), count(distinct ts_code) FROM {quarantine}
                        WHERE source_code NOT IN ({SZSE_B_SHARE_SQL})""").fetchall()[0]
                    result["scope_exclusions"][name] = conn.execute(f"""
                        SELECT count(*), count(distinct source_code) FROM {quarantine}
                        WHERE source_code IN ({SZSE_B_SHARE_SQL})""").fetchall()[0]
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
        finally:
            conn.unregister("_identity_map")
        result["table_coverage"] = {}
        for name in ("daily_raw", "daily_qfq", "daily_hfq", "index_daily"):
            if name in tables:
                count, first, last = conn.execute(f"SELECT count(*),min(trade_date),max(trade_date) FROM {name}").fetchall()[0]
                result["table_coverage"][name] = {"rows": count, "min_date": first, "max_date": last}
        result["failures"] = [dict(item, table=name)
                              for name, part in result.items() if isinstance(part, dict)
                              for item in part.get("failures", [])]
        if result["failures"]:
            result["status"] = "PARTIAL"
        manifest_path = source / "manifest.json"
        if manifest_path.exists() and not limit_files and not any(
                part.get("failures") for part in result.values() if isinstance(part, dict)):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            for item in manifest.get("files", []):
                package_file = (source / item["stored_name"]).resolve()
                if source not in package_file.parents or package_file.stat().st_size != item["bytes"] or sha256(package_file) != item["sha256"]:
                    raise ValueError("research package checksum mismatch")
            conn.execute("CREATE OR REPLACE TABLE research_package_metadata (key VARCHAR PRIMARY KEY,value VARCHAR)")
            for key, value in (("package_version", manifest.get("version")), ("package_hash", manifest.get("package_sha256"))):
                if value:
                    conn.execute("INSERT INTO research_package_metadata VALUES (?,?)", [key, str(value)])
        conn.execute("CHECKPOINT")
        (db.with_suffix(db.suffix + ".status.json")).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        conn.close()
        succeeded = not result["failures"]
        return result
    finally:
        if conn is not None:
            conn.close()
        stopped.set()
        heartbeat.join(timeout=15)
        progress("COMPLETE" if succeeded else "FAILED_RESUMABLE")
        lock.release()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--limit-files", type=int)
    ap.add_argument("--include-adjusted", action="store_true")
    ap.add_argument("--memory-limit", default=None)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    result = import_staging(Path(args.source_dir), Path(args.db), args.limit_files, args.include_adjusted,
                            memory_limit=args.memory_limit, threads=args.threads)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result.get("failures") else 1

if __name__ == "__main__":
    raise SystemExit(main())
