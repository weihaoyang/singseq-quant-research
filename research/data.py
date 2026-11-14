"""Read-only real-data preflight and a small causal price/volume feature set."""
from __future__ import annotations
import duckdb
import numpy as np
import pandas as pd
from evaluation.research_contract import forward_open_labels

FEATURES = ["reversal_5", "momentum_20", "volatility_20", "volume_ratio", "intraday", "range", "log_amount", "market_return"]


def _normalise_calendar(values) -> list[str]:
    """Return sorted YYYYMMDD values without silently inventing sessions."""
    raw = [value[0] if isinstance(value, (tuple, list)) else value for value in values]
    dates = pd.to_datetime(pd.Series(raw), errors="coerce").dt.strftime("%Y%m%d")
    return sorted(dates.dropna().unique().tolist())


def inspect_staging_data(conn, start: str, end: str) -> dict:
    """Read-only preflight for the raw staging archive.

    The staging archive is intentionally a different contract from the API
    DuckDB: it can support causal research labels, but it cannot certify
    adjustment, limit, PIT or executable portfolio fields.
    """
    tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    required = {"daily_raw", "trade_cal_snapshot", "stock_basic_snapshot", "delisted_basic_snapshot"}
    missing = sorted(required - tables)
    if missing:
        return {"ready": False, "data_mode": "staging", "errors": [f"missing tables: {missing}"],
                "execution_ready": False, "calendar": []}
    calendar = _normalise_calendar(conn.execute(
        'SELECT "日期" FROM trade_cal_snapshot WHERE "是否交易" = \'交易\''
    ).fetchall())
    calendar = [day for day in calendar if start <= day <= end]
    errors: list[str] = []
    if not calendar:
        errors.append("empty staging trading calendar")
    rows = conn.execute(
        "SELECT trade_date, count(*) AS n, count(DISTINCT ts_code) AS u "
        "FROM daily_raw WHERE trade_date BETWEEN ? AND ? GROUP BY 1", [start, end]
    ).fetchall()
    by_day = {str(day).replace("-", ""): int(n) for day, n, _ in rows}
    missing_days = sorted(set(calendar) - set(by_day))
    if missing_days:
        errors.append(f"daily_raw missing calendar sessions: {len(missing_days)}")
    duplicate_keys = int(conn.execute(
        "SELECT count(*) - count(DISTINCT ts_code || '|' || trade_date) "
        "FROM daily_raw WHERE trade_date BETWEEN ? AND ?", [start, end]
    ).fetchone()[0] or 0)
    if duplicate_keys:
        errors.append(f"daily_raw duplicate keys: {duplicate_keys}")
    invalid_ohlcv = int(conn.execute(
        "SELECT count(*) FROM daily_raw WHERE trade_date BETWEEN ? AND ? "
        "AND (open IS NULL OR close IS NULL OR open <= 0 OR close <= 0 "
        "OR high IS NULL OR low IS NULL OR high < low)", [start, end]
    ).fetchone()[0] or 0)
    if invalid_ohlcv:
        errors.append(f"daily_raw invalid OHLC rows: {invalid_ohlcv}")
    return {
        "ready": not errors,
        "data_mode": "staging",
        "execution_ready": False,
        "errors": errors,
        "start": start,
        "end": end,
        "calendar_sessions": len(calendar),
        "calendar": calendar,
        "counts": {"daily_raw_rows": int(sum(n for _, n, _ in rows)),
                   "daily_raw_days": len(rows), "duplicate_keys": duplicate_keys,
                   "invalid_ohlcv": invalid_ohlcv, "missing_days": len(missing_days)},
        "missing_days_sample": missing_days[:20],
        "limitations": ["no adj_factor", "no up/down limits", "no point-in-time industry or market cap",
                        "no historical ST/delist execution state", "no authoritative transaction costs"],
    }


def forward_open_labels_raw(frame: pd.DataFrame, calendar, horizon: int = 5) -> pd.DataFrame:
    """Create raw OHLCV research labels, explicitly without execution claims."""
    if horizon < 1:
        raise ValueError("horizon must be at least one")
    days = pd.Index(sorted(set(map(str, calendar))))
    df = frame.sort_values(["ts_code", "trade_date"]).copy()
    if df.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("duplicate panel keys")
    positions = days.get_indexer(df.trade_date.astype(str))
    if (positions < 0).any():
        raise ValueError("panel date absent from trading calendar")
    df["_session"] = positions
    grouped = df.groupby("ts_code", sort=False)
    by_code_open = df.open.groupby(df.ts_code, sort=False)
    entry = by_code_open.shift(-1)
    exit_price = by_code_open.shift(-(horizon + 1))
    contiguous = grouped["_session"].shift(-(horizon + 1)).eq(df._session + horizon + 1)
    valid = (contiguous & np.isfinite(entry) & np.isfinite(exit_price)
             & entry.gt(0) & exit_price.gt(0)
             & grouped.vol.shift(-1).gt(0) & grouped.vol.shift(-(horizon + 1)).gt(0))
    df["target_ret"] = (exit_price / entry - 1).where(valid)
    for name, offset in [("entry_date", 1), ("label_end_date", horizon + 1)]:
        df[name] = [days[p + offset] if p + offset < len(days) else None for p in positions]
    df["label_horizon"] = horizon
    df["adj_close"] = df.close
    return df.drop(columns="_session")


def load_staging_panel(conn, start: str, end: str, calendar, max_stocks: int = 128, horizon: int = 5):
    """Load a deterministic point-in-time bounded panel from staging."""
    anchor = calendar[0]
    anchor_codes = conn.execute(
        "SELECT DISTINCT ts_code FROM daily_raw WHERE trade_date = ?", [anchor]
    ).df()
    active = conn.execute("SELECT ts_code, list_date, delist_date FROM stock_basic_snapshot").df()
    delisted = conn.execute(
        'SELECT "TS代码" AS ts_code, "上市日期" AS list_date, "退市日期" AS delist_date '
        'FROM delisted_basic_snapshot WHERE "TS代码" IS NOT NULL'
    ).df()
    meta = pd.concat([active, delisted], ignore_index=True).drop_duplicates("ts_code", keep="first")
    for col in ("list_date", "delist_date"):
        meta[col] = pd.to_datetime(meta[col], errors="coerce").dt.strftime("%Y%m%d")
    meta["list_date"] = meta["list_date"].fillna("99999999")
    meta["delist_date"] = meta["delist_date"].fillna("99999999")
    eligible = anchor_codes.merge(meta, on="ts_code", how="left")
    eligible = eligible[(eligible.list_date <= anchor) & (eligible.delist_date >= anchor)]
    codes = eligible.sort_values("ts_code").ts_code.head(max_stocks).tolist()
    if len(codes) < 50:
        raise ValueError(f"need at least 50 staging stocks, found {len(codes)}")
    placeholders = ",".join("?" for _ in codes)
    panel = conn.execute(
        f"SELECT ts_code, trade_date, open, high, low, close, vol, amount "
        f"FROM daily_raw WHERE trade_date BETWEEN ? AND ? AND ts_code IN ({placeholders}) "
        "ORDER BY ts_code, trade_date", [start, end, *codes]
    ).df()
    panel["trade_date"] = panel.trade_date.astype(str).str.replace("-", "", regex=False)
    panel = forward_open_labels_raw(panel, calendar, horizon)
    panel["session_id"] = pd.Index(calendar).get_indexer(panel.trade_date)
    return panel, codes


def inspect_data(conn, start, end):
    tables = {x[0] for x in conn.execute("SHOW TABLES").fetchall()}
    required = {"daily", "daily_basic", "adj_factor", "trade_cal", "stk_limit", "stock_basic"}
    if required - tables:
        return {"ready": False, "errors": ["missing tables: " + str(sorted(required-tables))]}
    days = [str(r[0]) for r in conn.execute(
        "SELECT DISTINCT cal_date FROM trade_cal WHERE is_open=1 AND cal_date BETWEEN ? AND ? ORDER BY 1", [start,end]).fetchall()]
    errors, counts = [], {}
    if not days: errors.append("empty trading calendar")
    # Detect obvious calendar truncation independently of market row counts.
    for year in range(int(start[:4]), int(end[:4])+1):
        if start <= f"{year}0101" and end >= f"{year}1231":
            if sum(d.startswith(str(year)) for d in days) < 200: errors.append(f"calendar incomplete: {year}")
    # Per-table strictness: the base only guarantees stk_limit density from 2021
    # onward (older shards are sampled); joinability is asserted separately below.
    STRICT_FROM = {"stk_limit": "20210104"}
    for table in ["daily", "daily_basic", "adj_factor", "stk_limit"]:
        rows = conn.execute(f"SELECT trade_date, count(*), count(DISTINCT ts_code) FROM {table} WHERE trade_date BETWEEN ? AND ? GROUP BY 1", [start,end]).fetchall()
        by_day = {str(d):int(n) for d,n,_ in rows}
        missing = sorted(set(days)-set(by_day))
        hard = [d for d in missing if d >= STRICT_FROM.get(table, "00000000")]
        duplicates = sum(int(n-unique) for _,n,unique in rows)
        counts[table] = {"rows":sum(by_day.values()),"days":len(rows),"missing_days":len(missing),
                         "missing_before_guarantee":len(missing)-len(hard),
                         "missing_sample":missing[:10],"duplicate_keys":duplicates}
        if hard or duplicates: errors.append(f"{table}: missing={len(hard)} duplicates={duplicates}")
    # Joinability, not equality: adjustment/limit tables may include suspended
    # listings with no daily bar. This gate cannot unblock the official API.
    invalid = conn.execute("""
        SELECT count(*) FROM daily d
        LEFT JOIN adj_factor a USING(ts_code,trade_date)
        LEFT JOIN daily_basic b USING(ts_code,trade_date)
        WHERE d.trade_date BETWEEN ? AND ? AND (d.ts_code LIKE '%.SH' OR d.ts_code LIKE '%.SZ')
        AND (a.adj_factor IS NULL OR a.adj_factor<=0 OR b.circ_mv IS NULL OR b.circ_mv<=0)
        """,[start,end]).fetchone()[0]
    if invalid: errors.append(f"unjoinable or invalid SH/SZ daily keys: {invalid}")
    # stk_limit is only guaranteed dense from 2021 onward (data-package samples
    # cover scattered earlier dates); require it where the base promises it.
    if end >= "20210104":
        lo = max(start, "20210104")
        limit_invalid = conn.execute("""
            SELECT count(*) FROM daily d
            LEFT JOIN stk_limit l USING(ts_code,trade_date)
            WHERE d.trade_date BETWEEN ? AND ? AND d.trade_date >= '20210104'
              AND (d.ts_code LIKE '%.SH' OR d.ts_code LIKE '%.SZ')
            AND (l.up_limit IS NULL OR l.down_limit IS NULL
                 OR l.down_limit<=0 OR l.up_limit<=l.down_limit)
            """,[lo, end]).fetchone()[0]
        if limit_invalid: errors.append(f"unjoinable or invalid SH/SZ stk_limit keys: {limit_invalid}")
    else:
        limit_invalid = 0
    daily_counts = pd.Series({str(d):int(n) for d,n in conn.execute(
        "SELECT trade_date,count(*) FROM daily WHERE trade_date BETWEEN ? AND ? GROUP BY 1 ORDER BY 1",[start,end]).fetchall()})
    thin_days: list[str] = []
    if len(daily_counts):
        median = daily_counts.rolling(21,min_periods=1,center=True).median()
        thin = daily_counts < .9*median
        thin_days = [str(d) for d in thin[thin].index]
        # A mass-suspension session legitimately trades fewer names; only rows
        # the suspend event table cannot explain count as page truncation.
        if thin_days and "suspend_d" in tables:
            susp = dict(conn.execute(
                "SELECT trade_date, count(DISTINCT ts_code) FROM suspend_d "
                "WHERE trade_date BETWEEN ? AND ? GROUP BY 1",[start,end]).fetchall())
            unexplained = [d for d in thin_days
                           if int(susp.get(d,0)) < .8*(float(median.get(d,0))-float(daily_counts.get(d,0)))]
        else:
            unexplained = thin_days
        if unexplained: errors.append(f"possible truncated daily pages: {len(unexplained)}")
        counts.setdefault("daily", {})["thin_days"] = thin_days
    return {"ready":not errors,"errors":errors,"start":start,"end":end,"calendar_sessions":len(days),
            "counts":counts,"unjoinable_keys":int(invalid),"unjoinable_limit_keys":int(limit_invalid),
            "calendar":days,
            "scope":"development data gate; not official certification or external-calendar validation"}


def load_panel(conn, start, end, calendar, max_stocks=128, horizon=5):
    # Universe fixed using only the first signal date, never validation outcomes.
    codes = [r[0] for r in conn.execute("""
        SELECT DISTINCT d.ts_code FROM daily d JOIN stock_basic s USING(ts_code)
        WHERE d.trade_date=? AND (d.ts_code LIKE '%.SH' OR d.ts_code LIKE '%.SZ')
          AND strptime(s.list_date,'%Y%m%d') <= strptime(?,'%Y%m%d') - INTERVAL 120 DAY
        ORDER BY d.ts_code LIMIT ?
        """,[calendar[0],calendar[0],max_stocks]).fetchall()]
    if not codes: raise ValueError("empty initial-date universe")
    placeholders = ','.join('?' for _ in codes)
    frame = conn.execute(f"""
        SELECT d.ts_code,d.trade_date,d.open,d.high,d.low,d.close,d.vol,d.amount,
               a.adj_factor,b.circ_mv,l.up_limit,l.down_limit
        FROM daily d LEFT JOIN adj_factor a USING(ts_code,trade_date)
        LEFT JOIN daily_basic b USING(ts_code,trade_date)
        LEFT JOIN stk_limit l USING(ts_code,trade_date)
        WHERE d.trade_date BETWEEN ? AND ? AND d.ts_code IN ({placeholders})
        ORDER BY d.ts_code,d.trade_date
        """,[start,end,*codes]).df()
    frame["trade_date"] = frame.trade_date.astype(str)
    frame = forward_open_labels(frame,calendar,horizon)
    frame["session_id"] = pd.Index(calendar).get_indexer(frame.trade_date)
    frame["adj_close"] = frame.close*frame.adj_factor
    return frame, codes


def build_features(frame):
    df = frame.sort_values(["ts_code","trade_date"]).copy()
    g = df.groupby("ts_code",sort=False)
    returns = g.adj_close.pct_change(fill_method=None)
    df["reversal_5"] = -g.adj_close.pct_change(5,fill_method=None)
    df["momentum_20"] = g.adj_close.pct_change(20,fill_method=None)
    df["volatility_20"] = returns.groupby(df.ts_code).transform(lambda x:x.rolling(20,min_periods=20).std())
    df["volume_ratio"] = g.vol.transform(lambda x:x.rolling(5).mean()/x.rolling(20).mean())
    df["intraday"] = df.close/df.open-1
    df["range"] = (df.high-df.low)/df.open
    df["log_amount"] = np.log(df.amount.where(df.amount>0))
    # Leave-one-out equal-weight return of the fixed research universe.
    total = returns.groupby(df.trade_date).transform('sum')
    count = returns.groupby(df.trade_date).transform('count')
    df["market_return"] = ((total-returns)/(count-1)).where(count>1)
    # Refuse windows that bridge absent trading records.
    continuous = g.session_id.diff(20).eq(20)
    df.loc[~continuous,FEATURES] = np.nan
    df[FEATURES] = df[FEATURES].replace([np.inf,-np.inf],np.nan)
    return df
