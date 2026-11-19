"""Provider-neutral local data connectors.

Capabilities are discoverable before a query. Connector credentials come from
the local environment and are never copied into results, logs or diagnostic
snapshots. Raw provider data is normalised into the same long research-panel
contract used by the local Alpha Graph engine.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_sdk.panel_contract import (
    BAR_VERSION,
    PANEL_VERSION,
    TABLE_KINDS,
    as_of_join,
    bundle_manifest,
    panel_metadata,
)
from quant_sdk.research_workflow import _compact_date

CONNECTOR_FORMAT = "qconnector/1"
CONNECTOR_VERSION = "1"

_IDENTITY_ALIASES = {
    "symbol": "ts_code", "ticker": "ts_code", "asset": "ts_code",
    "code": "ts_code", "instrument": "ts_code",
    "date": "trade_date", "datetime": "trade_date", "time": "trade_date",
    "timestamp": "trade_date",
}
_FIELD_ALIASES = {
    "volume": "vol", "base_volume": "vol", "quote_volume": "amount",
    "turnover": "amount", "turnover_value": "amount",
    "vw": "vwap", "vwap": "vwap", "adj_close": "adj_close",
}
_POINT_IN_TIME = {
    "csv": "user responsibility",
    "parquet": "user responsibility",
    "duckdb": "user responsibility",
    "tushare": "provider trade-date semantics",
    "polygon": "provider aggregate timestamp; adjustment mode is explicit",
    "binance-public": "exchange candle open time in UTC",
    "yahoo-public": "provider daily chart timestamp in the listing market timezone",
    "databento": "event timestamp retained by provider schema",
    "ccxt": "exchange candle open time in UTC",
    "snowflake": "query-defined; caller must use an as-of join",
    "bigquery": "query-defined; caller must use an as-of join",
    "wrds": "query-defined; caller must use an as-of join",
}

_CONNECTORS: dict[str, dict[str, Any]] = {
    "csv": {
        "title": "Local CSV panel",
        "kind": "local_file",
        "market": "USER",
        "requires": [],
        "credentials": [],
        "fields": "OHLCV + amount when present",
    },
    "parquet": {
        "title": "Local Parquet panel",
        "kind": "local_file",
        "market": "USER",
        "requires": ["pyarrow"],
        "credentials": [],
        "fields": "OHLCV + amount when present",
    },
    "duckdb": {
        "title": "Local DuckDB panel",
        "kind": "local_file",
        "market": "USER",
        "requires": ["duckdb"],
        "credentials": [],
        "fields": "OHLCV + amount when present",
    },
    "tushare": {
        "title": "Tushare Pro",
        "kind": "provider_api",
        "market": "CN",
        "requires": ["tushare"],
        "credentials": ["TUSHARE_TOKEN"],
        "fields": "daily OHLCVA",
    },
    "polygon": {
        "title": "Polygon.io aggregates",
        "kind": "provider_api",
        "market": "US",
        "requires": [],
        "credentials": ["POLYGON_API_KEY"],
        "fields": "daily OHLCV, VWAP-derived amount",
    },
    "binance-public": {
        "title": "Binance public market data",
        "kind": "provider_api",
        "market": "CRYPTO",
        "requires": [],
        "credentials": [],
        "fields": "public OHLCV plus quote-asset volume as amount",
    },
    "yahoo-public": {
        "title": "Yahoo public daily equities",
        "kind": "provider_api",
        "market": "GLOBAL_EQUITY",
        "requires": [],
        "credentials": [],
        "fields": "public daily OHLCV with adjusted close and quote-currency amount",
    },
    "databento": {
        "title": "Databento historical",
        "kind": "provider_api",
        "market": "GLOBAL",
        "requires": ["databento"],
        "credentials": ["DATABENTO_API_KEY"],
        "fields": "provider OHLCV, optionally adjusted",
    },
    "ccxt": {
        "title": "CCXT crypto exchanges",
        "kind": "provider_api",
        "market": "CRYPTO",
        "requires": ["ccxt"],
        "credentials": [],
        "fields": "exchange OHLCV, quote volume as amount",
    },
    "snowflake": {
        "title": "Snowflake query",
        "kind": "warehouse",
        "market": "GLOBAL",
        "requires": ["snowflake.connector"],
        "credentials": [
            "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
            "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA",
        ],
        "fields": "query-defined; normalised after retrieval",
    },
    "bigquery": {
        "title": "Google BigQuery",
        "kind": "warehouse",
        "market": "GLOBAL",
        "requires": ["google.cloud.bigquery"],
        "credentials": ["GOOGLE_APPLICATION_CREDENTIALS"],
        "fields": "query-defined; normalised after retrieval",
    },
    "wrds": {
        "title": "WRDS",
        "kind": "warehouse",
        "market": "GLOBAL",
        "requires": ["wrds"],
        "credentials": ["WRDS_USERNAME", "WRDS_PASSWORD"],
        "fields": "query-defined; normalised after retrieval",
    },
}


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _credential_state(names: Iterable[str]) -> dict[str, bool]:
    return {name: bool(str(os.environ.get(name, "")).strip()) for name in names}


def list_connectors() -> dict[str, Any]:
    """Return capabilities only; credential values never leave the process."""
    items = []
    for name, spec in sorted(_CONNECTORS.items()):
        credentials = _credential_state(spec["credentials"])
        packages = {
            package: _module_available(package)
            for package in spec["requires"]
        }
        ready = (
            all(packages.values())
            and all(credentials.values())
        )
        items.append({
            "id": name,
            "title": spec["title"],
            "kind": spec["kind"],
            "market": spec["market"],
            "fields": spec["fields"],
            "pointInTime": _POINT_IN_TIME.get(name, "provider-defined"),
            "packages": packages,
            "credentials": credentials,
            "ready": ready,
            "networkRequired": spec["kind"] != "local_file",
        })
    try:
        from quant_sdk.capabilities import installed_connector_specs

        installed = installed_connector_specs()
    except Exception:
        installed = []
    for spec in installed:
        credentials = _credential_state(spec.get("credentials") or [])
        packages = {
            package: _module_available(package)
            for package in spec.get("requires") or []
        }
        items.append({
            "id": spec["id"],
            "title": spec["title"],
            "kind": spec["kind"],
            "market": spec["market"],
            "fields": ", ".join(spec.get("fields") or []),
            "pointInTime": spec["pointInTime"],
            "packages": packages,
            "credentials": credentials,
            "ready": all(packages.values()) and all(credentials.values()),
            "networkRequired": spec["kind"] != "local_file",
            "implementation": "installed_connector_pack",
            "pack": spec.get("pack"),
            "allowedHosts": list(spec.get("allowedHosts") or []),
        })
    return {
        "format": CONNECTOR_FORMAT,
        "version": CONNECTOR_VERSION,
        "status": "CAPABILITIES",
        "connectors": items,
        "panelContract": {
            "format": PANEL_VERSION,
            "barWireFormat": BAR_VERSION,
            "identity": ["ts_code", "trade_date"],
            "required": ["ts_code", "trade_date", "open", "high", "low", "close", "vol"],
            "optional": ["amount", "vwap", "adj_close", "industry_id", "circ_mv"],
            "supportedExtensions": {
                "adjustment": ["adj_factor", "benchmark_ret"],
                "universe": [
                    "up_limit", "down_limit", "st", "list_date",
                    "pool_membership", "shortable", "borrow_rate",
                ],
                "corporateActions": ["cash_dividend"],
                "fundamental": ["pb", "turnover_rate"],
                "flows": ["stock_flow", "sector_flow"],
                "rotation": [
                    "sector_close", "market_close", "sector_up_count",
                    "sector_total_count", "leader_close", "laggard_close",
                ],
                "events": [
                    "event_direction", "event_intensity", "event_confidence",
                    "event_impact", "event_symbol_hits", "event_sector_hits",
                    "regulatory_event", "shareholder_event", "policy_event",
                ],
                "sentiment": [
                    "sentiment", "board_break", "market_gap",
                    "market_reversal", "hotword",
                ],
                "intraday": [
                    "event_age_minutes", "turnover_30m", "vwap_30m",
                ],
            },
            "pointInTime": {
                "prices": "provider_timestamp",
                "fundamental": "known_at_lte_trade_date",
                "universe": "membership_effective_at_lte_trade_date",
                "events": "known_at_lte_decision_time",
            },
            "multiTable": {
                "kinds": ["bars", "reference", "fundamental", "event", "flow"],
                "identity": ["ts_code"],
                "eventTime": "explicit per table",
                "availableAt": "required for PIT joins",
                "joinMode": "as_of",
                "futureAvailabilityRejected": True,
                "futureAvailabilityEnforcement": "runtime",
                "nonBarPullProviders": [
                    "csv", "parquet", "duckdb",
                    "snowflake", "bigquery", "wrds",
                ],
                "joinCommand": "connector.join",
            },
            "unknownColumns": "preserved",
            "dateFormat": "YYYYMMDD",
            "dataBoundary": "local_or_provider_side_query",
        },
        "secretPolicy": "environment-only; values are never returned in diagnostics",
    }


def _require_ready(provider: str, *, credentials: bool = True) -> dict[str, Any]:
    key = str(provider or "").strip().lower()
    if key not in _CONNECTORS:
        raise ValueError(f"未知 data connector: {provider}")
    spec = _CONNECTORS[key]
    missing_packages = [
        name for name in spec["requires"] if not _module_available(name)
    ]
    if missing_packages:
        raise RuntimeError(
            f"{key} 缺少本地 SDK: {', '.join(missing_packages)}；请安装后再查询。")
    if credentials:
        missing = [
            name for name, present in _credential_state(spec["credentials"]).items()
            if not present
        ]
        if missing:
            raise RuntimeError(
                f"{key} 缺少环境变量: {', '.join(missing)}；值不会写入客户端。")
    return spec


def _timestamp_date(value: Any) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        integer_text = str(int(number))
        if integer_text.isdigit() and len(integer_text) == 8:
            return integer_text
        if number > 10_000_000_000:
            number /= 1000.0
        return datetime.fromtimestamp(number, tz=UTC).strftime("%Y%m%d")
    text = str(value or "").strip()
    if not text:
        return ""
    if text.isdigit() and len(text) == 8:
        return text
    if text.isdigit() and len(text) >= 13:
        return _timestamp_date(int(text))
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    return parsed.strftime("%Y%m%d") if pd.notna(parsed) else ""


def normalize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise provider output into the local long-panel contract."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("connector returned an empty frame")
    panel = frame.copy()
    for alias, target in {**_IDENTITY_ALIASES, **_FIELD_ALIASES}.items():
        if alias not in panel or alias == target:
            continue
        if target in panel:
            alternate = panel.pop(alias)
            panel[target] = panel[target].where(
                panel[target].notna(), alternate,
            )
        else:
            panel = panel.rename(columns={alias: target})
    if "ts_code" not in panel:
        raise ValueError("connector output is missing ts_code/symbol")
    if "trade_date" not in panel:
        raise ValueError("connector output is missing trade_date/date")
    panel["ts_code"] = panel["ts_code"].astype(str)
    panel["trade_date"] = panel["trade_date"].map(_timestamp_date)
    panel = panel[panel["trade_date"].str.fullmatch(r"\d{8}", na=False)]
    if panel.empty:
        raise ValueError("connector output has no valid dates")
    for column in ("open", "high", "low", "close", "vol", "amount", "vwap", "adj_close"):
        if column in panel:
            panel[column] = pd.to_numeric(panel[column], errors="coerce")
    if not {"open", "high", "low", "close"}.issubset(panel.columns):
        raise ValueError("connector output must contain open/high/low/close")
    if "vol" not in panel:
        panel["vol"] = np.nan
    panel = panel.drop_duplicates(["ts_code", "trade_date"], keep="last")
    return panel.sort_values(
        ["ts_code", "trade_date"], kind="mergesort",
    ).reset_index(drop=True)


def _normalize_time_values(values: pd.Series, name: str) -> pd.Series:
    text = values.map(
        lambda value: str(int(value))
        if isinstance(value, (int, float, np.integer, np.floating))
        and np.isfinite(float(value))
        else str(value).strip()
    )
    daily = text.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    if daily.any():
        parsed.loc[daily] = pd.to_datetime(
            text.loc[daily],
            format="%Y%m%d",
            errors="coerce",
            utc=True,
        )
    if (~daily).any():
        parsed.loc[~daily] = pd.to_datetime(
            text.loc[~daily],
            errors="coerce",
            utc=True,
        )
    if parsed.isna().any():
        bad = int(parsed.isna().sum())
        raise ValueError(f"{name} has {bad} missing or invalid timestamps")
    normalized = parsed.dt.tz_convert(UTC)
    midnight = (
        normalized.dt.hour.eq(0)
        & normalized.dt.minute.eq(0)
        & normalized.dt.second.eq(0)
        & normalized.dt.microsecond.eq(0)
    )
    result = normalized.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    result = result.where(
        ~midnight,
        normalized.dt.strftime("%Y%m%d"),
    )
    return result


def normalize_table(
    frame: pd.DataFrame,
    *,
    kind: str,
    identity: Sequence[str],
    event_time: str,
    available_at: str | None,
) -> pd.DataFrame:
    """Normalize one non-bar qpanel table without disguising it as OHLCV."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("connector returned an empty table")
    table = frame.copy()
    for alias, target in {**_IDENTITY_ALIASES, **_FIELD_ALIASES}.items():
        if alias not in table or alias == target:
            continue
        if target in table:
            alternate = table.pop(alias)
            table[target] = table[target].where(
                table[target].notna(), alternate,
            )
        else:
            table = table.rename(columns={alias: target})
    keys = [str(item).strip() for item in identity if str(item).strip()]
    if not keys:
        raise ValueError("multi-table connector requires identity fields")
    required = [*keys, event_time]
    if str(kind) != "bars" and available_at:
        required.append(available_at)
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(
            "connector table is missing required fields: "
            + ", ".join(missing)
        )
    for key in keys:
        table[key] = table[key].astype(str)
    identity_valid = pd.Series(True, index=table.index)
    for key in keys:
        identity_valid &= table[key].str.strip().ne("")
    table = table[identity_valid].copy()
    if table.empty:
        raise ValueError("connector table has no valid identity rows")
    table[event_time] = _normalize_time_values(
        table[event_time], event_time,
    )
    if available_at:
        table[available_at] = _normalize_time_values(
            table[available_at], available_at,
        )
        impossible = (
            pd.to_datetime(table[available_at], utc=True)
            < pd.to_datetime(table[event_time], utc=True)
        )
        if impossible.any():
            raise ValueError(
                f"connector table has {int(impossible.sum())} rows whose "
                "availability is before their event time"
            )
    table = table.drop_duplicates()
    ordering = [*keys]
    if available_at:
        ordering.append(available_at)
    ordering.append(event_time)
    return table.sort_values(
        ordering,
        kind="mergesort",
    ).reset_index(drop=True)


def _write_table_metadata(
    output: Path,
    metadata: dict[str, Any],
) -> Path:
    target = output.with_name(output.name + ".qpanel.json")
    temporary = target.with_name(target.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _write_panel(frame: pd.DataFrame, output: str | Path, *, force: bool) -> Path:
    target = Path(output).expanduser().resolve()
    suffix = target.suffix.lower()
    if suffix not in {".csv", ".parquet", ".duckdb"}:
        raise ValueError("connector output must be CSV, Parquet or DuckDB")
    if target.exists() and not force:
        raise ValueError(f"输出文件已存在: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        if suffix == ".csv":
            frame.to_csv(temporary, index=False)
        elif suffix == ".parquet":
            frame.to_parquet(temporary, index=False)
        else:
            import duckdb

            connection = duckdb.connect(str(temporary))
            try:
                connection.register("normalized_panel", frame)
                connection.execute(
                    "CREATE TABLE daily_raw AS SELECT * FROM normalized_panel")
            finally:
                connection.close()
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _symbols(values: Sequence[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = values.split(",")
    return list(dict.fromkeys(
        str(item).strip() for item in values if str(item).strip()
    ))


def _validate_read_only_query(query: str) -> str:
    """Reject multi-statement and mutating warehouse SQL.

    Provider-side permissions remain the final control. This client-side
    contract prevents accidental or agent-generated DDL/DML from being sent
    under a user's warehouse credentials.
    """
    text = str(query or "").strip()
    if not text:
        raise ValueError("warehouse query must not be empty")
    scrubbed = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    scrubbed = re.sub(r"--[^\r\n]*", " ", scrubbed)
    scrubbed = re.sub(r"'(?:''|[^'])*'", "''", scrubbed)
    scrubbed = re.sub(r'"(?:""|[^"])*"', '""', scrubbed)
    without_trailing = scrubbed.rstrip()
    if without_trailing.endswith(";"):
        without_trailing = without_trailing[:-1].rstrip()
    if ";" in without_trailing:
        raise ValueError("warehouse query must contain exactly one statement")
    if not re.match(r"^(?:select|with)\b", without_trailing, flags=re.IGNORECASE):
        raise ValueError("warehouse connector accepts SELECT or WITH queries only")
    forbidden = re.search(
        r"\b(?:alter|analyze|begin|call|commit|copy|create|delete|drop|"
        r"execute|exec|grant|insert|into|merge|put|refresh|remove|replace|"
        r"revoke|rollback|truncate|unload|update|upsert|vacuum)\b",
        without_trailing,
        flags=re.IGNORECASE,
    )
    if forbidden:
        raise ValueError(
            f"warehouse query contains a forbidden operation: {forbidden.group(0)}"
        )
    if re.search(r"\bfor\s+(?:update|share)\b", without_trailing, flags=re.IGNORECASE):
        raise ValueError("warehouse query cannot lock rows")
    return text


def _warehouse_limits(options: dict[str, Any]) -> tuple[int, int, int | None]:
    max_rows = options.get("maxRows", 1_000_000)
    timeout_seconds = options.get("timeoutSeconds", 300)
    max_bytes = options.get("maxBytesBilled")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= 5_000_000:
        raise ValueError("maxRows must be an integer between 1 and 5,000,000")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or not 1 <= timeout_seconds <= 3600
    ):
        raise ValueError("timeoutSeconds must be an integer between 1 and 3600")
    if max_bytes is not None and (
        isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0
    ):
        raise ValueError("maxBytesBilled must be a non-negative integer")
    return max_rows, timeout_seconds, max_bytes


def _local_pull(
    provider: str, *, input_path: str | Path, start: str, end: str,
    table_contract: dict[str, Any] | None = None,
) -> pd.DataFrame:
    path = Path(input_path).expanduser().resolve()
    if provider == "csv" and path.suffix.lower() != ".csv":
        raise ValueError("csv connector requires a .csv input")
    if provider == "parquet" and path.suffix.lower() != ".parquet":
        raise ValueError("parquet connector requires a .parquet input")
    if provider == "duckdb" and path.suffix.lower() not in {".duckdb", ".db"}:
        raise ValueError("duckdb connector requires .duckdb or .db input")
    if provider == "csv":
        frame = pd.read_csv(path)
    elif provider == "parquet":
        frame = pd.read_parquet(path)
    else:
        import duckdb

        from quant_sdk.research_workflow import _relation_for_duckdb

        connection = duckdb.connect(str(path), read_only=True)
        try:
            relation, _ = _relation_for_duckdb(connection, path)
            columns = [
                str(row[0])
                for row in connection.execute(
                    f"DESCRIBE SELECT * FROM {relation}"
                ).fetchall()
            ]
            date_column = next(
                (name for name in ("trade_date", "date", "datetime", "time", "timestamp")
                 if name in columns),
                None,
            )
            if date_column:
                quoted = '"' + date_column.replace('"', '""') + '"'
                frame = connection.execute(
                    f"SELECT * FROM {relation} "
                    f"WHERE replace(CAST({quoted} AS VARCHAR), '-', '') BETWEEN ? AND ?",
                    [start, end],
                ).df()
            else:
                frame = connection.execute(f"SELECT * FROM {relation}").df()
        finally:
            connection.close()
    if table_contract is not None:
        table = normalize_table(frame, **table_contract)
        event_time = str(table_contract["event_time"])
        event = pd.to_datetime(table[event_time], errors="coerce", utc=True)
        lower = pd.Timestamp(start, tz="UTC")
        upper = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        return table[
            event.ge(lower) & event.lt(upper)
        ].reset_index(drop=True)
    panel = normalize_panel(frame)
    return panel[panel["trade_date"].between(start, end)].reset_index(drop=True)


def _pull_tushare(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    if not symbols:
        raise ValueError("Tushare pull requires symbols")
    import tushare as ts

    pro = ts.pro_api(os.environ["TUSHARE_TOKEN"].strip())
    frames = [
        pro.daily(ts_code=symbol, start_date=start, end_date=end)
        for symbol in symbols
    ]
    return normalize_panel(pd.concat(
        [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty],
        ignore_index=True,
    ))


def _pull_polygon(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    if not symbols:
        raise ValueError("Polygon pull requires symbols")
    import requests

    headers = {"Authorization": f"Bearer {os.environ['POLYGON_API_KEY'].strip()}"}
    frames = []
    for symbol in symbols:
        response = requests.get(
            "https://api.polygon.io/v2/aggs/ticker/"
            f"{symbol}/range/1/day/{start}/{end}",
            headers=headers,
            params={"adjusted": "true", "sort": "asc", "limit": 50000},
            timeout=60,
        )
        response.raise_for_status()
        rows = response.json().get("results") or []
        if rows:
            frame = pd.DataFrame(rows).rename(columns={
                "t": "trade_date", "o": "open", "h": "high",
                "l": "low", "c": "close", "v": "vol", "vw": "vwap",
            })
            frame["ts_code"] = symbol
            frame["amount"] = frame["vwap"] * frame["vol"]
            frames.append(frame)
    if not frames:
        raise ValueError("Polygon returned no rows")
    return normalize_panel(pd.concat(frames, ignore_index=True))


def _get_public_json(
    url: str,
    params: dict[str, Any],
    *,
    attempts: int = 3,
    timeout: float = 30.0,
    backoff_seconds: float = 0.5,
) -> Any:
    """Read a public JSON endpoint with bounded retries for transient failures."""
    query = urllib.parse.urlencode(params)
    request_url = f"{url}?{query}"
    attempts = max(1, min(int(attempts), 5))
    timeout = max(1.0, min(float(timeout), 120.0))
    backoff_seconds = max(0.0, min(float(backoff_seconds), 10.0))
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            request_url,
            headers={
                "Accept": "application/json",
                "User-Agent": "quant-research-client/2",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = int(getattr(response, "status", 200))
                if status != 200:
                    error = RuntimeError(f"provider returned HTTP {status}")
                    last_error = error
                    retryable = status == 429 or 500 <= status < 600
                    if not retryable or attempt + 1 >= attempts:
                        raise error
                else:
                    return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            last_error = error
            retryable = error.code == 429 or 500 <= error.code < 600
            if not retryable or attempt + 1 >= attempts:
                raise RuntimeError(
                    f"provider returned HTTP {error.code}"
                ) from error
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            json.JSONDecodeError,
        ) as error:
            last_error = error
            if attempt + 1 >= attempts:
                raise RuntimeError(
                    f"provider request failed after {attempts} attempts: "
                    f"{type(error).__name__}"
                ) from error
        if backoff_seconds:
            time.sleep(backoff_seconds * (2**attempt))
    raise RuntimeError(
        f"provider request failed: {type(last_error).__name__ if last_error else 'unknown'}"
    )


def _connector_http_json(
    url: str,
    *,
    allowed_hosts: Iterable[str] | None = None,
    method: str = "GET",
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    body: Any = None,
    timeout_seconds: float = 30.0,
    attempts: int = 3,
) -> Any:
    """Bounded JSON HTTP helper constrained by a connector's declared hosts."""
    verb = str(method or "GET").upper()
    if verb not in {"GET", "POST"}:
        raise ValueError("connector http_json supports GET and POST only")
    request_url = str(url)
    if params:
        separator = "&" if "?" in request_url else "?"
        request_url += separator + urllib.parse.urlencode(params)
    parsed = urllib.parse.urlsplit(request_url)
    hostname = str(parsed.hostname or "").lower().rstrip(".")
    permitted = {
        str(item).strip().lower().rstrip(".")
        for item in (allowed_hosts or ())
        if str(item).strip()
    }
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("connector http_json requires an http or https URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("connector http_json URL cannot contain credentials")
    if parsed.fragment:
        raise ValueError("connector http_json URL cannot contain a fragment")
    if not hostname or hostname not in permitted:
        raise ValueError(
            "connector http_json host is not declared in allowedHosts: "
            + (hostname or "<missing>")
        )
    try:
        _ = parsed.port
    except ValueError as error:
        raise ValueError("connector http_json URL has an invalid port") from error
    payload = None
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "quant-research-connector/1",
        **{
            str(key): str(value)
            for key, value in (headers or {}).items()
        },
    }
    if body is not None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    bounded_attempts = max(1, min(int(attempts), 5))
    bounded_timeout = max(1.0, min(float(timeout_seconds), 120.0))
    last_error: Exception | None = None
    opener = urllib.request.build_opener(_NoRedirectHandler())
    for attempt_index in range(bounded_attempts):
        request = urllib.request.Request(
            request_url,
            data=payload,
            headers=request_headers,
            method=verb,
        )
        try:
            with opener.open(
                request,
                timeout=bounded_timeout,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            last_error = error
            retryable = error.code == 429 or 500 <= error.code < 600
            if not retryable or attempt_index + 1 >= bounded_attempts:
                raise RuntimeError(
                    f"connector provider returned HTTP {error.code}"
                ) from error
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            json.JSONDecodeError,
        ) as error:
            last_error = error
            if attempt_index + 1 >= bounded_attempts:
                break
        time.sleep(min(0.5 * (2 ** attempt_index), 4.0))
    raise RuntimeError(
        "connector provider request failed: "
        f"{type(last_error).__name__ if last_error else 'unknown'}"
    ) from last_error


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep one declared host from redirecting credentials to another origin."""

    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _installed_connector(provider: str) -> tuple[dict[str, Any], Any] | None:
    from quant_sdk.capabilities import (
        installed_connector_callable,
        installed_connector_specs,
    )

    spec = next(
        (
            item for item in installed_connector_specs()
            if str(item.get("id") or "").lower() == provider
        ),
        None,
    )
    if spec is None:
        return None
    function = installed_connector_callable(provider)
    if function is None:
        return None
    return spec, function


def _pull_installed_connector(
    spec: dict[str, Any],
    function: Any,
    *,
    start: str,
    end: str,
    symbols: list[str],
    options: dict[str, Any],
) -> pd.DataFrame:
    credentials = {
        name: str(os.environ.get(name, "")).strip()
        for name in spec.get("credentials") or []
    }
    missing = [name for name, value in credentials.items() if not value]
    if missing:
        raise RuntimeError(
            f"{spec['id']} 缺少环境变量: {', '.join(missing)}"
        )
    def connector_http_json(url: str, **kwargs: Any) -> Any:
        return _connector_http_json(
            url,
            allowed_hosts=spec.get("allowedHosts") or [],
            **kwargs,
        )

    payload = {
        "start": start,
        "end": end,
        "symbols": symbols,
        "options": options,
        "credentials": credentials,
        "http_json": connector_http_json,
    }
    parameters = inspect.signature(function).parameters
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        kwargs = payload
    else:
        kwargs = {
            name: payload[name]
            for name in parameters
            if name in payload
        }
    frame = function(**kwargs)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{spec['id']} connector must return a pandas DataFrame"
        )
    panel = normalize_panel(frame)
    return panel[panel["trade_date"].between(start, end)].reset_index(drop=True)


def _public_request_options(options: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempts": max(1, min(int(options.get("attempts", 3)), 5)),
        "timeout": max(1.0, min(float(options.get("timeoutSeconds", 30)), 120.0)),
        "backoff_seconds": max(
            0.0, min(float(options.get("backoffSeconds", 0.5)), 10.0),
        ),
    }


def _pull_binance_public(
    symbols: list[str], start: str, end: str, options: dict[str, Any],
) -> pd.DataFrame:
    if not symbols:
        raise ValueError("Binance public pull requires symbols")
    interval = str(options.get("interval") or "1d").strip()
    if interval not in {"1d", "3d", "1w", "1M"}:
        raise ValueError(
            "Binance public connector currently supports 1d, 3d, 1w or 1M "
            "so the normalized panel keeps one identity per trade_date")
    since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    until = int((
        pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    ).timestamp() * 1000)
    frames = []
    warnings: list[str] = []
    request_options = _public_request_options(options)
    for symbol in symbols:
        normalized_symbol = str(symbol).replace("/", "").replace("-", "").upper()
        try:
            rows = []
            cursor = since
            while cursor <= until:
                batch = _get_public_json(
                    "https://api.binance.com/api/v3/klines",
                    {
                        "symbol": normalized_symbol,
                        "interval": interval,
                        "startTime": cursor,
                        "endTime": until,
                        "limit": 1000,
                    },
                    **request_options,
                )
                if not isinstance(batch, list):
                    raise RuntimeError("response is not a kline array")
                if not batch:
                    break
                rows.extend(batch)
                next_cursor = int(batch[-1][0]) + 1
                if next_cursor <= cursor or len(batch) < 1000:
                    break
                cursor = next_cursor
            if rows:
                frame = pd.DataFrame(rows)
                if frame.shape[1] < 9:
                    raise RuntimeError("kline rows have fewer than 9 fields")
                frame = frame.iloc[:, :9]
                frame.columns = [
                    "trade_date", "open", "high", "low", "close", "vol",
                    "close_time", "amount", "trades",
                ]
                frame["ts_code"] = normalized_symbol
                volume = pd.to_numeric(frame["vol"], errors="coerce")
                amount = pd.to_numeric(frame["amount"], errors="coerce")
                frame["vwap"] = amount.div(volume.where(volume > 0))
                frames.append(frame)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            warnings.append(f"{normalized_symbol}: {type(error).__name__}: {error}")
    if not frames:
        detail = "; ".join(warnings[:8])
        raise ValueError(
            "Binance public API returned no rows"
            + (f" ({detail})" if detail else "")
        )
    panel = normalize_panel(pd.concat(frames, ignore_index=True))
    panel = panel[panel["trade_date"].between(start, end)].reset_index(drop=True)
    panel.attrs["sourceWarnings"] = warnings
    return panel


def _pull_yahoo_public(
    symbols: list[str], start: str, end: str, options: dict[str, Any],
) -> pd.DataFrame:
    if not symbols:
        raise ValueError("Yahoo public pull requires symbols")
    interval = str(options.get("interval") or "1d").strip()
    if interval != "1d":
        raise ValueError("Yahoo public connector currently supports 1d only")
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    end_ts = int((
        pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    ).timestamp())
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    request_options = _public_request_options(options)
    for symbol in symbols:
        ticker = str(symbol).strip().upper()
        try:
            payload = _get_public_json(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                {
                    "period1": start_ts,
                    "period2": end_ts,
                    "interval": interval,
                    "events": "history",
                    "includeAdjustedClose": "true",
                },
                **request_options,
            )
            chart = payload.get("chart") if isinstance(payload, dict) else None
            if not isinstance(chart, dict):
                raise ValueError("response has no chart object")
            provider_error = chart.get("error")
            if provider_error:
                raise ValueError(
                    "provider error: "
                    + str(provider_error.get("description") or provider_error)
                )
            result = chart.get("result")
            if not isinstance(result, list) or not result or not isinstance(result[0], dict):
                raise ValueError("response has no chart result")
            item = result[0]
            timestamps = item.get("timestamp") or []
            indicators = item.get("indicators") or {}
            quote_rows = indicators.get("quote") or []
            quote = (
                quote_rows[0]
                if isinstance(quote_rows, list) and quote_rows
                and isinstance(quote_rows[0], dict)
                else {}
            )
            adjusted_rows = indicators.get("adjclose") or []
            adjusted = (
                adjusted_rows[0].get("adjclose")
                if isinstance(adjusted_rows, list) and adjusted_rows
                and isinstance(adjusted_rows[0], dict)
                else []
            ) or []
            required = ("open", "high", "low", "close")
            if not timestamps or not quote or any(
                not isinstance(quote.get(name), list)
                for name in (*required, "volume")
            ):
                raise ValueError("response is missing timestamps or OHLCV arrays")
            length = min(
                len(timestamps),
                *(len(quote.get(name) or []) for name in (*required, "volume")),
            )
            if length <= 0:
                raise ValueError("response has no aligned OHLCV rows")
            adjusted_values = list(adjusted[:length])
            adjusted_values.extend([None] * (length - len(adjusted_values)))
            frame = pd.DataFrame({
                "trade_date": timestamps[:length],
                "open": (quote.get("open") or [])[:length],
                "high": (quote.get("high") or [])[:length],
                "low": (quote.get("low") or [])[:length],
                "close": (quote.get("close") or [])[:length],
                "vol": (quote.get("volume") or [])[:length],
                "adj_close": adjusted_values,
            })
            for column in (*required, "close", "vol", "adj_close"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            valid = frame[list(required)].notna().all(axis=1) & (frame["close"] > 0)
            dropped = int((~valid).sum())
            if dropped:
                warnings.append(f"{ticker}: dropped {dropped} rows with invalid OHLC")
            frame = frame[valid]
            if frame.empty:
                raise ValueError("response has no valid OHLC rows")
            frame["ts_code"] = ticker
            frame["amount"] = frame["close"] * frame["vol"]
            frame["vwap"] = frame["close"]
            timezone_name = str(
                (item.get("meta") or {}).get("exchangeTimezoneName") or "UTC"
            )
            try:
                from zoneinfo import ZoneInfo

                dates = pd.to_datetime(
                    frame["trade_date"], unit="s", utc=True,
                ).dt.tz_convert(ZoneInfo(timezone_name))
            except (ImportError, ValueError, TypeError):
                dates = pd.to_datetime(frame["trade_date"], unit="s", utc=True)
            frame["trade_date"] = dates.dt.strftime("%Y%m%d")
            frames.append(frame)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            warnings.append(f"{ticker}: {type(error).__name__}: {error}")
    if not frames:
        detail = "; ".join(warnings[:8])
        raise ValueError(
            "Yahoo public API returned no rows"
            + (f" ({detail})" if detail else "")
        )
    panel = normalize_panel(pd.concat(frames, ignore_index=True))
    panel = panel[panel["trade_date"].between(start, end)].reset_index(drop=True)
    panel.attrs["sourceWarnings"] = warnings
    return panel


def _pull_databento(
    symbols: list[str], start: str, end: str, options: dict[str, Any],
) -> pd.DataFrame:
    if not symbols:
        raise ValueError("Databento pull requires symbols")
    import databento as db

    client = db.Historical(os.environ["DATABENTO_API_KEY"].strip())
    dataset = str(options.get("dataset") or "").strip()
    if not dataset:
        raise ValueError("Databento pull requires options.dataset")
    start_dt = pd.Timestamp(start).tz_localize("UTC")
    end_dt = pd.Timestamp(end).tz_localize("UTC") + pd.Timedelta(days=1)
    data = client.timeseries.get_range(
        dataset=dataset, schema="ohlcv-1d", symbols=symbols,
        start=start_dt, end=end_dt,
    )
    frame = data.to_df().reset_index()
    if "symbol" not in frame and "ts_code" not in frame:
        raise ValueError("Databento output has no symbol")
    if "ts_event" in frame:
        frame = frame.rename(columns={"ts_event": "trade_date"})
    return normalize_panel(frame)


def _pull_ccxt(
    symbols: list[str], start: str, end: str, options: dict[str, Any],
) -> pd.DataFrame:
    if not symbols:
        raise ValueError("CCXT pull requires symbols")
    import ccxt

    exchange_id = str(options.get("exchange") or "binance").strip()
    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"CCXT exchange not found: {exchange_id}")
    exchange = exchange_class({"enableRateLimit": True})
    timeframe = str(options.get("timeframe") or "1d")
    since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    until = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp() * 1000)
    frames = []
    for symbol in symbols:
        rows = []
        cursor = since
        while cursor < until:
            batch = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=cursor, limit=1000,
            )
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1][0]) + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
        if rows:
            frame = pd.DataFrame(
                rows, columns=["trade_date", "open", "high", "low", "close", "vol"],
            )
            frame["ts_code"] = symbol
            frames.append(frame)
    if not frames:
        raise ValueError("CCXT returned no rows")
    panel = normalize_panel(pd.concat(frames, ignore_index=True))
    return panel[panel["trade_date"].between(start, end)].reset_index(drop=True)


def _pull_snowflake(
    query: str,
    options: dict[str, Any],
    *,
    table_contract: dict[str, Any] | None = None,
) -> pd.DataFrame:
    query = _validate_read_only_query(query)
    max_rows, timeout_seconds, _ = _warehouse_limits(options)
    import snowflake.connector

    connection = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
        role=options.get("role") or os.environ.get("SNOWFLAKE_ROLE"),
    )
    try:
        cursor = connection.cursor()
        try:
            cursor.execute(
                f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {timeout_seconds}"
            )
            cursor.execute(query, options.get("parameters") or ())
            columns = [str(item[0]) for item in cursor.description]
            rows = cursor.fetchmany(max_rows + 1)
            if len(rows) > max_rows:
                raise ValueError(
                    f"Snowflake query exceeded maxRows={max_rows}; add a narrower filter"
                )
            frame = pd.DataFrame(rows, columns=columns)
            return (
                normalize_table(frame, **table_contract)
                if table_contract is not None
                else normalize_panel(frame)
            )
        finally:
            cursor.close()
    finally:
        connection.close()


def _pull_bigquery(
    query: str,
    options: dict[str, Any],
    *,
    table_contract: dict[str, Any] | None = None,
) -> pd.DataFrame:
    query = _validate_read_only_query(query)
    max_rows, timeout_seconds, max_bytes = _warehouse_limits(options)
    from google.cloud import bigquery

    client = bigquery.Client(project=options.get("project") or None)
    if max_bytes is not None:
        dry_run = client.query(
            query,
            job_config=bigquery.QueryJobConfig(
                dry_run=True,
                use_query_cache=False,
            ),
        )
        processed = int(dry_run.total_bytes_processed or 0)
        if processed > int(max_bytes):
            raise ValueError(
                f"BigQuery dry run would process {processed} bytes, above "
                f"maxBytesBilled={max_bytes}"
            )
    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=max_bytes,
        job_timeout_ms=timeout_seconds * 1000,
    )
    result = client.query(query, job_config=job_config).result(
        max_results=max_rows + 1,
    )
    if result.total_rows > max_rows:
        raise ValueError(
            f"BigQuery query exceeded maxRows={max_rows}; add a narrower filter"
        )
    frame = result.to_dataframe()
    return (
        normalize_table(frame, **table_contract)
        if table_contract is not None
        else normalize_panel(frame)
    )


def _pull_wrds(
    query: str,
    options: dict[str, Any],
    *,
    table_contract: dict[str, Any] | None = None,
) -> pd.DataFrame:
    query = _validate_read_only_query(query)
    max_rows, _, _ = _warehouse_limits(options)
    import wrds

    connection = wrds.Connection(
        wrds_username=os.environ["WRDS_USERNAME"],
        wrds_password=os.environ["WRDS_PASSWORD"],
    )
    try:
        bounded = query.rstrip().rstrip(";")
        frame = connection.raw_sql(
            f"SELECT * FROM ({bounded}) AS quant_read_only_query "
            f"LIMIT {max_rows + 1}"
        )
        if len(frame) > max_rows:
            raise ValueError(
                f"WRDS query exceeded maxRows={max_rows}; add a narrower filter"
            )
        return (
            normalize_table(frame, **table_contract)
            if table_contract is not None
            else normalize_panel(frame)
        )
    finally:
        connection.close()


def pull_connector(
    provider: str,
    *,
    output: str | Path,
    start: str,
    end: str,
    input_path: str | Path | None = None,
    symbols: Sequence[str] | str | None = None,
    query: str | None = None,
    options: dict[str, Any] | None = None,
    force: bool = False,
    table_kind: str = "bars",
    identity: Sequence[str] | None = None,
    event_time: str | None = None,
    available_at: str | None = None,
) -> dict[str, Any]:
    """Pull/query provider data locally and write one normalised panel."""
    key = str(provider or "").strip().lower()
    installed_connector = None
    if key in _CONNECTORS:
        _require_ready(
            key,
            credentials=key not in {
                "csv", "parquet", "duckdb", "ccxt", "binance-public", "yahoo-public",
            },
        )
    else:
        installed_connector = _installed_connector(key)
        if installed_connector is None:
            raise ValueError(
                f"未知 data connector: {provider}；请先安装对应 connector-pack"
            )
    start = _compact_date(start, "start")
    end = _compact_date(end, "end")
    if start > end:
        raise ValueError("start must not be after end")
    codes = _symbols(symbols)
    settings = dict(options or {})
    table_kind = str(
        settings.pop("tableKind", table_kind) or "bars",
    ).strip().lower()
    raw_identity = settings.pop("identity", identity)
    if raw_identity is None:
        identity_values = ["ts_code"]
    elif isinstance(raw_identity, str):
        identity_values = [
            item.strip() for item in raw_identity.split(",") if item.strip()
        ]
    else:
        identity_values = [
            str(item).strip() for item in raw_identity if str(item).strip()
        ]
    event_time = str(
        settings.pop("eventTime", event_time)
        or ("trade_date" if table_kind == "bars" else "")
    ).strip()
    available_at = settings.pop("availableAt", available_at)
    available_at = str(available_at).strip() if available_at else None
    if table_kind not in TABLE_KINDS:
        raise ValueError(
            "tableKind must be one of: " + ", ".join(sorted(TABLE_KINDS))
        )
    table_contract = None
    if table_kind != "bars":
        if key not in {
            "csv", "parquet", "duckdb", "snowflake", "bigquery", "wrds",
        }:
            raise ValueError(
                f"{key} currently returns provider bar tables; "
                "use a local/warehouse connector for non-bar qpanel/2 tables"
            )
        if not available_at:
            raise ValueError(
                "non-bar connector pull requires availableAt"
            )
        if not event_time:
            raise ValueError(
                "non-bar connector pull requires eventTime"
            )
        table_contract = {
            "kind": table_kind,
            "identity": identity_values,
            "event_time": event_time,
            "available_at": available_at,
        }
    if key in {"csv", "parquet", "duckdb"}:
        if not input_path:
            raise ValueError(f"{key} connector requires inputPath")
        frame = _local_pull(
            key,
            input_path=input_path,
            start=start,
            end=end,
            table_contract=table_contract,
        )
    elif key == "tushare":
        frame = _pull_tushare(codes, start, end)
    elif key == "polygon":
        frame = _pull_polygon(codes, start, end)
    elif key == "binance-public":
        frame = _pull_binance_public(codes, start, end, settings)
    elif key == "yahoo-public":
        frame = _pull_yahoo_public(codes, start, end, settings)
    elif key == "databento":
        frame = _pull_databento(codes, start, end, settings)
    elif key == "ccxt":
        frame = _pull_ccxt(codes, start, end, settings)
    elif key == "snowflake":
        frame = _pull_snowflake(
            str(query or ""),
            settings,
            table_contract=table_contract,
        )
    elif key == "bigquery":
        frame = _pull_bigquery(
            str(query or ""),
            settings,
            table_contract=table_contract,
        )
    elif key == "wrds":
        frame = _pull_wrds(
            str(query or ""),
            settings,
            table_contract=table_contract,
        )
    else:
        spec, function = installed_connector
        frame = _pull_installed_connector(
            spec,
            function,
            start=start,
            end=end,
            symbols=codes,
            options=settings,
        )
    if key in {"snowflake", "bigquery", "wrds"}:
        if table_contract is None:
            frame = frame[
                frame["trade_date"].between(start, end)
            ].reset_index(drop=True)
        else:
            event_values = pd.to_datetime(
                frame[event_time],
                errors="coerce",
                utc=True,
            )
            frame = frame[
                event_values.ge(pd.Timestamp(start, tz="UTC"))
                & event_values.lt(
                    pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
                )
            ].reset_index(drop=True)
        max_rows, timeout_seconds, max_bytes = _warehouse_limits(settings)
    else:
        max_rows = timeout_seconds = max_bytes = None
    warnings = list(frame.attrs.get("sourceWarnings") or [])
    if table_contract is not None:
        metadata_target = Path(output).expanduser().resolve().with_name(
            Path(output).name + ".qpanel.json"
        )
        if metadata_target.exists() and not force:
            raise ValueError(f"输出元数据已存在: {metadata_target}")
    target = _write_panel(frame, output, force=force)
    table = None
    metadata_path = None
    if table_contract is not None:
        table = panel_metadata(frame, **table_contract)
        metadata_path = _write_table_metadata(target, table)
    return {
        "format": CONNECTOR_FORMAT,
        "version": CONNECTOR_VERSION,
        "status": "PULLED_TABLE" if table is not None else "PULLED",
        "provider": key,
        "output": str(target),
        "window": {"start": start, "end": end},
        "rows": len(frame),
        "assets": int(table["assets"]) if table is not None else int(
            frame["ts_code"].nunique()
        ),
        "dates": int(
            pd.to_datetime(frame[event_time], utc=True).dt.date.nunique()
        ) if table is not None else int(frame["trade_date"].nunique()),
        "columns": [str(column) for column in frame.columns],
        "partial": bool(warnings),
        "warnings": warnings,
        "networkRequired": key not in {"csv", "parquet", "duckdb"},
        "credentialsReturned": False,
        **({
            "table": table,
            "metadata": str(metadata_path),
        } if table is not None else {}),
        **({
            "implementation": "installed_connector_pack",
            "pack": installed_connector[0].get("pack"),
        } if installed_connector is not None else {}),
        **({
            "queryPolicy": {
                "readOnly": True,
                "singleStatement": True,
                "maxRows": max_rows,
                "timeoutSeconds": timeout_seconds,
                **({"maxBytesBilled": max_bytes} if max_bytes is not None else {}),
            }
        } if key in {"snowflake", "bigquery", "wrds"} else {}),
    }


def _read_table_path(path: str | Path) -> pd.DataFrame:
    target = Path(path).expanduser().resolve()
    suffix = target.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(target)
    if suffix == ".parquet":
        return pd.read_parquet(target)
    if suffix in {".duckdb", ".db"}:
        import duckdb

        from quant_sdk.research_workflow import _relation_for_duckdb

        connection = duckdb.connect(str(target), read_only=True)
        try:
            relation, _ = _relation_for_duckdb(connection, target)
            return connection.execute(
                f"SELECT * FROM {relation}"
            ).df()
        finally:
            connection.close()
    raise ValueError("qpanel join input must be CSV, Parquet or DuckDB")


def join_connector_tables(
    *,
    left: str | Path,
    right: str | Path,
    output: str | Path,
    identity: Sequence[str] | str = ("ts_code",),
    left_time: str = "trade_date",
    right_event_time: str = "event_time",
    right_available_at: str = "available_at",
    right_kind: str = "reference",
    columns: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """PIT-join two local tables and write a reproducible qpanel dataset."""
    keys = [identity] if isinstance(identity, str) else [
        str(item).strip() for item in identity if str(item).strip()
    ]
    if right_kind not in TABLE_KINDS:
        raise ValueError("rightKind is not a supported qpanel table kind")
    left_frame = _read_table_path(left)
    right_frame = _read_table_path(right)
    joined = as_of_join(
        left_frame,
        right_frame,
        identity=keys,
        left_time=left_time,
        right_event_time=right_event_time,
        right_available_at=right_available_at,
        columns=columns,
    )
    output_path = Path(output).expanduser().resolve()
    metadata_path = output_path.with_name(output_path.name + ".join.json")
    if metadata_path.exists() and not force:
        raise ValueError(f"连接元数据已存在: {metadata_path}")
    target = _write_panel(joined, output, force=force)
    bundle = bundle_manifest([
        {
            "name": "left",
            "kind": "bars",
            "identity": keys,
            "eventTime": left_time,
            "availableAt": None,
        },
        {
            "name": "right",
            "kind": right_kind,
            "identity": keys,
            "eventTime": right_event_time,
            "availableAt": right_available_at,
        },
    ])
    metadata = {
        "format": "qpanel-join/1",
        "status": "JOINED",
        "bundle": bundle,
        "inputs": {
            "left": str(Path(left).expanduser().resolve()),
            "right": str(Path(right).expanduser().resolve()),
        },
        "output": str(target),
        "rows": len(joined),
        "columns": [str(column) for column in joined.columns],
        "futureAvailabilityAccepted": False,
    }
    temporary = metadata_path.with_name(metadata_path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, metadata_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        **metadata,
        "metadata": str(metadata_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local data connectors")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--json", action="store_true")
    pull = sub.add_parser("pull")
    pull.add_argument("--provider", required=True)
    pull.add_argument("--output", required=True)
    pull.add_argument("--start", required=True)
    pull.add_argument("--end", required=True)
    pull.add_argument("--input")
    pull.add_argument("--symbols")
    pull.add_argument("--query")
    pull.add_argument("--options")
    pull.add_argument("--force", action="store_true")
    pull.add_argument("--table-kind", default="bars")
    pull.add_argument("--identity", default=None)
    pull.add_argument("--event-time", default=None)
    pull.add_argument("--available-at", default=None)
    join = sub.add_parser("join")
    join.add_argument("--left", required=True)
    join.add_argument("--right", required=True)
    join.add_argument("--output", required=True)
    join.add_argument("--identity", default="ts_code")
    join.add_argument("--left-time", default="trade_date")
    join.add_argument("--right-event-time", default="event_time")
    join.add_argument("--right-available-at", default="available_at")
    join.add_argument(
        "--right-kind",
        choices=sorted(TABLE_KINDS - {"bars"}),
        default="reference",
    )
    join.add_argument("--columns")
    join.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "list":
        result = list_connectors()
    elif args.command == "pull":
        result = pull_connector(
            args.provider, output=args.output, start=args.start, end=args.end,
            input_path=args.input, symbols=args.symbols, query=args.query,
            options=json.loads(args.options) if args.options else {},
            force=bool(args.force),
            table_kind=args.table_kind,
            identity=args.identity,
            event_time=args.event_time,
            available_at=args.available_at,
        )
    else:
        result = join_connector_tables(
            left=args.left,
            right=args.right,
            output=args.output,
            identity=args.identity,
            left_time=args.left_time,
            right_event_time=args.right_event_time,
            right_available_at=args.right_available_at,
            right_kind=args.right_kind,
            columns=(
                [item.strip() for item in args.columns.split(",") if item.strip()]
                if args.columns else None
            ),
            force=bool(args.force),
        )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
