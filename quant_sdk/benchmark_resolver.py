"""Market-aware benchmark selection for the portable local research runtime.

The protocol used to inherit an A-share production default (``000300.SH``) in
every market. That made a crypto or US study look valid while silently lacking
the matching benchmark return series. This resolver keeps the legacy default
readable but replaces it when the observed panel clearly belongs to another
market. Mixed or unknown markets remain explicit and never receive a fabricated
market index.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

LEGACY_A_SHARE_DEFAULT = "000300.SH"
BENCHMARK_FORMAT = "qbench-resolution/1"

_CRYPTO = re.compile(
    r"(?:USDT|USDC|BUSD|FDUSD|BTC|ETH|USD)$|/USD$|-USD$",
    re.IGNORECASE,
)
_A_SHARE_SUFFIX = re.compile(r"\.(?:SH|SZ|BJ)$", re.IGNORECASE)
_US_SUFFIX = re.compile(r"\.US$", re.IGNORECASE)
_US_TICKER = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


def _market(symbols: Iterable[Any]) -> str:
    values = {str(value).strip() for value in symbols if str(value).strip()}
    if not values:
        return "UNKNOWN"
    crypto = sum(1 for value in values if _CRYPTO.search(value))
    a_share = sum(1 for value in values if _A_SHARE_SUFFIX.search(value))
    us = sum(
        1 for value in values
        if _US_SUFFIX.search(value)
        or ("/" not in value and not _A_SHARE_SUFFIX.search(value)
            and _US_TICKER.fullmatch(value.upper()))
    )
    total = len(values)
    if a_share == total:
        return "CN"
    if crypto == total:
        return "CRYPTO"
    if us == total:
        return "US"
    if max(a_share, crypto, us) / total >= 0.8:
        return (
            "CN" if a_share > crypto and a_share > us
            else "CRYPTO" if crypto > a_share and crypto > us
            else "US"
        )
    return "MIXED"


def resolve_benchmark(
    symbols: Iterable[Any],
    *,
    explicit: str | None = None,
) -> dict[str, Any]:
    """Resolve one benchmark without hiding a missing market series."""
    observed = sorted({str(value).strip() for value in symbols if str(value).strip()})
    market = _market(observed)
    requested = str(explicit or "").strip()
    if requested and requested.upper() not in {"AUTO", "MARKET"}:
        legacy_replaced = (
            requested.upper() == LEGACY_A_SHARE_DEFAULT
            and market in {"CRYPTO", "US"}
        )
        if not legacy_replaced:
            return {
                "format": BENCHMARK_FORMAT,
                "market": market,
                "code": requested,
                "basis": "explicit",
                "legacyDefaultReplaced": False,
                "requiresMarketSeries": True,
            }
    else:
        legacy_replaced = False

    if market == "CN":
        code = "000300.SH"
        basis = "market_inference"
    elif market == "CRYPTO":
        code = "BTCUSDT"
        basis = "market_inference"
    elif market == "US":
        code = "SPY"
        basis = "market_inference"
    else:
        code = None
        basis = "equal_weight_fallback"
    return {
        "format": BENCHMARK_FORMAT,
        "market": market,
        "code": code,
        "basis": basis,
        "legacyDefaultReplaced": legacy_replaced,
        "requiresMarketSeries": code is not None,
        "observedSymbols": observed[:24],
    }


def sample_symbols(
    data_path: str | Path,
    *,
    start: str | None = None,
    end: str | None = None,
    limit: int = 200,
) -> list[str]:
    """Read a bounded symbol sample without loading or executing the panel."""
    path = Path(data_path).expanduser().resolve()
    if not path.is_file():
        return []
    try:
        import duckdb

        connection = (
            duckdb.connect(str(path), read_only=True)
            if path.suffix.lower() in {".duckdb", ".db"}
            else duckdb.connect()
        )
        try:
            if path.suffix.lower() in {".csv", ".parquet"}:
                escaped = str(path).replace("'", "''")
                relation = (
                    f"read_parquet('{escaped}')"
                    if path.suffix.lower() == ".parquet"
                    else f"read_csv_auto('{escaped}', sample_size=20000)"
                )
            else:
                tables = [str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()]
                if not tables:
                    return []
                relation = None
                for table in ("daily_raw", *tables):
                    quoted = '"' + table.replace('"', '""') + '"'
                    try:
                        columns = {
                            str(row[0])
                            for row in connection.execute(f"DESCRIBE {quoted}").fetchall()
                        }
                    except Exception:
                        continue
                    if "ts_code" in columns:
                        relation = quoted
                        break
                if relation is None:
                    return []
            where: list[str] = []
            params: list[Any] = []
            if start:
                where.append("replace(CAST(trade_date AS VARCHAR), '-', '') >= ?")
                params.append(str(start).replace("-", ""))
            if end:
                where.append("replace(CAST(trade_date AS VARCHAR), '-', '') <= ?")
                params.append(str(end).replace("-", ""))
            clause = f" WHERE {' AND '.join(where)}" if where else ""
            rows = connection.execute(
                f"SELECT DISTINCT ts_code FROM {relation}{clause} "
                "ORDER BY ts_code LIMIT ?",
                [*params, max(1, min(int(limit), 1000))],
            ).fetchall()
            return [str(row[0]) for row in rows if str(row[0]).strip()]
        finally:
            connection.close()
    except Exception:
        # Resolution is advisory. A malformed preview must not prevent the
        # eventual loader from returning its own precise error.
        return []
