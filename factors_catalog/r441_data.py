"""DuckDB adapters for the complete r441 registry-factor oracle."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factors_catalog.r441_oracle.fundamental_context import (
    load_symbol_fundamental_factor_context,
)

_FINANCIAL_FIELDS = {
    "income": {
        "report_period": "end_date",
        "ann_date": "ann_date",
        "f_ann_date": "f_ann_date",
        "revenue": "revenue",
        "operate_profit": "operate_profit",
        "net_income_parent": "n_income_attr_p",
        "oper_cost": "oper_cost",
        "ebitda": "ebitda",
    },
    "balancesheet": {
        "report_period": "end_date",
        "ann_date": "ann_date",
        "f_ann_date": "f_ann_date",
        "total_assets": "total_assets",
        "total_liabilities": "total_liab",
        "cash": "money_cap",
        "accounts_receivable": "accounts_receiv",
        "inventories": "inventories",
        "fixed_assets": "fix_assets",
        "current_assets": "total_cur_assets",
        "current_liabilities": "total_cur_liab",
        "retained_earnings": "undistr_porfit",
    },
    "cashflow": {
        "report_period": "end_date",
        "ann_date": "ann_date",
        "f_ann_date": "f_ann_date",
        "operating_cashflow": "n_cashflow_act",
        "financing_cashflow": "n_cash_flows_fnc_act",
    },
}


@dataclass(frozen=True)
class ReferenceFactorContext:
    panel: pd.DataFrame
    fundamental_by_symbol_date: dict[str, dict[str, dict[str, float]]]
    metadata: dict[str, Any]


class _FinancialRows:
    def __init__(
        self,
        income: Mapping[str, list[dict[str, Any]]],
        balancesheet: Mapping[str, list[dict[str, Any]]],
        cashflow: Mapping[str, list[dict[str, Any]]],
    ) -> None:
        self.rows = {
            "income": income,
            "balancesheet": balancesheet,
            "cashflow": cashflow,
        }

    def list_financial_rows_bulk(
        self, symbols: list[str], dataset: str, *, limit: int
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            symbol: list(self.rows[dataset].get(symbol, ()))[:limit]
            for symbol in symbols
        }

    def list_financial_rows(
        self, symbol: str, dataset: str, *, limit: int
    ) -> list[dict[str, Any]]:
        return list(self.rows[dataset].get(symbol, ()))[:limit]


def _table_exists(conn: Any, name: str) -> bool:
    return name in {
        str(row[0])
        for row in conn.execute("SHOW TABLES").fetchall()
    }


def _query(conn: Any, sql: str, params: Iterable[Any] = ()) -> pd.DataFrame:
    return conn.execute(sql, list(params)).df()


def _date_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("-", "", regex=False).str[:8],
        errors="coerce",
    )


def _attach_benchmark(
    conn: Any,
    panel: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    if not _table_exists(conn, "index_daily"):
        raise ValueError("r441_benchmark_unavailable:index_daily_missing")
    rows = _query(
        conn,
        """
        SELECT CAST(trade_date AS VARCHAR) AS trade_date,
               CAST(close AS DOUBLE) AS close,
               CAST(open AS DOUBLE) AS open,
               CAST(pre_close AS DOUBLE) AS pre_close
        FROM index_daily
        WHERE ts_code='000300.SH' AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date
        """,
        (start, end),
    )
    if rows.empty:
        raise ValueError("r441_benchmark_unavailable:000300.SH")
    rows["trade_date"] = rows["trade_date"].astype(str)
    lookup = rows.set_index("trade_date")
    result = panel.copy()
    result["benchmark_close"] = (
        result["trade_date"].astype(str).map(lookup["close"]).ffill()
    )
    if not np.isfinite(result["benchmark_close"]).any():
        raise ValueError("r441_benchmark_unavailable:no_aligned_close")
    return result


def _attach_moneyflow(
    conn: Any,
    panel: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not _table_exists(conn, "moneyflow"):
        return panel.assign(stock_flow=np.nan), None
    rows = _query(
        conn,
        """
        SELECT CAST(ts_code AS VARCHAR) AS ts_code,
               CAST(trade_date AS VARCHAR) AS trade_date,
               CAST(net_mf_amount AS DOUBLE) AS net_mf_amount
        FROM moneyflow
        WHERE trade_date BETWEEN ? AND ?
        """,
        (start, end),
    )
    if rows.empty:
        return panel.assign(stock_flow=np.nan), None
    rows["stock_flow"] = (
        pd.to_numeric(rows["net_mf_amount"], errors="coerce") * 10_000.0
    )
    merge = rows[["ts_code", "trade_date", "stock_flow"]]
    result = panel.merge(
        merge,
        on=["ts_code", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    return result, rows[["ts_code", "trade_date", "stock_flow"]].copy()


def _load_membership(
    conn: Any,
    *,
    start: str,
    end: str,
) -> pd.DataFrame | None:
    if not _table_exists(conn, "industry_membership"):
        return None
    columns = {
        str(row[0])
        for row in conn.execute('DESCRIBE "industry_membership"').fetchall()
    }
    required = {"ts_code", "start_date", "end_date", "industry_code"}
    if not required.issubset(columns):
        return None
    frame = _query(
        conn,
        """
        SELECT CAST(ts_code AS VARCHAR) AS ts_code,
               CAST(start_date AS VARCHAR) AS start_date,
               CAST(end_date AS VARCHAR) AS end_date,
               CAST(industry_code AS VARCHAR) AS industry_code
        FROM industry_membership
        WHERE CAST(start_date AS VARCHAR) <= ?
          AND (end_date IS NULL OR CAST(end_date AS VARCHAR) >= ?)
        """,
        (end, start),
    )
    if frame.empty:
        return None
    for column in ("start_date", "end_date"):
        frame[column] = pd.to_numeric(
            frame[column].astype(str).str.replace("-", "", regex=False).str[:8],
            errors="coerce",
        )
    frame = frame.dropna(subset=["ts_code", "start_date", "industry_code"])
    frame = frame.sort_values(["ts_code", "start_date"], kind="mergesort")
    return frame


def _attach_industry(
    panel: pd.DataFrame,
    membership: pd.DataFrame | None,
) -> pd.DataFrame:
    result = panel.copy()
    if membership is None:
        result["industry_code"] = pd.NA
        return result
    left = result[["ts_code", "trade_date"]].copy()
    left["_panel_index"] = result.index
    left["_date_i"] = _date_int(left["trade_date"])
    left = left.sort_values(["_date_i", "ts_code"], kind="mergesort")
    right = membership.rename(
        columns={"start_date": "_date_i"}
    ).sort_values(["_date_i", "ts_code"], kind="mergesort")
    merged = pd.merge_asof(
        left,
        right,
        on="_date_i",
        by="ts_code",
        direction="backward",
        allow_exact_matches=True,
    )
    valid = (
        merged["end_date"].isna()
        | (merged["end_date"] >= merged["_date_i"])
    )
    merged.loc[~valid, "industry_code"] = pd.NA
    industry = merged.set_index("_panel_index")["industry_code"]
    result["industry_code"] = industry.reindex(result.index)
    duplicates = result[["ts_code", "trade_date"]].duplicated().any()
    if duplicates:
        raise ValueError("r441_industry_join_duplicated_panel_keys")
    return result


def _attach_sector_context(
    conn: Any,
    panel: pd.DataFrame,
    moneyflow: pd.DataFrame | None,
    *,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "industry_membership": "UNAVAILABLE",
        "sector_daily": "UNAVAILABLE",
        "sector_flow": "UNAVAILABLE",
        "sector_breadth": "UNAVAILABLE",
    }
    if "industry_code" not in panel or panel["industry_code"].isna().all():
        return panel, metadata
    result = panel
    membership_status = {
        str(value): int(count)
        for value, count in result["industry_code"].value_counts(dropna=False).items()
    }
    metadata["industry_membership"] = "AVAILABLE"
    metadata["industry_membership_coverage"] = {
        "rows": len(result),
        "assigned": int(result["industry_code"].notna().sum()),
        "codes": len(membership_status),
    }

    if _table_exists(conn, "sw_daily"):
        sector_rows = _query(
            conn,
            """
            SELECT CAST(ts_code AS VARCHAR) AS industry_code,
                   CAST(trade_date AS VARCHAR) AS trade_date,
                   CAST(close AS DOUBLE) AS close
            FROM sw_daily
            WHERE trade_date BETWEEN ? AND ?
            """,
            (start, end),
        )
        if not sector_rows.empty:
            candidate = result.merge(
                sector_rows.rename(columns={"close": "sector_close"}),
                on=["industry_code", "trade_date"],
                how="left",
                validate="many_to_one",
            )
            coverage = float(candidate["sector_close"].notna().mean())
            if coverage > 0.0:
                result = candidate
                metadata["sector_daily"] = "AVAILABLE"
                metadata["sector_daily_match_coverage"] = round(coverage, 6)

    if moneyflow is not None and not moneyflow.empty:
        attributed = moneyflow.merge(
            result[["ts_code", "trade_date", "industry_code"]].drop_duplicates(),
            on=["ts_code", "trade_date"],
            how="left",
            validate="one_to_one",
        )
        sector_flow = (
            attributed.dropna(subset=["industry_code"])
            .groupby(["industry_code", "trade_date"], as_index=False)["stock_flow"]
            .sum()
            .rename(columns={"stock_flow": "sector_flow"})
        )
        if not sector_flow.empty:
            result = result.merge(
                sector_flow,
                on=["industry_code", "trade_date"],
                how="left",
                validate="many_to_one",
            )
            metadata["sector_flow"] = "AVAILABLE"

    ordered = result.sort_values(
        ["ts_code", "trade_date"], kind="mergesort"
    ).copy()
    ordered["_stock_return"] = (
        ordered.groupby("ts_code", sort=False)["close"]
        .pct_change(fill_method=None)
    )
    ordered["_trailing_return_20"] = (
        ordered.groupby("ts_code", sort=False)["close"]
        .pct_change(20, fill_method=None)
        .groupby(ordered["ts_code"], sort=False)
        .shift(1)
    )
    sector_inputs = ordered.dropna(
        subset=["industry_code", "_stock_return"]
    )
    breadth = (
        sector_inputs.assign(
            is_up=(sector_inputs["_stock_return"] > 0.0).astype(float),
            is_observed=1.0,
        )
        .groupby(["industry_code", "trade_date"], as_index=False)[
            ["is_up", "is_observed"]
        ]
        .sum()
        .rename(
            columns={
                "is_up": "sector_up_count",
                "is_observed": "sector_total_count",
            }
        )
    )
    if not breadth.empty:
        result = result.merge(
            breadth,
            on=["industry_code", "trade_date"],
            how="left",
            validate="many_to_one",
        )
        metadata["sector_breadth"] = "AVAILABLE"

    sector_inputs["_signal_rank"] = (
        sector_inputs.groupby(["industry_code", "trade_date"])[
            "_trailing_return_20"
        ]
        .rank(method="average", pct=True)
    )
    sector_inputs["_group_count"] = (
        sector_inputs.groupby(["industry_code", "trade_date"])[
            "_stock_return"
        ]
        .transform("count")
    )
    sector_inputs["_leader_component"] = np.where(
        (sector_inputs["_group_count"] >= 5)
        & (sector_inputs["_signal_rank"] > 0.8),
        sector_inputs["_stock_return"],
        np.nan,
    )
    sector_inputs["_laggard_component"] = np.where(
        (sector_inputs["_group_count"] >= 5)
        & (sector_inputs["_signal_rank"] <= 0.2),
        sector_inputs["_stock_return"],
        np.nan,
    )
    sector_returns = (
        sector_inputs.groupby(["industry_code", "trade_date"], as_index=False)
        .agg(
            _sector_return=("_stock_return", "mean"),
            _leader_return=("_leader_component", "mean"),
            _laggard_return=("_laggard_component", "mean"),
        )
    )
    if not sector_returns.empty:
        sector_returns["_leader_return"] = sector_returns[
            "_leader_return"
        ].fillna(sector_returns["_sector_return"])
        sector_returns["_laggard_return"] = sector_returns[
            "_laggard_return"
        ].fillna(sector_returns["_sector_return"])
    if not sector_returns.empty:
        sector_returns = sector_returns.sort_values(
            ["industry_code", "trade_date"], kind="mergesort"
        )
        for column, output in (
            ("_sector_return", "sector_close"),
            ("_leader_return", "leader_close"),
            ("_laggard_return", "laggard_close"),
        ):
            sector_returns[output] = (
                sector_returns.groupby("industry_code", sort=False)[column]
                .transform(lambda values: 100.0 * (1.0 + values.fillna(0.0)).cumprod())
            )
        derived_columns = [
            "industry_code",
            "trade_date",
            "sector_close",
            "leader_close",
            "laggard_close",
        ]
        if "sector_close" in result:
            result = result.drop(columns=["sector_close"])
        result = result.merge(
            sector_returns[derived_columns],
            on=["industry_code", "trade_date"],
            how="left",
            validate="many_to_one",
        )
        metadata["sector_daily"] = "AVAILABLE_DERIVED_PIT_EQUAL_WEIGHT"
        metadata["leader_laggard"] = "AVAILABLE_DERIVED_TRAILING_20D"

    result["market_close"] = result["benchmark_close"]
    return result, metadata


def _attach_market_and_events(
    conn: Any,
    panel: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = panel.copy()
    metadata: dict[str, Any] = {
        "sentiment": "DERIVED_REAL_MARKET_BREADTH",
        "limit_events": "UNAVAILABLE",
        "top_list_events": "UNAVAILABLE",
        "regulatory_events": "UNAVAILABLE",
        "shareholder_events": "UNAVAILABLE",
        "policy_events": "UNAVAILABLE",
        "hotwords": "UNAVAILABLE",
    }
    returns = (
        result.sort_values(["ts_code", "trade_date"], kind="mergesort")
        .groupby("ts_code", sort=False)["close"]
        .pct_change(fill_method=None)
    )
    breadth = result[["trade_date"]].copy()
    breadth["up"] = (returns > 0.0).astype(float)
    breadth["observed"] = returns.notna().astype(float)
    daily = (
        breadth.groupby("trade_date", as_index=False)[["up", "observed"]]
        .sum()
        .sort_values("trade_date")
    )
    daily["sentiment"] = daily["up"] / daily["observed"].clip(lower=1.0)
    daily["market_reversal"] = daily["sentiment"].diff().fillna(0.0)

    benchmark_gap = (
        result.sort_values(["ts_code", "trade_date"])
        .groupby("trade_date", as_index=False)["benchmark_close"]
        .first()
    )
    benchmark_gap["market_gap"] = benchmark_gap["benchmark_close"].pct_change(
        fill_method=None
    )
    daily = daily.merge(benchmark_gap, on="trade_date", how="left")

    if _table_exists(conn, "limit_list_d"):
        limits = _query(
            conn,
            """
            SELECT CAST(trade_date AS VARCHAR) AS trade_date,
                   CAST("limit" AS VARCHAR) AS limit_type,
                   CAST(open_times AS DOUBLE) AS open_times
            FROM limit_list_d
            WHERE trade_date BETWEEN ? AND ?
            """,
            (start, end),
        )
        if not limits.empty:
            limits["up"] = limits["limit_type"].astype(str).str.upper().eq("U")
            limits["_opened"] = limits["open_times"].fillna(0.0) > 0.0
            grouped = (
                limits.groupby("trade_date", as_index=False)
                .agg(
                    limit_up_count=("up", "sum"),
                    limit_break_count=("_opened", "sum"),
                )
            )
            grouped["board_break"] = (
                grouped["limit_break_count"]
                / grouped["limit_up_count"].clip(lower=1.0)
            ).clip(0.0, 1.0)
            daily = daily.merge(grouped, on="trade_date", how="left")
            metadata["limit_events"] = "AVAILABLE"

    result = result.merge(
        daily[
            [
                "trade_date",
                "sentiment",
                "market_reversal",
                "market_gap",
                *(
                    ["board_break"]
                    if "board_break" in daily.columns
                    else []
                ),
            ]
        ],
        on="trade_date",
        how="left",
        validate="many_to_one",
    )

    if _table_exists(conn, "top_list"):
        events = _query(
            conn,
            """
            SELECT CAST(ts_code AS VARCHAR) AS ts_code,
                   CAST(trade_date AS VARCHAR) AS trade_date,
                   CAST(net_amount AS DOUBLE) AS net_amount,
                   CAST(amount AS DOUBLE) AS event_amount,
                   CAST(amount_rate AS DOUBLE) AS amount_rate
            FROM top_list
            WHERE trade_date BETWEEN ? AND ?
            """,
            (start, end),
        )
        if not events.empty:
            events = (
                events.groupby(
                    ["ts_code", "trade_date"], as_index=False
                )
                .agg(
                    net_amount=("net_amount", "sum"),
                    event_amount=("event_amount", "sum"),
                    amount_rate=("amount_rate", "mean"),
                )
            )
            events["event_direction"] = np.sign(
                pd.to_numeric(events["net_amount"], errors="coerce").fillna(0.0)
            )
            events["event_intensity"] = (
                pd.to_numeric(events["net_amount"], errors="coerce").abs()
                / pd.to_numeric(events["event_amount"], errors="coerce")
                .abs()
                .clip(lower=1.0)
            ).clip(0.0, 1.5)
            events["event_confidence"] = (
                pd.to_numeric(events["amount_rate"], errors="coerce")
                .abs()
                .div(100.0)
                .clip(0.0, 1.0)
            )
            events["event_impact"] = (
                events["event_direction"] * events["event_intensity"]
            )
            events["event_symbol_hits"] = 1.0
            event_columns = [
                "ts_code",
                "trade_date",
                "event_direction",
                "event_intensity",
                "event_confidence",
                "event_impact",
                "event_symbol_hits",
            ]
            result = result.merge(
                events[event_columns],
                on=["ts_code", "trade_date"],
                how="left",
                validate="one_to_one",
            )
            for column in event_columns[2:]:
                result[column] = result[column].fillna(0.0)
            metadata["top_list_events"] = "AVAILABLE"
    return result, metadata


def _financial_rows(
    conn: Any,
    table: str,
    *,
    end: str,
    symbols: list[str],
) -> dict[str, list[dict[str, Any]]]:
    if not _table_exists(conn, table):
        return {}
    mapping = _FINANCIAL_FIELDS[table]
    columns = {
        str(row[0])
        for row in conn.execute(f'DESCRIBE "{table}"').fetchall()
    }
    available = {
        target: source
        for target, source in mapping.items()
        if source in columns
    }
    select = ", ".join(
        f'CAST("{source}" AS VARCHAR) AS "{target}"'
        if target in {"report_period", "ann_date", "f_ann_date"}
        else f'TRY_CAST("{source}" AS DOUBLE) AS "{target}"'
        for target, source in available.items()
    )
    rows = _query(
        conn,
        f"""
        WITH ranked AS (
          SELECT ts_code, {select},
                 ROW_NUMBER() OVER (
                   PARTITION BY ts_code, end_date
                   ORDER BY COALESCE(f_ann_date, ann_date, end_date) DESC,
                            update_flag DESC
                 ) AS revision,
                 ROW_NUMBER() OVER (
                   PARTITION BY ts_code
                   ORDER BY end_date DESC
                 ) AS period_sequence
          FROM "{table}"
          WHERE COALESCE(f_ann_date, ann_date, end_date) <= ?
            AND ts_code IN (SELECT UNNEST(?))
        )
        SELECT * FROM ranked
        WHERE revision=1 AND period_sequence<=24
        ORDER BY ts_code, report_period
        """,
        (end, symbols),
    )
    if rows.empty:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows.to_dict("records"):
        symbol = str(row.pop("ts_code"))
        row.pop("revision", None)
        row.pop("period_sequence", None)
        grouped.setdefault(symbol, []).append(row)
    return grouped


def _supplemental_financial_rows(
    root: str | Path | None,
    table: str,
) -> dict[str, list[dict[str, Any]]]:
    if not root:
        return {}
    directory = Path(root)
    if not directory.is_dir():
        return {}
    frames = [
        pd.read_parquet(path)
        for path in sorted(directory.glob(f"{table}_*.parquet"))
    ]
    if not frames:
        return {}
    frame = pd.concat(frames, ignore_index=True)
    if {
        "ts_code",
        "end_date",
        "f_ann_date",
        "ann_date",
    }.issubset(frame.columns):
        frame = (
            frame.assign(
                _effective_announcement=frame["f_ann_date"]
                .fillna(frame["ann_date"])
                .fillna(frame["end_date"])
                .astype(str)
            )
            .sort_values(
                ["ts_code", "end_date", "_effective_announcement"],
                kind="mergesort",
            )
            .drop_duplicates(["ts_code", "end_date"], keep="last")
            .drop(columns="_effective_announcement")
        )
    mapping = _FINANCIAL_FIELDS[table]
    available = {
        target: source
        for target, source in mapping.items()
        if source in frame.columns
    }
    source_columns = ["ts_code", *available.values()]
    rows = frame[source_columns].rename(
        columns={source: target for target, source in available.items()}
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in rows.to_dict("records"):
        symbol = str(record.pop("ts_code", ""))
        if symbol:
            grouped.setdefault(symbol, []).append(record)
    return grouped


def _load_fundamentals(
    conn: Any,
    panel: pd.DataFrame,
    *,
    start: str,
    end: str,
    supplement_root: str | Path | None,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, Any]]:
    symbols = sorted(panel["ts_code"].astype(str).unique())
    income = _financial_rows(
        conn, "income", end=end, symbols=symbols
    )
    balancesheet = _financial_rows(
        conn, "balancesheet", end=end, symbols=symbols
    )
    cashflow = _financial_rows(
        conn, "cashflow", end=end, symbols=symbols
    )
    supplements = {
        table: _supplemental_financial_rows(supplement_root, table)
        for table in _FINANCIAL_FIELDS
    }
    for symbol, rows in supplements["income"].items():
        income.setdefault(symbol, []).extend(rows)
    for symbol, rows in supplements["balancesheet"].items():
        balancesheet.setdefault(symbol, []).extend(rows)
    for symbol, rows in supplements["cashflow"].items():
        cashflow.setdefault(symbol, []).extend(rows)
    if not (income or balancesheet or cashflow):
        return {}, {"status": "UNAVAILABLE", "reason": "financial_tables_empty"}
    calendar = _query(
        conn,
        """
        SELECT CAST(cal_date AS VARCHAR) AS trade_date
        FROM trade_cal
        WHERE is_open=1 AND cal_date<=?
        ORDER BY cal_date
        """,
        (end,),
    )
    trading_days = tuple(
        f"{value[:4]}-{value[4:6]}-{value[6:8]}"
        for value in calendar["trade_date"].astype(str)
    )
    by_symbol = load_symbol_fundamental_factor_context(
        _FinancialRows(income, balancesheet, cashflow),
        symbols=symbols,
        start_day=f"{start[:4]}-{start[4:6]}-{start[6:8]}",
        end_day=f"{end[:4]}-{end[4:6]}-{end[6:8]}",
        trading_days=trading_days,
    )
    missing = [symbol for symbol in symbols if symbol not in by_symbol]
    normalized = {
        symbol: {
            str(day).replace("-", "")[:8]: values
            for day, values in days.items()
        }
        for symbol, days in by_symbol.items()
    }
    return normalized, {
        "status": "AVAILABLE" if by_symbol else "UNAVAILABLE",
        "symbols": len(by_symbol),
        "missing_symbols": len(missing),
        "income_symbols": len(income),
        "balancesheet_symbols": len(balancesheet),
        "cashflow_symbols": len(cashflow),
        "supplement_root": str(supplement_root) if supplement_root else None,
        "supplement_rows": {
            table: sum(len(rows) for rows in values.values())
            for table, values in supplements.items()
        },
        "latest_reported_period": max(
            (
                str(row.get("report_period") or "")
                for rows in income.values()
                for row in rows
            ),
            default=None,
        ),
    }


def load_reference_factor_context(
    conn: Any,
    panel: pd.DataFrame,
    *,
    start: str,
    end: str,
    supplemental_financial_root: str | Path | None = None,
) -> ReferenceFactorContext:
    """Attach all available real inputs required by the registry oracle."""

    frame = panel.copy()
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["trade_date"] = frame["trade_date"].astype(str).str.replace(
        "-", "", regex=False
    ).str[:8]
    frame = _attach_benchmark(conn, frame, start=start, end=end)
    frame, moneyflow = _attach_moneyflow(conn, frame, start=start, end=end)
    membership = _load_membership(conn, start=start, end=end)
    frame = _attach_industry(frame, membership)
    frame, sector_metadata = _attach_sector_context(
        conn,
        frame,
        moneyflow,
        start=start,
        end=end,
    )
    frame, event_metadata = _attach_market_and_events(
        conn,
        frame,
        start=start,
        end=end,
    )
    fundamentals, fundamental_metadata = _load_fundamentals(
        conn,
        frame,
        start=start,
        end=end,
        supplement_root=supplemental_financial_root,
    )
    metadata = {
        "benchmark": "AVAILABLE",
        "moneyflow": (
            "AVAILABLE" if moneyflow is not None else "UNAVAILABLE"
        ),
        **sector_metadata,
        **event_metadata,
        "fundamentals": fundamental_metadata,
        "event_proxy_contract": "daily-real-source-v1",
        "interaction_contract": "interaction-v1",
    }
    return ReferenceFactorContext(
        panel=frame,
        fundamental_by_symbol_date=fundamentals,
        metadata=metadata,
    )
