"""Point-in-Time (PIT) Fundamentals Query Engine & Multi-Asset Access.

Provides 100% JoinQuant-compatible and institutional-grade Point-in-Time
fundamental data querying with ZERO lookahead bias (严格杜绝未来函数):
- Financial statements strictly aligned to announcement dates (ann_date <= query_date)
- Daily valuation metrics (PE, PB, PS, turnover, market cap) from daily_basic
- Fluent Query builder mimicking JoinQuant: `query(valuation.code, valuation.pe, indicator.roe).filter(...)`
- Multi-asset capabilities: Convertible Bonds (`get_cb_daily` with Double-Low factor) & ETFs (`get_fund_daily`)
"""
from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = [
    "Field",
    "FundamentalsQuery",
    "TableNamespace",
    "balance",
    "cash_flow",
    "get_cb_daily",
    "get_fund_daily",
    "get_fundamentals",
    "income",
    "indicator",
    "query",
    "valuation",
]

_DB_DEFAULT = "tushare_data.duckdb"


# --------------------------------------------------------------------------- #
# Table & Field Abstractions for JoinQuant Syntax Emulation
# --------------------------------------------------------------------------- #
class Field:
    """Represents a selectable/filterable table column in fundamentals query."""

    def __init__(self, table_name: str, column_name: str, alias: str = ""):
        self.table_name = table_name
        self.column_name = column_name
        self.alias = alias or column_name

    def __str__(self) -> str:
        return f"{self.table_name}.{self.column_name}"

    def __repr__(self) -> str:
        return f"<Field {self.table_name}.{self.column_name}>"

    # Operator overloading for fluent query filters
    def __eq__(self, other: Any) -> FilterCondition:
        return FilterCondition(self, "=", other)

    def __ne__(self, other: Any) -> FilterCondition:
        return FilterCondition(self, "!=", other)

    def __lt__(self, other: Any) -> FilterCondition:
        return FilterCondition(self, "<", other)

    def __le__(self, other: Any) -> FilterCondition:
        return FilterCondition(self, "<=", other)

    def __gt__(self, other: Any) -> FilterCondition:
        return FilterCondition(self, ">", other)

    def __ge__(self, other: Any) -> FilterCondition:
        return FilterCondition(self, ">=", other)

    def in_(self, values: list[Any]) -> FilterCondition:
        return FilterCondition(self, "IN", values)


class FilterCondition:
    """Represents a boolean condition on a Field."""

    def __init__(self, field_obj: Field | str, op: str, value: Any):
        self.field_obj = field_obj
        self.op = op
        self.value = value

    def to_sql(self) -> str:
        col_str = str(self.field_obj)
        if self.op.upper() == "IN":
            if isinstance(self.value, (list, tuple, set)):
                val_list = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in self.value)
                return f"{col_str} IN ({val_list})"
            return f"{col_str} IN ({self.value})"

        if isinstance(self.value, str):
            return f"{col_str} {self.op} '{self.value}'"
        return f"{col_str} {self.op} {self.value}"


class TableNamespace:
    """Dynamic attribute resolver for JoinQuant-like namespaces."""

    _COLUMN_MAP: dict[str, str] = {
        # valuation aliases
        "code": "ts_code",
        "pe_ratio": "pe",
        "pe_ttm": "pe_ttm",
        "pb_ratio": "pb",
        "ps_ratio": "ps",
        "market_cap": "total_mv",
        "circulating_market_cap": "circ_mv",
        "turnover_ratio": "turnover_rate",
        # indicator aliases
        "inc_net_profit_year_on_year": "profit_dedt",
        "gross_profit_margin": "gross_margin",
        "roe": "roe",
        "eps": "eps",
    }

    def __init__(self, table_name: str, underlying_table: str):
        self._table_name = table_name
        self._underlying_table = underlying_table

    def __getattr__(self, name: str) -> Field:
        real_col = self._COLUMN_MAP.get(name, name)
        return Field(self._table_name, real_col, alias=name)


# Built-in JoinQuant query namespaces
valuation = TableNamespace("valuation", "daily_basic")
indicator = TableNamespace("indicator", "fina_indicator")
income = TableNamespace("income", "income")
balance = TableNamespace("balance", "balancesheet")
cash_flow = TableNamespace("cash_flow", "cashflow")


# --------------------------------------------------------------------------- #
# Fundamentals Query Builder
# --------------------------------------------------------------------------- #
class FundamentalsQuery:
    """Fluent query builder for financial fundamentals."""

    def __init__(self, *fields: Field | str):
        self.selected_fields: list[Field] = []
        for f in fields:
            if isinstance(f, Field):
                self.selected_fields.append(f)
            elif isinstance(f, str):
                parts = f.split(".")
                if len(parts) == 2:
                    self.selected_fields.append(Field(parts[0], parts[1]))
                else:
                    self.selected_fields.append(Field("valuation", parts[0]))
        self.conditions: list[FilterCondition | str] = []
        self.order_by_clause: str | None = None
        self.limit_count: int | None = None

    def filter(self, *conditions: FilterCondition | str) -> FundamentalsQuery:
        for c in conditions:
            self.conditions.append(c)
        return self

    def order_by(self, field_or_expr: Field | str, ascending: bool = True) -> FundamentalsQuery:
        direction = "ASC" if ascending else "DESC"
        f_name = str(field_or_expr)
        self.order_by_clause = f"{f_name} {direction}"
        return self

    def limit(self, count: int) -> FundamentalsQuery:
        self.limit_count = count
        return self


def query(*fields: Field | str) -> FundamentalsQuery:
    """Entrypoint mimicking JoinQuant `query(...)`."""
    return FundamentalsQuery(*fields)


# --------------------------------------------------------------------------- #
# Point-in-Time (PIT) Fundamentals Execution
# --------------------------------------------------------------------------- #
def get_fundamentals(
    query_obj: FundamentalsQuery | list[str] | str,
    date: str | None = None,
    statDate: str | None = None,  # noqa: N803
    db_path: str = _DB_DEFAULT,
) -> pd.DataFrame:
    """Point-in-Time fundamentals query with zero lookahead bias.

    Ensures financial statement data (income, balance, cashflow, indicator)
    is strictly joined where `ann_date <= date`, matching only what was
    publicly known at the market close on that trading day.
    """
    # 1. Resolve date
    if date is None:
        import quant_sdk.strategy as strat_mod
        if strat_mod._ACTIVE_ENGINE is not None:
            eng = strat_mod._ACTIVE_ENGINE
            date = getattr(eng, "as_of_date", None) or getattr(getattr(eng, "context", None), "current_dt", None)
        if not date:
            date = "20230630"  # default reference
    date_clean = str(date).replace("-", "")

    # Normalize query object
    if not isinstance(query_obj, FundamentalsQuery):
        if isinstance(query_obj, str):
            f_list = [f.strip() for f in query_obj.split(",")]
        else:
            f_list = list(query_obj)
        q = FundamentalsQuery(*f_list)
    else:
        q = query_obj

    if not q.selected_fields:
        # Default institutional selection
        q.selected_fields = [
            Field("valuation", "ts_code", "code"),
            Field("valuation", "pe_ttm", "pe_ratio"),
            Field("valuation", "pb", "pb_ratio"),
            Field("valuation", "circ_mv", "circulating_market_cap"),
            Field("indicator", "roe", "roe"),
            Field("indicator", "gross_margin", "gross_margin"),
        ]

    # Tables referenced
    tables_used = {f.table_name for f in q.selected_fields}
    for c in q.conditions:
        if isinstance(c, FilterCondition) and isinstance(c.field_obj, Field):
            tables_used.add(c.field_obj.table_name)
        elif isinstance(c, str):
            for t in ["valuation", "indicator", "income", "balance", "cash_flow"]:
                if f"{t}." in c:
                    tables_used.add(t)

    # 2. Build Point-in-Time SQL query
    from data_foundation import catalog
    conn = catalog.connect(db_path, read_only=True)
    try:
        select_cols = []
        has_code = False
        for f in q.selected_fields:
            if f.column_name == "ts_code" or f.alias == "code":
                has_code = True
            select_cols.append(f"{f.table_name}.{f.column_name} AS {f.alias}")
        if not has_code:
            select_cols.insert(0, "valuation.ts_code AS code")

        select_sql = ", ".join(select_cols)

        # Base is valuation (daily_basic on date)
        sql = f"""
        SELECT {select_sql}
        FROM daily_basic valuation
        """

        # Point-in-Time join for financial statements
        if "indicator" in tables_used:
            sql += f"""
            LEFT JOIN (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY ann_date DESC, end_date DESC) as _rn
                FROM fina_indicator
                WHERE ann_date <= '{date_clean}'
            ) indicator ON valuation.ts_code = indicator.ts_code AND indicator._rn = 1
            """

        if "income" in tables_used:
            sql += f"""
            LEFT JOIN (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY ann_date DESC, end_date DESC) as _rn
                FROM income
                WHERE ann_date <= '{date_clean}'
            ) income ON valuation.ts_code = income.ts_code AND income._rn = 1
            """

        if "balance" in tables_used:
            sql += f"""
            LEFT JOIN (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY ann_date DESC, end_date DESC) as _rn
                FROM balancesheet
                WHERE ann_date <= '{date_clean}'
            ) balance ON valuation.ts_code = balance.ts_code AND balance._rn = 1
            """

        if "cash_flow" in tables_used:
            sql += f"""
            LEFT JOIN (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY ann_date DESC, end_date DESC) as _rn
                FROM cashflow
                WHERE ann_date <= '{date_clean}'
            ) cash_flow ON valuation.ts_code = cash_flow.ts_code AND cash_flow._rn = 1
            """

        # Where clause
        where_clauses = [f"valuation.trade_date = '{date_clean}'"]
        for c in q.conditions:
            if isinstance(c, FilterCondition):
                where_clauses.append(c.to_sql())
            elif isinstance(c, str):
                where_clauses.append(c)

        sql += " WHERE " + " AND ".join(where_clauses)

        if q.order_by_clause:
            sql += f" ORDER BY {q.order_by_clause}"

        if q.limit_count:
            sql += f" LIMIT {q.limit_count}"

        df = conn.execute(sql).df()
        return df

    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Multi-Asset Access Helpers (Convertible Bonds & ETFs)
# --------------------------------------------------------------------------- #
def get_cb_daily(
    ts_code: str | list[str] | None = None,
    trade_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    with_double_low: bool = True,
    db_path: str = _DB_DEFAULT,
) -> pd.DataFrame:
    """Retrieve Convertible Bond (可转债) daily quotes and calculate Double-Low (双低指标).

    Double-Low Formula:
        Conversion Value = (stock_close / conv_price) * 100
        Premium Rate (%) = ((cb_close - conv_value) / conv_value) * 100
        Double Low Score = cb_close + Premium Rate
    """
    from data_foundation import catalog
    conn = catalog.connect(db_path, read_only=True)
    try:
        where_parts = []
        if ts_code:
            if isinstance(ts_code, str):
                where_parts.append(f"cb.ts_code = '{ts_code}'")
            else:
                codes_str = ", ".join(f"'{c}'" for c in ts_code)
                where_parts.append(f"cb.ts_code IN ({codes_str})")
        if trade_date:
            d = str(trade_date).replace("-", "")
            where_parts.append(f"cb.trade_date = '{d}'")
        if start_date:
            d = str(start_date).replace("-", "")
            where_parts.append(f"cb.trade_date >= '{d}'")
        if end_date:
            d = str(end_date).replace("-", "")
            where_parts.append(f"cb.trade_date <= '{d}'")

        where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

        if with_double_low:
            query = f"""
            SELECT
                cb.ts_code,
                cb.trade_date,
                cb.open,
                cb.high,
                cb.low,
                cb.close,
                cb.vol,
                cb.amount,
                b.stk_code,
                b.bond_short_name,
                b.conv_price,
                stk.close AS stk_close,
                ROUND((stk.close / NULLIF(b.conv_price, 0)) * 100, 3) AS conv_value,
                ROUND(((cb.close - (stk.close / NULLIF(b.conv_price, 0)) * 100) / NULLIF((stk.close / NULLIF(b.conv_price, 0)) * 100, 0)) * 100, 3) AS premium_rate,
                ROUND(cb.close + ((cb.close - (stk.close / NULLIF(b.conv_price, 0)) * 100) / NULLIF((stk.close / NULLIF(b.conv_price, 0)) * 100, 0)) * 100, 3) AS double_low
            FROM cb_daily cb
            LEFT JOIN cb_basic b ON cb.ts_code = b.ts_code
            LEFT JOIN daily stk ON b.stk_code = stk.ts_code AND cb.trade_date = stk.trade_date
            {where_sql}
            ORDER BY cb.trade_date ASC, cb.ts_code ASC
            """
        else:
            query = f"SELECT * FROM cb_daily cb {where_sql} ORDER BY trade_date ASC, ts_code ASC"

        return conn.execute(query).df()
    finally:
        conn.close()


def get_fund_daily(
    ts_code: str | list[str] | None = None,
    trade_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    db_path: str = _DB_DEFAULT,
) -> pd.DataFrame:
    """Retrieve ETF and mutual fund daily bars for fund rotation strategies."""
    from data_foundation import catalog
    conn = catalog.connect(db_path, read_only=True)
    try:
        where_parts = []
        if ts_code:
            if isinstance(ts_code, str):
                where_parts.append(f"f.ts_code = '{ts_code}'")
            else:
                codes_str = ", ".join(f"'{c}'" for c in ts_code)
                where_parts.append(f"f.ts_code IN ({codes_str})")
        if trade_date:
            d = str(trade_date).replace("-", "")
            where_parts.append(f"f.trade_date = '{d}'")
        if start_date:
            d = str(start_date).replace("-", "")
            where_parts.append(f"f.trade_date >= '{d}'")
        if end_date:
            d = str(end_date).replace("-", "")
            where_parts.append(f"f.trade_date <= '{d}'")

        where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
        query = f"SELECT * FROM fund_daily f {where_sql} ORDER BY trade_date ASC, ts_code ASC"
        return conn.execute(query).df()
    finally:
        conn.close()
