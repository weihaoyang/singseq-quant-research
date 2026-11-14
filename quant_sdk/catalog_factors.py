"""Run named factors that have an implementation in the shipped factor catalog.

Named factors execute trusted toolkit code against a researcher-owned panel. Their
results remain local and unverified; catalog metadata without an implementation is
rejected instead of being converted into a guessed formula.
"""
from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd


_LABEL_COLUMNS = ("benchmark_ret", "target_ret", "fwd_ret", "forward_return")
_BAR_COLUMNS = ("open", "high", "low", "close")


def _prepared_panel(frame: pd.DataFrame) -> pd.DataFrame:
    missing = {"ts_code", "trade_date", *_BAR_COLUMNS}.difference(frame.columns)
    if missing:
        raise ValueError(f"面板缺少具名因子所需列: {', '.join(sorted(missing))}")
    panel = frame.drop(columns=[name for name in _LABEL_COLUMNS if name in frame.columns],
                       errors="ignore").copy()
    panel["ts_code"] = panel["ts_code"].astype(str)
    panel["trade_date"] = panel["trade_date"].astype(str)
    panel["vol"] = panel["vol"] if "vol" in panel else np.nan
    panel["amount"] = panel["amount"] if "amount" in panel else np.nan
    if "adj_close" in panel:
        close = pd.to_numeric(panel["close"], errors="coerce").replace(0.0, np.nan)
        adjustment = pd.to_numeric(panel["adj_close"], errors="coerce") / close
        for column in ("open", "high", "low"):
            panel[column] = pd.to_numeric(panel[column], errors="coerce") * adjustment
        panel["close"] = pd.to_numeric(panel["adj_close"], errors="coerce")
    panel = panel.sort_values(["ts_code", "trade_date"], kind="mergesort")
    panel["stock_id"] = panel["ts_code"]
    panel["volume"] = panel["vol"]
    return panel.reset_index(drop=True)


@lru_cache(maxsize=1)
def _registry_formulas() -> dict[str, Any]:
    from factors_catalog.factor_registry import FACTOR_DEFINITIONS
    from factors_catalog import r441_daily_formulas

    return {
        name: getattr(r441_daily_formulas, name)
        for name in FACTOR_DEFINITIONS
        if callable(getattr(r441_daily_formulas, name, None))
    }


def _compute_registry_factor(name: str, panel: pd.DataFrame) -> pd.Series:
    function = _registry_formulas()[name]
    price_columns = {
        "prices": "close", "highs": "high", "lows": "low",
        "closes": "close", "opens": "open", "opens_1d": "open",
        "closes_1d": "close", "turnover": "amount",
    }
    parameters = list(inspect.signature(function).parameters)
    missing = [price_columns[param] for param in parameters
               if param not in price_columns or price_columns[param] not in panel]
    if missing:
        raise ValueError(f"因子 {name} 缺少输入字段: {', '.join(sorted(set(missing)))}")

    output = np.full(len(panel), np.nan, dtype=float)
    groups = panel.groupby("ts_code", sort=False).indices
    for positions in groups.values():
        positions = np.asarray(positions, dtype=np.int64)
        values = {param: pd.to_numeric(panel.iloc[positions][column], errors="coerce").to_numpy(dtype=float)
                  for param, column in price_columns.items() if param in parameters}
        for offset, position in enumerate(positions):
            end = offset + 1
            # The frozen r441 daily formulas consume at most 40 returns
            # (41 prices). Avoid copying an ever-growing history for every bar.
            begin = max(0, end - 41)
            args = {param: series[begin:end].tolist() for param, series in values.items()}
            try:
                result = float(function(**args))
            except (ArithmeticError, TypeError, ValueError, ZeroDivisionError):
                continue
            if np.isfinite(result):
                output[position] = result
    if not np.isfinite(output).any():
        raise ValueError(f"因子 {name} 在当前面板没有有效输出")
    return pd.Series(output, index=panel.index, name=name)


def _family_factor(name: str, panel: pd.DataFrame) -> pd.Series | None:
    from factors_catalog.alpha158 import compute_alpha158
    from factors_catalog.alpha101_operators import compute_alpha101_selected
    from factors_catalog.factor_zoo import compute_factor_zoo
    from factors_catalog.ohlc_volatility import compute_ohlc_volatility

    for compute in (compute_alpha158, compute_alpha101_selected,
                    compute_factor_zoo, compute_ohlc_volatility):
        frame, names = compute(panel)
        if name in names:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).any():
                raise ValueError(f"因子 {name} 在当前面板没有有效输出")
            return pd.Series(values, index=panel.index, name=name)
    return None


def compute_named_factor(name: str, frame: pd.DataFrame) -> pd.Series:
    """Compute one implemented catalog factor, aligned to sorted panel rows."""
    normalized = str(name or "").strip()
    if not normalized:
        raise ValueError("需要提供因子名")
    panel = _prepared_panel(frame)
    formulas = _registry_formulas()
    if normalized in formulas:
        return _compute_registry_factor(normalized, panel)
    values = _family_factor(normalized, panel)
    if values is None:
        raise ValueError(
            f"因子 {normalized} 尚无可运行实现；目录描述不是计算公式，请选择已有实现或提交源码。")
    return values


def runnable_factor_names(frame: pd.DataFrame) -> list[str]:
    """Return implementation-backed names for this panel's available columns."""
    panel = _prepared_panel(frame)
    names = set(_registry_formulas())
    from factors_catalog.alpha158 import compute_alpha158
    from factors_catalog.alpha101_operators import compute_alpha101_selected
    from factors_catalog.factor_zoo import compute_factor_zoo
    from factors_catalog.ohlc_volatility import compute_ohlc_volatility

    for compute in (compute_alpha158, compute_alpha101_selected,
                    compute_factor_zoo, compute_ohlc_volatility):
        _, produced = compute(panel)
        names.update(produced)
    return sorted(names)
