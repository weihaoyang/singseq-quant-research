"""Dated labels, purged splits and cross-sectional metrics shared by experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_open_labels(frame: pd.DataFrame, calendar, horizon: int = 5) -> pd.DataFrame:
    """Close-t signal; next open entry; open t+1+h exit. Never skip a gap.

    Prices use same-day adjustment factors. This is a prediction target, not
    a promise of fills; the portfolio replay applies execution constraints.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least one (A-share T+1)")
    days = pd.Index(sorted(set(map(str, calendar))))
    df = frame.sort_values(["ts_code", "trade_date"]).copy()
    if df.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("duplicate panel keys")
    positions = days.get_indexer(df.trade_date.astype(str))
    if (positions < 0).any():
        raise ValueError("panel date absent from trading calendar")
    df["_session"] = positions
    g = df.groupby("ts_code", sort=False)
    adj_open = df.open * df.adj_factor
    by_code = adj_open.groupby(df.ts_code, sort=False)
    entry = by_code.shift(-1)
    exit_price = by_code.shift(-(horizon + 1))
    contiguous = g["_session"].shift(-(horizon + 1)).eq(df._session + horizon + 1)
    valid = (contiguous & np.isfinite(entry) & np.isfinite(exit_price)
             & entry.gt(0) & exit_price.gt(0)
             & g.vol.shift(-1).gt(0) & g.vol.shift(-(horizon + 1)).gt(0))
    df["target_ret"] = (exit_price / entry - 1).where(valid)
    for name, offset in [("entry_date", 1), ("label_end_date", horizon + 1)]:
        df[name] = [days[p + offset] if p + offset < len(days) else None for p in positions]
    df["label_horizon"] = horizon
    return df.drop(columns="_session")


def purged_split(frame, train_end: str, valid_start: str, valid_end: str):
    if not train_end < valid_start <= valid_end:
        raise ValueError("split dates must be ordered and disjoint")
    mature = frame.label_end_date.notna()
    train = frame.trade_date.le(train_end) & mature & frame.label_end_date.lt(valid_start)
    valid = frame.trade_date.between(valid_start, valid_end) & mature & frame.label_end_date.le(valid_end)
    return train, valid


def daily_rank_ic(frame, score="score", target="target_ret", min_stocks=50):
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("duplicate prediction keys")
    rows = []
    for date, group in frame.groupby("trade_date", sort=True):
        good = group[[score, target]].replace([np.inf, -np.inf], np.nan).dropna()
        ic = np.nan
        if len(good) >= min_stocks and good[score].nunique() > 1 and good[target].nunique() > 1:
            ic = good[score].corr(good[target], method="spearman")
        rows.append({"trade_date": date, "n": len(good), "rank_ic": ic})
    return pd.DataFrame(rows, columns=["trade_date", "n", "rank_ic"])


def summarize_ic(daily, block_size=10, seed=42):
    """Moving-block bootstrap; descriptive CI, not multiple-testing correction."""
    values = daily.rank_ic.dropna().to_numpy()
    if not len(values):
        return {"days": 0, "mean_rank_ic": None, "icir_daily": None, "positive_share": None, "mean_ci95": None}
    sd = values.std(ddof=1) if len(values) > 1 else 0
    ci = None
    if len(values) >= 2 * block_size:
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, len(values) - block_size + 1, size=(500, int(np.ceil(len(values) / block_size))))
        draws = values[(starts[..., None] + np.arange(block_size)).reshape(500, -1)[:, :len(values)]]
        ci = np.quantile(draws.mean(axis=1), [.025, .975]).tolist()
    return {"days": len(values), "mean_rank_ic": float(values.mean()),
            "icir_daily": float(values.mean() / sd) if sd > 0 else None,
            "positive_share": float((values > 0).mean()), "mean_ci95": ci,
            "bootstrap_block_sessions": block_size}
