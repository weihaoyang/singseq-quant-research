"""Vectorized Qlib Alpha158 Representation for A-Share Quant Research.

Ported from BDC2026 research audit (audit_2026_alpha158_lightgbm.py).
Computes 105+ causal, point-in-time Alpha158 formulaic features:
- 9 K-bar candle geometry features (k_kmid, k_klen, k_kmid2, k_kup, k_kup2, k_klow, k_klow2, k_ksft, k_ksft2)
- Multi-window rolling features across windows (5, 10, 20, 60):
  * Price/Trend: roc, ma, std, max, min, qtlu, qtld, rsv, tsrank
  * Directional Count: cntp, cntn, cntd
  * Accumulated Magnitude: sump, sumn, sumd
  * Volume & Volatility: vma, vstd, vsump, vsumd
  * Volume-Price Correlation: corr, cord
  * Extrema Location: imax, imin, imxd
  * Regression Trend Quality: beta (slope), rsqr (R2), resi (normalized residual)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd


WINDOWS = (5, 10, 20, 60)


def _trend(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate slope, R2, and normalized residual for rolling window."""
    n = len(values)
    result = np.full((n, 3), np.nan, dtype=np.float32)
    if n < window:
        return result[:, 0], result[:, 1], result[:, 2]

    windows = np.lib.stride_tricks.sliding_window_view(values.astype(float), window)
    x = np.arange(window, dtype=float)
    xc = x - x.mean()
    denom = float(np.sum(xc * xc))
    if denom <= 0:
        return result[:, 0], result[:, 1], result[:, 2]

    mean = windows.mean(axis=1)
    slope = (windows @ xc) / denom
    fitted = mean + slope * (window - 1 - x.mean())
    centered = windows - mean[:, None]
    sst = np.sum(centered * centered, axis=1)
    ssr = slope * slope * denom
    r2 = np.divide(ssr, sst, out=np.zeros_like(ssr), where=sst > 1e-12)
    residual = windows[:, -1] - fitted
    std = np.std(windows, axis=1)

    result[window - 1 :, 0] = slope / np.maximum(np.abs(windows[:, -1]), 1e-8)
    result[window - 1 :, 1] = r2
    result[window - 1 :, 2] = residual / np.maximum(std, 1e-8)
    return result[:, 0], result[:, 1], result[:, 2]


def compute_alpha158(df: pd.DataFrame, windows: tuple[int, ...] = WINDOWS) -> tuple[pd.DataFrame, list[str]]:
    """Compute Alpha158 features for a panel DataFrame cleanly without fragmentation."""
    panel = df.copy()

    # Column name normalization
    col_map = {}
    if "ts_code" in panel.columns and "stock_id" not in panel.columns:
        col_map["ts_code"] = "stock_id"
    if "code" in panel.columns and "stock_id" not in panel.columns:
        col_map["code"] = "stock_id"
    if "date" in panel.columns and "trade_date" not in panel.columns:
        col_map["date"] = "trade_date"
    if "vol" in panel.columns and "volume" not in panel.columns:
        col_map["vol"] = "volume"
    if col_map:
        panel = panel.rename(columns=col_map)

    panel = panel.sort_values(["stock_id", "trade_date"]).reset_index(drop=True)

    # 1. K-bar Geometry features
    c = panel["close"]
    o = panel["open"]
    h = panel["high"]
    l = panel["low"]
    hl_diff = (h - l).replace(0.0, np.nan)
    max_oc = panel[["open", "close"]].max(axis=1)
    min_oc = panel[["open", "close"]].min(axis=1)

    kbar_dict = {
        "k_kmid": ((c - o) / o).astype(np.float32),
        "k_klen": ((h - l) / o).astype(np.float32),
        "k_kmid2": ((c - o) / hl_diff).astype(np.float32),
        "k_kup": ((h - max_oc) / o).astype(np.float32),
        "k_kup2": ((h - max_oc) / hl_diff).astype(np.float32),
        "k_klow": ((min_oc - l) / o).astype(np.float32),
        "k_klow2": ((min_oc - l) / hl_diff).astype(np.float32),
        "k_ksft": ((2.0 * c - h - l) / o).astype(np.float32),
        "k_ksft2": ((2.0 * c - h - l) / hl_diff).astype(np.float32),
    }
    kbar_df = pd.DataFrame(kbar_dict, index=panel.index)

    # 2. Per-stock rolling windows buffered in dict
    def _per_stock(group: pd.DataFrame) -> pd.DataFrame:
        d: dict[str, np.ndarray | pd.Series] = {}
        close = group["close"].astype(float)
        high = group["high"].astype(float)
        low = group["low"].astype(float)
        ret = close.pct_change()
        volume = group["volume"].fillna(0.0).astype(float)
        log_volume = np.log1p(volume)
        d_log_vol = log_volume.diff()
        d_ret = ret.diff()

        abs_ret = ret.abs()
        positive = ret.clip(lower=0.0)
        negative = ret.clip(upper=0.0)

        for w in windows:
            roll_close = close.rolling(w, min_periods=w)
            roll_high = high.rolling(w, min_periods=w)
            roll_low = low.rolling(w, min_periods=w)
            roll_ret = ret.rolling(w, min_periods=w)
            roll_volume = volume.rolling(w, min_periods=w)

            # Price / Momentum
            d[f"roc_{w}"] = close.pct_change(w).astype(np.float32)
            d[f"ma_{w}"] = (close / roll_close.mean().replace(0.0, np.nan) - 1.0).astype(np.float32)
            d[f"std_{w}"] = roll_ret.std().astype(np.float32)
            d[f"max_{w}"] = (roll_high.max() / close.replace(0.0, np.nan) - 1.0).astype(np.float32)
            d[f"min_{w}"] = (roll_low.min() / close.replace(0.0, np.nan) - 1.0).astype(np.float32)
            d[f"qtlu_{w}"] = (close / roll_close.quantile(0.8).replace(0.0, np.nan) - 1.0).astype(np.float32)
            d[f"qtld_{w}"] = (close / roll_close.quantile(0.2).replace(0.0, np.nan) - 1.0).astype(np.float32)
            spread = (roll_high.max() - roll_low.min()).replace(0.0, np.nan)
            d[f"rsv_{w}"] = ((close - roll_low.min()) / spread).astype(np.float32)
            d[f"tsrank_{w}"] = roll_close.rank(pct=True).astype(np.float32)

            # Directional count
            d[f"cntp_{w}"] = (ret > 0.0).rolling(w, min_periods=w).mean().astype(np.float32)
            d[f"cntn_{w}"] = (ret < 0.0).rolling(w, min_periods=w).mean().astype(np.float32)
            d[f"cntd_{w}"] = (d[f"cntp_{w}"] - d[f"cntn_{w}"]).astype(np.float32)

            # Accumulated magnitude
            pos_sum = positive.rolling(w, min_periods=w).sum()
            neg_sum = negative.abs().rolling(w, min_periods=w).sum()
            tot_sum = (pos_sum + neg_sum).replace(0.0, np.nan)
            d[f"sump_{w}"] = (pos_sum / tot_sum).astype(np.float32)
            d[f"sumn_{w}"] = (neg_sum / tot_sum).astype(np.float32)
            d[f"sumd_{w}"] = (d[f"sump_{w}"] - d[f"sumn_{w}"]).astype(np.float32)

            # Volume dynamics
            v_mean = roll_volume.mean().replace(0.0, np.nan)
            d[f"vma_{w}"] = (volume / v_mean - 1.0).astype(np.float32)
            d[f"vstd_{w}"] = (roll_volume.std() / v_mean).astype(np.float32)

            vol_abs_ret_sum = (volume * abs_ret).rolling(w, min_periods=w).sum().replace(0.0, np.nan)
            d[f"vsump_{w}"] = ((volume * positive).rolling(w, min_periods=w).sum() / vol_abs_ret_sum).astype(np.float32)
            d[f"vsumd_{w}"] = ((volume * ret).rolling(w, min_periods=w).sum() / vol_abs_ret_sum).astype(np.float32)

            # Volume-price correlation
            d[f"corr_{w}"] = ret.rolling(w, min_periods=w).corr(d_log_vol).astype(np.float32)
            d[f"cord_{w}"] = d_ret.rolling(w, min_periods=w).corr(d_log_vol).astype(np.float32)

            # Extrema indicators
            d[f"imax_{w}"] = (close == roll_close.max()).astype(np.float32)
            d[f"imin_{w}"] = (close == roll_close.min()).astype(np.float32)
            d[f"imxd_{w}"] = (d[f"imax_{w}"] - d[f"imin_{w}"]).astype(np.float32)

            # Trend regression
            slope, rsq, resi = _trend(np.log(np.maximum(close.to_numpy(float), 1e-8)), w)
            d[f"beta_{w}"] = slope
            d[f"rsqr_{w}"] = rsq
            d[f"resi_{w}"] = resi

        return pd.DataFrame(d, index=group.index)

    chunks = []
    for _, group in panel.groupby("stock_id", sort=False):
        chunks.append(_per_stock(group))

    derived = pd.concat(chunks).sort_index()

    # Cleanly concatenate all at once
    panel = pd.concat([panel, kbar_df, derived], axis=1)
    feature_names = list(kbar_df.columns) + list(derived.columns)
    feature_names = list(dict.fromkeys(feature_names))

    return panel.replace([np.inf, -np.inf], np.nan), feature_names
