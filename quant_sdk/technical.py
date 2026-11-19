"""Unified Technical Analysis Engine combining C-accelerated TA-Lib and TongDaXin MyTT.

Features:
- TA-Lib C-speed indicators (MACD, RSI, Bollinger Bands, ATR, Stochastics, ADX, OBV, KAMA)
- MyTT TongDaXin (通达信) / Flush (同花顺) formula primitives (CROSS, REF, SMA, EMA, KDJ, BOLL, BIAS, etc.)
- One-line DataFrame technical feature enricher: `add_technical_indicators(df)`
"""

from __future__ import annotations

import logging

import MyTT
import numpy as np
import pandas as pd
import talib

logger = logging.getLogger(__name__)

# Re-export classic MyTT primitives for TongDaXin algorithmic strategy parity
RD = MyTT.RD
RET = MyTT.RET
ABS = MyTT.ABS
MAX = MyTT.MAX
MIN = MyTT.MIN
MA = MyTT.MA
EMA = MyTT.EMA
SMA = MyTT.SMA
DMA = MyTT.DMA
WMA = MyTT.WMA
REF = MyTT.REF
DIFF = MyTT.DIFF
HHV = MyTT.HHV
LLV = MyTT.LLV
SUM = MyTT.SUM
COUNT = MyTT.COUNT
EVERY = MyTT.EVERY
EXIST = MyTT.EXIST
LAST = MyTT.LAST
CROSS = MyTT.CROSS
LONGCROSS = MyTT.LONGCROSS
VALUEWHEN = MyTT.VALUEWHEN
BARSLAST = MyTT.BARSLAST
BARSLASTCOUNT = MyTT.BARSLASTCOUNT
BARSSINCEN = MyTT.BARSSINCEN
CONST = MyTT.CONST
HHVBARS = MyTT.HHVBARS
LLVBARS = MyTT.LLVBARS


def macd(
    close: pd.Series | np.ndarray,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate MACD using TA-Lib C-engine.

    Returns:
        (macd_line, signal_line, macd_hist)
    """
    c = np.asarray(close, dtype=np.float64)
    return talib.MACD(c, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)


def rsi(close: pd.Series | np.ndarray, timeperiod: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index (RSI) using TA-Lib."""
    c = np.asarray(close, dtype=np.float64)
    return talib.RSI(c, timeperiod=timeperiod)


def bollinger_bands(
    close: pd.Series | np.ndarray,
    timeperiod: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Bollinger Bands (upper, middle, lower)."""
    c = np.asarray(close, dtype=np.float64)
    return talib.BBANDS(c, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)


def atr(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    timeperiod: int = 14,
) -> np.ndarray:
    """Calculate Average True Range (ATR)."""
    h = np.asarray(high, dtype=np.float64)
    l_ = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    return talib.ATR(h, l_, c, timeperiod=timeperiod)


def kdj(
    close: pd.Series | np.ndarray,
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Chinese standard KDJ indicator (TongDaXin standard).

    Returns:
        (k, d, j)
    """
    c = np.asarray(close, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l_ = np.asarray(low, dtype=np.float64)
    return MyTT.KDJ(c, h, l_, N=n, M1=m1, M2=m2)


def add_technical_indicators(
    df: pd.DataFrame,
    include_kdj: bool = True,
    include_macd: bool = True,
    include_boll: bool = True,
    include_rsi: bool = True,
    include_atr: bool = True,
) -> pd.DataFrame:
    """Enrich an OHLCV DataFrame with standard technical indicators in-place or copy.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', and optional 'volume'.

    Returns:
        New DataFrame with enriched technical columns.
    """
    res = df.copy()
    close = res["close"].to_numpy(dtype=np.float64)
    high = res["high"].to_numpy(dtype=np.float64)
    low = res["low"].to_numpy(dtype=np.float64)

    if include_macd:
        macd_line, signal_line, hist = macd(close)
        res["macd"] = macd_line
        res["macd_signal"] = signal_line
        res["macd_hist"] = hist * 2.0  # Chinese A-share standard (2 * (DIFF - DEA))

    if include_rsi:
        res["rsi_6"] = rsi(close, timeperiod=6)
        res["rsi_12"] = rsi(close, timeperiod=12)
        res["rsi_24"] = rsi(close, timeperiod=24)

    if include_boll:
        up, mid, low_b = bollinger_bands(close, timeperiod=20)
        res["boll_upper"] = up
        res["boll_mid"] = mid
        res["boll_lower"] = low_b

    if include_atr:
        res["atr_14"] = atr(high, low, close, timeperiod=14)

    if include_kdj:
        k_val, d_val, j_val = kdj(close, high, low)
        res["kdj_k"] = k_val
        res["kdj_d"] = d_val
        res["kdj_j"] = j_val

    return res
