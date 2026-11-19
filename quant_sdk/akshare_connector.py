"""Institutional Open-Source Data Connector powered by AkShare.

Provides high-frequency, macro, alternative, and market data interfaces:
- Realtime market quotes (EastMoney / Sina)
- A-share historical daily/minute OHLCV (qfq/hfq adjusted)
- Northbound capital flow (北向资金 - 沪股通/深股通/北向合计)
- Institutional Dragon & Tiger List (机构龙虎榜交易明细)
- Margin Financing & Securities Lending (融资融券)
- China Treasury Yield Curve (中债国债收益率曲线)
- Macroeconomic indicators (CPI, PPI, PMI, M2, LPR)
"""

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_symbol(symbol: str) -> str:
    """Extract pure 6-digit stock code, e.g. '600519.SH' -> '600519'."""
    clean = re.sub(r"[^0-9]", "", str(symbol))
    return clean[-6:] if len(clean) >= 6 else clean


def get_realtime_quotes(symbols: list[str] | None = None) -> pd.DataFrame:
    """Fetch realtime market quote snapshot from AkShare.

    Args:
        symbols: Optional list of stock codes (e.g. ['600519.SH', '000858.SZ']).
                 If None, returns all A-share quotes.

    Returns:
        DataFrame with standardized columns:
        ['symbol', 'name', 'price', 'change_pct', 'open', 'high', 'low', 'volume', 'amount', 'pe', 'pb', 'total_mv']
    """
    import akshare as ak

    try:
        df = ak.stock_zh_a_spot_em()
    except Exception as exc:
        logger.error("Failed to fetch spot quotes from AkShare: %s", exc)
        return pd.DataFrame()

    col_map = {
        "代码": "symbol",
        "名称": "name",
        "最新价": "price",
        "涨跌幅": "change_pct",
        "今开": "open",
        "最高": "high",
        "最低": "low",
        "昨收": "pre_close",
        "成交量": "volume",
        "成交额": "amount",
        "市盈率-动态": "pe",
        "市净率": "pb",
        "总市值": "total_mv",
    }
    available_cols = {k: v for k, v in col_map.items() if k in df.columns}
    out = df[list(available_cols.keys())].rename(columns=available_cols).copy()

    # Format symbol with standard suffix
    def _add_suffix(code: str) -> str:
        s = str(code).zfill(6)
        if s.startswith(("60", "68", "90")):
            return f"{s}.SH"
        elif s.startswith(("00", "30", "20")):
            return f"{s}.SZ"
        elif s.startswith(("8", "4")):
            return f"{s}.BJ"
        return s

    out["symbol"] = out["symbol"].apply(_add_suffix)

    if symbols:
        target_set = {s.upper() for s in symbols}
        # Also match bare 6-digit codes
        bare_set = {_normalize_symbol(s) for s in symbols}
        out = out[out["symbol"].isin(target_set) | out["symbol"].apply(_normalize_symbol).isin(bare_set)].copy()

    return out.reset_index(drop=True)


def get_stock_hist(
    symbol: str,
    period: str = "daily",
    start_date: str = "20200101",
    end_date: str = "20261231",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """Fetch historical OHLCV bar data with dividend reinvestment adjustment.

    Args:
        symbol: Stock symbol (e.g. '600519' or '600519.SH').
        period: 'daily', 'weekly', 'monthly', or '1', '5', '15', '30', '60' for minutes.
        start_date: 'YYYYMMDD' format.
        end_date: 'YYYYMMDD' format.
        adjust: 'qfq' (forward adjusted), 'hfq' (backward adjusted), or '' (raw).

    Returns:
        DataFrame indexed by pd.DatetimeIndex with columns:
        ['open', 'close', 'high', 'low', 'volume', 'amount', 'turnover']
    """
    import akshare as ak

    clean_code = _normalize_symbol(symbol)
    start_fmt = start_date.replace("-", "")
    end_fmt = end_date.replace("-", "")

    try:
        if period in {"1", "5", "15", "30", "60", "1m", "5m", "15m", "30m", "60m"}:
            period_num = period.replace("m", "")
            df = ak.stock_zh_a_hist_min_em(
                symbol=clean_code,
                period=period_num,
                start_date=start_fmt,
                end_date=end_fmt,
                adjust=adjust,
            )
        else:
            df = ak.stock_zh_a_hist_em(
                symbol=clean_code,
                period=period,
                start_date=start_fmt,
                end_date=end_fmt,
                adjust=adjust,
            )
    except Exception as exc:
        logger.error("Failed to fetch historical bars for %s: %s", symbol, exc)
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    col_map = {
        "日期": "datetime",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover",
    }
    cols = [c for c in col_map if c in df.columns]
    res = df[cols].rename(columns=col_map).copy()
    res["datetime"] = pd.to_datetime(res["datetime"])
    res = res.set_index("datetime").sort_index()
    return res


def get_northbound_money_flow() -> pd.DataFrame:
    """Fetch historical Northbound smart money flow (北向资金 - 沪深股通流入明细).

    Returns:
        DataFrame indexed by date containing:
        - sh_hk_net_inflow (沪股通净流入)
        - sz_hk_net_inflow (深股通净流入)
        - total_net_inflow (北向资金合计净流入)
    """
    import akshare as ak

    try:
        df = ak.stock_hsgt_hist_em()
    except Exception as exc:
        logger.error("Failed to fetch northbound data: %s", exc)
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Normalize column names
    col_map = {
        "日期": "date",
        "沪股通净流入": "sh_hk_net_inflow",
        "深股通净流入": "sz_hk_net_inflow",
        "北向资金净流入": "total_net_inflow",
        "北向资金": "total_net_inflow",
    }
    present = {c: col_map[c] for c in df.columns if c in col_map}
    out = df[list(present.keys())].rename(columns=present).copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out = out.set_index("date").sort_index()
    return out


def get_dragon_tiger_inst_list(date: str) -> pd.DataFrame:
    """Fetch Institutional Dragon & Tiger List (机构专用席位买卖明细).

    Args:
        date: Format 'YYYYMMDD' (e.g. '20240415').
    """
    import akshare as ak

    d_clean = date.replace("-", "")
    try:
        df = ak.stock_lhb_jgzz_em(date=d_clean)
        return df if not df.empty else pd.DataFrame()
    except Exception as exc:
        logger.warning("No Dragon-Tiger institutional data for %s: %s", date, exc)
        return pd.DataFrame()


def get_treasury_yield_curve(start_date: str = "20230101", end_date: str = "20261231") -> pd.DataFrame:
    """Fetch China Interbank Treasury Bond Yield Curve (中债国债收益率曲线).

    Essential for:
    - Dynamic risk-free rate calculation
    - Term structure / Macro yield curve slope (10Y - 2Y / 10Y - 1Y)
    """
    import akshare as ak

    try:
        df = ak.bond_china_yield(start_date=start_date.replace("-", ""), end_date=end_date.replace("-", ""))
        return df if not df.empty else pd.DataFrame()
    except Exception as exc:
        logger.warning("Could not fetch China bond yield curve: %s", exc)
        return pd.DataFrame()


def get_macro_indicator(indicator: str = "cpi") -> pd.DataFrame:
    """Fetch macroeconomic indicators.

    Supported indicators:
    - 'cpi': China Consumer Price Index (居民消费价格指数)
    - 'ppi': China Producer Price Index (工业品出厂价格指数)
    - 'pmi': China Purchasing Managers Index (制造业采购经理人指数)
    - 'm2': China Money Supply M2 / M1 (广义货币供应量)
    - 'lpr': China Loan Prime Rate (贷款市场报价利率)
    """
    import akshare as ak

    ind = indicator.lower().strip()
    try:
        if ind == "cpi":
            return ak.macro_china_cpi()
        elif ind == "ppi":
            return ak.macro_china_ppi()
        elif ind == "pmi":
            return ak.macro_china_pmi()
        elif ind == "m2":
            return ak.macro_china_money_supply()
        elif ind == "lpr":
            return ak.macro_china_lpr()
        else:
            raise ValueError(f"Unsupported macro indicator: {indicator}")
    except Exception as exc:
        logger.warning("Failed to fetch macro indicator %s: %s", indicator, exc)
        return pd.DataFrame()
