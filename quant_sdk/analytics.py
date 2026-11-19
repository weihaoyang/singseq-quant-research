"""
quant_sdk/analytics.py
--------------------------------------------------
高级量化分析与股票专业终端助手 (Professional Quant Analytics & Stock Terminal Bridge)

功能特性：
1. 一键拉取完整复权行情与技术指标 DataFrame (get_kline)
   - 前复权 (QFQ) / 后复权 (HFQ) / 原始不复权 (RAW)
   - 经典均线 (MA5/10/20/60)、成交量均线 (VOL_MA5/10)
   - 经典 MACD (12, 26, 9)、布林线 (BOLL 20, 2)
   - 估值与市值衍生 (PE_TTM, PB, 换手率, 流通市值)
2. 全市场截面选股雷达 (get_screener)
3. 唤醒专业级交互式股票图形软件 (browse)
"""

from __future__ import annotations

import webbrowser

import numpy as np
import pandas as pd

from quant_sdk.provider import pro_api


def get_kline(
    ts_code: str = "000001.SZ",
    start_date: str | None = None,
    end_date: str | None = None,
    adjust: str = "qfq",  # "qfq" | "hfq" | "none"
    limit: int = 1000,
) -> pd.DataFrame:
    """
    拉取专业级量化 K 线与指标面板
    :param ts_code: 股票代码，如 "000001.SZ", "600519.SH"
    :param start_date: 开始日期，如 "20230101"
    :param end_date: 结束日期，如 "20231231"
    :param adjust: 复权模式，默认 'qfq' 前复权
    :return: 包含 OHLCV、复权价、均线、MACD、BOLL及估值的 DataFrame
    """
    pro = pro_api()
    # The generic query path returns rows in **ascending** date order and pages from
    # the earliest session, so `LIMIT` alone yields the OLDEST window (a request with
    # no dates used to come back as ~2008).  Bound the window by the requested number
    # of sessions instead, so `.tail(n)` really is the most recent `n` sessions.
    sessions = int(limit) if limit else 1000
    params = {"ts_code": ts_code, "limit": sessions}
    if not start_date:
        from datetime import date as _date
        from datetime import timedelta as _timedelta
        # ~1.3 calendar days per trading day, so the window holds <= `sessions` rows
        # and a single ascending page still reaches the latest session.
        span_days = int(sessions * 1.30) + 30
        params["start_date"] = (_date.today() - _timedelta(days=span_days)).strftime("%Y%m%d")
    else:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    # 1. 行情与复权数据
    df = pro.daily(**params)
    if df.empty:
        return df

    df = df.sort_values("trade_date").reset_index(drop=True)

    # 2. 补齐复权因子。请求复权数据时不得把失败静默降级成原始价格。
    if adjust in {"qfq", "hfq"}:
        try:
            adj_df = pro.adj_factor(**params)
        except Exception as exc:
            raise RuntimeError(
                "复权因子查询失败，拒绝把未复权价格标记为复权结果"
            ) from exc
        if adj_df.empty or not {"trade_date", "adj_factor"}.issubset(adj_df.columns):
            raise RuntimeError("复权因子覆盖为空，拒绝生成复权行情")
        df = pd.merge(df, adj_df[["trade_date", "adj_factor"]],
                      on="trade_date", how="left", validate="one_to_one")
        factors = pd.to_numeric(df["adj_factor"], errors="coerce")
        if not np.isfinite(factors).all() or (factors <= 0).any():
            raise RuntimeError("复权因子覆盖不完整，拒绝生成复权行情")
        df["adj_factor"] = factors
    else:
        df["adj_factor"] = 1.0

    # 计算前复权/后复权
    if adjust == "qfq":
        latest_adj = df["adj_factor"].iloc[-1] if not df.empty else 1.0
        ratio = df["adj_factor"] / (latest_adj if latest_adj > 0 else 1.0)
        df["open_qfq"] = np.round(df["open"] * ratio, 2)
        df["high_qfq"] = np.round(df["high"] * ratio, 2)
        df["low_qfq"] = np.round(df["low"] * ratio, 2)
        df["close_qfq"] = np.round(df["close"] * ratio, 2)
        base_close = df["close_qfq"]
    elif adjust == "hfq":
        ratio = df["adj_factor"]
        df["open_hfq"] = np.round(df["open"] * ratio, 2)
        df["high_hfq"] = np.round(df["high"] * ratio, 2)
        df["low_hfq"] = np.round(df["low"] * ratio, 2)
        df["close_hfq"] = np.round(df["close"] * ratio, 2)
        base_close = df["close_hfq"]
    else:
        base_close = df["close"]

    # 3. 计算经典技术指标
    # 均线
    df["ma5"] = np.round(base_close.rolling(5).mean(), 2)
    df["ma10"] = np.round(base_close.rolling(10).mean(), 2)
    df["ma20"] = np.round(base_close.rolling(20).mean(), 2)
    df["ma60"] = np.round(base_close.rolling(60).mean(), 2)

    # 成交量均线
    df["vol_ma5"] = np.round(df["vol"].rolling(5).mean(), 1)
    df["vol_ma10"] = np.round(df["vol"].rolling(10).mean(), 1)

    # MACD (12, 26, 9)
    ema12 = base_close.ewm(span=12, adjust=False).mean()
    ema26 = base_close.ewm(span=26, adjust=False).mean()
    df["dif"] = np.round(ema12 - ema26, 3)
    df["dea"] = np.round(df["dif"].ewm(span=9, adjust=False).mean(), 3)
    df["macd"] = np.round((df["dif"] - df["dea"]) * 2, 3)

    # 布林带 BOLL (20, 2)
    boll_mid = base_close.rolling(20).mean()
    boll_std = base_close.rolling(20).std()
    df["boll_mid"] = np.round(boll_mid, 2)
    df["boll_upper"] = np.round(boll_mid + 2.0 * boll_std, 2)
    df["boll_lower"] = np.round(boll_mid - 2.0 * boll_std, 2)

    # 4. 尝试关联 daily_basic 估值
    try:
        basic_df = pro.daily_basic(**params)
        if not basic_df.empty:
            cols = [c for c in ["trade_date", "pe_ttm", "pb", "turnover_rate", "total_mv", "circ_mv"] if c in basic_df.columns]
            df = pd.merge(df, basic_df[cols], on="trade_date", how="left")
    except Exception:
        pass

    return df


def get_screener(
    trade_date: str | None = None,
    limit: int = 100
) -> pd.DataFrame:
    """
    全市场横截面选股与行情透视

    未指定 ``trade_date`` 时，窗口被限制在最近一段时间内（通用查询路径按日期
    **升序**从最早一段分页，不设窗口会返回 2000 年附近的截面）；此时返回的是
    该窗口内**最早一个交易日**的截面，而不是"今天"。
    """
    pro = pro_api()
    params = {"limit": limit}
    if trade_date:
        params["trade_date"] = trade_date
    else:
        from datetime import date as _date
        from datetime import timedelta as _timedelta
        params["start_date"] = (_date.today() - _timedelta(days=45)).strftime("%Y%m%d")

    # 获取全市场截面
    daily_df = pro.daily(**params)
    if daily_df.empty:
        return daily_df

    # 关联股票基本信息
    try:
        stock_basic = pro.stock_basic(limit=6000)
        if not stock_basic.empty:
            daily_df = pd.merge(daily_df, stock_basic[["ts_code", "name", "industry"]], on="ts_code", how="left")
    except Exception:
        pass

    return daily_df


def browse(ts_code: str = "000001.SZ", host: str = "127.0.0.1", port: int = 8000) -> None:
    """
    在默认浏览器中一键唤起专业级可视化股票工作站 (Interactive Stock Pro Terminal)
    :param ts_code: 目标股票代码 (如 000001.SZ, 600519.SH)
    :param host: 服务主机地址
    :param port: 服务端口
    """
    url = f"http://{host}:{port}/portal?tab=terminal&code={ts_code}"
    print(f"\n[Quant SDK] 正在打开专业股票全息可视化终端 -> {url}")
    webbrowser.open(url)
