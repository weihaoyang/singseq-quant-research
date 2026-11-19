"""Ready-to-run Strategy Templates (Native Modern OOP Architecture).

Provides reference strategies implemented using `quant_sdk.Strategy`:
1. Dual Moving Average Trend Following (双均线择时策略)
2. Small-Cap Value & High-ROE Fundamentals (小市值低估值高ROE基本面轮动)
3. Convertible Bond Double-Low Rotation (可转债双低轮动策略)
4. Multi-Asset ETF Momentum Rotation (大类资产ETF动量轮动策略)
5. miniQMT / XtQuant Live Broker Execution (极简实盘对接与风控模板)
"""
from __future__ import annotations

from typing import Any

__all__ = ["STRATEGY_TEMPLATES", "get_strategy_template", "list_strategy_templates"]

STRATEGY_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "dual_ma",
        "title": "双均线择时趋势跟踪",
        "category": "技术择时",
        "description": "经典5日与20日均线金叉做多、死叉平仓，纯面向对象架构，内置A股T+1及滑点佣金微观结构。",
        "code": '''import quant_sdk as ts


class DualMAStrategy(ts.Strategy):
    def on_init(self, context: ts.Context) -> None:
        context.set_benchmark("000300.SH")
        ts.set_commission(ts.AshareCommission(rate=0.0003, min_commission=5.0, stamp_duty=0.0005))
        ts.set_slippage(ts.FixedSlippage(0.001))
        context.g.security = "000001.SZ"

    def on_bar(self, context: ts.Context, data: ts.CurrentData) -> None:
        sec = context.g.security
        if sec not in data:
            return
        # 获取过去25个交易日的收盘价
        hist = ts.attribute_history(sec, count=25, fields=['close'])
        if len(hist) < 20:
            return

        ma5 = hist['close'][-5:].mean()
        ma20 = hist['close'][-20:].mean()
        curr_pos = context.portfolio.positions.get(sec)
        holding = curr_pos.total_amount if curr_pos else 0

        if ma5 > ma20 and holding == 0:
            # 5日上穿20日均线，全仓买入
            self.order_target_percent(sec, 0.95)
        elif ma5 < ma20 and holding > 0:
            # 5日下穿20日均线，平仓
            self.order_target(sec, 0)


if __name__ == "__main__":
    result = ts.run_strategy(
        strategy=DualMAStrategy(),
        start_date="20230101",
        end_date="20231231",
        initial_cash=1_000_000,
    )
    result.summary()
    result.generate_html_report("dual_ma_report.html")
''',
    },
    {
        "id": "small_cap_pit_value",
        "title": "小市值低估值高ROE基本面策略 (零未来函数)",
        "category": "基本面选股",
        "description": "严格Point-in-Time对齐财报公告日，筛选PE合理、ROE>10%且市值偏小的前5只股票，月度调仓。",
        "code": '''import quant_sdk as ts


class SmallCapValueStrategy(ts.Strategy):
    def on_init(self, context: ts.Context) -> None:
        context.set_benchmark("000905.SH")
        ts.set_commission(ts.AshareCommission())
        context.g.max_holdings = 5
        # 每月第一个交易日开盘执行调仓
        ts.run_monthly(self.rebalance, monthday=1)

    def rebalance(self, context: ts.Context) -> None:
        # 严格 Point-in-Time 无未来函数基本面查询（自动识别当前交易日）
        q = (
            ts.query(
                ts.valuation.code,
                ts.valuation.pe_ratio,
                ts.valuation.circulating_market_cap,
                ts.indicator.roe,
            )
            .filter(
                ts.valuation.pe_ratio > 0,
                ts.valuation.pe_ratio < 35,
                ts.indicator.roe > 10,
            )
            .order_by(ts.valuation.circulating_market_cap, ascending=True)
            .limit(context.g.max_holdings)
        )
        df = ts.get_fundamentals(q)
        target_stocks = set(df['code'].tolist())

        # 1. 卖出不再符合选股条件的持仓
        for sec, pos in list(context.portfolio.positions.items()):
            if sec not in target_stocks and pos.closeable_amount > 0:
                self.order_target(sec, 0)

        # 2. 等权重买入目标股票
        weight = 0.95 / len(target_stocks) if target_stocks else 0.0
        for sec in target_stocks:
            self.order_target_percent(sec, weight)


if __name__ == "__main__":
    result = ts.run_strategy(
        strategy=SmallCapValueStrategy(),
        start_date="20230101",
        end_date="20231231",
        initial_cash=1_000_000,
    )
    result.summary()
''',
    },
    {
        "id": "cb_double_low",
        "title": "可转债双低轮动策略 (T+0灵活对冲)",
        "category": "固收+",
        "description": "基于转债价格+溢价率计算双低指标，每周动态买入全市场双低分最低、下行保护极强的转债篮子。",
        "code": '''import quant_sdk as ts


class ConvertibleBondDoubleLowStrategy(ts.Strategy):
    def on_init(self, context: ts.Context) -> None:
        context.set_benchmark("000300.SH")
        context.g.target_cb_count = 10
        ts.run_weekly(self.rebalance, weekday=1)

    def rebalance(self, context: ts.Context) -> None:
        # 获取当期全市场可转债双低指标
        df_cb = ts.get_cb_daily(trade_date=context.current_dt, with_double_low=True)
        if df_cb.empty:
            return

        # 筛选价格在合理防御区间(90~130元)且双低分最低的前N只
        valid = df_cb[(df_cb['close'] >= 90) & (df_cb['close'] <= 130)]
        valid = valid.sort_values('double_low', ascending=True).head(context.g.target_cb_count)
        targets = set(valid['ts_code'].tolist())

        # 卖出不在目标池的转债
        for sec in list(context.portfolio.positions.keys()):
            if sec not in targets:
                self.order_target(sec, 0)

        # 等权重建仓
        weight = 0.95 / len(targets) if targets else 0.0
        for sec in targets:
            self.order_target_percent(sec, weight)


if __name__ == "__main__":
    result = ts.run_strategy(
        strategy=ConvertibleBondDoubleLowStrategy(),
        start_date="20230101",
        end_date="20230630",
        initial_cash=1_000_000,
    )
    result.summary()
''',
    },
    {
        "id": "etf_momentum_rotation",
        "title": "大类资产 ETF 动量轮动策略",
        "category": "资产配置",
        "description": "覆盖A股大盘(300ETF)、中小盘(500ETF)、黄金与避险国债，按20日动量持有强势资产。",
        "code": '''import quant_sdk as ts


class ETFMomentumStrategy(ts.Strategy):
    def on_init(self, context: ts.Context) -> None:
        context.set_benchmark("000300.SH")
        context.g.etf_pool = [
            "510300.SH",  # 沪深300ETF
            "510500.SH",  # 中证500ETF
            "518880.SH",  # 黄金ETF
            "511010.SH",  # 国债ETF
        ]
        ts.run_weekly(self.rebalance, weekday=1)

    def rebalance(self, context: ts.Context) -> None:
        mom_scores = {}
        for etf in context.g.etf_pool:
            hist = ts.get_fund_daily(ts_code=etf, end_date=context.current_dt)
            if len(hist) >= 20:
                p_now = hist['close'].iloc[-1]
                p_prev = hist['close'].iloc[-20]
                mom_scores[etf] = (p_now - p_prev) / p_prev

        if not mom_scores:
            return

        # 选取动量最高的ETF（若全为负则防御持有国债ETF）
        best_etf = max(mom_scores, key=mom_scores.get)
        if mom_scores[best_etf] < 0:
            best_etf = "511010.SH"

        for sec in list(context.portfolio.positions.keys()):
            if sec != best_etf:
                self.order_target(sec, 0)

        self.order_target_percent(best_etf, 0.98)


if __name__ == "__main__":
    result = ts.run_strategy(
        strategy=ETFMomentumStrategy(),
        start_date="20230101",
        end_date="20231231",
        initial_cash=500_000,
    )
    result.summary()
''',
    },
    {
        "id": "live_miniqmt_execution",
        "title": "miniQMT 实盘自动化对接与风控模板",
        "category": "实盘网关",
        "description": "直连券商本地 miniQMT (XtQuant) 客户端，内置单笔上限、日内限额与ST黑名单硬拦截。",
        "code": '''import quant_sdk as ts

# 1. 声明机构级本地安全风控规则（单笔最大10万，日累计最大50万，禁止买入ST）
risk_rules = ts.TraderRiskController(
    max_order_notional=100_000.0,
    max_daily_notional=500_000.0,
    max_orders_per_minute=20,
    block_st=True,
    security_blacklist=["*ST*"],
)

# 2. 初始化券商本地交易网关（MockQmtTrader可用于零配置回放测试，生产接入传入QmtTrader）
trader = ts.MockQmtTrader(
    account_id="55001234",
    initial_cash=1_000_000.0,
    risk_controller=risk_rules,
)

# 3. 查询资产与执行委托
asset = trader.query_asset()
print(f"账户初始资金: {asset.cash:.2f}, 总资产: {asset.total_asset:.2f}")

order = trader.order("000001.SZ", 2000, price=10.50)
print(f"委托状态: {order.status}, 委托单号: {order.order_id}")

positions = trader.query_positions()
print(f"当前持仓股票数: {len(positions)}")
''',
    },
    {
        "id": "nautilus_high_frequency",
        "title": "Nautilus Trader Rust纳秒级日内事件驱动回测",
        "category": "高频与事件驱动",
        "description": "基于Nautilus Trader Rust内核的高性能纳秒级事件驱动回测引擎，专为A股T+1与微观流动性设计。",
        "code": '''import quant_sdk as ts
import pandas as pd
import numpy as np

# 1. 构造纳秒级行情数据 (或使用 ts.load_minute_bars / ts.get_stock_hist)
dates = pd.date_range("2024-01-02 09:30:00", "2024-01-02 15:00:00", freq="1min")
prices = 100 * np.cumprod(1 + np.random.normal(0.0001, 0.001, size=len(dates)))
df = pd.DataFrame({
    "datetime": dates,
    "open": prices,
    "high": prices * 1.001,
    "low": prices * 0.999,
    "close": prices,
    "volume": np.random.uniform(1000, 5000, size=len(dates)),
})

# 2. 转换数据为 Nautilus Trader Bar 纳秒流
instrument_id, bars = ts.convert_df_to_nautilus_bars(df, symbol="600519", exchange="XSHG")

# 3. 启动 Rust 引擎加载与事件回测
engine = ts.AshareNautilusEngine(initial_cash=1_000_000.0)
engine.add_bars(instrument_id, bars)
print("Nautilus Trader 引擎初始化完成，开始纳秒事件流撮合回测...")
engine.run()
print("回测完成！已无缝应用 A 股每手 100 股最小交易量与 CNY 结算体系。")
''',
    },
    {
        "id": "alphalens_pyfolio_tearsheet",
        "title": "Alphalens + PyFolio 顶级机构单因子与组合风险全息报告",
        "category": "因子与风险分析",
        "description": "对冲基金级因子检验(IC/IR、分位数收益、多期衰减)与投资组合全息风险指标(CAGR、Sharpe、Drawdown、VaR)。",
        "code": '''import quant_sdk as ts
import pandas as pd
import numpy as np

# 1. 生成因子值与多资产价格历史
dates = pd.date_range("2023-01-01", periods=180, freq="B")
symbols = ["600519.SH", "000858.SZ", "601318.SH"]

prices_data = {}
factor_rows = []
for sym in symbols:
    p = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, size=len(dates)))
    prices_data[sym] = p
    for dt, price in zip(dates, p):
        factor_rows.append({"datetime": dt, "symbol": sym, "factor": -np.random.normal(0, 1)})

prices_df = pd.DataFrame(prices_data, index=dates)
factor_df = pd.DataFrame(factor_rows).set_index(["datetime", "symbol"])["factor"]

# 2. 生成 Alphalens 因子评价与独立 HTML 交互报告
metrics = ts.calc_factor_metrics(factor_df, prices_df, periods=(1, 5, 10))
print("Alphalens 因子核心指标:")
print(f"Mean IC: {metrics.get('ic_mean', 0.0):.4f}, IC IR: {metrics.get('ic_ir', 0.0):.4f}")

html_report = ts.generate_factor_tear_sheet(factor_df, prices_df, title="Alphalens Factor Analysis")
with open("factor_tear_sheet.html", "w", encoding="utf-8") as f:
    f.write(html_report)
print("已成功输出 factor_tear_sheet.html 因子报告！")

# 3. 计算 PyFolio 机构风险指标与水下回撤
returns = prices_df.pct_change().mean(axis=1).dropna()
risk = ts.calc_pyfolio_risk_metrics(returns)
print("PyFolio 风险指标:")
print(f"Sharpe: {risk['sharpe_ratio']}, Max Drawdown: {risk['max_drawdown']*100:.2f}%")
''',
    },
    {
        "id": "hrp_portfolio_optimization",
        "title": "PyPortfolioOpt 层次风险平价 (HRP) 与 Black-Litterman 配置",
        "category": "资产配置",
        "description": "基于机器学习图论层次聚类的风险平价模型，无需协方差矩阵求逆，防病态奇异矩阵。",
        "code": '''import quant_sdk as ts
import pandas as pd
import numpy as np

# 构造多资产历史行情
dates = pd.date_range("2023-01-01", periods=200, freq="B")
assets = {
    "沪深300ETF": 100 * np.cumprod(1 + np.random.normal(0.0004, 0.012, size=len(dates))),
    "中证500ETF": 100 * np.cumprod(1 + np.random.normal(0.0006, 0.016, size=len(dates))),
    "国债ETF": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.002, size=len(dates))),
    "黄金ETF": 100 * np.cumprod(1 + np.random.normal(0.0005, 0.008, size=len(dates))),
}
prices = pd.DataFrame(assets, index=dates)

# 1. 运行 HRP 层次风险平价优化
hrp_weights = ts.optimize_hrp(prices)
print("HRP 层次风险平价最优配置权重:")
for asset, wt in hrp_weights.items():
    print(f"  {asset}: {wt * 100:.2f}%")

# 2. 评估组合表现
perf = ts.evaluate_portfolio(hrp_weights, prices)
print(f"组合年化预期收益: {perf['expected_return']*100:.2f}%")
print(f"组合年化波动率: {perf['volatility']*100:.2f}%")
print(f"夏普比率 (Sharpe): {perf['sharpe_ratio']:.2f}")

# 3. 输出自包含资产配置 HTML 报告
html = ts.generate_portfolio_report_html(hrp_weights, perf, title="大类资产配置 HRP 风险平价分析")
with open("portfolio_allocation.html", "w", encoding="utf-8") as f:
    f.write(html)
print("已成功输出 portfolio_allocation.html 报告！")
''',
    },
    {
        "id": "qlib_lightgbm_ml_alpha",
        "title": "Qlib + LightGBM 时序机器学习 Alpha 挖掘",
        "category": "AI量化与机器学习",
        "description": "内置防未来函数与防重叠泄漏的 Purged & Embargoed 时序交叉验证，端到端挖掘非线性超额收益。",
        "code": '''import quant_sdk as ts
import pandas as pd
import numpy as np

# 1. 构造股票历史量价数据
dates = pd.date_range("2023-01-01", periods=250, freq="B")
close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, size=250))
df = pd.DataFrame({
    "datetime": dates,
    "open": close * 0.99,
    "high": close * 1.02,
    "low": close * 0.98,
    "close": close,
    "volume": np.random.uniform(10000, 50000, size=250),
})

# 2. 端到端运行防泄漏时序 Walk-Forward 机器学习流水线
print("正在执行 Purged & Embargoed 时序切分与 LightGBM GBDT 模型训练...")
results = ts.run_purged_walkforward_alpha(df, horizons=(5, 10, 20), target_horizon=5, n_splits=4)

print("样本外评价结果:")
print(f"样本外平均 IC (Pearson): {results['mean_ic']}")
print(f"样本外平均 Rank IC (Spearman): {results['mean_rank_ic']}")
print("\\n核心特征重要性 (Gain Importance):")
for feat, imp in list(results['feature_importances'].items())[:5]:
    print(f"  {feat}: {imp:.2f}")
''',
    },
]


def list_strategy_templates() -> list[dict[str, Any]]:
    """Return catalog of built-in ready-made strategy templates."""
    return [
        {
            "id": t["id"],
            "title": t["title"],
            "category": t["category"],
            "description": t["description"],
        }
        for t in STRATEGY_TEMPLATES
    ]


def get_strategy_template(template_id: str) -> dict[str, Any]:
    """Retrieve full code and metadata of a specific strategy template."""
    for t in STRATEGY_TEMPLATES:
        if t["id"] == template_id:
            return t
    raise KeyError(f"Strategy template '{template_id}' not found.")
