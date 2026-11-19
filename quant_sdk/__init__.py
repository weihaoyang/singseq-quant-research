"""
quant_sdk
--------------------------------------------------
机构级量化数据中枢、全功能算法研究与Rust级事件驱动算力沙箱

集成顶级量化生态与开箱即用能力：
1. Nautilus Trader: Rust级纳秒高频与日内事件驱动回测引擎
2. Alphalens-Reloaded: 宽客与对冲基金级多期因子检验与 IC/IR 报告
3. PyFolio & Empyrical: 顶级机构风险分析、水下回撤图与 VaR/CVaR 评估
4. PyPortfolioOpt: Markowitz 均值-方差、HRP 层次风险平价、Black-Litterman 资产配置
5. AkShare: 宏观经济(CPI/PMI/LPR)、北向资金净流入、龙虎榜席位与实时行情
6. LightGBM / Qlib: 去除数据穿越的 Purged & Embargoed 时序机器学习 Alpha 挖掘
7. TA-Lib & MyTT: C语言极速指标库与通达信/同花顺公式语法无缝兼容
8. DuckDB & Polars: 纳秒/微秒级内存零拷贝数据湖与面板极速查询
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "2.2.0"

__all__ = [
    # Core Architecture & Strategy
    "AshareCommission",
    "BaseTrader",
    "Context",
    "Field",
    "FixedSlippage",
    "FundamentalsQuery",
    "LiveOrder",
    "LivePosition",
    "LocalFactorRunner",
    "MockQmtTrader",
    "PaperAccount",
    "PaperTradingManager",
    "Portfolio",
    "Position",
    "ProAPI",
    "QmtTrader",
    "QuantClient",
    "STRATEGY_TEMPLATES",
    "Strategy",
    "StrategyResult",
    "TraderRiskController",
    "VolumeShareSlippage",
    "as_of_join",
    "attribute_history",
    "audit_factor",
    "balance",
    "browse",
    "cancel_order",
    "cash_flow",
    "create_paper_account",
    "get_cb_daily",
    "get_current_data",
    "get_fund_daily",
    "get_fundamentals",
    "get_kline",
    "get_paper_account",
    "get_price",
    "get_screener",
    "get_strategy_template",
    "history",
    "income",
    "indicator",
    "init",
    "list_paper_accounts",
    "list_strategy_templates",
    "load_minute_bars",
    "load_minute_universe",
    "load_panel",
    "operators",
    "order",
    "order_target",
    "order_target_percent",
    "order_target_value",
    "order_value",
    "panel_from_frame",
    "pro_api",
    "query",
    "run_daily",
    "run_local_backtest",
    "run_local_factor",
    "run_local_intraday",
    "run_local_intraday_cross_section",
    "run_monthly",
    "run_strategy",
    "run_weekly",
    "set_benchmark",
    "set_commission",
    "set_option",
    "set_slippage",
    "step_paper_account",
    "submit_factor",
    "test_factor",
    "valuation",
    # Nautilus Trader Rust Engine
    "AshareNautilusEngine",
    "convert_df_to_nautilus_bars",
    # Alphalens Factor Analysis
    "prepare_factor_data",
    "calc_factor_metrics",
    "generate_factor_tear_sheet",
    # PyFolio & Empyrical Risk Tear Sheet
    "calc_pyfolio_risk_metrics",
    "get_drawdown_table",
    "generate_pyfolio_tear_sheet",
    # PyPortfolioOpt Asset Allocation
    "optimize_max_sharpe",
    "optimize_min_volatility",
    "optimize_hrp",
    "optimize_black_litterman",
    "evaluate_portfolio",
    "generate_portfolio_report_html",
    # AkShare Real-time & Macro Data
    "get_realtime_quotes",
    "get_stock_hist",
    "get_northbound_money_flow",
    "get_dragon_tiger_inst_list",
    "get_treasury_yield_curve",
    "get_macro_indicator",
    # Machine Learning Alpha (Qlib / LightGBM)
    "build_alpha_features",
    "PurgedTimeSeriesSplit",
    "LGBMAlphaModel",
    "run_purged_walkforward_alpha",
    # Technical Indicators (TA-Lib & MyTT)
    "macd",
    "rsi",
    "bollinger_bands",
    "atr",
    "kdj",
    "add_technical_indicators",
    "CROSS",
    "MA",
    "EMA",
    "SMA",
    "DIFF",
    "HHV",
    "LLV",
    "COUNT",
    "EVERY",
    "EXIST",
]

_EXPORTS = {
    # Existing core
    "query": ("quant_sdk.fundamentals", "query"),
    "get_fundamentals": ("quant_sdk.fundamentals", "get_fundamentals"),
    "valuation": ("quant_sdk.fundamentals", "valuation"),
    "indicator": ("quant_sdk.fundamentals", "indicator"),
    "income": ("quant_sdk.fundamentals", "income"),
    "balance": ("quant_sdk.fundamentals", "balance"),
    "cash_flow": ("quant_sdk.fundamentals", "cash_flow"),
    "get_cb_daily": ("quant_sdk.fundamentals", "get_cb_daily"),
    "get_fund_daily": ("quant_sdk.fundamentals", "get_fund_daily"),
    "FundamentalsQuery": ("quant_sdk.fundamentals", "FundamentalsQuery"),
    "Field": ("quant_sdk.fundamentals", "Field"),
    "STRATEGY_TEMPLATES": ("quant_sdk.strategy_templates", "STRATEGY_TEMPLATES"),
    "list_strategy_templates": ("quant_sdk.strategy_templates", "list_strategy_templates"),
    "get_strategy_template": ("quant_sdk.strategy_templates", "get_strategy_template"),
    "BaseTrader": ("quant_sdk.trader", "BaseTrader"),
    "QmtTrader": ("quant_sdk.trader", "QmtTrader"),
    "MockQmtTrader": ("quant_sdk.trader", "MockQmtTrader"),
    "TraderRiskController": ("quant_sdk.trader", "TraderRiskController"),
    "LiveOrder": ("quant_sdk.trader", "LiveOrder"),
    "LivePosition": ("quant_sdk.trader", "LivePosition"),
    "PaperAccount": ("quant_sdk.paper_trading", "PaperAccount"),
    "PaperTradingManager": ("quant_sdk.paper_trading", "PaperTradingManager"),
    "create_paper_account": ("quant_sdk.paper_trading", "create_paper_account"),
    "get_paper_account": ("quant_sdk.paper_trading", "get_paper_account"),
    "list_paper_accounts": ("quant_sdk.paper_trading", "list_paper_accounts"),
    "step_paper_account": ("quant_sdk.paper_trading", "step_paper_account"),
    "Strategy": ("quant_sdk.strategy", "Strategy"),
    "StrategyResult": ("quant_sdk.strategy", "StrategyResult"),
    "run_strategy": ("quant_sdk.strategy", "run_strategy"),
    "order": ("quant_sdk.strategy", "order"),
    "order_target": ("quant_sdk.strategy", "order_target"),
    "order_value": ("quant_sdk.strategy", "order_value"),
    "order_target_value": ("quant_sdk.strategy", "order_target_value"),
    "order_target_percent": ("quant_sdk.strategy", "order_target_percent"),
    "cancel_order": ("quant_sdk.strategy", "cancel_order"),
    "get_price": ("quant_sdk.strategy", "get_price"),
    "get_current_data": ("quant_sdk.strategy", "get_current_data"),
    "history": ("quant_sdk.strategy", "history"),
    "attribute_history": ("quant_sdk.strategy", "attribute_history"),
    "run_daily": ("quant_sdk.strategy", "run_daily"),
    "run_weekly": ("quant_sdk.strategy", "run_weekly"),
    "run_monthly": ("quant_sdk.strategy", "run_monthly"),
    "set_benchmark": ("quant_sdk.strategy", "set_benchmark"),
    "set_slippage": ("quant_sdk.strategy", "set_slippage"),
    "set_commission": ("quant_sdk.strategy", "set_commission"),
    "set_option": ("quant_sdk.strategy", "set_option"),
    "FixedSlippage": ("quant_sdk.strategy", "FixedSlippage"),
    "VolumeShareSlippage": ("quant_sdk.strategy", "VolumeShareSlippage"),
    "AshareCommission": ("quant_sdk.strategy", "AshareCommission"),
    "Context": ("quant_sdk.strategy", "Context"),
    "Position": ("quant_sdk.strategy", "Position"),
    "Portfolio": ("quant_sdk.strategy", "Portfolio"),
    "QuantClient": ("quant_sdk.provider", "QuantClient"),
    "ProAPI": ("quant_sdk.provider", "ProAPI"),
    "pro_api": ("quant_sdk.provider", "pro_api"),
    "init": ("quant_sdk.provider", "init"),
    "get_kline": ("quant_sdk.analytics", "get_kline"),
    "get_screener": ("quant_sdk.analytics", "get_screener"),
    "browse": ("quant_sdk.analytics", "browse"),
    "test_factor": ("quant_sdk.preflight", "test_factor"),
    "submit_factor": ("quant_sdk.preflight", "submit_factor"),
    "LocalFactorRunner": ("quant_sdk.local", "LocalFactorRunner"),
    "run_local_factor": ("quant_sdk.local", "run_local_factor"),
    "load_panel": ("quant_sdk.local_backtest", "load_panel"),
    "panel_from_frame": ("quant_sdk.local_backtest", "panel_from_frame"),
    "as_of_join": ("quant_sdk.panel_contract", "as_of_join"),
    "run_local_backtest": ("quant_sdk.local_backtest", "run_local_backtest"),
    "audit_factor": ("quant_sdk.evidence_audit", "audit_factor"),
    "load_minute_bars": ("quant_sdk.local_intraday", "load_minute_bars"),
    "load_minute_universe": ("quant_sdk.local_intraday", "load_minute_universe"),
    "run_local_intraday": ("quant_sdk.local_intraday", "run_local_intraday"),
    "run_local_intraday_cross_section": (
        "quant_sdk.local_intraday",
        "run_local_intraday_cross_section",
    ),
    "operators": ("quant_sdk.operators", None),
    # Nautilus Trader Rust Engine
    "AshareNautilusEngine": ("quant_sdk.nautilus_adapter", "AshareNautilusEngine"),
    "convert_df_to_nautilus_bars": ("quant_sdk.nautilus_adapter", "convert_df_to_nautilus_bars"),
    # Alphalens
    "prepare_factor_data": ("quant_sdk.alphalens_tear_sheet", "prepare_factor_data"),
    "calc_factor_metrics": ("quant_sdk.alphalens_tear_sheet", "calc_factor_metrics"),
    "generate_factor_tear_sheet": ("quant_sdk.alphalens_tear_sheet", "generate_factor_tear_sheet"),
    # PyFolio
    "calc_pyfolio_risk_metrics": ("quant_sdk.pyfolio_tear_sheet", "calc_pyfolio_risk_metrics"),
    "get_drawdown_table": ("quant_sdk.pyfolio_tear_sheet", "get_drawdown_table"),
    "generate_pyfolio_tear_sheet": ("quant_sdk.pyfolio_tear_sheet", "generate_pyfolio_tear_sheet"),
    # PyPortfolioOpt
    "optimize_max_sharpe": ("quant_sdk.portfolio_optimizer", "optimize_max_sharpe"),
    "optimize_min_volatility": ("quant_sdk.portfolio_optimizer", "optimize_min_volatility"),
    "optimize_hrp": ("quant_sdk.portfolio_optimizer", "optimize_hrp"),
    "optimize_black_litterman": ("quant_sdk.portfolio_optimizer", "optimize_black_litterman"),
    "evaluate_portfolio": ("quant_sdk.portfolio_optimizer", "evaluate_portfolio"),
    "generate_portfolio_report_html": ("quant_sdk.portfolio_optimizer", "generate_portfolio_report_html"),
    # AkShare
    "get_realtime_quotes": ("quant_sdk.akshare_connector", "get_realtime_quotes"),
    "get_stock_hist": ("quant_sdk.akshare_connector", "get_stock_hist"),
    "get_northbound_money_flow": ("quant_sdk.akshare_connector", "get_northbound_money_flow"),
    "get_dragon_tiger_inst_list": ("quant_sdk.akshare_connector", "get_dragon_tiger_inst_list"),
    "get_treasury_yield_curve": ("quant_sdk.akshare_connector", "get_treasury_yield_curve"),
    "get_macro_indicator": ("quant_sdk.akshare_connector", "get_macro_indicator"),
    # Machine Learning Alpha
    "build_alpha_features": ("quant_sdk.ml_alpha", "build_alpha_features"),
    "PurgedTimeSeriesSplit": ("quant_sdk.ml_alpha", "PurgedTimeSeriesSplit"),
    "LGBMAlphaModel": ("quant_sdk.ml_alpha", "LGBMAlphaModel"),
    "run_purged_walkforward_alpha": ("quant_sdk.ml_alpha", "run_purged_walkforward_alpha"),
    # Technical Indicators
    "macd": ("quant_sdk.technical", "macd"),
    "rsi": ("quant_sdk.technical", "rsi"),
    "bollinger_bands": ("quant_sdk.technical", "bollinger_bands"),
    "atr": ("quant_sdk.technical", "atr"),
    "kdj": ("quant_sdk.technical", "kdj"),
    "add_technical_indicators": ("quant_sdk.technical", "add_technical_indicators"),
    "CROSS": ("quant_sdk.technical", "CROSS"),
    "MA": ("quant_sdk.technical", "MA"),
    "EMA": ("quant_sdk.technical", "EMA"),
    "SMA": ("quant_sdk.technical", "SMA"),
    "DIFF": ("quant_sdk.technical", "DIFF"),
    "HHV": ("quant_sdk.technical", "HHV"),
    "LLV": ("quant_sdk.technical", "LLV"),
    "COUNT": ("quant_sdk.technical", "COUNT"),
    "EVERY": ("quant_sdk.technical", "EVERY"),
    "EXIST": ("quant_sdk.technical", "EXIST"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'quant_sdk' has no attribute {name!r}")
    module_name, attribute = target
    value = importlib.import_module(module_name)
    if attribute is not None:
        value = getattr(value, attribute)
    globals()[name] = value
    return value
