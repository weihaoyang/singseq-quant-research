"""Local quantitative research tools for data access, backtests and evidence."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="quant-sdk",
        description="本地量化数据访问、因子预检、回测与证据工具"
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="子命令")

    # 1. status
    p_status = subparsers.add_parser("status", help="查看集群算力、硬件状态与数据湖概况")
    p_status.add_argument("--url", default=None, help="指定服务端地址")

    # 2. browse
    p_browse = subparsers.add_parser("browse", help="在浏览器中直接打开专业级股票可视化全息看板")
    p_browse.add_argument("code", nargs="?", default="000001.SZ", help="股票代码 (如 000001.SZ, 600519.SH)")
    p_browse.add_argument("--port", type=int, default=8000, help="服务端口")

    # 3. kline
    p_kline = subparsers.add_parser("kline", help="快速查看指定标的近期复权行情与技术指标摘要")
    p_kline.add_argument("code", help="股票代码 (如 000001.SZ)")
    p_kline.add_argument("--limit", type=int, default=10, help="输出最近天数")

    # 4. screener
    p_screen = subparsers.add_parser("screener", help="查看全市场选股截面与涨跌分布")
    p_screen.add_argument("--date", default=None, help="交易日期 (如 20240105)")
    p_screen.add_argument("--limit", type=int, default=15, help="展示行数")

    # 5. preflight
    p_pref = subparsers.add_parser("preflight", help="秒级本地因子语法与输出格式预审")
    p_pref.add_argument("file", help="因子 Python 文件路径")
    p_pref.add_argument("--data", required=True,
                        help="本机 CSV/Parquet/DuckDB/DB 面板；预检不连接研究服务器")

    p_local = subparsers.add_parser(
        "local-mine", help="用你的 Token 在本地拉数并计算因子（计算不离开本机）")
    p_local.add_argument("--file", required=True, help="因子源码文件路径")
    p_local.add_argument("--codes", required=True, help="逗号分隔 ts_code，如 000001.SZ,600519.SH")
    p_local.add_argument("--start", required=True, help="起始日 YYYYMMDD")
    p_local.add_argument("--end", required=True, help="结束日 YYYYMMDD")
    p_local.add_argument("--url", default=None, help="指定服务端地址")
    p_local.add_argument("--token", default=None, help="你的 API Token（或 QUANT_API_TOKEN）")
    p_local.add_argument("--api", default="daily", help="数据接口名（默认 daily）")
    p_local.add_argument("--price-col", default="open", help="预测收益使用的价格列")
    p_local.add_argument("--result-csv", default=None, help="可选：导出本地因子值 CSV")

    p_bt = subparsers.add_parser(
        "local-backtest", help="用 canonical 引擎在本地回测（计算不离开本机）")
    p_bt.add_argument("--file", required=True, help="因子源码文件路径")
    p_bt.add_argument("--data", required=True, help="面板 CSV/Parquet，或本地 staging DuckDB 路径")
    p_bt.add_argument("--start", required=True, help="起始日 YYYYMMDD")
    p_bt.add_argument("--end", required=True, help="结束日 YYYYMMDD")
    p_bt.add_argument("--allow-short", action="store_true", help="默认 long-only（与产品一致）")
    p_bt.add_argument("--groups", type=int, default=5, help="分组数，默认 5")
    p_bt.add_argument("--no-costs", action="store_true", help="关闭成本（默认开启）")
    p_bt.add_argument("--neutralize", action="store_true", help="行业+市值中性化")
    p_bt.add_argument(
        "--backend",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="计算后端；auto 优先 GPU，不可用时回退 CPU",
    )
    p_bt.add_argument("--benchmark-code", default="000300.SH", help="基准指数代码")

    p_id = subparsers.add_parser(
        "local-intraday", help="本地日内（Bar 级）回测（canonical intraday 引擎）")
    p_id.add_argument("--file", required=True, help="因子源码文件路径")
    p_id.add_argument("--data", required=True, help="分钟 parquet/csv 文件或目录")
    p_id.add_argument("--code", default=None, help="选择回测证券，或为无代码数据指定证券")
    p_id.add_argument("--buy-threshold", type=float, default=0.0)
    p_id.add_argument("--sell-threshold", type=float, default=0.0)
    p_id.add_argument("--order-shares", type=int, default=10000)
    p_id.add_argument("--base-shares", type=int, default=0)
    p_id.add_argument("--cross-sectional", action="store_true",
                      help="多标的日内截面（日内信号聚合为日频因子）")
    p_id.add_argument("--agg", default="last",
                      choices=["last", "mean", "first", "max", "min", "median"],
                      help="日内聚合方式")
    p_id.add_argument("--groups", type=int, default=5, help="分组数")
    p_id.add_argument("--no-costs", action="store_true", help="关闭成本")
    p_id.add_argument("--neutralize", action="store_true", help="按行业 + log 市值中性化")

    args = parser.parse_args()

    if args.subcommand == "status":
        from quant_sdk.provider import QuantClient
        client = QuantClient(base_url=args.url)
        print(f"\n[Quant SDK] 当前连接模式: {getattr(client, 'mode', 'remote').upper()}")
        print(f"[Quant SDK] API 网关: {client.base_url}")
        try:
            hw = client.check_health()
            h = hw.get("hardware", {})
            print(f"[Hardware] CPU 负载: {h.get('cpu_percent')}% | 内存: {h.get('ram_used_gb')}/{h.get('ram_total_gb')} GB ({h.get('ram_percent')}%)")
            gpu = h.get("gpu", {})
            print(f"[Hardware] GPU: {gpu.get('device_name')} (显存: {gpu.get('allocated_mb')}/{gpu.get('total_mb')} MB | bfloat16: {gpu.get('bf16_supported')})")
            list_apis = getattr(client, "list_apis", None)
            if callable(list_apis):
                apis = list_apis()
                print(f"[Data Lake] 可用数据表总数: {len(apis)} 张 (含 daily, daily_basic, balancesheet 等)")
        except Exception as e:
            print(f"[Error] 获取状态失败: {e}", file=sys.stderr)
            # A scripted/agent caller reads the exit code: reporting success after a
            # failed status probe was silent.
            sys.exit(1)

    elif args.subcommand == "browse":
        from quant_sdk.analytics import browse
        browse(args.code, port=args.port)

    elif args.subcommand == "kline":
        from quant_sdk.analytics import get_kline
        df = get_kline(args.code)
        if df.empty:
            print(f"未找到标的 [{args.code}] 行情数据")
            return
        tail = df.tail(args.limit)
        show_cols = [c for c in ["trade_date", "close", "pct_chg", "ma5", "ma20", "vol", "macd", "pe_ttm", "pb"] if c in tail.columns]
        print(f"\n===== [{args.code}] 最近 {len(tail)} 个交易日行情面板 =====")
        print(tail[show_cols].to_string(index=False))

    elif args.subcommand == "screener":
        from quant_sdk.analytics import get_screener
        df = get_screener(trade_date=args.date, limit=args.limit)
        if df.empty:
            print("未能获取截面数据")
            return
        show_cols = [c for c in ["ts_code", "name", "industry", "close", "pct_chg", "vol", "amount"] if c in df.columns]
        print(f"\n===== 全市场行情截面扫描 (前 {len(df)} 支标的) =====")
        print(df[show_cols].to_string(index=False))

    elif args.subcommand == "preflight":
        from quant_sdk.preflight import test_factor
        p = Path(args.file)
        if not p.exists():
            print(f"文件不存在: {args.file}")
            sys.exit(1)
        res = test_factor(str(p), data=args.data)
        if not (res.get("valid", res.get("ok", False))):
            sys.exit(1)

    elif args.subcommand == "local-mine":
        from quant_sdk.local import run_local_factor
        try:
            metrics = run_local_factor(
                Path(args.file).read_text(encoding="utf-8"),
                ts_codes=[c.strip() for c in args.codes.split(",") if c.strip()],
                start=args.start, end=args.end, token=args.token,
                base_url=args.url, api=args.api, price_col=args.price_col,
                result_csv=args.result_csv,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"本地计算失败: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"[LOCAL_UNVERIFIED] {metrics['scored_rows']}/{metrics['rows']} 行 | "
              f"IC 天数 {metrics['ic_days']} | mean Rank IC {metrics['mean_rank_ic']} | "
              f"IC_IR {metrics['ic_ir']}")
        if metrics.get("result_csv"):
            print(f"因子值已导出: {metrics['result_csv']}")

    elif args.subcommand == "local-backtest":
        import json as _json

        from quant_sdk.local_backtest import load_panel, run_local_backtest
        path = Path(args.data)
        if path.suffix.lower() in {".duckdb", ".db"}:
            frame = load_panel(path, args.start, args.end,
                               benchmark_code=args.benchmark_code)
        elif path.suffix.lower() == ".parquet":
            import pandas as pd
            frame = pd.read_parquet(path)
        else:
            import pandas as pd
            frame = pd.read_csv(path)
        result = run_local_backtest(
            Path(args.file).read_text(encoding="utf-8"), frame,
            allow_short=args.allow_short, n_groups=args.groups,
            costs=not args.no_costs, neutralize=bool(args.neutralize),
            backend=args.backend)
        print(_json.dumps(result, ensure_ascii=False, indent=2))

    elif args.subcommand == "local-intraday":
        import json as _json

        from quant_sdk.local_intraday import (
            load_minute_bars,
            load_minute_universe,
            run_local_intraday,
            run_local_intraday_cross_section,
        )
        try:
            source = Path(args.file).read_text(encoding="utf-8")
            if getattr(args, "cross_sectional", False):
                bars = load_minute_universe(args.data)
                result = run_local_intraday_cross_section(
                    source, bars, agg=args.agg, n_groups=args.groups,
                    costs=not args.no_costs, neutralize=bool(args.neutralize))
            else:
                bars = load_minute_bars(args.data, code=args.code)
                result = run_local_intraday(
                    source, bars,
                    buy_threshold=args.buy_threshold, sell_threshold=args.sell_threshold,
                    order_shares=args.order_shares, base_shares=args.base_shares)
        except Exception as exc:  # noqa: BLE001
            print(f"日内回测失败: {exc}", file=sys.stderr)
            sys.exit(1)
        print(_json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
