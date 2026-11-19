"""One-command local public-market Alpha demo.

This is an onboarding workflow, not a production research job. It pulls public
Binance or Yahoo data on the user's machine, screens local factors, runs one
canonical local backtest, creates a signed ``.qalpha`` pack, replays it on the
same data slice, and builds a trusted-only local index.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_SYMBOLS = (
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "DOT/USDT",
    "LTC/USDT",
    "BCH/USDT",
)
DEFAULT_EQUITY_SYMBOLS = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AVGO",
    "BRK-B",
    "JPM",
    "LLY",
    "UNH",
)


def _public_key_file(private_key: str | Path, output: Path) -> Path:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
    except ImportError as error:  # pragma: no cover - dependency error
        raise RuntimeError("需要安装 cryptography 才能生成可信公钥") from error
    key = serialization.load_pem_private_key(
        Path(private_key).expanduser().read_bytes(),
        password=None,
    )
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("demo signing key 必须是 Ed25519 私钥")
    raw = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(base64.b64encode(raw).decode("ascii") + "\n", encoding="utf-8")
    return output


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(
    *,
    output_dir: str | Path,
    start: str = "20240101",
    end: str = "20240630",
    symbols: Sequence[str] | str | None = None,
    top: int = 20,
    scan_limit: int = 300,
    force: bool = False,
    signing_key: str | Path | None = None,
    signer: str = "local-quant-demo",
    panel: str | Path | None = None,
    provider: str = "binance-public",
    benchmark_panel: str | Path | None = None,
    benchmark_symbol: str | None = None,
    benchmark_name: str | None = None,
) -> dict[str, Any]:
    """Run the complete demo locally and return a machine-readable receipt."""
    from quant_sdk.alpha_graph import graph_from_factor_names
    from quant_sdk.alpha_screen import screen_factors
    from quant_sdk.data_connectors import pull_connector
    from quant_sdk.evidence_audit import (
        attach_equal_weight_benchmark,
        attach_market_benchmark,
        audit_factor,
        research_split,
    )
    from quant_sdk.evidence_pack import (
        build_index,
        create_pack,
        generate_signing_key,
        reproduce_pack,
    )
    from quant_sdk.research_workflow import (
        _load_frame,
        batch_evaluate,
        inspect_data,
    )

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    provider = str(provider or "binance-public").strip().lower()
    if provider not in {"binance-public", "yahoo-public", "a-share-local"}:
        raise ValueError(
            "provider 必须是 binance-public、yahoo-public 或 a-share-local")
    default_tickers = (
        DEFAULT_EQUITY_SYMBOLS
        if provider == "yahoo-public"
        else DEFAULT_SYMBOLS if provider == "binance-public" else ()
    )
    tickers = (
        list(default_tickers)
        if symbols is None
        else [str(item).strip() for item in (
            symbols.split(",") if isinstance(symbols, str) else symbols
        ) if str(item).strip()]
    )
    if provider != "a-share-local" and not tickers:
        raise ValueError("symbols 不能为空")

    start = str(start).strip().replace("-", "")
    end = str(end).strip().replace("-", "")
    if len(start) != 8 or len(end) != 8 or start > end:
        raise ValueError("start/end 必须使用 YYYYMMDD，且 start <= end")

    if panel:
        panel_path = Path(panel).expanduser().resolve()
        if not panel_path.is_file():
            raise ValueError(f"panel 不存在: {panel_path}")
        pulled = {
            "status": "REUSED_PANEL",
            "provider": "local-panel",
            "output": str(panel_path),
            "credentialsReturned": False,
        }
    else:
        if provider == "a-share-local":
            raise ValueError("a-share-local 需要 --panel <本地标准化 A 股面板>")
        panel_name = (
            "yahoo-panel.parquet"
            if provider == "yahoo-public"
            else "binance-panel.parquet"
        )
        panel_path = root / panel_name
        pulled = pull_connector(
            provider,
            output=panel_path,
            start=start,
            end=end,
            symbols=tickers,
            force=bool(force),
        )
    source_frame = _load_frame(panel_path, start, end)
    if benchmark_panel:
        import pandas as pd

        benchmark_path = Path(benchmark_panel).expanduser().resolve()
        if not benchmark_path.is_file():
            raise ValueError(f"benchmark panel 不存在: {benchmark_path}")
        if benchmark_path.suffix.lower() == ".parquet":
            benchmark_source = pd.read_parquet(benchmark_path)
        elif benchmark_path.suffix.lower() == ".csv":
            benchmark_source = pd.read_csv(benchmark_path)
        else:
            raise ValueError("benchmark panel 必须是 CSV 或 Parquet")
        benchmark_frame, benchmark_evidence = attach_market_benchmark(
            source_frame,
            benchmark_source,
            symbol=benchmark_symbol,
            source=benchmark_name or str(benchmark_path),
        )
    else:
        benchmark_frame, benchmark_evidence = attach_equal_weight_benchmark(
            source_frame
        )
    if benchmark_evidence.get("status") == "ATTACHED":
        benchmark_panel = root / "benchmark-panel.parquet"
        temporary = benchmark_panel.with_name(
            benchmark_panel.name + ".tmp"
        )
        benchmark_frame.to_parquet(temporary, index=False)
        os.replace(temporary, benchmark_panel)
        panel_path = benchmark_panel
        pulled["normalizedPanel"] = str(panel_path)
        pulled["benchmark"] = benchmark_evidence
    inspected = inspect_data(panel_path, start=start, end=end)
    if not inspected.get("readyForBacktest"):
        raise ValueError("公开数据面板未通过本地回测就绪检查")

    frame_for_split = _load_frame(panel_path, start, end)
    split = research_split(
        frame_for_split["trade_date"].astype(str).tolist(),
        selection_ratio=0.7,
        purge_days=1,
        embargo_days=1,
    )
    screen = screen_factors(
        data_path=str(panel_path),
        start=start,
        end=str(split["selectionEnd"]),
        families=["alpha101", "alpha191"],
        scan_limit=int(scan_limit),
        top=int(top),
        chunk_size=64,
        minimum_coverage=0.5,
        minimum_ic_days=10,
        min_names=5,
        output=root / "screen.json",
    )
    selected = list(screen.get("selected") or [])
    if not selected:
        raise ValueError("本地筛选没有产生可回测候选")

    trial_count = int(screen.get("counts", {}).get("computed") or 1)
    selection_candidates: list[dict[str, Any]] = []
    directions: dict[str, int] = {}
    for item in selected[:10]:
        candidate = str(item.get("name") or "").strip()
        if not candidate:
            continue
        selection_ic = item.get("meanRankIc")
        direction = (
            -1
            if selection_ic is not None and float(selection_ic) < 0.0
            else 1
        )
        directions[candidate] = direction
        selection_candidates.append({
            "name": candidate,
            "factorName": candidate,
            "direction": direction,
        })
    selection_batch = batch_evaluate(
        data_path=str(panel_path),
        start=start,
        end=str(split["selectionEnd"]),
        candidates=selection_candidates,
        out=root / "selection-batch.json",
        n_groups=3,
        costs=True,
        allow_short=False,
        neutralize=False,
        n_trials=trial_count,
        reports=False,
    )
    viable = []
    batch_errors: list[dict[str, Any]] = []
    for row in selection_batch.get("candidates", []):
        if not row.get("ok"):
            batch_errors.append({
                "name": row.get("name"),
                "error": row.get("error"),
            })
            continue
        metrics = (row.get("summary") or {}).get("metrics") or {}
        sharpe = metrics.get("sharpe")
        annualized = metrics.get("annualized")
        viable.append({
            "name": row.get("name"),
            "factorName": row.get("factorName"),
            "direction": int(row.get("direction") or directions.get(
                str(row.get("factorName") or ""), 1
            )),
            "selectionSharpe": sharpe,
            "selectionAnnualized": annualized,
        })
    viable.sort(key=lambda item: (
        item["selectionSharpe"] is not None,
        float(item["selectionSharpe"] or -math.inf),
        float(item["selectionAnnualized"] or -math.inf),
        str(item["name"]),
    ), reverse=True)
    if not viable:
        raise RuntimeError(
            "筛选候选均未通过选择期 canonical 回测: "
            + json.dumps(batch_errors[:10], ensure_ascii=False)
        )
    selected_winner = viable[0]
    chosen_direction = int(selected_winner["direction"])
    chosen_name = str(selected_winner["factorName"])
    batch = batch_evaluate(
        data_path=str(panel_path),
        start=start,
        end=end,
        candidates=[{
            "name": chosen_name,
            "factorName": chosen_name,
            "direction": chosen_direction,
        }],
        out=root / "batch.json",
        n_groups=3,
        costs=True,
        allow_short=False,
        neutralize=False,
        n_trials=trial_count,
        reports=True,
    )
    chosen = next(
        (
            row for row in batch.get("candidates", [])
            if row.get("ok") and row.get("factorName") == chosen_name
        ),
        None,
    )
    if chosen is None or batch is None:
        failed = next(
            (
                row for row in batch.get("candidates", [])
                if row.get("factorName") == chosen_name
            ),
            None,
        )
        batch_errors.append({
            "name": chosen_name,
            "error": (failed or {}).get("error")
            or "selection winner failed the full canonical backtest",
        })
        raise RuntimeError(
            "选择期最优候选未通过完整 canonical 回测: "
            + json.dumps(batch_errors[:5], ensure_ascii=False)
        )
    benchmark_code = str(
        benchmark_symbol
        or {
            "binance-public": "BTCUSDT",
            "yahoo-public": "SPY",
            "a-share-local": "000300.SH",
        }[provider]
    )
    audit = audit_factor(
        data_path=str(panel_path),
        start=start,
        end=end,
        factor_name=str(chosen["factorName"]),
        name=str(chosen["factorName"]),
        selection_ratio=0.7,
        purge_days=1,
        embargo_days=1,
        horizon=1,
        min_names=5,
        n_groups=3,
        costs=True,
        allow_short=False,
        n_trials=trial_count,
        direction=chosen_direction,
        benchmark_code=benchmark_code,
        output=root / "audit.json",
    )
    if audit.get("status") == "REJECTED_LEAKAGE":
        raise RuntimeError(
            "候选未通过因果/泄漏审计，拒绝封装为可发布证据："
            + json.dumps(
                audit.get("runtimeCausality") or audit.get("staticAudit"),
                ensure_ascii=False,
            )[:1000]
        )

    key_path = Path(signing_key).expanduser().resolve() if signing_key else root / "demo-signing.pem"
    if not key_path.exists():
        generated = generate_signing_key(key_path)
        key_path = Path(generated["privateKey"])
    trusted_path = root / "trusted-demo.pub"
    _public_key_file(key_path, trusted_path)

    candidate_name = str(chosen["factorName"])
    base_graph = graph_from_factor_names([candidate_name])
    if chosen_direction < 0:
        factor_graph = {
            **base_graph,
            "nodes": [
                *base_graph["nodes"],
                {"id": "direction", "op": "neg", "input": "f1"},
            ],
            "outputs": ["direction"],
        }
    else:
        factor_graph = base_graph
    market_label = {
        "binance-public": "crypto",
        "yahoo-public": "equity",
        "a-share-local": "ashare",
    }[provider]
    market_name = {
        "binance-public": "crypto",
        "yahoo-public": "US equity",
        "a-share-local": "A-share",
    }[provider]
    point_in_time = {
        "binance-public": "Binance daily candle open time in UTC",
        "yahoo-public": (
            "Yahoo daily chart timestamp in the listing-market timezone"
        ),
        "a-share-local": (
            "caller-supplied A-share panel; trading-date semantics remain "
            "the data owner's responsibility"
        ),
    }[provider]
    pack_path = root / f"{market_label}-{candidate_name}.qalpha"
    pack = create_pack(
        pack_path,
        name=f"{market_label}-{candidate_name}",
        graph=factor_graph,
        factor_name=candidate_name,
        data_path=panel_path,
        data_start=start,
        data_end=end,
        result_file=chosen["factorCsv"],
        audit_file=root / "audit.json",
        signing_key_file=key_path,
        signer=signer,
        metadata={
            "market": {
                "id": market_label,
                "name": market_name,
                "provider": provider,
            },
            "pointInTime": point_in_time,
            "costModel": f"canonical local backtest with explicit {market_name} proxy approximations",
            "benchmark": {
                **(audit.get("benchmark") or {}),
                "definition": benchmark_evidence,
            },
            "selectionProtocol": {
                "screenWindow": {
                    "start": start,
                    "end": split["selectionEnd"],
                },
                "screenDidNotUseHoldout": True,
                "split": audit.get("split"),
                "direction": chosen_direction,
                "directionSource": "selection-period mean rank IC sign",
            },
            "outOfSample": audit.get("outOfSample"),
            "capacity": audit.get("capacity"),
            "leakageChecks": {
                "status": (audit.get("runtimeCausality") or {}).get("status"),
                "staticSourceAudit": (audit.get("staticAudit") or {}).get("status"),
                "runtimeCausality": audit.get("runtimeCausality"),
                "checks": audit.get("checks"),
            },
        },
    )
    replay = reproduce_pack(
        pack["path"],
        data_path=panel_path,
        start=start,
        end=end,
    )
    index = build_index(
        [root],
        trusted_keys=[trusted_path],
        trusted_only=True,
        output_path=root / "qalpha-index.json",
    )
    index_rows = list(index.get("rows") or [])
    status = (
        "DEMO_READY"
        if replay.get("status") == "REPRODUCED"
        and index.get("status") == "LOCAL_VERIFIED"
        and index.get("source", {}).get("uniquePacks") == 1
        else "DEMO_INCOMPLETE"
    )
    result = {
        "format": "quant-public-demo/1",
        "demo": market_label,
        "market": market_name,
        "provider": provider,
        "status": status,
        "alphaValidation": audit.get("status"),
        "alphaValidated": bool(audit.get("validated")),
        "executionBoundary": "local",
        "data": pulled,
        "inspection": inspected,
        "screen": {
            "counts": screen.get("counts"),
            "selected": selected[:5],
            "window": screen.get("window"),
        },
        "audit": {
            "status": audit.get("status"),
            "validated": audit.get("validated"),
            "checks": audit.get("checks"),
            "split": audit.get("split"),
            "direction": chosen_direction,
            "outOfSample": {
                "selection": (audit.get("outOfSample") or {}).get("selection"),
                "holdout": (audit.get("outOfSample") or {}).get("holdout"),
            },
            "benchmark": audit.get("benchmark"),
            "capacity": audit.get("capacity"),
            "leakage": {
                "status": (audit.get("runtimeCausality") or {}).get("status"),
                "probes": (audit.get("runtimeCausality") or {}).get("probes"),
            },
            "path": (audit.get("output")),
        },
        "backtest": {
            "candidate": chosen,
            "batch": batch.get("panel"),
            "selectionWinner": selected_winner,
            "selectionAttempts": viable,
            "selectionBatch": selection_batch.get("panel"),
            "attemptErrors": batch_errors,
        },
        "artifact": pack,
        "replay": {
            "status": replay.get("status"),
            "reproduced": replay.get("reproduced"),
            "expectedCanonicalSha256": replay.get("expectedCanonicalSha256"),
            "actualCanonicalSha256": replay.get("actualCanonicalSha256"),
            "valueColumnMapping": replay.get("valueColumnMapping"),
        },
        "index": {
            "status": index.get("status"),
            "packs": index.get("source", {}).get("packs"),
            "trustedPacks": sum(
                1 for row in index_rows if row.get("status") == "TRUSTED"
            ),
            "paths": [row.get("path") for row in index_rows],
            "uploaded": index.get("uploaded"),
            "submitted": index.get("submitted"),
            "published": index.get("published"),
        },
        "paths": {
            "outputDir": str(root),
            "panel": str(panel_path),
            "artifact": str(pack_path),
            "signingKey": str(key_path),
            "trustedKey": str(trusted_path),
            "index": str(root / "qalpha-index.json"),
            "audit": str(root / "audit.json"),
        },
        "note": (
            f"{provider} 数据、本地计算、显式 canonical 近似；"
            "选择期与 OOS 已隔离，结果带因果、成本、基准与容量证据；"
            "没有提交、发布或排名，也不是投资建议。"
        ),
    }
    summary_path = root / "demo-summary.json"
    summary_path.write_text(
        json.dumps(_jsonable(result), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    result["paths"]["summary"] = str(summary_path)
    return _jsonable(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local public-market Alpha demo")
    parser.add_argument("--provider", default="binance-public")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", default="20240101")
    parser.add_argument("--end", default="20240630")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--scan-limit", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--signing-key")
    parser.add_argument("--signer", default="local-quant-demo")
    parser.add_argument("--panel")
    parser.add_argument("--benchmark-panel")
    parser.add_argument("--benchmark-symbol")
    parser.add_argument("--benchmark-name")
    args = parser.parse_args(argv)
    result = run(
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
        symbols=args.symbols or None,
        top=args.top,
        scan_limit=args.scan_limit,
        force=args.force,
        signing_key=args.signing_key,
        signer=args.signer,
        panel=args.panel,
        provider=args.provider,
        benchmark_panel=args.benchmark_panel,
        benchmark_symbol=args.benchmark_symbol,
        benchmark_name=args.benchmark_name,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result.get("status") == "DEMO_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
