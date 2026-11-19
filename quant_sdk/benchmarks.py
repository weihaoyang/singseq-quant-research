"""Portable public benchmarks and local evaluation contracts.

A benchmark publishes the data contract and evaluation rule, not factor source
code or market data. Evaluation runs on the researcher's machine unless an
explicit external evaluation service is requested through a separate product.
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_sdk.alpha_graph import evaluate_graph, graph_from_factor_names
from quant_sdk.evidence_pack import (
    MANIFEST_NAME,
    _frame_slice_identity,
    verify_pack,
)
from quant_sdk.labels import label_contract
from quant_sdk.research_workflow import _compact_date, _load_frame

BENCHMARK_FORMAT = "qbench/1"
BENCHMARK_VERSION = "1"

_BENCHMARKS: dict[str, dict[str, Any]] = {
    "a-share-daily-cross-sectional-v1": {
        "id": "a-share-daily-cross-sectional-v1",
        "title": "A-share daily cross-sectional alpha",
        "market": "CN",
        "assetClass": "equity",
        "frequency": "1d",
        "dataContract": {
            "required": ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"],
            "identity": ["ts_code", "trade_date"],
            "pointInTime": {
                "fundamentalAvailability": "provider responsibility",
                "delistingReturns": "provider responsibility",
                "priceAdjustment": "declare adj_close or document the approximation",
            },
        },
        "evaluation": {
            "target": "adjusted next-open forward return",
            "labelVersion": "next-open-adjusted/1",
            "horizon": 1,
            "minimumNamesPerDate": 20,
            "minimumCoverage": 0.5,
            "selectionRatio": 0.7,
            "metrics": ["rankIc", "icIr", "icPositiveShare", "coverage", "autocorrelation"],
        },
    },
    "global-equity-daily-cross-sectional-v1": {
        "id": "global-equity-daily-cross-sectional-v1",
        "title": "Global equity daily cross-sectional alpha",
        "market": "GLOBAL",
        "assetClass": "equity",
        "frequency": "1d",
        "dataContract": {
            "required": ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"],
            "identity": ["ts_code", "trade_date"],
            "pointInTime": {
                "corporateActions": "provider responsibility",
                "survivorship": "provider responsibility",
                "timeZone": "daily session close in the listing market",
            },
        },
        "evaluation": {
            "target": "adjusted listing-market next-open forward return",
            "labelVersion": "next-open-adjusted/1",
            "horizon": 1,
            "minimumNamesPerDate": 20,
            "minimumCoverage": 0.5,
            "selectionRatio": 0.7,
            "metrics": ["rankIc", "icIr", "icPositiveShare", "coverage", "autocorrelation"],
        },
    },
    "crypto-daily-cross-sectional-v1": {
        "id": "crypto-daily-cross-sectional-v1",
        "title": "Crypto daily cross-sectional alpha",
        "market": "GLOBAL",
        "assetClass": "crypto",
        "frequency": "1d",
        "dataContract": {
            "required": ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"],
            "identity": ["ts_code", "trade_date"],
            "pointInTime": {
                "exchange": "provider responsibility",
                "quoteAsset": "declare before comparison",
                "sessionCutoff": "UTC 00:00",
            },
        },
        "evaluation": {
            "target": "adjusted UTC next-open forward return",
            "labelVersion": "next-open-adjusted/1",
            "horizon": 1,
            "minimumNamesPerDate": 10,
            "minimumCoverage": 0.5,
            "selectionRatio": 0.7,
            "metrics": ["rankIc", "icIr", "icPositiveShare", "coverage", "autocorrelation"],
        },
    },
}

_PRIVACY_PROTOCOL = {
    "defaultMode": "local_only",
    "modes": {
        "local_only": {
            "description": "Data and factor execution remain on the researcher device.",
            "automaticUpload": False,
        },
        "evidence_only": {
            "description": "Share a signed .qalpha pack containing source or graph, metadata and result digests; market data is not embedded.",
            "automaticUpload": False,
        },
        "secure_compute": {
            "description": "Optional provider-side compute-to-data protocol for Alpha Graph and catalog factors. The provider returns a signed execution receipt; source-only packs must be converted to a graph first.",
            "automaticUpload": False,
            "implementedLocally": "isolated_subprocess_or_oci",
            "hardwareConfidential": False,
            "runner": "private.run",
            "request": "private.request",
            "verify": "private.verify",
        },
    },
    "submissionIsExplicit": True,
}


def _benchmark_map() -> dict[str, dict[str, Any]]:
    from quant_sdk.capabilities import installed_benchmark_specs

    benchmarks = {
        key: json.loads(json.dumps(value))
        for key, value in _BENCHMARKS.items()
    }
    for benchmark in installed_benchmark_specs():
        benchmark_id = str(benchmark.get("id") or "")
        if not benchmark_id:
            continue
        if benchmark_id in benchmarks:
            raise ValueError(
                f"installed benchmark conflicts with a built-in id: {benchmark_id}"
            )
        benchmarks[benchmark_id] = benchmark
    return benchmarks


def benchmark_manifest(benchmark_id: str) -> dict[str, Any]:
    key = str(benchmark_id or "").strip()
    benchmarks = _benchmark_map()
    if key not in benchmarks:
        raise ValueError(f"未知 benchmark: {key}")
    return {
        "format": BENCHMARK_FORMAT,
        "version": BENCHMARK_VERSION,
        "status": "PUBLIC_RULE",
        "officialRanking": None,
        **benchmarks[key],
        "artifactContract": {
            "evidencePack": "qalpha/1",
            "factorIdentity": ["source", "graph", "catalogName"],
            "marketDataEmbedded": False,
            "requiresDataSliceIdentity": True,
        },
        "privacy": _PRIVACY_PROTOCOL,
        "executionBoundary": "researcher_local_by_default",
        "submission": {
            "explicitOnly": True,
            "automatic": False,
            "note": "Local evaluation and benchmark pass/fail do not submit or publish a factor.",
        },
    }


def list_benchmarks() -> dict[str, Any]:
    benchmarks = _benchmark_map()
    return {
        "format": BENCHMARK_FORMAT,
        "version": BENCHMARK_VERSION,
        "count": len(benchmarks),
        "benchmarks": [
            {
                "id": item["id"],
                "title": item["title"],
                "market": item["market"],
                "assetClass": item["assetClass"],
                "frequency": item["frequency"],
            }
            for item in benchmarks.values()
        ],
        "privacy": _PRIVACY_PROTOCOL,
    }


def _graph_from_input(
    graph: dict[str, Any] | None,
    factor_names: Iterable[str] | None,
) -> dict[str, Any]:
    if graph is not None:
        return graph
    return graph_from_factor_names(factor_names or [])


def _window(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    work = frame[
        frame["trade_date"].astype(str).between(str(start), str(end))
    ].reset_index(drop=True)
    if work.empty:
        raise ValueError("benchmark evaluation window is empty")
    return work


def _metric(frame: pd.DataFrame, graph: dict[str, Any], *,
            horizon: int, min_names: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    return evaluate_graph(frame, graph, horizon=horizon, min_names=min_names)


def evaluate_benchmark(
    *,
    benchmark_id: str,
    data_path: str | Path,
    start: str,
    end: str,
    graph: dict[str, Any] | None = None,
    factor_names: Iterable[str] | None = None,
    evidence_pack_path: str | Path | None = None,
    benchmark_code: str = "000300.SH",
    minimum_names: int | None = None,
    minimum_coverage: float | None = None,
    selection_ratio: float | None = None,
) -> dict[str, Any]:
    """Evaluate one graph/catalog factor locally under a public benchmark rule."""
    contract = benchmark_manifest(benchmark_id)
    policy = contract["evaluation"]
    min_names = int(
        minimum_names
        if minimum_names is not None
        else policy["minimumNamesPerDate"]
    )
    min_coverage = float(
        minimum_coverage
        if minimum_coverage is not None
        else policy["minimumCoverage"]
    )
    ratio = float(
        selection_ratio
        if selection_ratio is not None
        else policy["selectionRatio"]
    )
    if not 0.1 <= ratio <= 0.9:
        raise ValueError("selectionRatio must be between 0.1 and 0.9")
    start = _compact_date(start, "start")
    end = _compact_date(end, "end")
    source_path = Path(data_path).expanduser().resolve()
    artifact = None
    expected_data_sha = None
    if evidence_pack_path:
        if graph is not None or factor_names:
            raise ValueError(
                "evidencePackPath cannot be combined with graph or factorNames")
        pack_path = Path(evidence_pack_path).expanduser().resolve()
        checked = verify_pack(pack_path)
        if not checked["valid"]:
            raise ValueError(
                "Evidence Pack verification failed: "
                + ", ".join(checked.get("errors") or []))
        with zipfile.ZipFile(pack_path) as archive:
            manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
            factor = manifest.get("factor") or {}
            graph_member = factor.get("graph")
            if graph_member and graph_member in archive.namelist():
                graph = json.loads(archive.read(graph_member).decode("utf-8"))
            elif factor.get("catalogName"):
                graph = graph_from_factor_names([str(factor["catalogName"])])
            else:
                raise ValueError(
                    "Evidence Pack needs a graph or catalogName for benchmark evaluation")
        expected_data_sha = (
            ((manifest.get("data") or {}).get("slice") or {}).get("sha256")
        )
        artifact = {
            "path": str(pack_path),
            "sha256": checked["sha256"],
            "name": checked.get("name"),
            "version": checked.get("version"),
            "signatureValid": checked.get("signatureValid"),
            "signerFingerprint": checked.get("signerFingerprint"),
            "publisherVerified": checked.get("publisherVerified"),
        }
    requested_benchmark = str(benchmark_code or "").strip()
    contract_market = str(contract.get("market") or "").upper()
    contract_asset = str(contract.get("assetClass") or "").lower()
    if (
        not requested_benchmark
        or requested_benchmark.upper() in {"000300.SH", "AUTO", "MARKET"}
    ):
        if contract_market == "CN" and contract_asset == "equity":
            requested_benchmark = "000300.SH"
        elif contract_asset == "crypto":
            requested_benchmark = "BTCUSDT"
        elif contract_asset == "equity":
            requested_benchmark = "SPY"
    frame = _load_frame(
        source_path, start, end, benchmark_code=requested_benchmark,
    )
    benchmark_resolution = frame.attrs.get("benchmarkResolution") or {}
    observed_market = str(benchmark_resolution.get("market") or "UNKNOWN")
    expected_market = (
        "CRYPTO" if contract_asset == "crypto"
        else "CN" if contract_market == "CN" and contract_asset == "equity"
        else "US" if contract_asset == "equity"
        else "GLOBAL"
    )
    compatibility = observed_market in {expected_market, "GLOBAL"}
    if expected_market == "US" and observed_market in {"US", "GLOBAL"}:
        compatibility = True
    data_identity = _frame_slice_identity(frame)
    if expected_data_sha and data_identity["sha256"] != expected_data_sha:
        raise ValueError("benchmark data does not match the Evidence Pack slice")
    normalized_graph = _graph_from_input(graph, factor_names)
    dates = sorted(frame["trade_date"].astype(str).unique())
    split_index = max(1, min(len(dates) - 1, int(len(dates) * ratio)))
    train_end = dates[split_index - 1]
    test_start = dates[split_index]
    train = _window(frame, dates[0], train_end)
    test = _window(frame, test_start, dates[-1])
    horizon = int(policy["horizon"])
    _, full_summary = _metric(
        frame, normalized_graph, horizon=horizon, min_names=min_names,
    )
    _, train_summary = _metric(
        train, normalized_graph, horizon=horizon, min_names=min_names,
    )
    _, test_summary = _metric(
        test, normalized_graph, horizon=horizon, min_names=min_names,
    )
    outputs = []
    full_metric = full_summary["outputs"]
    train_metric = train_summary["outputs"]
    test_metric = test_summary["outputs"]
    for name in normalized_graph["outputs"]:
        full = full_metric[name]
        training = train_metric[name]
        holdout = test_metric[name]
        train_ic = training.get("meanRankIc")
        test_ic = holdout.get("meanRankIc")
        checks = {
            "marketBenchmarkMatches": compatibility,
            "coverage": bool(float(full.get("coverage") or 0.0) >= min_coverage),
            "minimumIcDays": bool(int(full.get("icDays") or 0) >= 10),
            "holdoutObserved": bool(holdout.get("icDays", 0) > 0),
            "holdoutSameDirection": bool(
                train_ic is not None
                and test_ic is not None
                and np.sign(float(train_ic)) == np.sign(float(test_ic))
            ),
        }
        outputs.append({
            "name": name,
            "full": full,
            "selection": training,
            "holdout": holdout,
            "checks": checks,
            "selectionChecksPassed": all(checks.values()),
            "selectionScore": (
                round(
                    abs(float(full["meanRankIc"]))
                    * math.sqrt(float(full["icDays"]))
                    * math.sqrt(max(float(full.get("coverage") or 0.0), 0.0)),
                    6,
                )
                if full.get("meanRankIc") is not None else None
            ),
        })
    return {
        "format": BENCHMARK_FORMAT,
        "version": BENCHMARK_VERSION,
        "status": "LOCAL_BENCHMARK_COMPLETE",
        "official": False,
        "benchmark": contract,
        "benchmarkResolution": {
            **benchmark_resolution,
            "expectedMarket": expected_market,
            "dataMatchesBenchmarkMarket": compatibility,
        },
        "data": {
            "path": str(source_path),
            "window": {"start": start, "end": end},
            "identity": data_identity,
        },
        "split": {
            "selectionRatio": ratio,
            "selectionEnd": train_end,
            "holdoutStart": test_start,
        },
        "label": label_contract(frame, horizon),
        "graph": normalized_graph,
        "artifact": artifact,
        "outputs": outputs,
        "submitted": False,
        "published": False,
        "note": "本地 benchmark 结果不是官方排名，也不会自动上传或提交。",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local benchmark protocol")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--json", action="store_true")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--benchmark", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--benchmark", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--start", required=True)
    evaluate.add_argument("--end", required=True)
    evaluate.add_argument("--factor-names", default="")
    evaluate.add_argument("--spec")
    evaluate.add_argument("--evidence-pack")
    evaluate.add_argument("--benchmark-code", default="000300.SH")
    evaluate.add_argument("--minimum-names", type=int)
    evaluate.add_argument("--minimum-coverage", type=float)
    evaluate.add_argument("--selection-ratio", type=float)
    args = parser.parse_args(argv)
    if args.command == "list":
        result = list_benchmarks()
    elif args.command == "inspect":
        result = benchmark_manifest(args.benchmark)
    else:
        graph = (
            json.loads(Path(args.spec).read_text(encoding="utf-8"))
            if args.spec else None
        )
        names = [
            item.strip() for item in str(args.factor_names).split(",")
            if item.strip()
        ]
        result = evaluate_benchmark(
            benchmark_id=args.benchmark, data_path=args.data,
            start=args.start, end=args.end, graph=graph,
            factor_names=names, evidence_pack_path=args.evidence_pack,
            benchmark_code=args.benchmark_code,
            minimum_names=args.minimum_names,
            minimum_coverage=args.minimum_coverage,
            selection_ratio=args.selection_ratio,
        )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
