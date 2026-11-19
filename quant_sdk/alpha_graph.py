"""Local Alpha Graph evaluation.

Factor leaves reference the existing named-factor catalogue; algebraic,
cross-sectional and causal-operator nodes compose those leaves after the panel
has been loaded once. The graph is a compiler boundary, not a second execution
engine.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from quant_sdk.labels import canonical_forward_returns, label_contract

GRAPH_VERSION = "alpha-graph-1"
GRAPH_COMPILER_VERSION = "alpha-graph-compiler-1"
_OPERATOR_CONTEXT = {"stocks", "dates", "groups", "industries"}
_OPERATOR_CONTEXT_FIELDS = {
    "stocks": "ts_code",
    "dates": "trade_date",
    "groups": None,
    "industries": None,
}


@lru_cache(maxsize=1)
def _numba_rank_matrix_kernel() -> Any:
    try:
        import numba
    except ImportError:
        return None
    numba.set_num_threads(
        max(1, min(16, int(numba.config.NUMBA_NUM_THREADS))),
    )

    @numba.njit(parallel=True)
    def kernel(values: np.ndarray) -> np.ndarray:
        output = np.full(values.shape, np.nan)
        rows, columns = values.shape
        for column in numba.prange(columns):
            order = np.argsort(values[:, column])
            count = 0
            for row in range(rows):
                if np.isfinite(values[row, column]):
                    count += 1
            position = 0
            while position < count:
                end = position + 1
                value = values[order[position], column]
                while (
                    end < count
                    and values[order[end], column] == value
                ):
                    end += 1
                average_rank = (position + 1 + end) / 2.0
                for rank_position in range(position, end):
                    output[order[rank_position], column] = average_rank
                position = end
        return output

    return kernel


def _average_rank_matrix(values: np.ndarray) -> np.ndarray:
    """Average-rank every column, using local JIT when available."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("rank matrix must be two-dimensional")
    # Numba's first compilation can cost more than scipy for a small output
    # set. Reserve it for matrices where the parallel kernel can amortise JIT.
    if matrix.size < 500_000 or matrix.shape[1] < 8:
        return rankdata(
            matrix,
            axis=0,
            method="average",
            nan_policy="omit",
        )
    kernel = _numba_rank_matrix_kernel()
    if kernel is not None:
        return kernel(np.ascontiguousarray(matrix))
    return rankdata(
        matrix,
        axis=0,
        method="average",
        nan_policy="omit",
    )


@lru_cache(maxsize=1)
def _numa_grouped_rank_kernel() -> Any:
    try:
        import numba
    except ImportError:
        return None
    numba.set_num_threads(
        max(1, min(16, int(numba.config.NUMBA_NUM_THREADS))),
    )

    @numba.njit(parallel=True)
    def kernel(
        values: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
    ) -> np.ndarray:
        output = np.full(values.shape, np.nan)
        groups, columns = len(starts), values.shape[1]
        for task in numba.prange(groups * columns):
            group = task // columns
            column_index = task - group * columns
            start = starts[group]
            end = ends[group]
            column = values[start:end, column_index]
            order = np.argsort(column)
            count = 0
            for row in range(len(column)):
                if np.isfinite(column[row]):
                    count += 1
            position = 0
            while position < count:
                tie_end = position + 1
                value = column[order[position]]
                while (
                    tie_end < count
                    and column[order[tie_end]] == value
                ):
                    tie_end += 1
                average_rank = (position + 1 + tie_end) / 2.0
                for rank_position in range(position, tie_end):
                    output[
                        start + order[rank_position],
                        column_index,
                    ] = average_rank
                position = tie_end
        return output

    return kernel


def _grouped_average_rank_matrix(
    values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    matrix = np.ascontiguousarray(values, dtype=float)
    if matrix.size < 250_000:
        output = np.empty_like(matrix)
        for start, end in zip(starts, ends, strict=True):
            output[start:end] = rankdata(
                matrix[start:end],
                axis=0,
                method="average",
                nan_policy="omit",
            )
        return output
    kernel = _numa_grouped_rank_kernel()
    if kernel is not None:
        return kernel(matrix, starts.astype(np.int64), ends.astype(np.int64))
    output = np.empty_like(matrix)
    for start, end in zip(starts, ends, strict=True):
        output[start:end] = rankdata(
            matrix[start:end],
            axis=0,
            method="average",
            nan_policy="omit",
        )
    return output


@lru_cache(maxsize=1)
def _numa_rolling_batch_kernel() -> Any:
    try:
        import numba
    except ImportError:
        return None
    numba.set_num_threads(
        max(1, min(16, int(numba.config.NUMBA_NUM_THREADS))),
    )

    @numba.njit(parallel=True)
    def kernel(
        values: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        windows: np.ndarray,
        min_periods: np.ndarray,
        method_codes: np.ndarray,
    ) -> np.ndarray:
        groups = len(starts)
        outputs = len(windows)
        result = np.full((len(values), outputs), np.nan)
        for task in numba.prange(groups * outputs):
            group = task // outputs
            output_index = task - group * outputs
            start = starts[group]
            end = ends[group]
            window = windows[output_index]
            minimum = min_periods[output_index]
            method = method_codes[output_index]
            if end <= start or window < 1:
                continue
            length = end - start
            count_prefix = np.zeros(length + 1, dtype=np.float64)
            sum_prefix = np.zeros(length + 1, dtype=np.float64)
            square_prefix = np.zeros(length + 1, dtype=np.float64)
            for offset in range(length):
                value = values[start + offset]
                finite = np.isfinite(value)
                count_prefix[offset + 1] = count_prefix[offset] + (1.0 if finite else 0.0)
                if finite:
                    sum_prefix[offset + 1] = sum_prefix[offset] + value
                    square_prefix[offset + 1] = square_prefix[offset] + value * value
                else:
                    sum_prefix[offset + 1] = sum_prefix[offset]
                    square_prefix[offset + 1] = square_prefix[offset]
            for offset in range(minimum - 1, length):
                right = offset + 1
                left = max(0, right - window)
                count = count_prefix[right] - count_prefix[left]
                if count < minimum or count <= 0.0:
                    continue
                total = sum_prefix[right] - sum_prefix[left]
                mean = total / count
                if method == 0:
                    result[start + offset, output_index] = mean
                elif method == 1:
                    result[start + offset, output_index] = total
                else:
                    if count <= 1.0:
                        continue
                    squared = square_prefix[right] - square_prefix[left]
                    numerator = max(squared - total * total / count, 0.0)
                    variance = numerator / (count - 1.0)
                    result[start + offset, output_index] = (
                        variance if method == 3 else np.sqrt(variance)
                    )
        return result

    return kernel


def _rolling_batch(
    values: pd.Series,
    stocks: pd.Series,
    *,
    windows: list[int],
    min_periods: list[int],
    method: str,
) -> np.ndarray:
    """Compute several trailing windows over one stock-partitioned series."""
    data = np.asarray(values, dtype=float)
    labels = np.asarray(stocks.astype(str))
    stock_codes, _ = pd.factorize(labels, sort=False)
    order = np.argsort(stock_codes, kind="stable")
    sorted_codes = stock_codes[order]
    starts = np.r_[
        0,
        np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1,
    ]
    ends = np.r_[starts[1:], len(sorted_codes)]
    windows_array = np.asarray(windows, dtype=np.int64)
    minimum_array = np.asarray(min_periods, dtype=np.int64)
    method_code = {"mean": 0, "sum": 1, "std": 2, "var": 3}[method]
    codes_array = np.full(len(windows), method_code, dtype=np.int64)
    kernel = _numa_rolling_batch_kernel()
    if kernel is not None:
        # The kernel receives contiguous group slices.  Reorder the input into
        # stock order and map the result back to the original panel order.
        ranked = kernel(
            data[order],
            starts.astype(np.int64),
            ends.astype(np.int64),
            windows_array,
            minimum_array,
            codes_array,
        )
        output = np.empty_like(ranked)
        output[order] = ranked
        return output
    output = np.full((len(data), len(windows)), np.nan, dtype=float)
    for output_index, (window, minimum) in enumerate(
        zip(windows, min_periods, strict=True)
    ):
        for start, end in zip(starts, ends, strict=True):
            group = pd.Series(data[start:end])
            rolling = group.rolling(window, min_periods=minimum)
            calculated = getattr(rolling, method)()
            output[start:end, output_index] = calculated.to_numpy(dtype=float)
    return output


def _rank_ic_result(
    names: list[str],
    factor_values: np.ndarray,
    ic: np.ndarray,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for column_index, name in enumerate(names):
        factor = factor_values[:, column_index]
        values = ic[np.isfinite(ic[:, column_index]), column_index]
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result[name] = {
            "finiteRows": int(np.isfinite(factor).sum()),
            "coverage": (
                round(float(np.isfinite(factor).mean()), 6)
                if len(factor) else 0.0
            ),
            "icDays": len(values),
            "meanRankIc": (
                round(float(values.mean()), 6) if len(values) else None
            ),
            "icIr": (
                round(float(values.mean() / std), 6)
                if len(values) > 1 and std > 0.0 else None
            ),
            "icPositiveShare": (
                round(float((values > 0.0).mean()), 6)
                if len(values) else None
            ),
            "icTStat": (
                round(
                    float(values.mean() / std * math.sqrt(len(values))),
                    6,
                )
                if len(values) > 1 and std > 0.0 else None
            ),
        }
    return result


def _resolve_graph_engine(
    backend: str,
    *,
    rows: int,
    outputs: int,
) -> dict[str, Any]:
    normalized = str(backend or "auto").strip().lower()
    if normalized not in {"auto", "cpu", "gpu"}:
        raise ValueError("Alpha Graph backend 必须是 auto、cpu 或 gpu")
    if normalized == "cpu":
        return {
            "engine": None,
            "requested": normalized,
            "selected": "cpu",
            "reason": "explicit-cpu",
        }
    minimum_cells = 250_000
    try:
        minimum_cells = max(
            1,
            int(os.environ.get("QUANT_GRAPH_GPU_MIN_CELLS", "250000")),
        )
    except ValueError:
        pass
    if (
        normalized == "auto"
        and int(rows) * max(1, int(outputs)) < minimum_cells
    ):
        return {
            "engine": None,
            "requested": normalized,
            "selected": "cpu",
            "reason": "below-auto-gpu-threshold",
        }
    if importlib.util.find_spec("torch") is None:
        if normalized == "gpu":
            raise RuntimeError("Alpha Graph 请求 GPU，但本机未安装 PyTorch CUDA")
        return {
            "engine": None,
            "requested": normalized,
            "selected": "cpu",
            "reason": "torch-not-installed",
        }
    try:
        from quant_core.gpu import GPUTensorEngine, cuda_available
    except Exception as error:
        if normalized == "gpu":
            raise RuntimeError(f"Alpha Graph GPU 后端不可用: {error}") from error
        return {
            "engine": None,
            "requested": normalized,
            "selected": "cpu",
            "reason": "gpu-module-unavailable",
        }
    if not cuda_available():
        if normalized == "gpu":
            raise RuntimeError("Alpha Graph 请求 GPU，但 CUDA 不可用")
        return {
            "engine": None,
            "requested": normalized,
            "selected": "cpu",
            "reason": "cuda-not-available",
        }
    try:
        engine = GPUTensorEngine()
    except Exception as error:
        if normalized == "gpu":
            raise RuntimeError(f"Alpha Graph GPU 初始化失败: {error}") from error
        return {
            "engine": None,
            "requested": normalized,
            "selected": "cpu",
            "reason": "gpu-initialization-failed",
        }
    return {
        "engine": engine,
        "requested": normalized,
        "selected": "cuda",
        "reason": None,
    }


def _panel_matrix_codes(
    work: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int] | None:
    if work.duplicated(["trade_date", "ts_code"]).any():
        return None
    date_codes, _ = pd.factorize(
        work["trade_date"].astype(str), sort=False,
    )
    stock_codes, _ = pd.factorize(
        work["ts_code"].astype(str), sort=False,
    )
    return (
        np.asarray(date_codes, dtype=np.int64),
        np.asarray(stock_codes, dtype=np.int64),
        np.arange(len(work), dtype=np.int64),
        int(date_codes.max()) + 1,
        int(stock_codes.max()) + 1,
    )


def _rank_ic_metrics_tensor(
    engine: Any,
    factors: pd.DataFrame,
    work: pd.DataFrame,
    forward: pd.Series,
    *,
    min_names: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]] | None:
    codes = _panel_matrix_codes(work)
    if codes is None:
        return None
    date_codes, stock_codes, row_ids, dates, stocks = codes
    flat_positions = date_codes * stocks + stock_codes
    forward_matrix = np.full((dates, stocks), np.nan, dtype=np.float64)
    forward_matrix.reshape(-1)[flat_positions] = forward.to_numpy(dtype=float)
    result: dict[str, dict[str, Any]] = {}
    backend = "cuda"
    fallback_reason = None
    block_size = 64
    for offset in range(0, len(factors.columns), block_size):
        names = list(factors.columns[offset:offset + block_size])
        values = factors.iloc[:, offset:offset + block_size].to_numpy(
            dtype=np.float64,
        )
        cube = np.full(
            (dates * stocks, len(names)),
            np.nan,
            dtype=np.float64,
        )
        cube[flat_positions] = values[row_ids]
        ic, _ = engine.alpha_graph_rank_ic(
            cube.reshape(dates, stocks, len(names)),
            forward_matrix,
            min_names=int(min_names),
        )
        result.update(_rank_ic_result(names, values, ic))
        if engine.mode == "cpu-fallback":
            backend = "cpu-fallback"
            fallback_reason = engine.fallback_reason
    return result, {
        "backend": backend,
        "fallbackReason": fallback_reason,
        "shape": {"dates": dates, "stocks": stocks},
    }


def _autocorrelation_metrics_tensor(
    engine: Any,
    factors: pd.DataFrame,
    work: pd.DataFrame,
) -> tuple[dict[str, float | None], dict[str, Any]] | None:
    codes = _panel_matrix_codes(work)
    if codes is None:
        return None
    date_codes, stock_codes, row_ids, dates, stocks = codes
    flat_positions = stock_codes * dates + date_codes
    output: dict[str, float | None] = {}
    backend = "cuda"
    fallback_reason = None
    block_size = 32
    for offset in range(0, len(factors.columns), block_size):
        names = list(factors.columns[offset:offset + block_size])
        values = factors.iloc[:, offset:offset + block_size].to_numpy(
            dtype=np.float64,
        )
        cube = np.full(
            (stocks * dates, len(names)),
            np.nan,
            dtype=np.float64,
        )
        cube[flat_positions] = values[row_ids]
        correlations = engine.alpha_graph_autocorrelation(
            cube.reshape(stocks, dates, len(names)),
        )
        for index, name in enumerate(names):
            correlation = float(correlations[index])
            output[name] = (
                round(correlation, 6)
                if math.isfinite(correlation) else None
            )
        if engine.mode == "cpu-fallback":
            backend = "cpu-fallback"
            fallback_reason = engine.fallback_reason
    return output, {
        "backend": backend,
        "fallbackReason": fallback_reason,
        "shape": {"dates": dates, "stocks": stocks},
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def graph_from_factor_names(factor_names: Iterable[str]) -> dict[str, Any]:
    names = [str(name).strip() for name in factor_names if str(name).strip()]
    if not names:
        raise ValueError("Alpha Graph 至少需要一个因子")
    nodes = [
        {"id": f"f{index + 1}", "op": "factor", "factor": name}
        for index, name in enumerate(names)
    ]
    return {
        "version": GRAPH_VERSION,
        "nodes": nodes,
        "outputs": [node["id"] for node in nodes],
    }


def _load_frame(data_path: str | Path, start: str, end: str,
                benchmark_code: str) -> pd.DataFrame:
    from quant_sdk.research_workflow import _load_frame as load_frame

    return load_frame(data_path, start, end, benchmark_code=benchmark_code)


def _normalize_graph(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("Alpha Graph spec 必须是对象")  # noqa: TRY004
    nodes = spec.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Alpha Graph 缺少 nodes")
    normalized: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, raw in enumerate(nodes):
        if not isinstance(raw, dict):
            raise ValueError(f"node {index + 1} 必须是对象")  # noqa: TRY004
        node = dict(raw)
        node_id = str(node.get("id") or "").strip()
        if not node_id or node_id in ids:
            raise ValueError(f"node id 缺失或重复: {node_id!r}")
        ids.add(node_id)
        op = str(node.get("op") or "").strip().lower()
        if not op:
            raise ValueError(f"node {node_id} 缺少 op")
        node["id"] = node_id
        node["op"] = op
        normalized.append(node)
    outputs = spec.get("outputs")
    if outputs is None:
        outputs = [
            node["id"] for node in normalized
            if node["op"] not in {"factor", "field"}
        ] or [normalized[-1]["id"]]
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("Alpha Graph outputs 必须是非空数组")
    output_ids = [str(item) for item in outputs]
    if any(item not in ids for item in output_ids):
        raise ValueError("Alpha Graph outputs 引用了不存在的 node")
    return {
        "version": str(spec.get("version") or GRAPH_VERSION),
        "nodes": normalized,
        "outputs": output_ids,
    }


def _dependency_ids(
    node: dict[str, Any],
    nodes: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    op = node["op"]
    if op in {"factor", "field"}:
        return []
    if op == "operator":
        raw = node.get("inputs", node.get("input"))
        if isinstance(raw, str):
            dependencies = [raw]
        elif isinstance(raw, list) and raw and all(
            isinstance(item, str) and item.strip() for item in raw
        ):
            dependencies = [str(item) for item in raw]
        else:
            raise ValueError(
                f"{node['id']} 的 operator 需要 input 或非空 inputs 数组"
            )
        if nodes:
            for key, value in node.items():
                if (
                    key not in {
                        "id", "op", "input", "inputs", "operator",
                        "window", "period", "periods", "lag", "horizon",
                        "lower", "upper", "minPeriods", "min_periods",
                    }
                    and isinstance(value, str)
                    and value in nodes
                    and value not in dependencies
                ):
                    dependencies.append(value)
        return dependencies
    if op in {"add", "sub", "mul", "div"}:
        inputs = node.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ValueError(f"{node['id']} 的 {op} 需要 inputs")
        return [str(item) for item in inputs]
    if op in {"neg", "rank", "zscore", "scale", "clip"}:
        value = node.get("input")
        if not value:
            raise ValueError(f"{node['id']} 的 {op} 需要 input")
        return [str(value)]
    raise ValueError(f"不支持的 Alpha Graph op: {op}")


def _validate_dag(graph: dict[str, Any]) -> None:
    nodes = {node["id"]: node for node in graph["nodes"]}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError(f"Alpha Graph 存在环: {node_id}")
        if node_id in visited:
            return
        if node_id not in nodes:
            raise ValueError(f"Alpha Graph 引用了不存在的 node: {node_id}")
        visiting.add(node_id)
        for dependency in _dependency_ids(nodes[node_id], nodes):
            visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in nodes:
        visit(node_id)


def _cross_section(frame: pd.DataFrame, values: pd.Series, op: str,
                   node: dict[str, Any]) -> pd.Series:
    grouped = values.groupby(frame["trade_date"], sort=False)
    if op == "rank":
        return grouped.rank(pct=True)
    if op == "zscore":
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0.0, np.nan)
        return (values - mean) / std
    if op == "scale":
        denominator = grouped.transform(lambda item: item.abs().sum())
        return values / denominator.replace(0.0, np.nan)
    if op == "clip":
        lower = float(node.get("lower", 0.01))
        upper = float(node.get("upper", 0.99))
        if not (0.0 <= lower < upper <= 1.0):
            raise ValueError(f"{node['id']} clip 分位数非法")
        low = grouped.transform(lambda item: item.quantile(lower))
        high = grouped.transform(lambda item: item.quantile(upper))
        return values.clip(lower=low, upper=high)
    raise ValueError(f"不支持的截面 op: {op}")


def _reachable_nodes(graph: dict[str, Any]) -> set[str]:
    nodes = {node["id"]: node for node in graph["nodes"]}
    reached: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in reached:
            return
        reached.add(node_id)
        for dependency in _dependency_ids(nodes[node_id], nodes):
            visit(dependency)

    for output in graph["outputs"]:
        visit(output)
    return reached


def _operator_callable(name: str) -> Any:
    """Resolve one causal operator from the allow-listed public catalogue."""
    from quant_sdk import operators

    normalized = str(name or "").strip().lower()
    entry = getattr(operators, "_REGISTRY", {}).get(normalized)
    if entry:
        return entry[2]
    from quant_sdk.capabilities import installed_operator_callable

    installed = installed_operator_callable(normalized)
    if installed is not None:
        return installed
    raise ValueError(
        f"不支持的 Alpha Graph operator: {normalized or '<empty>'}"
    )


def _operator_context_value(
    parameter: str,
    node: dict[str, Any],
    work: pd.DataFrame,
    resolve_node: Any,
) -> Any:
    """Resolve stocks/dates/groups/industries context for an operator."""
    explicit = node.get(parameter)
    if explicit is None and parameter == "log_market_cap":
        explicit = node.get("logMarketCap")
    if explicit is not None:
        reference = str(explicit).strip()
        if parameter in {"stocks", "dates"}:
            if reference not in work.columns:
                raise ValueError(f"{node['id']} 引用了不存在的数据字段: {reference}")
            return work[reference]
        if reference in work.columns:
            return work[reference]
        try:
            return resolve_node(reference)
        except (KeyError, ValueError):
            raise ValueError(
                f"{node['id']} 的 {parameter} 必须引用数据字段或 graph node"
            ) from None
    field = _OPERATOR_CONTEXT_FIELDS.get(parameter)
    if field is not None:
        return work[field]
    raise ValueError(
        f"{node['id']} 的 operator 参数 {parameter} 需要字段或 graph node 引用"
    )


def _evaluate_operator(
    node: dict[str, Any],
    work: pd.DataFrame,
    dependency_values: list[pd.Series],
    resolve_node: Any,
) -> pd.Series:
    function = _operator_callable(str(node.get("operator") or ""))
    parameters = inspect.signature(function).parameters
    positional = list(dependency_values)
    supplied_context: dict[str, Any] = {}
    arguments: list[Any] = []
    keywords: dict[str, Any] = {}
    input_index = 0

    for parameter in parameters.values():
        name = parameter.name
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            raise ValueError(f"{node['id']} 不支持可变位置参数")
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        value = node.get(name)
        if value is None and name == "log_market_cap":
            value = node.get("logMarketCap")
        if name in _OPERATOR_CONTEXT:
            value = _operator_context_value(name, node, work, resolve_node)
            supplied_context[name] = value
        elif value is None and input_index < len(positional):
            value = positional[input_index]
            input_index += 1
        elif value is None:
            if parameter.default is inspect.Parameter.empty:
                raise ValueError(
                    f"{node['id']} 的 operator {node.get('operator')} "
                    f"缺少参数 {name}"
                )
            continue
        if isinstance(value, str) and value in work.columns:
            value = work[value]
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            arguments.append(value)
        else:
            keywords[name] = value

    if input_index != len(positional):
        raise ValueError(
            f"{node['id']} 提供了多余的 graph inputs；"
            f"operator {node.get('operator')} 只消费 {input_index} 个"
        )
    return pd.Series(function(*arguments, **keywords), index=work.index, dtype=float)


def _semantic_signature(
    node: dict[str, Any], signatures: dict[str, tuple[Any, ...]],
) -> tuple[Any, ...]:
    op = node["op"]
    if op == "factor":
        return ("factor", str(node.get("factor") or "").strip())
    if op == "field":
        return ("field", str(node.get("field") or "").strip())
    if op == "operator":
        parameters = {
            str(key): value
            for key, value in node.items()
            if key not in {"id", "op", "input", "inputs", "operator"}
        }
        return (
            "operator",
            str(node.get("operator") or "").strip().lower(),
            tuple(signatures[item] for item in _dependency_ids(node)),
            json.dumps(parameters, sort_keys=True, default=str, separators=(",", ":")),
        )
    dependencies = tuple(signatures[item] for item in _dependency_ids(node))
    if op in {"add", "sub", "mul", "div"}:
        return (op, dependencies)
    if op in {"neg", "rank", "zscore", "scale"}:
        return (op, dependencies)
    if op == "clip":
        return (
            op,
            dependencies,
            float(node.get("lower", 0.01)),
            float(node.get("upper", 0.99)),
        )
    raise ValueError(f"不支持的 Alpha Graph op: {op}")


def _stable_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compile_graph(
    spec: dict[str, Any],
    *,
    backend: str = "auto",
) -> dict[str, Any]:
    """Compile a normalized graph into a deterministic typed execution plan."""
    requested_backend = str(backend or "auto").strip().lower()
    if requested_backend not in {"auto", "cpu", "gpu"}:
        raise ValueError("Alpha Graph backend 必须是 auto、cpu 或 gpu")
    normalized = _normalize_graph(spec)
    _validate_dag(normalized)
    nodes = {node["id"]: node for node in normalized["nodes"]}
    reachable = _reachable_nodes(normalized)
    order: list[str] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            raise ValueError(f"Alpha Graph 存在环: {node_id}")
        visiting.add(node_id)
        for dependency in _dependency_ids(nodes[node_id], nodes):
            visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)
        order.append(node_id)

    for output in normalized["outputs"]:
        visit(output)

    signatures: dict[str, tuple[Any, ...]] = {}
    semantic_keys: dict[str, str] = {}
    first_node_by_key: dict[str, str] = {}
    cse_aliases: dict[str, str] = {}
    plan_nodes: list[dict[str, Any]] = []
    for position, node_id in enumerate(order):
        node = nodes[node_id]
        dependencies = _dependency_ids(node, nodes)
        signature = _semantic_signature(node, signatures)
        signatures[node_id] = signature
        semantic_key = _stable_sha256(signature)
        semantic_keys[node_id] = semantic_key
        if semantic_key in first_node_by_key:
            canonical = first_node_by_key[semantic_key]
            cse_aliases[node_id] = canonical
        else:
            first_node_by_key[semantic_key] = node_id
        op = node["op"]
        plan_nodes.append({
            "id": node_id,
            "position": position,
            "op": op,
            "dependencies": dependencies,
            "valueType": "series[float64]",
            "outputScope": "cross-section" if op in {
                "rank", "zscore", "scale", "clip",
            } else "panel",
            "semanticKey": semantic_key,
            "cseAlias": cse_aliases.get(node_id),
        })

    factor_names = sorted({
        str(nodes[node_id].get("factor") or "").strip()
        for node_id in reachable
        if nodes[node_id]["op"] == "factor"
        and str(nodes[node_id].get("factor") or "").strip()
    })
    cse_groups = {
        key: [
            node_id for node_id in order
            if semantic_keys[node_id] == key
        ]
        for key in first_node_by_key
    }
    cse_groups = {
        key: members for key, members in cse_groups.items()
        if len(members) > 1
    }
    plan = {
        "format": GRAPH_COMPILER_VERSION,
        "graphVersion": normalized["version"],
        "nodes": plan_nodes,
        "outputs": list(normalized["outputs"]),
        "reachableNodes": len(reachable),
        "factorLeaves": factor_names,
        "operatorNodes": sum(
            1 for node_id in reachable if nodes[node_id]["op"] == "operator"
        ),
        "cseGroups": cse_groups,
        "executionBackend": (
            "cpu-topological"
            if requested_backend == "cpu"
            else "adaptive-cpu-cuda"
        ),
        "backendRequested": requested_backend,
        "gpuEligible": True,
        "gpuSupportedStages": [
            "rank-ic",
            "autocorrelation",
            "correlation",
        ],
    }
    plan["planSha256"] = _stable_sha256(plan)
    return {
        "graph": normalized,
        "plan": plan,
        "nodes": nodes,
        "reachable": reachable,
        "order": order,
        "semanticKeys": semantic_keys,
    }


def _rank_ic_metrics(
    factors: pd.DataFrame, work: pd.DataFrame, forward: pd.Series,
    *, min_names: int,
) -> dict[str, dict[str, Any]]:
    """Compute rank IC with date-contiguous, factor-block matrix operations."""
    if factors.empty:
        return {}
    dates = work["trade_date"].astype(str)
    date_codes, _unique_dates = pd.factorize(dates, sort=False)
    order = np.argsort(date_codes, kind="stable")
    sorted_codes = date_codes[order]
    starts = np.r_[
        0,
        np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1,
    ]
    ends = np.r_[starts[1:], len(sorted_codes)]

    def grouped_rank(values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        ranked_sorted = _grouped_average_rank_matrix(
            matrix[order],
            starts,
            ends,
        )
        ranked = np.empty_like(ranked_sorted)
        ranked[order] = ranked_sorted
        return ranked

    forward_rank = grouped_rank(
        forward.to_numpy(dtype=float),
    )[:, 0]
    result: dict[str, dict[str, Any]] = {}
    block_size = 64
    for offset in range(0, len(factors.columns), block_size):
        names = list(factors.columns[offset:offset + block_size])
        block = factors.iloc[:, offset:offset + block_size]
        factor_values = block.to_numpy(dtype=float)
        factor_rank = grouped_rank(factor_values)[order]
        y_rank = forward_rank[order]
        y_valid = np.isfinite(y_rank)
        valid = np.isfinite(factor_rank) & y_valid[:, None]
        x = np.where(valid, factor_rank, 0.0)
        y = np.where(valid, y_rank[:, None], 0.0)
        count = np.add.reduceat(
            valid.astype(np.int64), starts, axis=0,
        ).astype(float)
        sum_x = np.add.reduceat(x, starts, axis=0)
        sum_y = np.add.reduceat(y, starts, axis=0)
        sum_xx = np.add.reduceat(np.square(x), starts, axis=0)
        sum_yy = np.add.reduceat(np.square(y), starts, axis=0)
        sum_xy = np.add.reduceat(x * y, starts, axis=0)
        eligible = count >= int(min_names)
        safe_count = np.where(eligible, count, np.nan)
        mean_x = sum_x / safe_count
        mean_y = sum_y / safe_count
        covariance = sum_xy / safe_count - mean_x * mean_y
        variance_x = np.maximum(
            sum_xx / safe_count - np.square(mean_x),
            0.0,
        )
        variance_y = np.maximum(
            sum_yy / safe_count - np.square(mean_y),
            0.0,
        )
        denominator = np.sqrt(variance_x * variance_y)
        ic = np.divide(
            covariance,
            denominator,
            out=np.full_like(covariance, np.nan),
            where=eligible & np.isfinite(denominator) & (denominator > 0.0),
        )
        result.update(_rank_ic_result(names, factor_values, ic))
    return result


def _autocorrelation_metrics(
    factors: pd.DataFrame, stocks: pd.Series,
) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    if factors.empty:
        return output
    stock_codes, _ = pd.factorize(stocks.astype(str), sort=False)
    order = np.argsort(stock_codes, kind="stable")
    sorted_codes = stock_codes[order]
    starts = np.r_[
        0,
        np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1,
    ]
    ends = np.r_[starts[1:], len(sorted_codes)]
    block_size = 32
    for offset in range(0, len(factors.columns), block_size):
        names = list(factors.columns[offset:offset + block_size])
        values = factors.iloc[:, offset:offset + block_size].to_numpy(
            dtype=float,
        )
        sorted_values = values[order]
        ranked_sorted = _grouped_average_rank_matrix(
            sorted_values, starts, ends,
        )
        counts_by_group = np.add.reduceat(
            np.isfinite(sorted_values).astype(np.int64),
            starts,
            axis=0,
        )
        counts_by_row = np.repeat(
            counts_by_group,
            np.diff(np.r_[starts, len(sorted_values)]),
            axis=0,
        )
        ranked_sorted = np.divide(
            ranked_sorted,
            counts_by_row,
            out=np.full_like(ranked_sorted, np.nan),
            where=counts_by_row > 0,
        )
        ranked = np.empty_like(ranked_sorted)
        ranked[order] = ranked_sorted
        lagged_sorted = np.full_like(ranked_sorted, np.nan)
        if len(ranked_sorted) > 1:
            lagged_sorted[1:] = ranked_sorted[:-1]
            lagged_sorted[starts[1:]] = np.nan
        lagged = np.empty_like(lagged_sorted)
        lagged[order] = lagged_sorted
        valid = np.isfinite(ranked) & np.isfinite(lagged)
        x = np.where(valid, ranked, 0.0)
        y = np.where(valid, lagged, 0.0)
        count = valid.sum(axis=0).astype(float)
        sum_x = x.sum(axis=0)
        sum_y = y.sum(axis=0)
        sum_xx = np.square(x).sum(axis=0)
        sum_yy = np.square(y).sum(axis=0)
        sum_xy = (x * y).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            covariance = sum_xy / count - (sum_x / count) * (sum_y / count)
            variance_x = np.maximum(
                sum_xx / count - np.square(sum_x / count),
                0.0,
            )
            variance_y = np.maximum(
                sum_yy / count - np.square(sum_y / count),
                0.0,
            )
        denominator = np.sqrt(variance_x * variance_y)
        correlations = np.divide(
            covariance,
            denominator,
            out=np.full_like(covariance, np.nan),
            where=(
                (count > 2)
                & np.isfinite(denominator)
                & (denominator > 0.0)
            ),
        )
        for index, name in enumerate(names):
            correlation = correlations[index]
            output[name] = (
                round(float(correlation), 6)
                if math.isfinite(float(correlation)) else None
            )
    return output


def _rank_correlation_matrix(
    factors: pd.DataFrame,
    *,
    max_rows: int | None = None,
    engine: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute a reproducible, bounded pairwise Spearman matrix."""
    columns = [str(column) for column in factors.columns]
    if not columns:
        return pd.DataFrame(), {
            "rows": 0,
            "panelRows": len(factors),
            "sampling": "empty",
        }
    if max_rows is None:
        try:
            configured = int(os.environ.get(
                "QUANT_GRAPH_CORRELATION_ROWS", "200000",
            ))
        except ValueError:
            configured = 200_000
        max_rows = configured if configured > 0 else len(factors)
    # Keep the sampled matrix near two million cells. This bounds working
    # memory while retaining a deterministic panel-wide view for ranking.
    memory_bounded_rows = max(1_000, 2_000_000 // len(columns))
    row_limit = max(1, min(int(max_rows), memory_bounded_rows))
    if len(factors) > row_limit:
        positions = np.linspace(
            0,
            len(factors) - 1,
            row_limit,
            dtype=np.int64,
        )
        sample = factors.iloc[positions]
        sampling = "deterministic-even-sample"
    else:
        sample = factors
        positions = np.arange(len(factors), dtype=np.int64)
        sampling = "full-panel"
    raw = sample.to_numpy(dtype=float)
    if engine is not None:
        try:
            correlations, sampling_rows, common_complete = (
                engine.alpha_graph_correlation(raw)
            )
            return pd.DataFrame(
                correlations,
                index=columns,
                columns=columns,
            ), {
                "rows": sampling_rows,
                "panelRows": len(factors),
                "sampling": (
                    "common-complete-plus-deterministic-even-sample"
                    if common_complete else sampling
                ),
                "commonComplete": common_complete,
                "executionBackend": (
                    "cpu-fallback"
                    if engine.mode == "cpu-fallback" else "cuda"
                ),
            }
        except ValueError:
            # Sparse pairwise-complete columns keep the pandas reference path.
            pass
    ranked = _average_rank_matrix(raw)
    valid = np.isfinite(ranked)
    common_rows = valid.all(axis=1)
    use_common_complete = (
        len(columns) > 64
        and not common_rows.all()
        and int(common_rows.sum()) >= max(1_000, int(len(sample) * 0.5))
    )
    if valid.all() or use_common_complete:
        dense = ranked[common_rows] if use_common_complete else ranked
        centered = dense - dense.mean(axis=0, keepdims=True)
        norms = np.sqrt(np.einsum("ij,ij->j", centered, centered))
        correlations = np.divide(
            centered.T @ centered,
            norms[:, None] * norms[None, :],
            out=np.full((len(columns), len(columns)), np.nan),
            where=(norms[:, None] > 0.0) & (norms[None, :] > 0.0),
        )
        if use_common_complete:
            sampling = "common-complete-plus-deterministic-even-sample"
            sampling_rows = int(common_rows.sum())
        else:
            sampling_rows = len(sample)
        np.fill_diagonal(correlations, 1.0)
        correlations = pd.DataFrame(
            correlations,
            index=columns,
            columns=columns,
        )
    else:
        # Spearman's pairwise-complete definition ranks each pair on its own
        # valid subset.  Keep the reference implementation for sparse
        # columns; the dense fast path above is the common graph case.
        correlations = sample.corr(method="spearman")
        sampling_rows = len(sample)
    return correlations, {
        "rows": sampling_rows,
        "panelRows": len(factors),
        "sampling": (
            "common-complete-plus-deterministic-even-sample"
            if use_common_complete else sampling
        ),
        "commonComplete": bool(use_common_complete),
        "executionBackend": (
            "cpu-fallback"
            if engine is not None and engine.mode == "cpu-fallback"
            else "cpu"
        ),
    }


def evaluate_graph(frame: pd.DataFrame, graph: dict[str, Any], *,
                   horizon: int = 1, min_names: int = 5,
                   backend: str = "auto") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate a graph against an already-loaded long panel."""
    from quant_sdk.catalog_factors import compute_named_factors

    started = time.perf_counter()
    compiled = compile_graph(graph, backend=backend)
    normalized = compiled["graph"]
    plan = compiled["plan"]
    work = frame.copy()
    work["ts_code"] = work["ts_code"].astype(str)
    work["trade_date"] = work["trade_date"].astype(str)
    work = work.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    prepare_seconds = time.perf_counter() - started
    values: dict[str, pd.Series] = {}
    nodes = compiled["nodes"]
    reachable = compiled["reachable"]
    factor_names = plan["factorLeaves"]
    if factor_names:
        started = time.perf_counter()
        batch = compute_named_factors(factor_names, work)
        factor_cache = {
            name: batch[name].reset_index(drop=True).astype(float)
            for name in batch.columns
        }
        factor_seconds = time.perf_counter() - started
    else:
        factor_cache = {}
        factor_seconds = 0.0
    semantic_values: dict[str, pd.Series] = {}
    trace = {
        "reachableNodes": len(reachable),
        "factorLeaves": len(factor_names),
        "operatorNodes": plan["operatorNodes"],
        "nodesEvaluated": 0,
        "nodeCacheHits": 0,
        "semanticCacheHits": 0,
        "batchFactorLoads": 1 if factor_names else 0,
        "executionMode": "typed-topological-plan",
        "backendRequested": str(backend or "auto").strip().lower(),
        "executionBackend": "cpu",
        "gpuDevice": None,
        "gpuUploads": [],
        "gpuFallbackReason": None,
        "planSha256": plan["planSha256"],
        "cseGroupCount": len(plan["cseGroups"]),
        "timingsMs": {},
    }
    execution_started = time.perf_counter()

    rolling_groups: dict[
        tuple[str, str],
        list[tuple[str, int, int]],
    ] = defaultdict(list)
    for planned in plan["nodes"]:
        node = nodes[planned["id"]]
        if node.get("op") != "operator":
            continue
        operator = str(node.get("operator") or "").strip().lower()
        operation = {
            "ts_mean": "mean",
            "adv": "mean",
            "ts_sum": "sum",
            "ts_std": "std",
            "ts_var": "var",
        }.get(operator)
        dependencies = planned.get("dependencies") or []
        if operation is None or len(dependencies) != 1:
            continue
        stocks_parameter = node.get("stocks")
        if stocks_parameter not in (None, "ts_code"):
            continue
        try:
            window = int(node["window"])
        except (KeyError, TypeError, ValueError):
            continue
        if window < 1:
            continue
        minimum = node.get("min_periods", node.get("minPeriods", window))
        try:
            minimum = int(minimum)
        except (TypeError, ValueError):
            continue
        if not 1 <= minimum <= window:
            continue
        rolling_groups[
            (
                operation,
                compiled["semanticKeys"][dependencies[0]],
            )
        ].append((planned["id"], window, minimum))

    batched_nodes: dict[str, np.ndarray] = {}
    trace["batchedRollingGroups"] = 0
    trace["batchedRollingNodes"] = 0
    for planned in plan["nodes"]:
        node_id = planned["id"]
        if node_id in batched_nodes:
            if planned["semanticKey"] in semantic_values:
                values[node_id] = semantic_values[planned["semanticKey"]]
                trace["semanticCacheHits"] += 1
            else:
                values[node_id] = pd.Series(
                    batched_nodes[node_id], index=work.index, dtype=float,
                )
                semantic_values[planned["semanticKey"]] = values[node_id]
                trace["nodesEvaluated"] += 1
            continue
        node = nodes[node_id]
        op = node["op"]
        semantic_key = planned["semanticKey"]
        if semantic_key in semantic_values:
            values[node_id] = semantic_values[semantic_key]
            trace["semanticCacheHits"] += 1
            continue
        if op == "factor":
            factor_name = str(node.get("factor") or "").strip()
            if not factor_name:
                raise ValueError(f"{node_id} 缺少 factor")
            values[node_id] = factor_cache[factor_name]
            semantic_values[semantic_key] = values[node_id]
            trace["nodesEvaluated"] += 1
            continue
        elif op == "field":
            field = str(node.get("field") or "").strip()
            if not field:
                raise ValueError(f"{node_id} 缺少 field")
            if field not in work.columns:
                raise ValueError(f"{node_id} 引用了不存在的数据字段: {field}")
            values[node_id] = work[field].reset_index(drop=True).astype(float)
            semantic_values[semantic_key] = values[node_id]
            trace["nodesEvaluated"] += 1
            continue
        dependency_values = [values[item] for item in planned["dependencies"]]
        if op in {"add", "sub", "mul", "div"}:
            series = [item.astype(float) for item in dependency_values]
            result = series[0]
            for item in series[1:]:
                if op == "add":
                    result = result + item
                elif op == "sub":
                    result = result - item
                elif op == "mul":
                    result = result * item
                else:
                    result = result / item.replace(0.0, np.nan)
            values[node_id] = result
        elif op == "neg":
            values[node_id] = -dependency_values[0].astype(float)
        elif op == "operator":
            operator = str(node.get("operator") or "").strip().lower()
            operation = {
                "ts_mean": "mean",
                "adv": "mean",
                "ts_sum": "sum",
                "ts_std": "std",
                "ts_var": "var",
            }.get(operator)
            group_key = (
                operation,
                compiled["semanticKeys"][planned["dependencies"][0]],
            ) if operation and len(planned["dependencies"]) == 1 else None
            group = rolling_groups.get(group_key, []) if group_key else []
            if (
                group_key is not None
                and len(group) >= 2
                and node.get("stocks") in (None, "ts_code")
            ):
                input_value = dependency_values[0]
                arrays = _rolling_batch(
                    input_value.astype(float),
                    work["ts_code"],
                    windows=[item[1] for item in group],
                    min_periods=[item[2] for item in group],
                    method=operation,
                )
                for index, (group_node_id, _, _) in enumerate(group):
                    batched_nodes[group_node_id] = arrays[:, index]
                values[node_id] = pd.Series(
                    batched_nodes[node_id], index=work.index, dtype=float,
                )
                trace["batchedRollingGroups"] += 1
                trace["batchedRollingNodes"] += len(group)
            else:
                values[node_id] = _evaluate_operator(
                    node, work, dependency_values, lambda item: values[item],
                )
        elif op in {"rank", "zscore", "scale", "clip"}:
            values[node_id] = _cross_section(
                work, dependency_values[0].astype(float), op, node,
            )
        semantic_values[semantic_key] = values[node_id]
        trace["nodesEvaluated"] += 1
    graph_seconds = time.perf_counter() - execution_started
    output_started = time.perf_counter()
    output_frame = pd.DataFrame(
        {name: values[name].replace([np.inf, -np.inf], np.nan)
         for name in normalized["outputs"]},
        index=work.index,
    )
    output_frame_seconds = time.perf_counter() - output_started
    # The topological executor already shares equal nodes through CSE.  Carry
    # the same identity into the diagnostic phase: rank IC, persistence and
    # correlation are pure functions of the output series, so recalculating
    # them once per alias only makes large graphs slower without adding
    # information.
    canonical_outputs: dict[str, str] = {}
    for name in normalized["outputs"]:
        planned = next(
            item for item in plan["nodes"] if item["id"] == name
        )
        canonical_outputs[name] = planned.get("cseAlias") or name
    unique_names = list(dict.fromkeys(canonical_outputs.values()))
    unique_frame = pd.DataFrame(
        {
            name: output_frame[name]
            for name in unique_names
            if name in output_frame.columns
        },
        index=output_frame.index,
    )
    backend_resolution = _resolve_graph_engine(
        backend,
        rows=len(work),
        outputs=len(unique_names),
    )
    engine = backend_resolution["engine"]
    trace["backendSelectionReason"] = backend_resolution["reason"]
    trace["executionBackend"] = (
        "cuda-diagnostics"
        if backend_resolution["selected"] == "cuda"
        else backend_resolution["selected"]
    )
    if engine is not None:
        try:
            trace["gpuDevice"] = engine.device_info()
        except Exception:
            trace["gpuDevice"] = None
    forward = canonical_forward_returns(work, int(horizon))
    started = time.perf_counter()
    tensor_rank = (
        _rank_ic_metrics_tensor(
            engine,
            unique_frame,
            work,
            forward,
            min_names=int(min_names),
        )
        if engine is not None else None
    )
    if tensor_rank is None:
        unique_metrics = _rank_ic_metrics(
            unique_frame, work, forward, min_names=int(min_names),
        )
        if engine is not None:
            trace["executionBackend"] = "cpu-fallback"
            trace["gpuFallbackReason"] = "panel-has-duplicate-date-stock-keys"
    else:
        unique_metrics, rank_trace = tensor_rank
        if rank_trace["backend"] == "cpu-fallback":
            trace["executionBackend"] = "cpu-fallback"
            trace["gpuFallbackReason"] = rank_trace["fallbackReason"]
    rank_ic_seconds = time.perf_counter() - started
    started = time.perf_counter()
    tensor_autocorrelation = (
        _autocorrelation_metrics_tensor(engine, unique_frame, work)
        if engine is not None else None
    )
    if tensor_autocorrelation is None:
        unique_persistence = _autocorrelation_metrics(
            unique_frame, work["ts_code"],
        )
    else:
        unique_persistence, persistence_trace = tensor_autocorrelation
        if persistence_trace["backend"] == "cpu-fallback":
            trace["executionBackend"] = "cpu-fallback"
            trace["gpuFallbackReason"] = persistence_trace["fallbackReason"]
    autocorrelation_seconds = time.perf_counter() - started
    for name in unique_frame.columns:
        if name in unique_metrics:
            unique_metrics[name]["autocorrelation"] = (
                unique_persistence.get(name)
            )
    output_metrics = {
        name: dict(unique_metrics.get(canonical_outputs[name], {}))
        for name in normalized["outputs"]
    }
    started = time.perf_counter()
    correlation, correlation_trace = _rank_correlation_matrix(
        unique_frame,
        engine=engine,
    )
    correlation_seconds = time.perf_counter() - started
    if len(unique_names) != len(normalized["outputs"]):
        correlation_trace["uniqueOutputs"] = len(unique_names)
        correlation_trace["expandedOutputs"] = len(normalized["outputs"])
        correlation_trace["expansionMethod"] = "semantic-cse"
        correlation = correlation.reindex(
            index=unique_names, columns=unique_names,
        )
    correlation = correlation.round(6)
    trace["uniqueOutputCount"] = len(unique_names)
    trace["cseAliasExpanded"] = len(normalized["outputs"]) - len(unique_names)
    if engine is not None:
        trace["gpuUploads"] = list(engine.uploads)
        if engine.mode == "cpu-fallback":
            trace["executionBackend"] = "cpu-fallback"
            trace["gpuFallbackReason"] = engine.fallback_reason
    trace["timingsMs"] = {
        "prepare": round(prepare_seconds * 1000.0, 3),
        "factorLeaves": round(factor_seconds * 1000.0, 3),
        "graphExecution": round(graph_seconds * 1000.0, 3),
        "outputFrame": round(output_frame_seconds * 1000.0, 3),
        "rankIc": round(rank_ic_seconds * 1000.0, 3),
        "autocorrelation": round(autocorrelation_seconds * 1000.0, 3),
        "correlation": round(correlation_seconds * 1000.0, 3),
        "total": round(
            (
                prepare_seconds
                + factor_seconds
                + graph_seconds
                + output_frame_seconds
                + rank_ic_seconds
                + autocorrelation_seconds
                + correlation_seconds
            ) * 1000.0,
            3,
        ),
    }
    summary = {
        "version": GRAPH_VERSION,
        "status": "LOCAL_UNVERIFIED",
        "panel": {
            "rows": len(work),
            "dates": int(work["trade_date"].nunique()),
            "stocks": int(work["ts_code"].nunique()),
        },
        "horizon": int(horizon),
        "minNames": int(min_names),
        "label": label_contract(work, int(horizon)),
        "graph": normalized,
        "compiler": plan,
        "engine": trace,
        "outputs": output_metrics,
        "correlation": json.loads(correlation.to_json()),
        "correlationSampling": correlation_trace,
    }
    return output_frame, _jsonable(summary)


def evaluate_graph_file(*, data_path: str, start: str, end: str,
                        graph: dict[str, Any] | None = None,
                        factor_names: Iterable[str] | None = None,
                        output_dir: str | None = None,
                        benchmark_code: str = "000300.SH",
                        horizon: int = 1, min_names: int = 5,
                        backend: str = "auto") -> dict[str, Any]:
    if graph is None:
        graph = graph_from_factor_names(factor_names or [])
    frame = _load_frame(data_path, start, end, benchmark_code)
    values, summary = evaluate_graph(
        frame,
        graph,
        horizon=horizon,
        min_names=min_names,
        backend=backend,
    )
    summary.update({
        "start": str(start),
        "end": str(end),
        "data": str(Path(data_path).expanduser().resolve()),
        "benchmark": frame.attrs.get("benchmarkResolution"),
    })
    if output_dir:
        directory = Path(output_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        work = frame.copy().sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
        export = work[["ts_code", "trade_date"]].copy()
        export = pd.concat([export, values.reset_index(drop=True)], axis=1)
        export.to_csv(directory / "factor_values.csv", index=False)
        _write_json(directory / "summary.json", summary)
        _write_json(directory / "graph.json", summary["graph"])
        summary["artifacts"] = {
            "directory": str(directory.resolve()),
            "values": str((directory / "factor_values.csv").resolve()),
            "summary": str((directory / "summary.json").resolve()),
            "graph": str((directory / "graph.json").resolve()),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Alpha Graph evaluator")
    parser.add_argument("--data", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--spec", default=None)
    parser.add_argument("--factor-names", default="")
    parser.add_argument("--out", default=None)
    parser.add_argument("--benchmark-code", default="000300.SH")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--min-names", type=int, default=5)
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "gpu"),
        default="auto",
    )
    args = parser.parse_args()
    spec = None
    if args.spec:
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    names = [item.strip() for item in str(args.factor_names).split(",") if item.strip()]
    result = evaluate_graph_file(
        data_path=args.data, start=args.start, end=args.end,
        graph=spec, factor_names=names, output_dir=args.out,
        benchmark_code=args.benchmark_code, horizon=args.horizon,
        min_names=args.min_names, backend=args.backend,
    )
    print(json.dumps(_jsonable(result), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

