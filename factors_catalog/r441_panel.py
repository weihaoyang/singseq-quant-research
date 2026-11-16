"""Panel execution for the complete r441 registry oracle."""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factors_catalog.factor_registry import FACTOR_DEFINITIONS
from factors_catalog.r441_oracle.factors import (
    auction_turnover_imbalance_30m,
    close_drive_strength_30m,
    intraday_range_close_location_30m,
    intraday_trend_consistency_30m,
    opening_impulse_retention_30m,
    vwap_dislocation_30m,
)
from factors_catalog.r441_registry import (
    EVENT_FACTOR_INPUTS,
    RegistryFactorInputs,
    compute_registry_factor_snapshot,
)

_SIDE_COLUMNS = {
    "stock_flows": "stock_flow",
    "sector_flows": "sector_flow",
    "sector_closes": "sector_close",
    "market_closes": "market_close",
    "sector_up_counts": "sector_up_count",
    "sector_total_counts": "sector_total_count",
    "leader_prices": "leader_close",
    "laggard_prices": "laggard_close",
    "sentiment_series": "sentiment",
    "board_break_series": "board_break",
    "gap_series": "market_gap",
    "reversal_series": "market_reversal",
    "event_direction_series": "event_direction",
    "event_intensity_series": "event_intensity",
    "event_confidence_series": "event_confidence",
    "event_age_minutes": "event_age_minutes",
    "event_impact_series": "event_impact",
    "event_sector_hits": "event_sector_hits",
    "event_symbol_hits": "event_symbol_hits",
    "regulatory_event_series": "regulatory_event",
    "shareholder_event_series": "shareholder_event",
    "policy_event_series": "policy_event",
    "hotword_series": "hotword",
}

MINUTE_FACTOR_NAMES = (
    "auction_turnover_imbalance_30m",
    "close_drive_strength_30m",
    "intraday_range_close_location_30m",
    "intraday_trend_consistency_30m",
    "opening_impulse_retention_30m",
    "vwap_dislocation_30m",
)

MINUTE_ARCHIVE_END = "20260410"


def _numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _window(values: np.ndarray, begin: int, end: int) -> list[float]:
    return values[begin:end].tolist()


def compute_registry_panel(
    prepared: pd.DataFrame,
    *,
    fundamental_by_symbol_date: Mapping[
        str, Mapping[str, Mapping[str, float]]
    ] | None = None,
    enable_numba_kernels: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute all computable registry factors for a prepared date panel.

    ``prepared`` must be sorted by ``ts_code, trade_date`` and already contain
    adjusted OHLC plus ``amount``.  Optional context columns are read when
    present.  Missing context remains missing and its dependent factors are
    left unavailable instead of receiving proxy values.
    """

    required = {"ts_code", "trade_date", "open", "high", "low", "close", "amount"}
    missing = sorted(required - set(prepared.columns))
    if missing:
        raise ValueError(f"registry panel missing columns: {', '.join(missing)}")
    frame = prepared.reset_index(drop=True)
    if not frame[["ts_code", "trade_date"]].equals(
        frame[["ts_code", "trade_date"]].sort_values(
            ["ts_code", "trade_date"], kind="mergesort"
        )
    ):
        raise ValueError("registry panel must be sorted by ts_code, trade_date")
    if "benchmark_close" not in frame:
        raise ValueError("registry panel missing benchmark_close")

    columns = sorted(FACTOR_DEFINITIONS)
    output = {
        name: np.full(len(frame), np.nan, dtype=np.float32)
        for name in columns
    }
    arrays = {
        name: _numeric_array(frame, name)
        for name in {
            "open",
            "high",
            "low",
            "close",
            "amount",
            "benchmark_close",
            *_SIDE_COLUMNS.values(),
        }
        if name in frame
    }
    dates = frame["trade_date"].astype(str).to_numpy()
    groups = frame.groupby("ts_code", sort=False).indices

    for raw_symbol, raw_positions in groups.items():
        symbol = str(raw_symbol)
        positions = np.asarray(raw_positions, dtype=np.int64)
        fundamentals = (
            (fundamental_by_symbol_date or {}).get(symbol, {})
        )
        for offset, position in enumerate(positions):
            begin = max(0, offset - 127)
            end = offset + 1
            side = {
                key: _window(values, begin, end)
                for key, column in _SIDE_COLUMNS.items()
                if (values := arrays.get(column)) is not None
            }
            day = str(dates[position])
            factors = compute_registry_factor_snapshot(
                RegistryFactorInputs(
                    prices=_window(arrays["close"], begin, end),
                    turnover=_window(arrays["amount"], begin, end),
                    opens=_window(arrays["open"], begin, end),
                    highs=_window(arrays["high"], begin, end),
                    lows=_window(arrays["low"], begin, end),
                    benchmark_prices=_window(
                        arrays["benchmark_close"], begin, end
                    ),
                    fundamental_factors=dict(fundamentals.get(day, {})),
                    enable_numba_kernels=enable_numba_kernels,
                    **side,
                )
            )
            for name, value in factors.items():
                output[name][position] = value

    generated = pd.DataFrame(output, index=frame.index)
    names = [
        name for name in columns
        if np.isfinite(generated[name].to_numpy(dtype=float)).any()
    ]
    return generated[names].copy(), names


def registry_panel_metadata(
    prepared: pd.DataFrame,
    *,
    fundamental_by_symbol_date: Mapping[
        str, Mapping[str, Mapping[str, float]]
    ] | None = None,
) -> dict[str, Any]:
    """Describe context availability without executing the factor oracle."""

    side_columns = {
        key: column
        for key, column in _SIDE_COLUMNS.items()
        if column in prepared
    }
    return {
        "rows": len(prepared),
        "symbols": int(prepared["ts_code"].nunique()),
        "side_columns": side_columns,
        "fundamental_symbols": len(fundamental_by_symbol_date or {}),
        "all_registry_names": len(FACTOR_DEFINITIONS),
    }


def registry_availability_report(
    prepared: pd.DataFrame,
    *,
    computed_names: set[str] | list[str],
    fundamental_by_symbol_date: Mapping[
        str, Mapping[str, Mapping[str, float]]
    ] | None = None,
    start: str,
    end: str,
    minute_source_root: str | None = None,
) -> dict[str, Any]:
    """Classify every registered factor with a concrete input/coverage reason."""

    computed = {str(name) for name in computed_names}
    available_inputs = {
        "prices_1d",
        "opens_1d",
        "lows_1d",
        "turnover_1d",
        "benchmark_prices",
    }
    input_columns = {
        "stock_flows": "stock_flow",
        "sector_flows": "sector_flow",
        "sector_closes": "sector_close",
        "market_closes": "market_close",
        "sector_up_counts": "sector_up_count",
        "sector_total_counts": "sector_total_count",
        "leader_prices": "leader_close",
        "laggard_prices": "laggard_close",
        "sentiment_proxy": "sentiment",
        "event_direction_series": "event_direction",
        "event_intensity_series": "event_intensity",
        "event_confidence_series": "event_confidence",
        "event_age_minutes": "event_age_minutes",
        "event_impact_series": "event_impact",
        "event_sector_hits": "event_sector_hits",
        "event_symbol_hits": "event_symbol_hits",
        "regulatory_event_series": "regulatory_event",
        "shareholder_event_series": "shareholder_event",
        "policy_event_series": "policy_event",
        "hotword_series": "hotword",
    }
    for input_name, column in input_columns.items():
        if column in prepared.columns and np.isfinite(
            _numeric_array(prepared, column)
        ).any():
            available_inputs.add(input_name)
    if "board_break" in prepared.columns and np.isfinite(
        _numeric_array(prepared, "board_break")
    ).any():
        available_inputs.add("sentiment_proxy")
    if fundamental_by_symbol_date:
        available_inputs.update(
            {
                "financial_income",
                "financial_balancesheet",
                "financial_cashflow",
            }
        )

    def required_inputs(
        name: str,
        seen: frozenset[str] = frozenset(),
    ) -> set[str]:
        if name in seen:
            return {f"circular_dependency:{name}"}
        event_inputs = EVENT_FACTOR_INPUTS.get(name)
        if event_inputs is not None:
            return set(event_inputs)
        definition = FACTOR_DEFINITIONS.get(name)
        if definition is None:
            return {name}
        missing: set[str] = set()
        for dependency in definition.dependencies:
            if dependency in FACTOR_DEFINITIONS:
                missing.update(
                    required_inputs(
                        dependency,
                        seen | {name},
                    )
                )
            else:
                missing.add(str(dependency))
        return missing

    try:
        start_day = str(start).replace("-", "")[:8]
        end_day = str(end).replace("-", "")[:8]
    except (TypeError, ValueError):
        start_day = str(start)
        end_day = str(end)
    rows: dict[str, dict[str, Any]] = {}
    status_counts: dict[str, int] = {}
    for name, definition in sorted(FACTOR_DEFINITIONS.items()):
        required = required_inputs(name)
        missing = sorted(required - available_inputs)
        coverage = None
        if name in MINUTE_FACTOR_NAMES:
            coverage = {
                "kind": "intraday_30m",
                "requested_start": start_day,
                "requested_end": end_day,
                "published_through": MINUTE_ARCHIVE_END,
                "source_root": minute_source_root,
            }
            if minute_source_root is None:
                status = "UNAVAILABLE"
                reason = (
                    "minute_source_unavailable: "
                    "QUANT_MINUTE_ARCHIVE_ROOT is not a directory"
                )
            elif name not in computed:
                status = "UNAVAILABLE"
                reason = (
                    "minute_source_present_but_factor_returned_no_finite_values: "
                    f"{start_day}..{end_day}"
                )
            elif end_day > MINUTE_ARCHIVE_END:
                status = "PARTIAL"
                reason = (
                    "minute_source_published_through="
                    f"{MINUTE_ARCHIVE_END}; requested_end={end_day}; "
                    f"uncovered_from={int(MINUTE_ARCHIVE_END) + 1!s}"
                )
            else:
                status = "COMPLETE"
                reason = (
                    f"minute_source_covers_{start_day}_{end_day}"
                )
        elif name in computed and not missing:
            status = "COMPLETE"
            reason = "real_registry_oracle_values_available"
        elif missing:
            status = "UNAVAILABLE"
            reason = "missing_required_inputs=" + ",".join(missing)
        else:
            status = "FAILED"
            reason = (
                "declared_inputs_present_but_registry_oracle_returned_"
                "no_finite_values"
            )
        row = {
            "factor_name": name,
            "family": definition.family,
            "dependencies": list(required),
            "missing_inputs": missing,
            "status": status,
            "reason": reason,
        }
        if coverage is not None:
            row["coverage"] = coverage
        rows[name] = row
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "window": {"start": start_day, "end": end_day},
        "minute_archive_end": MINUTE_ARCHIVE_END,
        "minute_source_root": minute_source_root,
        "total": len(rows),
        "computed": len(computed),
        "status_counts": dict(sorted(status_counts.items())),
        "rows": rows,
    }


def compute_registry_arrays(
    prepared: pd.DataFrame,
    output_directory: str | Path,
    *,
    fundamental_by_symbol_date: Mapping[
        str, Mapping[str, Mapping[str, float]]
    ] | None = None,
    enable_numba_kernels: bool = True,
    checkpoint_identity: str = "",
) -> list[str]:
    """Execute the registry oracle directly into per-factor ``.npy`` files."""

    required = {"ts_code", "trade_date", "open", "high", "low", "close", "amount"}
    missing = sorted(required - set(prepared.columns))
    if missing:
        raise ValueError(f"registry panel missing columns: {', '.join(missing)}")
    if "benchmark_close" not in prepared:
        raise ValueError("registry panel missing benchmark_close")
    frame = prepared.reset_index(drop=True)
    if not frame[["ts_code", "trade_date"]].equals(
        frame[["ts_code", "trade_date"]].sort_values(
            ["ts_code", "trade_date"], kind="mergesort"
        )
    ):
        raise ValueError("registry panel must be sorted by ts_code, trade_date")

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "r441-registry.checkpoint.json"
    completed_symbols: set[str] = set()
    saved_names: list[str] = []
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint_identity
            and checkpoint.get("identity") != checkpoint_identity
        ):
            raise ValueError("r441_registry_checkpoint_identity_mismatch")
        completed_symbols = {
            str(value) for value in checkpoint.get("completed_symbols", [])
        }
        saved_names = [
            str(value) for value in checkpoint.get("names", [])
        ]
    arrays = {
        name: _numeric_array(frame, name)
        for name in {
            "open",
            "high",
            "low",
            "close",
            "amount",
            "benchmark_close",
            *_SIDE_COLUMNS.values(),
        }
        if name in frame
    }
    dates = frame["trade_date"].astype(str).to_numpy()
    outputs: dict[str, np.memmap] = {}
    for name in saved_names:
        key = hashlib.sha256(name.encode()).hexdigest()
        path = directory / f"{key}.npy"
        if path.is_file():
            outputs[name] = np.lib.format.open_memmap(
                path, mode="r+"
            )
    groups = frame.groupby("ts_code", sort=False).indices

    for raw_symbol, raw_positions in groups.items():
        symbol = str(raw_symbol)
        if symbol in completed_symbols:
            continue
        positions = np.asarray(raw_positions, dtype=np.int64)
        fundamentals = (
            (fundamental_by_symbol_date or {}).get(symbol, {})
        )
        for offset, position in enumerate(positions):
            begin = max(0, offset - 127)
            end = offset + 1
            side = {
                key: _window(values, begin, end)
                for key, column in _SIDE_COLUMNS.items()
                if (values := arrays.get(column)) is not None
            }
            factors = compute_registry_factor_snapshot(
                RegistryFactorInputs(
                    prices=_window(arrays["close"], begin, end),
                    turnover=_window(arrays["amount"], begin, end),
                    opens=_window(arrays["open"], begin, end),
                    highs=_window(arrays["high"], begin, end),
                    lows=_window(arrays["low"], begin, end),
                    benchmark_prices=_window(
                        arrays["benchmark_close"], begin, end
                    ),
                    fundamental_factors=dict(
                        fundamentals.get(str(dates[position]), {})
                    ),
                    enable_numba_kernels=enable_numba_kernels,
                    **side,
                )
            )
            for name, value in factors.items():
                target = outputs.get(name)
                if target is None:
                    key = hashlib.sha256(name.encode()).hexdigest()
                    path = directory / f"{key}.npy"
                    if path.is_file():
                        target = np.lib.format.open_memmap(path, mode="r+")
                    else:
                        target = np.lib.format.open_memmap(
                            path,
                            mode="w+",
                            dtype=np.float32,
                            shape=(len(frame),),
                        )
                        target[:] = np.nan
                    outputs[name] = target
                target[position] = value
        completed_symbols.add(symbol)
        if len(completed_symbols) % 10 == 0:
            for target in outputs.values():
                target.flush()
            temporary = checkpoint_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "identity": checkpoint_identity,
                        "completed_symbols": sorted(completed_symbols),
                        "names": sorted(outputs),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            temporary.replace(checkpoint_path)

    for target in outputs.values():
        target.flush()
    names = sorted(outputs)
    temporary = checkpoint_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "identity": checkpoint_identity,
                "completed_symbols": sorted(completed_symbols),
                "names": names,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(checkpoint_path)
    return names


_PARALLEL_PANEL: pd.DataFrame | None = None
_PARALLEL_FUNDAMENTALS: Mapping[
    str, Mapping[str, Mapping[str, float]]
] | None = None
_PARALLEL_ROOT: Path | None = None
_PARALLEL_MINUTE_ROOT: str | None = None
_PARALLEL_START: str = ""
_PARALLEL_MINUTE_END: str = ""


def _minute_factor_rows(frame: pd.DataFrame) -> dict[str, float]:
    ordered = frame.sort_values("trade_time", kind="mergesort")
    opens = pd.to_numeric(ordered["open"], errors="coerce").tolist()
    highs = pd.to_numeric(ordered["high"], errors="coerce").tolist()
    lows = pd.to_numeric(ordered["low"], errors="coerce").tolist()
    closes = pd.to_numeric(ordered["close"], errors="coerce").tolist()
    volume = pd.to_numeric(ordered["vol"], errors="coerce")
    amount = pd.to_numeric(ordered["amount"], errors="coerce")
    vwap = (amount / (volume * 100.0)).where(
        (volume > 0.0) & (amount > 0.0),
        pd.Series(closes, index=ordered.index, dtype=float),
    )
    vwap_values = vwap.tolist()
    return {
        "auction_turnover_imbalance_30m": auction_turnover_imbalance_30m(amount.tolist()),
        "close_drive_strength_30m": close_drive_strength_30m(closes),
        "intraday_range_close_location_30m": intraday_range_close_location_30m(
            highs, lows, closes
        ),
        "intraday_trend_consistency_30m": intraday_trend_consistency_30m(closes),
        "opening_impulse_retention_30m": opening_impulse_retention_30m(
            opens, closes
        ),
        "vwap_dislocation_30m": vwap_dislocation_30m(
            closes, vwap_values
        ),
    }


def _compute_symbol_minute_factors(
    store: Any,
    *,
    symbol: str,
    dates: set[str],
    output: dict[str, np.ndarray],
    positions: Mapping[str, int],
) -> None:
    if not dates:
        return
    frame = store.load(
        [symbol], _PARALLEL_START, _PARALLEL_MINUTE_END, freq="30min"
    )
    if frame.empty:
        return
    frame = frame.copy()
    frame["_day"] = (
        frame["trade_time"].astype(str).str.slice(0, 10).str.replace(
            "-", "", regex=False
        )
    )
    for day, group in frame.groupby("_day", sort=False):
        day = str(day)
        if day not in dates or len(group) < 2:
            continue
        position = positions.get(day)
        if position is None:
            continue
        values = _minute_factor_rows(group)
        for name, value in values.items():
            output[name][position] = value


def _compute_partition(payload: tuple[int, list[str]]) -> dict[str, Any]:
    part, symbols = payload
    if _PARALLEL_PANEL is None or _PARALLEL_ROOT is None:
        raise RuntimeError("r441 registry worker was not initialized")
    part_root = _PARALLEL_ROOT / f"part-{part:03d}"
    complete = part_root / "complete.json"
    if complete.is_file():
        return json.loads(complete.read_text(encoding="utf-8"))
    part_root.mkdir(parents=True, exist_ok=True)
    selected = _PARALLEL_PANEL[
        _PARALLEL_PANEL["ts_code"].astype(str).isin(symbols)
    ].copy()
    positions = selected["_registry_global_position"].to_numpy(dtype=np.int64)
    selected = selected.drop(columns=["_registry_global_position"]).reset_index(
        drop=True
    )
    fundamentals = {
        symbol: (_PARALLEL_FUNDAMENTALS or {}).get(symbol, {})
        for symbol in symbols
    }
    generated, names = compute_registry_panel(
        selected,
        fundamental_by_symbol_date=fundamentals,
        enable_numba_kernels=True,
    )
    if _PARALLEL_MINUTE_ROOT:
        from data_foundation.minute import LocalMinuteStore

        store = LocalMinuteStore(_PARALLEL_MINUTE_ROOT)
        minute_output = {
            name: np.full(len(selected), np.nan, dtype=np.float32)
            for name in (
                "auction_turnover_imbalance_30m",
                "close_drive_strength_30m",
                "intraday_range_close_location_30m",
                "intraday_trend_consistency_30m",
                "opening_impulse_retention_30m",
                "vwap_dislocation_30m",
            )
        }
        for symbol in symbols:
            symbol_positions = selected.index[
                selected["ts_code"].astype(str) == symbol
            ].to_list()
            if not symbol_positions:
                continue
            positions_by_day = {
                str(selected.at[position, "trade_date"]): int(position)
                for position in symbol_positions
            }
            _compute_symbol_minute_factors(
                store,
                symbol=symbol,
                dates=set(positions_by_day),
                output=minute_output,
                positions=positions_by_day,
            )
        for name, values in minute_output.items():
            if np.isfinite(values).any():
                generated[name] = values
                names.append(name)
    np.save(part_root / "positions.npy", positions)
    for name in names:
        np.save(
            part_root / f"{hashlib.sha256(name.encode()).hexdigest()}.npy",
            generated[name].to_numpy(dtype=np.float32),
        )
    result = {
        "part": part,
        "names": names,
        "rows": len(selected),
        "first_position": int(positions.min()) if len(positions) else None,
        "last_position": int(positions.max()) if len(positions) else None,
    }
    temporary = complete.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(result, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(complete)
    return result


def _balanced_symbol_parts(
    panel: pd.DataFrame, workers: int
) -> list[list[str]]:
    groups = panel.groupby("ts_code", sort=False).size().sort_values(
        ascending=False
    )
    parts: list[list[str]] = [[] for _ in range(workers)]
    loads = [0] * workers
    for symbol, count in groups.items():
        index = min(range(workers), key=lambda item: loads[item])
        parts[index].append(str(symbol))
        loads[index] += int(count)
    return [sorted(part) for part in parts if part]


def compute_registry_arrays_parallel(
    prepared: pd.DataFrame,
    output_directory: str | Path,
    *,
    fundamental_by_symbol_date: Mapping[
        str, Mapping[str, Mapping[str, float]]
    ] | None = None,
    workers: int = 6,
    checkpoint_identity: str = "",
    minute_source_root: str | None = None,
    start: str = "",
    end: str = "",
) -> list[str]:
    """Compute registry arrays in stock partitions and merge by global row."""

    if not 1 <= int(workers) <= 8:
        raise ValueError("registry workers must be between 1 and 8")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "r441-registry.checkpoint.json"
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint_identity
            and checkpoint.get("identity") != checkpoint_identity
        ):
            raise ValueError("r441_registry_checkpoint_identity_mismatch")
        if checkpoint.get("mode") == "parallel" and checkpoint.get("complete"):
            names = [str(value) for value in checkpoint.get("names", [])]
            if all(
                (directory / f"{hashlib.sha256(name.encode()).hexdigest()}.npy").is_file()
                for name in names
            ):
                return names

    working = prepared.reset_index(drop=True).copy()
    working["_registry_global_position"] = np.arange(len(working), dtype=np.int64)
    split = _balanced_symbol_parts(working, int(workers))
    part_root = directory / "r441-registry-parts"
    part_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    global _PARALLEL_PANEL, _PARALLEL_FUNDAMENTALS, _PARALLEL_ROOT
    global _PARALLEL_MINUTE_ROOT, _PARALLEL_START, _PARALLEL_MINUTE_END
    _PARALLEL_PANEL = working
    _PARALLEL_FUNDAMENTALS = fundamental_by_symbol_date or {}
    _PARALLEL_ROOT = part_root
    _PARALLEL_MINUTE_ROOT = minute_source_root
    _PARALLEL_START = str(start or working["trade_date"].min())
    _PARALLEL_MINUTE_END = min(
        str(end or working["trade_date"].max()),
        "20260410",
    )
    execution_mode = "process-fork" if os.name == "posix" else "inline"
    if execution_mode == "process-fork":
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=len(split),
            mp_context=context,
        ) as executor:
            pending = {
                executor.submit(_compute_partition, (index, symbols)): index
                for index, symbols in enumerate(split)
            }
            for future in as_completed(pending):
                results.append(future.result())
    else:
        # Windows has no fork context. The public research client must still
        # produce one deterministic result; partition work inline rather than
        # hiding a platform-only failure behind the Linux batch runner.
        results.extend(
            _compute_partition((index, symbols))
            for index, symbols in enumerate(split)
        )
    results.sort(key=lambda item: int(item["part"]))

    names = sorted({
        str(name)
        for result in results
        for name in result.get("names", [])
    })
    for name in names:
        target_path = directory / f"{hashlib.sha256(name.encode()).hexdigest()}.npy"
        target = np.lib.format.open_memmap(
            target_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(working),),
        )
        target[:] = np.nan
        target.flush()
        del target
    for result in results:
        part = int(result["part"])
        part_dir = part_root / f"part-{part:03d}"
        positions = np.load(part_dir / "positions.npy")
        for name in result.get("names", []):
            target_path = (
                directory
                / f"{hashlib.sha256(str(name).encode()).hexdigest()}.npy"
            )
            target = np.lib.format.open_memmap(target_path, mode="r+")
            target[positions] = np.load(
                part_dir
                / f"{hashlib.sha256(str(name).encode()).hexdigest()}.npy"
            )
            target.flush()
            del target
    temporary = checkpoint_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "identity": checkpoint_identity,
                "mode": "parallel",
                "complete": True,
                "workers": len(split),
                "execution_mode": execution_mode,
                "names": names,
                "parts": results,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(checkpoint_path)
    return names
