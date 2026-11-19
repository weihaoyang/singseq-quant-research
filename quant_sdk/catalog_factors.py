"""Run named factors that have an implementation in the shipped factor catalog.

Named factors execute trusted toolkit code against a researcher-owned panel. Their
results remain local and unverified; catalog metadata without an implementation is
rejected instead of being converted into a guessed formula.
"""
from __future__ import annotations

import inspect
import json
import os
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_LABEL_COLUMNS = (
    "benchmark_ret",
    "benchmark_basis",
    "benchmark_grade",
    "target_ret",
    "fwd_ret",
    "forward_return",
)
_BAR_COLUMNS = ("open", "high", "low", "close")
_FACTOR_MANIFEST_FORMAT = "quant-factor-capability-manifest/1"
_FACTOR_MANIFEST_PATH = Path(__file__).with_name(
    "factor_capability_manifest.json",
)
_FORMULA_FIELD_MAP = {
    "prices": "close",
    "highs": "high",
    "lows": "low",
    "closes": "close",
    "opens": "open",
    "opens_1d": "open",
    "closes_1d": "close",
    "turnover": "amount",
}
_FACTOR_INPUT_COLUMNS = {
    "open": ("open",),
    "high": ("high",),
    "low": ("low",),
    "close": ("close",),
    "amount": ("amount",),
    "volume": ("vol",),
    "vwap": ("amount", "vol"),
    "returns": ("close",),
    "ret": ("close",),
    "banchmarkindexclose": ("close",),
    "banchmarkindexopen": ("open",),
    "cap": ("circ_mv",),
}


def _prepared_panel(frame: pd.DataFrame) -> pd.DataFrame:
    missing = {"ts_code", "trade_date", *_BAR_COLUMNS}.difference(frame.columns)
    if missing:
        raise ValueError(f"面板缺少具名因子所需列: {', '.join(sorted(missing))}")
    panel = frame.drop(columns=[name for name in _LABEL_COLUMNS if name in frame.columns],
                       errors="ignore").copy()
    panel["ts_code"] = panel["ts_code"].astype(str)
    panel["trade_date"] = panel["trade_date"].astype(str)
    panel["vol"] = panel.get("vol", np.nan)
    panel["amount"] = panel.get("amount", np.nan)
    for column in ("open", "high", "low", "close", "vol", "amount"):
        if column in panel:
            panel[column] = pd.to_numeric(
                panel[column], errors="coerce",
            ).astype(float)
    if "adj_close" in panel:
        close = panel["close"].replace(0.0, np.nan)
        adjustment = pd.to_numeric(panel["adj_close"], errors="coerce") / close
        for column in ("open", "high", "low"):
            panel[column] = pd.to_numeric(panel[column], errors="coerce") * adjustment
        panel["close"] = pd.to_numeric(panel["adj_close"], errors="coerce")
    panel = panel.sort_values(["ts_code", "trade_date"], kind="mergesort")
    panel["stock_id"] = panel["ts_code"]
    panel["volume"] = panel["vol"]
    return panel.reset_index(drop=True)


@lru_cache(maxsize=1)
def _registry_formulas() -> dict[str, Any]:
    from factors_catalog import r441_daily_formulas
    from factors_catalog.factor_registry import FACTOR_DEFINITIONS

    return {
        name: getattr(r441_daily_formulas, name)
        for name in FACTOR_DEFINITIONS
        if callable(getattr(r441_daily_formulas, name, None))
    }


def _registry_required_columns(name: str) -> list[str]:
    function = _registry_formulas().get(name)
    if function is None:
        return list(_BAR_COLUMNS)
    return sorted({
        _FORMULA_FIELD_MAP[parameter]
        for parameter in inspect.signature(function).parameters
        if parameter in _FORMULA_FIELD_MAP
    })


@lru_cache(maxsize=1)
def _factor_manifest() -> dict[str, dict[str, Any]] | None:
    if os.environ.get("QUANT_FACTOR_DISCOVERY", "").strip().lower() in {
        "dynamic", "source", "legacy",
    }:
        return None
    try:
        payload = json.loads(_FACTOR_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("format") != _FACTOR_MANIFEST_FORMAT
        or not isinstance(payload.get("factors"), list)
    ):
        return None
    manifest: dict[str, dict[str, Any]] = {}
    for item in payload["factors"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            return None
        name = item["name"]
        if not name or name in manifest:
            return None
        manifest[name] = item
    return manifest


def _manifest_supports(
    item: dict[str, Any],
    fields: set[str],
    history: int,
) -> bool:
    if item.get("executionValidated") is False:
        return False
    required = item.get("requiredColumns")
    if not isinstance(required, list):
        required = item.get("inputs")
    try:
        minimum_history = max(0, int(item.get("minimumHistory") or 0))
    except (TypeError, ValueError):
        return False
    return (
        isinstance(required, list)
        and set(str(value) for value in required).issubset(fields)
        and history >= minimum_history
    )


def _required_columns_from_inputs(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        lower = normalized.lower()
        mapped = _FACTOR_INPUT_COLUMNS.get(lower)
        if mapped is None and lower.startswith("adv"):
            mapped = ("amount", "vol")
        result.extend(mapped or (normalized,))
    return list(dict.fromkeys(item for item in result if item))


def _compute_registry_factor(name: str, panel: pd.DataFrame) -> pd.Series:
    function = _registry_formulas()[name]
    parameters = list(inspect.signature(function).parameters)
    missing: list[str] = []
    for parameter in parameters:
        column = _FORMULA_FIELD_MAP.get(parameter)
        if column is None:
            missing.append(parameter)
        elif column not in panel:
            missing.append(column)
    if missing:
        raise ValueError(f"因子 {name} 缺少输入字段: {', '.join(sorted(set(missing)))}")

    output = np.full(len(panel), np.nan, dtype=float)
    groups = panel.groupby("ts_code", sort=False).indices
    for positions in groups.values():
        positions = np.asarray(positions, dtype=np.int64)
        values = {param: pd.to_numeric(
            panel.iloc[positions][_FORMULA_FIELD_MAP[param]],
            errors="coerce",
        ).to_numpy(dtype=float)
                  for param in parameters if param in _FORMULA_FIELD_MAP}
        for offset, position in enumerate(positions):
            end = offset + 1
            # The frozen r441 daily formulas consume at most 40 returns
            # (41 prices). Avoid copying an ever-growing history for every bar.
            begin = max(0, end - 41)
            args = {param: series[begin:end].tolist() for param, series in values.items()}
            try:
                result = float(function(**args))
            except (ArithmeticError, TypeError, ValueError, ZeroDivisionError):
                continue
            if np.isfinite(result):
                output[position] = result
    if not np.isfinite(output).any():
        raise ValueError(f"因子 {name} 在当前面板没有有效输出")
    return pd.Series(output, index=panel.index, name=name)


def _compute_registry_factors(
    panel: pd.DataFrame,
    names: list[str],
) -> dict[str, pd.Series]:
    """Compute registry factors through whole-history kernels when economical."""
    formulas = _registry_formulas()
    requested = [name for name in names if name in formulas]
    if not requested:
        return {}
    try:
        minimum_bulk = max(1, int(os.environ.get("QUANT_REGISTRY_BULK_MIN", "16")))
    except ValueError:
        minimum_bulk = 16
    try:
        minimum_work = max(1, int(os.environ.get(
            "QUANT_REGISTRY_BULK_WORK", "250000",
        )))
    except ValueError:
        minimum_work = 250_000
    output: dict[str, pd.Series] = {}
    if (
        len(requested) >= minimum_bulk
        and len(panel) * len(requested) >= minimum_work
        and {"ts_code", "trade_date", "amount"}.issubset(panel.columns)
    ):
        try:
            from factors_catalog.r441_bulk import (
                compute_registry_panel_bulk,
                numba_available,
            )

            if numba_available():
                bulk_panel = panel
                if "benchmark_close" not in bulk_panel.columns:
                    bulk_panel = bulk_panel.copy()
                    if "benchmark_ret" in bulk_panel.columns:
                        daily = (
                            pd.to_numeric(
                                bulk_panel["benchmark_ret"],
                                errors="coerce",
                            )
                            .groupby(bulk_panel["trade_date"], sort=True)
                            .first()
                        )
                        benchmark_close = (1.0 + daily.fillna(0.0)).cumprod()
                        bulk_panel["benchmark_close"] = (
                            bulk_panel["trade_date"].map(benchmark_close)
                        )
                    else:
                        bulk_panel["benchmark_close"] = np.nan
                generated, produced = compute_registry_panel_bulk(
                    bulk_panel,
                )
                produced_set = set(produced)
                for name in requested:
                    if name in produced_set:
                        output[name] = pd.to_numeric(
                            generated[name], errors="coerce",
                        ).astype(float).reset_index(drop=True)
        except (ImportError, RuntimeError, TypeError, ValueError):
            output.clear()
    for name in requested:
        if name not in output:
            output[name] = _compute_registry_factor(name, panel)
    return output


def _family_factor(name: str, panel: pd.DataFrame) -> pd.Series | None:
    from factors_catalog.alpha101_operators import compute_alpha101_selected
    from factors_catalog.alpha158 import compute_alpha158
    from factors_catalog.alpha360 import compute_alpha360
    from factors_catalog.factor_zoo import compute_factor_zoo
    from factors_catalog.ohlc_volatility import compute_ohlc_volatility

    family = factor_family(name)
    candidates = []
    if family == "alpha158":
        candidates.append(compute_alpha158)
    elif family == "alpha360":
        candidates.append(compute_alpha360)
    elif family in {"alpha101", "alpha"}:
        candidates.append(compute_alpha101_selected)
    else:
        candidates.extend((
            compute_factor_zoo,
            compute_ohlc_volatility,
            compute_alpha158,
        ))
    for compute in candidates:
        frame, names = compute(panel)
        if name in names:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).any():
                raise ValueError(f"因子 {name} 在当前面板没有有效输出")
            return pd.Series(values, index=panel.index, name=name)
    return None


def compute_named_factor(name: str, frame: pd.DataFrame) -> pd.Series:
    """Compute one implemented catalog factor, aligned to sorted panel rows."""
    normalized = str(name or "").strip()
    if not normalized:
        raise ValueError("需要提供因子名")
    return compute_named_factors([normalized], frame).iloc[:, 0].rename(normalized)


def compute_named_factors(
    names: list[str] | tuple[str, ...],
    frame: pd.DataFrame,
    *,
    workers: int | None = None,
) -> pd.DataFrame:
    """Compute named factors in one pass per implementation family.

    The public single-factor API remains the compatibility boundary.  Batch
    callers use this function so Alpha360, Alpha101/191, Alpha158 and Factor Zoo
    do not repeatedly prepare the same panel, sort it and rebuild the same
    rolling inputs for every requested output.
    """
    requested = list(dict.fromkeys(
        str(name or "").strip()
        for name in names
        if str(name or "").strip()
    ))
    if not requested:
        return pd.DataFrame(index=frame.index)
    panel = _prepared_panel(frame)
    output: dict[str, pd.Series] = {}

    from quant_sdk.capabilities import (
        compute_pack_factor,
        installed_factor_names,
    )

    installed = set(installed_factor_names(fields=set(panel.columns)))
    pack_factors = [name for name in requested if name in installed]
    for name in pack_factors:
        try:
            output[name] = compute_pack_factor(name, panel)
        except Exception as error:
            raise ValueError(f"Factor Pack 因子 {name} 计算失败：{error}") from error

    external = [
        name for name in requested
        if name not in output
        and name.lower().startswith(("alpha101_", "alpha191_"))
    ]
    if external:
        from factors_catalog.open_adapters import compute_external_factors

        try:
            batch = compute_external_factors(
                external,
                panel,
                workers=workers,
            )
        except (ImportError, RuntimeError, ValueError) as error:
            raise ValueError(f"外部因子批处理失败；{error}") from error
        output.update({name: batch[name] for name in external})

    alpha360 = [
        name for name in requested
        if name not in output
        and name.lower().startswith("alpha360_")
    ]
    if alpha360:
        from factors_catalog.alpha360 import compute_alpha360_features

        batch = compute_alpha360_features(alpha360, panel)
        output.update({name: batch[name] for name in alpha360})

    remaining = [
        name for name in requested
        if name not in output
    ]
    registry = _compute_registry_factors(panel, remaining)
    if registry:
        output.update(registry)
        remaining = [name for name in remaining if name not in registry]

    if remaining:
        from factors_catalog.alpha101_operators import compute_alpha101_selected
        from factors_catalog.alpha158 import compute_alpha158
        from factors_catalog.factor_zoo import compute_factor_zoo
        from factors_catalog.ohlc_volatility import compute_ohlc_volatility

        remaining_set = set(remaining)
        remaining_families = {
            factor_family(name) for name in remaining_set
        }
        family_computes = []
        if "alpha158" in remaining_families:
            family_computes.append(compute_alpha158)
        if remaining_families.intersection({"alpha", "alpha101"}):
            family_computes.append(compute_alpha101_selected)
        unresolved_families = remaining_families.difference({
            "alpha", "alpha101", "alpha158", "alpha360",
            "talib", "mytt", "pta", "ta", "qtl", "talipp", "pyti",
        })
        if unresolved_families:
            family_computes.extend((
                compute_factor_zoo,
                compute_ohlc_volatility,
            ))
        for compute in family_computes:
            if not remaining_set:
                break
            family_frame, produced = compute(panel)
            matched = remaining_set.intersection(produced)
            for name in matched:
                output[name] = pd.to_numeric(
                    family_frame[name], errors="coerce",
                ).astype(float)
            remaining_set.difference_update(matched)
        remaining = [name for name in remaining if name in remaining_set]

    technical_names = [
        name for name in remaining
        if name.lower().startswith(
            ("talib_", "mytt_", "pta_", "ta_", "qtl_", "talipp_", "pyti_")
        )
    ]
    if technical_names:
        from factors_catalog.technical_adapters import (
            compute_technical_factors,
        )

        try:
            technical = compute_technical_factors(
                technical_names,
                panel,
                workers=workers,
            )
        except (ImportError, RuntimeError, ValueError) as error:
            raise ValueError(f"技术指标批处理失败；{error}") from error
        output.update({
            name: technical[name].reset_index(drop=True)
            for name in technical_names
        })
        remaining = [
            name for name in remaining
            if name not in technical_names
        ]

    for name in remaining:
        try:
            from factors_catalog.open_adapters import compute_external_factor

            output[name] = compute_external_factor(name, panel)
        except (ImportError, RuntimeError, ValueError) as error:
            raise ValueError(f"因子 {name} 尚无可运行实现；{error}") from error

    return pd.DataFrame(
        {name: output[name] for name in requested},
        index=panel.index,
    ).replace([np.inf, -np.inf], np.nan)


@lru_cache(maxsize=1)
def _registry_family_map() -> dict[str, str]:
    from factors_catalog.factor_registry import FACTOR_DEFINITIONS

    return {
        str(name): str(definition.family).strip().lower()
        for name, definition in FACTOR_DEFINITIONS.items()
    }


def factor_family(name: str) -> str:
    """Return one canonical family label for discovery and screening."""
    text = str(name or "").strip()
    lower = text.lower()
    if lower.startswith("alpha_") and lower[6:].isdigit():
        return "alpha101"
    registered = _registry_family_map().get(text)
    if registered:
        return registered
    if lower.startswith((
        "k_", "roc_", "ma_", "std_", "qtlu_", "qtld_", "rsv_",
        "tsrank_", "cnt", "sum", "vma_", "vstd_", "corr_", "cord_",
        "imax_", "imin_", "imxd_", "beta_", "rsqr_", "resi_",
    )):
        return "alpha158"
    for prefix in (
        "alpha101", "alpha191", "alpha360", "talib", "mytt",
        "pta", "qtl", "ta", "talipp", "pyti",
    ):
        if lower.startswith(prefix + "_"):
            return prefix
    return lower.split("_", 1)[0] if lower else "unknown"


def _family_requested(family: str, requested: set[str]) -> bool:
    if not requested:
        return True
    if family in requested:
        return True
    return family.startswith("alpha") and "alpha" in requested


def _dynamic_runnable_factor_names(
    frame: pd.DataFrame,
    families: Sequence[str] | None = None,
) -> list[str]:
    """Return implementation-backed names for this panel's available columns."""
    panel = _prepared_panel(frame)
    requested_families = {
        str(value).strip().lower() for value in (families or [])
        if str(value).strip()
    }

    def enabled(*names: str) -> bool:
        return not requested_families or bool(requested_families.intersection(names))

    names = {
        name for name in _registry_formulas()
        if _family_requested(factor_family(name), requested_families)
    }
    from quant_sdk.capabilities import installed_factor_specs

    names.update(
        item["name"]
        for item in installed_factor_specs(fields=set(panel.columns))
        if _family_requested(
            str(item.get("family") or factor_family(item["name"])).strip().lower(),
            requested_families,
        )
    )

    # Family discovery needs the output schema, not a full research run.  Two
    # observations per asset preserve the group structure while avoiding the
    # dominant rolling-window cost.
    discovery_panel = panel.groupby("ts_code", sort=False).head(2).copy()
    family_computes = []
    if enabled("alpha158"):
        from factors_catalog.alpha158 import compute_alpha158

        family_computes.append((compute_alpha158, False))
    if enabled("alpha360"):
        from factors_catalog.alpha360 import compute_alpha360

        family_computes.append((compute_alpha360, False))
    if enabled("alpha101", "alpha"):
        from factors_catalog.alpha101_operators import compute_alpha101_selected

        family_computes.append((compute_alpha101_selected, True))
    if enabled("factor_zoo", "zoo"):
        from factors_catalog.factor_zoo import compute_factor_zoo

        family_computes.append((compute_factor_zoo, False))
    if enabled("ohlc", "volatility"):
        from factors_catalog.ohlc_volatility import compute_ohlc_volatility

        family_computes.append((compute_ohlc_volatility, False))
    for compute, filter_family in family_computes:
        try:
            _, produced = compute(discovery_panel)
        except (ArithmeticError, KeyError, TypeError, ValueError):
            _, produced = compute(panel)
        names.update(
            name for name in produced
            if not filter_family
            or _family_requested(factor_family(name), requested_families)
        )
    if enabled("alpha101", "alpha191", "alpha"):
        from factors_catalog.open_adapters import available_factors

        try:
            names.update(
                name for name in available_factors(panel)
                if _family_requested(factor_family(name), requested_families)
            )
        except (ImportError, RuntimeError):
            pass
    technical_families = {
        "talib", "mytt", "pta", "ta", "qtl", "talipp", "pyti",
    }
    if not requested_families or requested_families.intersection(technical_families):
        from factors_catalog.technical_adapters import (
            available_factors as technical_factors,
        )

        try:
            names.update(
                name for name in technical_factors(panel)
                if _family_requested(factor_family(name), requested_families)
            )
        except (ImportError, RuntimeError):
            pass
    from factors_catalog.history import execution_validated, minimum_history

    history = int(panel.groupby("ts_code", sort=False).size().max())
    return sorted(
        name for name in names
        if execution_validated(name)
        and history >= minimum_history(name)
    )


def runnable_factor_names(
    frame: pd.DataFrame,
    families: Sequence[str] | None = None,
    *,
    use_manifest: bool = True,
) -> list[str]:
    """Return implementation-backed factor names for the panel schema.

    Release builds use a generated capability manifest so discovery does not
    import optional technical libraries or execute provisional factor code.
    Installed ``qcap/1`` packs are still merged live on every call.
    """
    if not use_manifest:
        return _dynamic_runnable_factor_names(frame, families=families)
    return [
        item["name"]
        for item in runnable_factor_specs(
            frame,
            families=families,
            use_manifest=True,
        )
    ]


def runnable_factor_specs(
    frame: pd.DataFrame,
    names: list[str] | None = None,
    *,
    families: Sequence[str] | None = None,
    use_manifest: bool = True,
) -> list[dict[str, Any]]:
    """Return factor names plus source, required fields and output metadata."""
    panel = _prepared_panel(frame)
    history = int(panel.groupby("ts_code", sort=False).size().max())
    requested_families = {
        str(value).strip().lower() for value in (families or [])
        if str(value).strip()
    }
    from quant_sdk.capabilities import installed_factor_specs

    installed = {
        item["name"]: item
        for item in installed_factor_specs(fields=set(panel.columns))
    }
    manifest = _factor_manifest() if use_manifest else None
    if names is None and manifest is not None:
        result = [
            json.loads(json.dumps(item))
            for name, item in sorted(manifest.items())
            if _manifest_supports(item, set(panel.columns), history)
            and _family_requested(
                str(item.get("family") or factor_family(name)).lower(),
                requested_families,
            )
        ]
        result.extend(
            item for name, item in sorted(installed.items())
            if _family_requested(
                str(item.get("family") or factor_family(name)).lower(),
                requested_families,
            )
        )
        return result
    if names is None:
        names = _dynamic_runnable_factor_names(panel, families=families)
    from factors_catalog.factor_registry import FACTOR_DEFINITIONS
    from factors_catalog.open_adapters import (
        factor_minimum_history,
        factor_requirements,
    )
    from factors_catalog.technical_adapters import technical_factor_specs

    technical = {
        item["name"]: item for item in technical_factor_specs(panel)
    }
    result: list[dict[str, Any]] = []
    for name in names:
        if manifest is not None and name in manifest and _manifest_supports(
            manifest[name], set(panel.columns), history,
        ):
            result.append(json.loads(json.dumps(manifest[name])))
            continue
        if name in installed:
            result.append(installed[name])
            continue
        if name in technical:
            result.append(technical[name])
            continue
        lower = name.lower()
        if lower.startswith("alpha101_"):
            inputs = tuple(factor_requirements(name))
            from factors_catalog.history import generated_description

            result.append({
                "name": name, "family": "Alpha101", "source": "ta_cn_alpha101",
                "library": "ta-cn", "license": "MIT",
                "inputs": list(dict.fromkeys(inputs)),
                "requiredColumns": _required_columns_from_inputs(inputs),
                "minimumHistory": factor_minimum_history(name),
                "output": "factor_value", "outputs": ["factor_value"],
                "implementation": "local_adapter",
                "description": generated_description({"name": name}),
            })
            continue
        if lower.startswith("alpha191_"):
            inputs = tuple(factor_requirements(name))
            from factors_catalog.history import generated_description

            result.append({
                "name": name, "family": "Alpha191", "source": "ta_cn_alpha191",
                "library": "ta-cn", "license": "MIT",
                "inputs": list(dict.fromkeys(inputs)),
                "requiredColumns": _required_columns_from_inputs(inputs),
                "minimumHistory": factor_minimum_history(name),
                "output": "factor_value", "outputs": ["factor_value"],
                "implementation": "local_adapter",
                "description": generated_description({"name": name}),
            })
            continue
        if lower.startswith("alpha360_"):
            from factors_catalog.history import generated_description, minimum_history

            result.append({
                "name": name, "family": "Alpha360", "source": "qlib_alpha360",
                "library": "Qlib Alpha360", "license": "MIT",
                "inputs": ["open", "high", "low", "close", "vol", "amount"],
                "requiredColumns": ["open", "high", "low", "close"],
                "output": "factor_value", "outputs": ["factor_value"],
                "implementation": "local_builtin",
                "minimumHistory": minimum_history(name),
                "description": generated_description({"name": name}),
            })
            continue
        definition = FACTOR_DEFINITIONS.get(name)
        from factors_catalog.history import generated_description, minimum_history

        result.append({
            "name": name,
            "family": (
                str(definition.family)
                if definition is not None
                else name.split("_", 1)[0]
            ),
            "source": "local_registry" if definition is not None else "local_catalog",
            "library": "Quant Research Toolkit",
            "license": "project",
            "inputs": list(definition.dependencies) if definition is not None else [],
            "requiredColumns": (
                _registry_required_columns(name)
                if definition is not None else list(_BAR_COLUMNS)
            ),
            "output": "factor_value",
            "outputs": ["factor_value"],
            "implementation": "local_builtin",
            "minimumHistory": minimum_history(name),
            "description": (
                definition.description
                if definition is not None and definition.description
                else generated_description({"name": name})
            ),
        })
    return result
