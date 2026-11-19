"""Portable panel and multi-table contracts for federated research data.

`qpanel/1` remains the bar-panel wire format consumed by Alpha Graph. The
`qpanel/2` metadata contract adds explicit temporal semantics for reference,
fundamental and event tables so a caller can perform point-in-time joins
without guessing whether a date is an observation date or an availability date.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

PANEL_VERSION = "qpanel/2"
BAR_VERSION = "qpanel/1"
TABLE_KINDS = {"bars", "reference", "fundamental", "event", "flow"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _pit_datetime(values: pd.Series, name: str) -> pd.Series:
    text = values.map(
        lambda value: str(int(value))
        if isinstance(value, (int, float))
        and pd.notna(value)
        and float(value).is_integer()
        else str(value).strip()
    )
    daily = text.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    if daily.any():
        parsed.loc[daily] = pd.to_datetime(
            text.loc[daily],
            format="%Y%m%d",
            errors="raise",
            utc=True,
        )
    if (~daily).any():
        parsed.loc[~daily] = pd.to_datetime(
            text.loc[~daily],
            errors="raise",
            utc=True,
        )
    if parsed.isna().any():
        raise ValueError(f"{name} contains missing timestamps")
    return parsed


def panel_metadata(
    frame: pd.DataFrame,
    *,
    kind: str = "bars",
    identity: Sequence[str] = ("ts_code",),
    event_time: str = "trade_date",
    available_at: str | None = None,
) -> dict[str, Any]:
    """Describe one table's identity and point-in-time availability."""
    normalized_kind = str(kind).strip().lower()
    if normalized_kind not in TABLE_KINDS:
        raise ValueError(f"unsupported panel table kind: {kind}")
    if not identity:
        raise ValueError("panel table requires at least one identity field")
    missing_identity = [
        name for name in identity if name not in frame.columns
    ]
    if missing_identity:
        raise ValueError(
            "panel table is missing identity fields: "
            + ", ".join(missing_identity)
        )
    if event_time not in frame.columns:
        raise ValueError(f"panel table is missing temporal field: {event_time}")
    if normalized_kind != "bars" and not available_at:
        raise ValueError(
            f"{normalized_kind} table requires an explicit availability field"
        )
    if available_at and available_at not in frame.columns:
        raise ValueError(f"panel table is missing availability field: {available_at}")
    if normalized_kind == "bars":
        missing = {
            "open", "high", "low", "close", "vol",
        }.difference(frame.columns)
        if missing:
            raise ValueError(
                "bar table is missing fields: " + ", ".join(sorted(missing))
            )
    return {
        "format": PANEL_VERSION,
        "barWireFormat": BAR_VERSION,
        "kind": normalized_kind,
        "identity": list(identity),
        "eventTime": event_time,
        "availableAt": available_at,
        "pitRule": (
            "known_at_lte_trade_date"
            if normalized_kind in {"fundamental", "reference"}
            else "known_at_lte_decision_time"
            if normalized_kind == "event"
            else "provider_timestamp"
        ),
        "columns": [str(column) for column in frame.columns],
        "rows": len(frame),
        "assets": int(frame[identity[0]].astype(str).nunique()),
    }


def bundle_manifest(tables: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate and seal a portable multi-table research bundle."""
    if not isinstance(tables, Sequence) or len(tables) == 0:
        raise ValueError("panel bundle requires at least one table")
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(tables):
        if not isinstance(raw, Mapping):
            raise ValueError(f"table[{index}] must be an object")
        table = dict(raw)
        name = str(table.get("name") or "").strip()
        if not name or name in names:
            raise ValueError(f"table[{index}] has a missing or duplicate name")
        names.add(name)
        kind = str(table.get("kind") or "").strip().lower()
        if kind not in TABLE_KINDS:
            raise ValueError(f"table {name} has invalid kind")
        identity = table.get("identity")
        event_time = str(table.get("eventTime") or "").strip()
        if not identity or not event_time:
            raise ValueError(f"table {name} requires identity and eventTime")
        if kind != "bars" and not table.get("availableAt"):
            raise ValueError(
                f"table {name} requires availableAt for point-in-time joins"
            )
        table["name"] = name
        table["kind"] = kind
        table["identity"] = list(identity)
        table["eventTime"] = event_time
        table["availableAt"] = (
            str(table["availableAt"]).strip()
            if table.get("availableAt") else None
        )
        normalized.append(table)
    body = {
        "format": PANEL_VERSION,
        "barWireFormat": BAR_VERSION,
        "tables": normalized,
        "joinContract": {
            "mode": "as_of",
            "observationTime": "eventTime",
            "availabilityTime": "availableAt",
            "neverUseFutureAvailability": True,
        },
    }
    body["sha256"] = hashlib.sha256(
        json.dumps(
            body, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return _jsonable(body)


def as_of_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    identity: str | Sequence[str] = ("ts_code",),
    left_time: str = "trade_date",
    right_event_time: str = "event_time",
    right_available_at: str = "available_at",
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Attach the latest record known at each left-hand decision time.

    The right-hand table is joined on ``available_at``, not on ``event_time``.
    A record whose availability is after a left row's decision time can never
    be selected, even when its event time is earlier. This is the executable
    counterpart of the ``qpanel/2`` metadata contract.
    """
    keys = [identity] if isinstance(identity, str) else list(identity)
    if not keys:
        raise ValueError("as-of join requires at least one identity field")
    missing_left = (set(keys) | {left_time}).difference(left.columns)
    missing_right = (
        set(keys) | {right_event_time, right_available_at}
    ).difference(right.columns)
    if missing_left:
        raise ValueError(
            "as-of left table is missing fields: "
            + ", ".join(sorted(missing_left))
        )
    if missing_right:
        raise ValueError(
            "as-of right table is missing fields: "
            + ", ".join(sorted(missing_right))
        )
    if columns is not None:
        selected = list(dict.fromkeys([
            *keys,
            right_event_time,
            right_available_at,
            *columns,
        ]))
        missing = set(selected).difference(right.columns)
        if missing:
            raise ValueError(
                "as-of right table is missing selected fields: "
                + ", ".join(sorted(missing))
            )
        right = right.loc[:, selected].copy()
    else:
        right = right.copy()
    left = left.copy()

    left["_qpanel_decision_time"] = _pit_datetime(
        left[left_time], left_time,
    )
    right["_qpanel_event_time"] = _pit_datetime(
        right[right_event_time], right_event_time,
    )
    right["_qpanel_available_at"] = _pit_datetime(
        right[right_available_at], right_available_at,
    )
    left["_qpanel_row_order"] = range(len(left))

    left_sorted = left.sort_values(
        [*keys, "_qpanel_decision_time"],
        kind="mergesort",
    ).reset_index(drop=True)
    right_sorted = right.sort_values(
        [*keys, "_qpanel_available_at", "_qpanel_event_time"],
        kind="mergesort",
    ).reset_index(drop=True)
    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on="_qpanel_decision_time",
        right_on="_qpanel_available_at",
        by=keys,
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_qpanel_right"),
    )
    available = merged["_qpanel_available_at"].notna()
    future = available & (
        merged["_qpanel_available_at"] > merged["_qpanel_decision_time"]
    )
    if bool(future.any()):
        raise ValueError("as-of join selected a future-available record")
    merged = merged.sort_values("_qpanel_row_order", kind="mergesort")
    merged = merged.drop(
        columns=[
            "_qpanel_decision_time",
            "_qpanel_event_time",
            "_qpanel_available_at",
            "_qpanel_row_order",
        ],
        errors="ignore",
    )
    return merged.reset_index(drop=True)


__all__ = [
    "BAR_VERSION",
    "PANEL_VERSION",
    "TABLE_KINDS",
    "as_of_join",
    "bundle_manifest",
    "panel_metadata",
]
