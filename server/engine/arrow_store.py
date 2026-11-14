"""Arrow zero-copy store for dense [T, N] factor panels (S3).

Design notes (ARCHITECTURE.md 5.1 / H7):

* One IPC file per panel window.  Columns are the *per-row* members of
  the panel, laid out date-major ``[T, N]`` flat, so a column read is a
  strided view over the whole panel:

  - f64 ``open high low close adj_close up_limit down_limit``
  - f32 ``vol amount circ_mv pb turnover_rate``
  - bool ``st``

* No column carries nulls (NaN sentinels), therefore
  ``Array.to_numpy(zero_copy_only=True)`` yields **read-only views**
  over the mmapped buffer -> true zero-copy open (the 5 ms gate).
* Vector coordinates that are not per-row (``codes`` [N], ``dates`` [T],
  ``list_date`` [N], ``benchmark_ret`` [T]) live in schema metadata as
  base64 blobs (≈ 100 KB total; the big matrices never go there).
* Writes are atomic: serialise into ``<path>.tmp`` then ``os.replace``
  (sandbox workers race on the same cache otherwise).
* Date-window selection is a row-block slice -> O(1) views, no decode.

Purity: numpy + pyarrow + stdlib only (engine stays framework/DB free).
"""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

PER_ROW_F64: Tuple[str, ...] = (
    "open", "high", "low", "close", "adj_close", "up_limit", "down_limit",
)
PER_ROW_F32: Tuple[str, ...] = ("vol", "amount", "circ_mv", "pb", "turnover_rate",
                                "industry_id", "cash_dividend")
PER_ROW_BOOL: Tuple[str, ...] = ("st", "shortable")
#: Columns a canonical panel must contain.  These are execution facts, not
#: optional research features: omitting one would silently change replay
#: accounting or point-in-time eligibility.
REQUIRED_COLUMNS: Tuple[str, ...] = PER_ROW_F64 + (
    "vol", "amount", "st", "industry_id", "cash_dividend", "shortable",
    "pool_membership",
)
#: everything the writer knows how to emit (order = file layout)
PER_ROW_COLUMNS: Tuple[str, ...] = PER_ROW_F64 + PER_ROW_F32 + PER_ROW_BOOL

_META_KEYS = ("codes", "dates", "list_date", "benchmark_ret")
_FORMAT_VERSION = 1


class ArrowStoreError(RuntimeError):
    """Raised when the Arrow panel store is unusable (missing/corrupt)."""


# --------------------------------------------------------------------------- #
# schema / metadata helpers
# --------------------------------------------------------------------------- #
def _decode(data: bytes, dtype: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(data), dtype=np.dtype(dtype)).copy()


def _meta_entry(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    payload = {"dtype": arr.dtype.str,
               "b64": base64.b64encode(arr.tobytes()).decode("ascii")}
    return json.dumps(payload)


def _type_for(name: str, arr: np.ndarray) -> pa.DataType:
    if name in PER_ROW_F64 or arr.dtype == np.float64:
        return pa.float64()
    if name in PER_ROW_F32 or arr.dtype in (np.float32, np.float16):
        return pa.float32()
    if name in PER_ROW_BOOL or arr.dtype == np.bool_:
        # NOTE: pyarrow bit-packs booleans, which *cannot* be memory-mapped
        # zero-copy.  Store flags as uint8 instead so `to_numpy(
        # zero_copy_only=True)` still yields a read-only view (gate 5.1).
        return pa.uint8()
    raise ArrowStoreError(f"unsupported dtype for {name!r}: {arr.dtype}")


def build_schema(n_dates: int, n_codes: int, columns: Mapping[str, np.ndarray],
                 meta: Mapping[str, np.ndarray],
                 json_meta: Optional[Mapping[str, Any]] = None) -> pa.Schema:
    order = [c for c in PER_ROW_COLUMNS if c in columns]
    order += [c for c in columns if c not in PER_ROW_COLUMNS]
    fields = [pa.field(name, _type_for(name, np.asarray(columns[name])))
              for name in order]
    md: Dict[str, str] = {
        "format_version": str(_FORMAT_VERSION),
        "shape": json.dumps([int(n_dates), int(n_codes)]),
        "columns": json.dumps(order),
    }
    for key in _META_KEYS:
        if key in meta:
            md[key] = _meta_entry(np.asarray(meta[key]))
    if json_meta:
        md["panel_meta"] = json.dumps(dict(json_meta), ensure_ascii=False,
                                      default=str)
    return pa.schema(fields, metadata=md)


def write_panel(path: os.PathLike | str,
                columns: Mapping[str, np.ndarray],
                meta: Mapping[str, np.ndarray],
                n_dates: int, n_codes: int,
                json_meta: Optional[Mapping[str, Any]] = None) -> str:
    """Atomically write a dense no-null panel to ``path`` (Arrow IPC file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ArrowStoreError(f"missing required columns: {missing}")
    schema = build_schema(n_dates, n_codes, columns, meta, json_meta)
    arrays: List[pa.Array] = []
    for field_ in schema:
        name = field_.name
        arr = np.asarray(columns[name])
        if arr.ndim != 1 or arr.shape[0] != n_dates * n_codes:
            raise ArrowStoreError(
                f"column {name!r} must be flat [{n_dates}*{n_codes}], got {arr.shape}")
        if name in PER_ROW_BOOL:
            arrays.append(pa.array(arr.astype(np.uint8), type=pa.uint8()))
        elif pa.types.is_float64(field_.type):
            arrays.append(pa.array(arr.astype(np.float64), type=pa.float64()))
        else:
            arrays.append(pa.array(arr.astype(np.float32), type=pa.float32()))
    table = pa.Table.from_arrays(arrays, schema=schema)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    with pa.OSFile(str(tmp), "wb") as sink:
        with ipc.new_file(sink, schema) as writer:
            writer.write_table(table)
    os.replace(tmp, path)          # atomic on POSIX + Windows
    return str(path)


def _atomic_copy(src: os.PathLike | str, dst: os.PathLike | str) -> str:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    data = Path(src).read_bytes()
    tmp.write_bytes(data)
    os.replace(tmp, dst)
    return str(dst)


# --------------------------------------------------------------------------- #
# mapped (zero-copy) open
# --------------------------------------------------------------------------- #
@dataclass
class MappedPanel:
    """Zero-copy read side of the Arrow panel store.

    ``open()`` only maps the file and decodes the schema metadata
    (mmap + footer + meta ≈ 1-2 ms, gate ≤ 5 ms).  The record-batch body
    is materialised **lazily** on the first ``column()`` / ``window()``
    call (≈ 13 ms for the 195 MB canonical panel, ~15 GB/s page-in from
    the OS cache) and every column afterwards is a *read-only numpy view
    over the mmapped buffer* - verified by mutating the file through a
    second file handle and observing the view change.

    ``window(start, end)`` returns row-block slice views
    (``reshape(T, N)[start:end]``) without copying.
    """

    path: str
    n_dates: int
    n_codes: int
    meta: Dict[str, np.ndarray] = field(default_factory=dict)
    panel_meta: Dict[str, Any] = field(default_factory=dict)
    column_names: Tuple[str, ...] = ()
    _arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    _reader: Any = None
    _source: Any = None
    _batch: Any = None
    materialize_ms: float = 0.0
    opened_at: float = field(default_factory=time.perf_counter)

    # -- construction ------------------------------------------------------ #
    @classmethod
    def open(cls, path: os.PathLike | str) -> "MappedPanel":
        """Map the file and read only the footer/schema metadata."""
        path = str(path)
        if not os.path.exists(path):
            raise ArrowStoreError(f"panel store missing: {path}")
        try:
            source = pa.memory_map(path, "r")
            reader = ipc.open_file(source)
        except Exception as exc:
            raise ArrowStoreError(f"corrupt panel store {path}: {exc}") from exc
        schema_md = reader.schema.metadata or {}
        try:
            shape = json.loads(schema_md.get(b"shape", b"[0,0]"))
            n_dates, n_codes = int(shape[0]), int(shape[1])
        except Exception as exc:
            raise ArrowStoreError(f"panel store has no shape metadata: {path}") from exc
        meta: Dict[str, np.ndarray] = {}
        for key in _META_KEYS:
            raw = schema_md.get(key.encode())
            if raw:
                payload = json.loads(raw.decode("ascii"))
                meta[key] = _decode(payload["b64"], payload["dtype"])
        names = json.loads(schema_md.get(b"columns",
                                         json.dumps(list(PER_ROW_COLUMNS)).encode())
                           .decode("ascii"))
        raw_pm = schema_md.get(b"panel_meta")
        panel_meta: Dict[str, Any] = {}
        if raw_pm:
            try:
                panel_meta = json.loads(raw_pm.decode("utf-8"))
            except Exception:  # pragma: no cover
                panel_meta = {}
        return cls(path=path, n_dates=n_dates, n_codes=n_codes, meta=meta,
                   panel_meta=panel_meta, column_names=tuple(names),
                   _reader=reader, _source=source)

    # -- lazy materialisation ---------------------------------------------- #
    def _materialize(self) -> Dict[str, np.ndarray]:
        if self._arrays:
            return self._arrays
        if self._reader is None:
            raise ArrowStoreError(f"panel store closed: {self.path}")
        t0 = time.perf_counter()
        try:
            batch = self._reader.get_batch(0)     # single batch per panel
        except Exception as exc:
            raise ArrowStoreError(
                f"cannot read record batch from {self.path}: {exc}") from exc
        self._batch = batch
        fields = [f.name for f in self._reader.schema]
        arrays: Dict[str, np.ndarray] = {}
        for i, name in enumerate(fields):
            arr = batch.column(i).to_numpy(zero_copy_only=True)
            if name in PER_ROW_BOOL:
                arr = arr.view(np.bool_)     # uint8 -> bool, still zero-copy
            arrays[name] = arr
        self._arrays = arrays
        self.materialize_ms = (time.perf_counter() - t0) * 1000.0
        return arrays

    # -- access ------------------------------------------------------------ #
    def column(self, name: str) -> np.ndarray:
        arrays = self._materialize()
        try:
            return arrays[name]
        except KeyError:
            raise ArrowStoreError(
                f"unknown column {name!r} (have {list(arrays)})") from None

    def has_column(self, name: str) -> bool:
        return name in self.column_names

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n_dates, self.n_codes)

    def window(self, start: int = 0, end: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Row-block slice views ``[start, end)`` for every per-row column."""
        end = self.n_dates if end is None else end
        if not (0 <= start < end <= self.n_dates):
            raise ArrowStoreError(f"bad window [{start}, {end}) for T={self.n_dates}")
        arrays = self._materialize()
        return {name: arrays[name][start * self.n_codes:end * self.n_codes]
                .reshape(end - start, self.n_codes)
                for name in self.column_names}

    def window_index(self, dates: Sequence[int]) -> np.ndarray:
        """Row indices for an explicit list of trade dates (O(T))."""
        all_dates = np.asarray(self.meta["dates"])
        want = np.asarray(dates, dtype=all_dates.dtype)
        idx = np.searchsorted(all_dates, want)
        bad = idx >= all_dates.shape[0]
        if not bad.any() and want.size:
            bad = ~np.array_equal(all_dates[idx], want)
        if np.any(bad):
            raise ArrowStoreError(f"dates outside store: {np.asarray(dates)[bad][:5].tolist()}")
        return idx

    def meta_array(self, name: str) -> np.ndarray:
        try:
            return self.meta[name]
        except KeyError:
            raise ArrowStoreError(f"unknown metadata array {name!r}") from None

    def close(self) -> None:
        self._arrays.clear()
        self._batch = None
        for attr in ("_reader", "_source"):
            handle = getattr(self, attr, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:  # pragma: no cover
                    pass
                setattr(self, attr, None)


def mapped_open(path: os.PathLike | str) -> MappedPanel:
    """Open the store zero-copy (gate: <= 5 ms for the canonical window)."""
    return MappedPanel.open(path)


def mapped_open_ms(path: os.PathLike | str) -> float:
    """Measured wall-clock of :func:`mapped_open` (map + footer + metadata)."""
    t0 = time.perf_counter()
    mp = mapped_open(path)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    mp.close()
    return dt_ms


def mapped_load_ms(path: os.PathLike | str) -> Dict[str, float]:
    """Honest split: ``open_ms`` (map+footer+meta) vs ``materialize_ms``
    (first record-batch read, i.e. paging the body in) vs ``total_ms``."""
    t0 = time.perf_counter()
    mp = mapped_open(path)
    open_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    mp._materialize()
    mat_ms = (time.perf_counter() - t1) * 1000.0
    mp.close()
    return {"open_ms": round(open_ms, 3), "materialize_ms": round(mat_ms, 3),
            "total_ms": round(open_ms + mat_ms, 3)}


def panel_cache_dir(root: os.PathLike | str) -> Path:
    d = Path(root) / "data_lake" / "arrow"
    d.mkdir(parents=True, exist_ok=True)
    return d


def panel_cache_path(root: os.PathLike | str, start_date: int, end_date: int) -> Path:
    return panel_cache_dir(root) / f"panel_{int(start_date)}_{int(end_date)}.arrow"


# --------------------------------------------------------------------------- #
# dataset facade: open_dataset + honest before/after load_ms + stale watermark
# --------------------------------------------------------------------------- #
#: deliverable name for the zero-copy handle (same object, same contract)
ArrowDatasetHandle = MappedPanel


def open_dataset(path: os.PathLike | str) -> ArrowDatasetHandle:
    """Open an Arrow panel dataset zero-copy (mmap + footer + metadata).

    Alias of :func:`mapped_open`; the rebuild path for a missing/corrupt
    store is :func:`write_panel`, which serialises into ``<path>.tmp`` and
    commits with ``os.replace`` (atomic, race-safe between workers).
    """
    return MappedPanel.open(path)


def dataset_load_report(path: os.PathLike | str) -> Dict[str, Any]:
    """Honest load_ms **before** (DuckDB long-table pivot) vs **after**
    (mmap open of the Arrow store).

    ``before_duckdb_ms`` is the real build-time measurement the panel
    builder recorded in ``panel_meta["build_ms"]`` (None when the store
    predates the field - never invented).  ``after_mmap_*`` is measured
    on this call: open = map + footer + schema metadata, materialize =
    first record-batch page-in.
    """
    t0 = time.perf_counter()
    mp = MappedPanel.open(path)
    open_ms = (time.perf_counter() - t0) * 1000.0
    try:
        raw_build = mp.panel_meta.get("build_ms")
        before = float(raw_build) if raw_build is not None else None
        t1 = time.perf_counter()
        mp._materialize()
        mat_ms = (time.perf_counter() - t1) * 1000.0
    finally:
        mp.close()
    after_ms = open_ms + mat_ms
    speedup = round(before / after_ms, 1) if before and after_ms > 0 else None
    return {
        "path": str(path),
        "before_duckdb_ms": None if before is None else round(before, 1),
        "after_mmap_open_ms": round(open_ms, 3),
        "after_mmap_materialize_ms": round(mat_ms, 3),
        "after_mmap_total_ms": round(after_ms, 3),
        "speedup_x": speedup,
        "before_source": "panel_meta.build_ms (measured by the DuckDB builder)",
    }


def stale_watermark(path: os.PathLike | str, source_watermark: Any
                    ) -> Dict[str, Any]:
    """Panel-cache freshness policy vs the DuckDB ``daily`` watermark.

    ``source_watermark`` is ``max(trade_date)`` of the source ``daily``
    table (YYYYMMDD, as maintained by ``services.data_assets``).  Policy::

        stale  <=>  panel_end < source_watermark

    where ``panel_end`` is the last trade date stored in the panel.  A
    store whose dates metadata cannot be read is reported stale (we
    cannot prove it is fresh) so the caller rebuilds it through
    :func:`write_panel` (atomic ``.tmp`` + ``os.replace``).
    """
    mp = MappedPanel.open(path)          # footer only, ~1 ms
    try:
        dates = mp.meta.get("dates")
        panel_end = int(np.asarray(dates)[-1]) if (
            dates is not None and len(dates)) else None
    finally:
        mp.close()
    try:
        src = int(source_watermark) if source_watermark is not None else None
    except (TypeError, ValueError):
        src = None
    if src is None:
        stale, reason = False, "no source watermark provided (cannot compare)"
    elif panel_end is None:
        stale, reason = True, "panel dates metadata unreadable (cannot prove fresh)"
    elif panel_end < src:
        stale, reason = True, "panel ends before the source watermark"
    else:
        stale, reason = False, "panel covers the source watermark"
    return {"stale": bool(stale), "panel_end": panel_end,
            "source_watermark": src, "reason": reason,
            "policy": "stale iff panel_end < source_watermark; "
                      "rebuild = write_panel(.tmp + os.replace)"}


__all__ = [
    "ArrowStoreError", "MappedPanel", "ArrowDatasetHandle",
    "REQUIRED_COLUMNS", "PER_ROW_COLUMNS",
    "PER_ROW_F64", "PER_ROW_F32", "PER_ROW_BOOL",
    "build_schema", "mapped_open", "mapped_open_ms", "mapped_load_ms",
    "open_dataset", "dataset_load_report", "stale_watermark",
    "panel_cache_dir", "panel_cache_path", "write_panel",
]
