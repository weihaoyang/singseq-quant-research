"""Deterministic provenance helpers for canonical replay artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from server.process_utils import hidden_process_kwargs

CANONICAL_ENGINE_VERSION = "duckdb-arrow-canonical-v2"


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str)


def sha256_file(path: str | os.PathLike, chunk_size: int = 8 * 1024 * 1024) -> str:
    p = Path(path)
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        while True:
            block = handle.read(int(chunk_size))
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=2,
            **hidden_process_kwargs(),
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


_REPLAY_SOURCE_FILES = ('evaluation/__init__.py', 'evaluation/alpha_zoo.py', 'evaluation/attribution.py', 'evaluation/library_correlation.py', 'evaluation/mining.py', 'evaluation/novelty.py', 'evaluation/novelty_scoring.py', 'evaluation/overfit.py', 'evaluation/portfolio.py', 'evaluation/portfolio_replay.py', 'evaluation/research_contract.py', 'evaluation/tearsheet.py', 'evaluation/validation_ledger.py', 'quant_sdk/archive_scope.py', 'quant_sdk/catalog_factors.py', 'quant_sdk/factor_templates.py', 'quant_sdk/local.py', 'quant_sdk/local_backtest.py', 'quant_sdk/local_intraday.py', 'quant_sdk/local_research.py', 'quant_sdk/research.py', 'server/engine/__init__.py', 'server/engine/arrow_store.py', 'server/engine/backtest.py', 'server/engine/friction.py', 'server/engine/gpu.py', 'server/engine/gpu_tensor.py', 'server/engine/intraday.py', 'server/engine/metrics.py', 'server/engine/provenance.py', 'server/engine/universe.py')


def source_tree_fingerprint(root: str | os.PathLike) -> dict[str, Any]:
    """Fingerprint the actual replay implementation, including dirty files."""
    base = Path(root).resolve()
    files: dict[str, str] = {}
    for relative in _REPLAY_SOURCE_FILES:
        digest = sha256_file(base / relative)
        files[relative] = digest
    aggregate = hashlib.sha256(stable_json(files).encode("utf-8")).hexdigest()
    return {"files": files, "sha256": aggregate}


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def runtime_provenance(seed: int = 0) -> dict[str, Any]:
    """Return stable runtime details without importing GPU frameworks."""
    return {
        "python": sys.version.split()[0],
        "implementation": sys.implementation.name,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {name: _version(name) for name in
                      ("numpy", "pandas", "pyarrow", "duckdb", "torch")},
        "git_revision": _git_revision(),
        "seed": int(seed),
    }


def build_manifest(*, request: Mapping[str, Any], data: Mapping[str, Any],
                   engine: Mapping[str, Any] | None = None,
                   seed: int = 0, root: str | os.PathLike | None = None) -> dict[str, Any]:
    """Build a replay manifest and a deterministic content hash.

    ``created_at_utc`` is informational and excluded from ``manifest_hash``;
    rerunning the same input/config therefore produces the same identity.
    """
    body = {
        "manifest_version": 1,
        "canonical_engine_version": CANONICAL_ENGINE_VERSION,
        "request": dict(request),
        "data": dict(data),
        "engine": {**dict(engine or {}),
                   "source_fingerprint": source_tree_fingerprint(
                       root or Path(__file__).resolve().parents[2])},
        "runtime": runtime_provenance(seed),
    }
    body["manifest_hash"] = hashlib.sha256(stable_json(body).encode("utf-8")).hexdigest()
    body["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    return body


__all__ = ["CANONICAL_ENGINE_VERSION", "build_manifest", "runtime_provenance",
           "sha256_file", "source_tree_fingerprint", "stable_json"]
