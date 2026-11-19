"""Installable local capability packs.

Capability discovery is metadata-only. Installed Factor Pack source is never
executed until a factor is explicitly computed, and installation rejects unsafe
paths, changed files and code that fails the existing factor AST contract.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

CAPABILITY_FORMAT = "qcap/1"
ACTIVE_POINTER_FORMAT = "qcap-active/1"
CAPABILITY_KINDS = (
    "factor-pack",
    "operator-pack",
    "benchmark-pack",
    "connector-pack",
)
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_PACK_BYTES = 64 * 1024 * 1024
MAX_PACK_FILES = 256
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,80}$")
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,79}$")
_FACTOR_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{2,127}$")
_BENCHMARK_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
_HASH_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_HOSTNAME_PATTERN = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_COMPUTE_STACK: ContextVar[tuple[str, ...]] = ContextVar(
    "quant_capability_factor_stack", default=()
)


class CapabilityError(ValueError):
    """Invalid or unsafe local capability pack."""


def _allowed_hosts(value: Any, connector_id: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise CapabilityError(
            f"connector {connector_id} allowedHosts must be an array"
        )
    hosts: list[str] = []
    for raw in value:
        if not isinstance(raw, str):
            raise CapabilityError(
                f"connector {connector_id} allowedHosts entries must be strings"
            )
        host = raw.strip().lower().rstrip(".")
        if not _HOSTNAME_PATTERN.fullmatch(host):
            raise CapabilityError(
                f"connector {connector_id} has an invalid allowedHosts entry: {raw!r}"
            )
        hosts.append(host)
    return list(dict.fromkeys(hosts))


def capability_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve()
    configured = os.environ.get("QUANT_CAPABILITIES_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    data_root = os.environ.get("QUANT_DATA_ROOT", "").strip()
    if data_root:
        return (Path(data_root).expanduser() / "capabilities").resolve()
    return (Path.home() / ".quant" / "capabilities").resolve()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _signable_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(manifest)
    for key in ("signature", "signatureInfo", "installedAt", "contentSha256"):
        body.pop(key, None)
    return body


def _trusted_public_key(value: str | Path | None) -> bytes | None:
    if value is None or str(value).strip() == "":
        return None
    candidate = Path(value).expanduser()
    try:
        raw = candidate.read_bytes() if candidate.is_file() else str(value).encode("ascii")
    except OSError:
        raw = str(value).encode("ascii")
    try:
        if raw.startswith(b"-----BEGIN"):
            from cryptography.hazmat.primitives import serialization

            key = serialization.load_pem_public_key(raw)
            raw = key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        else:
            raw = base64.b64decode(raw, validate=True)
    except Exception as error:
        raise CapabilityError("trusted publisher key is invalid") from error
    if len(raw) != 32:
        raise CapabilityError("trusted publisher key must be a 32-byte Ed25519 key")
    return raw


def _signature_state(
    manifest: Mapping[str, Any],
    *,
    trusted_key: str | Path | None = None,
) -> dict[str, Any]:
    signature = manifest.get("signature")
    expected = _trusted_public_key(trusted_key)
    if signature is None:
        if expected is not None:
            raise CapabilityError("trusted publisher key requires a signed capability")
        return {
            "signatureValid": None,
            "publisherVerified": False,
            "signerFingerprint": None,
            "trustedKeySupplied": False,
        }
    if not isinstance(signature, dict) or signature.get("algorithm") != "ed25519":
        raise CapabilityError("capability signature must use Ed25519")
    try:
        public_key = base64.b64decode(str(signature["publicKey"]), validate=True)
        signature_bytes = base64.b64decode(str(signature["value"]), validate=True)
        if len(public_key) != 32 or len(signature_bytes) != 64:
            raise ValueError("invalid Ed25519 key or signature length")
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )

        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature_bytes,
            _canonical_json(_signable_manifest(manifest)),
        )
    except Exception as error:
        raise CapabilityError("capability signature is invalid") from error
    fingerprint = _sha256_bytes(public_key)
    publisher_verified = expected is not None and expected == public_key
    if expected is not None and not publisher_verified:
        raise CapabilityError("capability signer does not match the trusted key")
    return {
        "signatureValid": True,
        "publisherVerified": publisher_verified,
        "signerFingerprint": fingerprint,
        "trustedKeySupplied": expected is not None,
    }


def _safe_relative(name: str) -> str:
    normalized = str(name).replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (
        not normalized
        or relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or "\\" in name
    ):
        raise CapabilityError(f"capability member path is unsafe: {name!r}")
    return relative.as_posix()


def _read_directory(source: Path) -> tuple[dict[str, bytes], dict[str, int]]:
    files: dict[str, bytes] = {}
    modes: dict[str, int] = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix()
        if path.is_symlink():
            raise CapabilityError(f"capability source contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise CapabilityError(f"capability source contains a non-file: {relative}")
        content = path.read_bytes()
        files[_safe_relative(relative)] = content
        modes[relative] = int(path.stat().st_mode)
    return files, modes


def _read_zip(source: Path) -> tuple[dict[str, bytes], dict[str, int]]:
    files: dict[str, bytes] = {}
    modes: dict[str, int] = {}
    total = 0
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        if len(members) > MAX_PACK_FILES:
            raise CapabilityError(
                f"capability pack has too many files: {len(members)} > {MAX_PACK_FILES}"
            )
        for info in members:
            relative = _safe_relative(info.filename)
            if info.is_dir():
                continue
            file_type = (int(info.external_attr) >> 16) & 0o170000
            if file_type == 0o120000:
                raise CapabilityError(f"capability pack contains a symlink: {relative}")
            total += int(info.file_size)
            if total > MAX_PACK_BYTES:
                raise CapabilityError("capability pack exceeds its uncompressed size limit")
            files[relative] = archive.read(info)
            modes[relative] = int(info.external_attr >> 16)
    return files, modes


def _read_source(source: str | Path) -> tuple[dict[str, bytes], dict[str, int]]:
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise CapabilityError(f"capability source does not exist: {path}")
    if path.is_dir():
        return _read_directory(path)
    if path.is_file() and path.suffix.lower() in {".zip", ".qcap"}:
        return _read_zip(path)
    raise CapabilityError("capability source must be a directory, .zip or .qcap file")


def _load_manifest(files: Mapping[str, bytes]) -> dict[str, Any]:
    raw = files.get("qcap.json")
    if raw is None:
        raise CapabilityError("capability pack is missing qcap.json")
    if len(raw) > MAX_MANIFEST_BYTES:
        raise CapabilityError("capability manifest is too large")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CapabilityError(f"capability manifest is not valid UTF-8 JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise CapabilityError("capability manifest must be an object")
    return manifest


def _checked_manifest(
    manifest: Mapping[str, Any],
    files: Mapping[str, bytes],
    *,
    trusted_key: str | Path | None = None,
) -> dict[str, Any]:
    if manifest.get("format") != CAPABILITY_FORMAT:
        raise CapabilityError(f"capability format must be {CAPABILITY_FORMAT}")
    kind = str(manifest.get("kind") or "")
    if kind not in CAPABILITY_KINDS:
        raise CapabilityError(f"unsupported capability kind: {kind or '<empty>'}")
    pack_id = str(manifest.get("id") or "")
    if not _ID_PATTERN.fullmatch(pack_id):
        raise CapabilityError("capability id must be a lowercase namespace identifier")
    version = str(manifest.get("version") or "")
    if not _VERSION_PATTERN.fullmatch(version):
        raise CapabilityError("capability version is invalid")

    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, dict):
        raise CapabilityError("capability manifest requires compatibility")
    declared_protocol = str(compatibility.get("protocol") or "")
    if declared_protocol not in {"1.1", "1.2", "1.3", "2.0"}:
        raise CapabilityError("capability requires an unsupported protocol version")
    if str(compatibility.get("alphaGraph") or "") not in {"", "alpha-graph-1"}:
        raise CapabilityError("capability requires an unsupported Alpha Graph version")
    if compatibility.get("operators") not in {None, 2}:
        raise CapabilityError("capability requires an unsupported operator catalogue")

    declared_files = manifest.get("files")
    if not isinstance(declared_files, list):
        raise CapabilityError("capability manifest requires files")
    if not declared_files and kind not in {"benchmark-pack", "factor-pack"}:
        raise CapabilityError("capability manifest requires files")
    declared: dict[str, dict[str, Any]] = {}
    for item in declared_files:
        if not isinstance(item, dict):
            raise CapabilityError("capability file entries must be objects")
        relative = _safe_relative(str(item.get("path") or ""))
        if relative == "qcap.json":
            raise CapabilityError("qcap.json must not be listed as a member file")
        if relative in declared:
            raise CapabilityError(f"duplicate capability member: {relative}")
        expected = str(item.get("sha256") or "").lower()
        if not _HASH_PATTERN.fullmatch(expected):
            raise CapabilityError(f"capability member lacks a valid SHA-256: {relative}")
        actual = files.get(relative)
        if actual is None:
            raise CapabilityError(f"declared capability member is missing: {relative}")
        if _sha256_bytes(actual) != expected:
            raise CapabilityError(f"capability member hash mismatch: {relative}")
        declared[relative] = item

    extra = sorted(set(files).difference(declared).difference({"qcap.json"}))
    if extra:
        raise CapabilityError(
            "capability pack contains undeclared files: " + ", ".join(extra[:8])
        )

    for relative in declared:
        if relative.endswith(".py"):
            from quant_sdk.local import audit_factor_source

            source = files[relative].decode("utf-8")
            try:
                audit_factor_source(source)
            except Exception as error:
                raise CapabilityError(
                    f"capability Python failed static audit: {relative}: {error}"
                ) from error

    def checked_entry(value: Any, label: str) -> str:
        entry = str(value or "").strip()
        relative, separator, function = entry.partition(":")
        relative = _safe_relative(relative)
        if (
            not separator
            or not function.isidentifier()
            or relative not in declared
            or not relative.endswith(".py")
        ):
            raise CapabilityError(f"{label} entry must be <file.py>:<function>")
        return f"{relative}:{function}"

    if kind == "factor-pack":
        factors = manifest.get("factors")
        if not isinstance(factors, list) or not factors:
            raise CapabilityError("factor pack requires at least one factor")
        names: set[str] = set()
        normalized: list[dict[str, Any]] = []
        for raw in factors:
            if not isinstance(raw, dict):
                raise CapabilityError("factor declarations must be objects")
            factor = dict(raw)
            name = str(factor.get("name") or "").strip()
            if not _FACTOR_NAME_PATTERN.fullmatch(name) or "." not in name:
                raise CapabilityError(
                    "factor pack names must be namespaced identifiers such as vendor.momentum20"
                )
            if name in names:
                raise CapabilityError(f"duplicate factor in pack: {name}")
            names.add(name)
            family = str(
                factor.get("family") or manifest.get("id") or "factor-pack"
            ).strip()
            if not family or len(family) > 128:
                raise CapabilityError(f"factor {name} has invalid family")
            factor["family"] = family
            required = factor.get("requiredFields")
            if (
                not isinstance(required, list)
                or not required
                or any(not isinstance(field, str) or not field.strip() for field in required)
            ):
                raise CapabilityError(f"factor {name} requires requiredFields")
            factor["requiredFields"] = list(dict.fromkeys(
                field.strip() for field in required
            ))
            minimum = factor.get("minimumHistory", 1)
            if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
                raise CapabilityError(f"factor {name} has invalid minimumHistory")
            factor["minimumHistory"] = minimum
            entry = str(factor.get("entry") or "").strip()
            graph = factor.get("graph")
            if bool(entry) == bool(graph):
                raise CapabilityError(f"factor {name} requires exactly one of entry or graph")
            if entry:
                factor["entry"] = checked_entry(entry, f"factor {name}")
            else:
                if not isinstance(graph, dict):
                    raise CapabilityError(f"factor {name} graph must be an object")
                outputs = graph.get("outputs")
                if not isinstance(outputs, list) or not outputs or not all(
                    isinstance(value, str) and value for value in outputs
                ):
                    raise CapabilityError(f"factor {name} graph requires outputs")
                output = str(factor.get("output") or outputs[0])
                if output not in outputs:
                    raise CapabilityError(f"factor {name} output is not a graph output")
                factor["output"] = output
            normalized.append(factor)
        checked_payload = {"factors": normalized}
    elif kind == "operator-pack":
        operators = manifest.get("operators")
        if not isinstance(operators, list) or not operators:
            raise CapabilityError("operator pack requires at least one operator")
        names = set()
        normalized = []
        for raw in operators:
            if not isinstance(raw, dict):
                raise CapabilityError("operator declarations must be objects")
            operator = dict(raw)
            name = str(operator.get("name") or "").strip().lower()
            if not _FACTOR_NAME_PATTERN.fullmatch(name) or "." not in name:
                raise CapabilityError(
                    "operator pack names must be namespaced identifiers such as vendor.zscore"
                )
            if name in names:
                raise CapabilityError(f"duplicate operator in pack: {name}")
            names.add(name)
            operator["name"] = name
            operator["entry"] = checked_entry(
                operator.get("entry"), f"operator {name}",
            )
            operator["category"] = str(
                operator.get("category") or "extension"
            ).strip()
            operator["description"] = str(
                operator.get("description") or "installed operator"
            ).strip()
            operator["signature"] = str(
                operator.get("signature") or name
            ).strip()
            normalized.append(operator)
        checked_payload = {"operators": normalized}
    elif kind == "benchmark-pack":
        benchmarks = manifest.get("benchmarks")
        if not isinstance(benchmarks, list) or not benchmarks:
            raise CapabilityError(
                "benchmark pack requires at least one benchmark"
            )
        ids: set[str] = set()
        normalized = []
        for raw in benchmarks:
            if not isinstance(raw, dict):
                raise CapabilityError("benchmark declarations must be objects")
            benchmark = dict(raw)
            benchmark_id = str(benchmark.get("id") or "").strip()
            if (
                not _BENCHMARK_ID_PATTERN.fullmatch(benchmark_id)
                or "." not in benchmark_id
            ):
                raise CapabilityError(
                    "benchmark pack ids must be namespaced, for example vendor.market-v1"
                )
            if benchmark_id in ids:
                raise CapabilityError(
                    f"duplicate benchmark in pack: {benchmark_id}"
                )
            ids.add(benchmark_id)
            for field in ("title", "market", "assetClass", "frequency"):
                value = str(benchmark.get(field) or "").strip()
                if not value:
                    raise CapabilityError(
                        f"benchmark {benchmark_id} requires {field}"
                    )
                benchmark[field] = value
            contract = benchmark.get("dataContract")
            if (
                not isinstance(contract, dict)
                or not isinstance(contract.get("required"), list)
                or not contract["required"]
                or not all(
                    isinstance(item, str) and item.strip()
                    for item in contract["required"]
                )
                or not isinstance(contract.get("identity"), list)
                or not {"ts_code", "trade_date"}.issubset(
                    set(contract["identity"])
                )
            ):
                raise CapabilityError(
                    f"benchmark {benchmark_id} has an invalid dataContract"
                )
            evaluation = benchmark.get("evaluation")
            if not isinstance(evaluation, dict):
                raise CapabilityError(
                    f"benchmark {benchmark_id} requires evaluation"
                )
            if str(evaluation.get("labelVersion") or "") != "next-open-adjusted/1":
                raise CapabilityError(
                    f"benchmark {benchmark_id} must use next-open-adjusted/1"
                )
            try:
                horizon = int(evaluation.get("horizon") or 1)
                minimum_names = int(
                    evaluation.get("minimumNamesPerDate") or 1
                )
                minimum_coverage = float(
                    evaluation.get("minimumCoverage") or 0.5
                )
                selection_ratio = float(
                    evaluation.get("selectionRatio") or 0.7
                )
            except (TypeError, ValueError) as error:
                raise CapabilityError(
                    f"benchmark {benchmark_id} has invalid evaluation limits"
                ) from error
            if (
                horizon < 1
                or minimum_names < 1
                or not 0.0 <= minimum_coverage <= 1.0
                or not 0.1 <= selection_ratio <= 0.9
            ):
                raise CapabilityError(
                    f"benchmark {benchmark_id} has invalid evaluation limits"
                )
            metrics = evaluation.get("metrics")
            if (
                not isinstance(metrics, list)
                or not metrics
                or not all(isinstance(item, str) and item for item in metrics)
            ):
                raise CapabilityError(
                    f"benchmark {benchmark_id} requires metrics"
                )
            benchmark["id"] = benchmark_id
            benchmark["evaluation"] = {
                **evaluation,
                "horizon": horizon,
                "minimumNamesPerDate": minimum_names,
                "minimumCoverage": minimum_coverage,
                "selectionRatio": selection_ratio,
            }
            normalized.append(benchmark)
        checked_payload = {"benchmarks": normalized}
    else:
        connectors = manifest.get("connectors")
        if not isinstance(connectors, list) or not connectors:
            raise CapabilityError(
                "connector pack requires at least one connector"
            )
        ids: set[str] = set()
        normalized = []
        for raw in connectors:
            if not isinstance(raw, dict):
                raise CapabilityError(
                    "connector declarations must be objects"
                )
            connector = dict(raw)
            connector_id = str(connector.get("id") or "").strip().lower()
            if (
                not _BENCHMARK_ID_PATTERN.fullmatch(connector_id)
                or "." not in connector_id
            ):
                raise CapabilityError(
                    "connector ids must be namespaced, for example vendor.market-v1"
                )
            if connector_id in ids:
                raise CapabilityError(
                    f"duplicate connector in pack: {connector_id}"
                )
            ids.add(connector_id)
            for field in ("title", "market", "kind", "pointInTime"):
                value = str(connector.get(field) or "").strip()
                if not value:
                    raise CapabilityError(
                        f"connector {connector_id} requires {field}"
                    )
                connector[field] = value
            connector["id"] = connector_id
            connector["kind"] = connector["kind"].lower()
            connector["entry"] = checked_entry(
                connector.get("entry"), f"connector {connector_id}",
            )
            connector["allowedHosts"] = _allowed_hosts(
                connector.get("allowedHosts"), connector_id,
            )
            if connector["kind"] != "local_file" and not connector["allowedHosts"]:
                raise CapabilityError(
                    f"connector {connector_id} requires allowedHosts for network access"
                )
            for field in ("credentials", "requires", "fields"):
                values = connector.get(field, [])
                if not isinstance(values, list) or not all(
                    isinstance(item, str) and item.strip()
                    for item in values
                ):
                    raise CapabilityError(
                        f"connector {connector_id} has invalid {field}"
                    )
                connector[field] = list(dict.fromkeys(
                    item.strip() for item in values
                ))
            normalized.append(connector)
        checked_payload = {"connectors": normalized}
    checked = dict(manifest)
    signature_info = _signature_state(manifest, trusted_key=trusted_key)
    checked.update(
        id=pack_id,
        version=version,
        compatibility={
            "protocol": declared_protocol,
            "alphaGraph": str(compatibility.get("alphaGraph") or "alpha-graph-1"),
            "operators": int(compatibility.get("operators") or 2),
        },
        signatureInfo=signature_info,
        **checked_payload,
    )
    return checked


def _manifest_at(path: Path) -> dict[str, Any] | None:
    manifest_path = path / "qcap.json"
    if not manifest_path.is_file():
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _pack_version_paths(pack_dir: Path) -> list[Path]:
    if not pack_dir.is_dir():
        return []
    return [
        path for path in sorted(pack_dir.iterdir())
        if path.is_dir() and (path / "qcap.json").is_file()
    ]


def _active_pointer(pack_dir: Path) -> Path:
    return pack_dir / "active.json"


def _active_version_path(pack_dir: Path) -> Path | None:
    versions = _pack_version_paths(pack_dir)
    if not versions:
        return None
    pointer = _active_pointer(pack_dir)
    try:
        value = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        value = None
    if (
        isinstance(value, dict)
        and value.get("format") == ACTIVE_POINTER_FORMAT
        and isinstance(value.get("version"), str)
    ):
        expected = str(value["version"])
        for path in versions:
            manifest = _manifest_at(path)
            if manifest is not None and str(manifest.get("version")) == expected:
                return path
    # Legacy installations may contain multiple versions without a pointer.
    # Prefer the most recently installed timestamp, then a deterministic
    # version name, instead of loading duplicate entries.
    def installed_order(path: Path) -> tuple[str, str]:
        manifest = _manifest_at(path) or {}
        return (str(manifest.get("installedAt") or ""), path.name)

    return max(versions, key=installed_order)


def _write_active_version(
    pack_dir: Path,
    *,
    kind: str,
    pack_id: str,
    version: str,
) -> None:
    pack_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": ACTIVE_POINTER_FORMAT,
        "kind": kind,
        "id": pack_id,
        "version": version,
        "updatedAt": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    temporary = pack_dir / f".active-{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, _active_pointer(pack_dir))


def _installed_pack_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for kind in CAPABILITY_KINDS:
        kind_root = root / kind
        if not kind_root.is_dir():
            continue
        for pack_dir in sorted(kind_root.iterdir()):
            if not pack_dir.is_dir():
                continue
            active = _active_version_path(pack_dir)
            if active is not None:
                paths.append(active)
    return paths


def list_capabilities(root: str | Path | None = None) -> dict[str, Any]:
    capability_dir = capability_root(root)
    packs: list[dict[str, Any]] = []
    for path in _installed_pack_paths(capability_dir):
        manifest = _manifest_at(path)
        if manifest is None:
            packs.append({
                "path": str(path),
                "status": "INVALID_MANIFEST",
            })
            continue
        try:
            signature_info = _signature_state(manifest)
        except CapabilityError as error:
            packs.append({
                "path": str(path),
                "status": "INVALID_SIGNATURE",
                "error": str(error),
            })
            continue
        factors = manifest.get("factors") if isinstance(manifest.get("factors"), list) else []
        operators = manifest.get("operators") if isinstance(
            manifest.get("operators"), list
        ) else []
        connectors = manifest.get("connectors") if isinstance(
            manifest.get("connectors"), list
        ) else []
        packs.append({
            "format": manifest.get("format"),
            "kind": manifest.get("kind"),
            "id": manifest.get("id"),
            "version": manifest.get("version"),
            "displayName": manifest.get("displayName") or manifest.get("id"),
            "status": "INSTALLED",
            "path": str(path),
            "factors": [
                str(item.get("name"))
                for item in factors
                if isinstance(item, dict) and item.get("name")
            ],
            "operators": [
                str(item.get("name"))
                for item in operators
                if isinstance(item, dict) and item.get("name")
            ],
            "connectors": [
                str(item.get("id"))
                for item in connectors
                if isinstance(item, dict) and item.get("id")
            ],
            "signature": signature_info,
            "compatibility": manifest.get("compatibility"),
            "availableVersions": [
                item.name for item in _pack_version_paths(path.parent)
            ],
            "retainedVersions": [
                item.name for item in _pack_version_paths(path.parent)
                if item != path
            ],
        })
    packs.sort(key=lambda item: (
        str(item.get("kind") or ""), str(item.get("id") or ""),
        str(item.get("version") or ""),
    ))
    return {
        "format": CAPABILITY_FORMAT,
        "status": "CAPABILITIES",
        "root": str(capability_dir),
        "count": len(packs),
        "packs": packs,
    }


def _entry_owners(
    root: Path,
    field: str,
    *,
    key: str = "name",
    exclude: Path | None = None,
    exclude_pack: tuple[str, str] | None = None,
) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    excluded = exclude.resolve() if exclude is not None else None
    for path in _installed_pack_paths(root):
        if excluded is not None and path.resolve() == excluded:
            continue
        manifest = _manifest_at(path)
        if manifest is None:
            continue
        identity_key = (
            str(manifest.get("kind") or ""),
            str(manifest.get("id") or ""),
        )
        if exclude_pack is not None and identity_key == exclude_pack:
            continue
        identity = f"{manifest.get('id')}@{manifest.get('version')}"
        entries = manifest.get(field)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and entry.get(key):
                owners.setdefault(str(entry[key]), []).append(identity)
    return owners


def install_capability(
    source: str | Path,
    *,
    root: str | Path | None = None,
    upgrade: bool = False,
    trusted_key: str | Path | None = None,
) -> dict[str, Any]:
    files, modes = _read_source(source)
    manifest = _checked_manifest(
        _load_manifest(files), files, trusted_key=trusted_key,
    )
    if sum(len(value) for value in files.values()) > MAX_PACK_BYTES:
        raise CapabilityError("capability pack exceeds its uncompressed size limit")
    capability_dir = capability_root(root)
    kind = str(manifest["kind"])
    pack_id = str(manifest["id"])
    version = str(manifest["version"])
    for other_kind in CAPABILITY_KINDS:
        if other_kind == kind:
            continue
        other_active = _active_version_path(
            capability_dir / other_kind / pack_id
        )
        if other_active is not None:
            raise CapabilityError(
                f"capability id is already active as {other_kind}: {pack_id}"
            )
    target = (
        capability_dir
        / kind
        / pack_id
        / version
    )
    pack_dir = target.parent
    active = _active_version_path(pack_dir)
    if active is not None and active.resolve() != target.resolve() and not upgrade:
        previous = _manifest_at(active) or {}
        raise CapabilityError(
            "capability version is already active: "
            f"{pack_id}@{previous.get('version')}; use upgrade to activate "
            f"{version} while retaining previous versions"
        )
    if target.exists() and not upgrade:
        raise CapabilityError(
            f"capability already installed: {manifest['id']}@{manifest['version']}"
        )
    replaced = target.exists()
    for field, label in (
        ("factors", "factor"),
        ("operators", "operator"),
        ("connectors", "connector"),
    ):
        key = "id" if field == "connectors" else "name"
        owners = _entry_owners(
            capability_dir, field, key=key, exclude=target,
            exclude_pack=(kind, pack_id),
        )
        conflicts = {
            entry[key]: owners.get(entry[key], [])
            for entry in manifest.get(field, [])
            if owners.get(entry[key])
        }
        if conflicts:
            detail = ", ".join(
                f"{name} ({', '.join(values)})"
                for name, values in sorted(conflicts.items())
            )
            raise CapabilityError(f"{label} names already installed: {detail}")

    if manifest["kind"] == "benchmark-pack":
        owners: dict[str, list[str]] = {}
        for path in _installed_pack_paths(capability_dir):
            other = _manifest_at(path)
            if (
                other is not None
                and other.get("kind") == kind
                and other.get("id") == pack_id
            ):
                continue
            if path.resolve() == target.resolve():
                continue
            if other is None or other.get("kind") != "benchmark-pack":
                continue
            identity = f"{other.get('id')}@{other.get('version')}"
            for benchmark in other.get("benchmarks") or []:
                if isinstance(benchmark, dict) and benchmark.get("id"):
                    owners.setdefault(str(benchmark["id"]), []).append(identity)
        conflicts = {
            benchmark["id"]: owners.get(benchmark["id"], [])
            for benchmark in manifest.get("benchmarks", [])
            if owners.get(benchmark["id"])
        }
        if conflicts:
            detail = ", ".join(
                f"{name} ({', '.join(values)})"
                for name, values in sorted(conflicts.items())
            )
            raise CapabilityError(
                f"benchmark ids already installed: {detail}"
            )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".install-", dir=str(target.parent)))
    try:
        for relative, content in files.items():
            destination = staging.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            mode = modes.get(relative, 0o644) & 0o777
            try:
                destination.chmod(mode or 0o644)
            except OSError:
                pass
        installed = dict(manifest)
        installed["installedAt"] = datetime.now(UTC).replace(microsecond=0).isoformat()
        installed["contentSha256"] = _sha256_bytes(_canonical_json({
            "manifest": manifest,
            "files": [
                {
                    "path": path,
                    "sha256": _sha256_bytes(files[path]),
                }
                for path in sorted(files)
            ],
        }))
        (staging / "qcap.json").write_text(
            json.dumps(installed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        backup = target.with_name(f".{target.name}.backup-{os.getpid()}")
        if target.exists():
            if backup.exists():
                shutil.rmtree(backup)
            target.replace(backup)
        try:
            staging.replace(target)
        except Exception:
            if backup.exists() and not target.exists():
                backup.replace(target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    _write_active_version(
        pack_dir,
        kind=kind,
        pack_id=pack_id,
        version=version,
    )
    retained = [
        path.name for path in _pack_version_paths(pack_dir)
        if path.resolve() != target.resolve()
    ]
    return {
        "format": CAPABILITY_FORMAT,
        "status": "INSTALLED",
        "root": str(capability_dir),
        "path": str(target),
        "id": manifest["id"],
        "version": manifest["version"],
        "kind": manifest["kind"],
        "factors": [
            factor["name"] for factor in manifest.get("factors", [])
        ],
        "operators": [
            operator["name"] for operator in manifest.get("operators", [])
        ],
        "benchmarks": [
            benchmark["id"] for benchmark in manifest.get("benchmarks", [])
        ],
        "connectors": [
            connector["id"] for connector in manifest.get("connectors", [])
        ],
        **manifest.get("signatureInfo", {}),
        "contentSha256": installed["contentSha256"],
        "activeVersion": version,
        "replaced": replaced,
        "retainedVersions": retained,
    }


def activate_capability(
    pack_id: str,
    version: str,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Activate one retained version without deleting other versions."""
    normalized = str(pack_id or "").strip()
    if not _ID_PATTERN.fullmatch(normalized):
        raise CapabilityError("capability id is invalid")
    normalized_version = str(version or "").strip()
    if not _VERSION_PATTERN.fullmatch(normalized_version):
        raise CapabilityError("capability version is invalid")
    capability_dir = capability_root(root)
    matches: list[Path] = []
    for kind in CAPABILITY_KINDS:
        pack_dir = capability_dir / kind / normalized
        for path in _pack_version_paths(pack_dir):
            manifest = _manifest_at(path) or {}
            if str(manifest.get("version")) == normalized_version:
                matches.append(path)
    if len(matches) != 1:
        raise CapabilityError(
            "retained capability version not found: "
            f"{normalized}@{normalized_version}"
        )
    path = matches[0]
    manifest = _manifest_at(path)
    if manifest is None:
        raise CapabilityError("retained capability manifest is unavailable")
    _signature_state(manifest)
    _write_active_version(
        path.parent,
        kind=str(manifest.get("kind") or ""),
        pack_id=normalized,
        version=normalized_version,
    )
    return {
        "format": CAPABILITY_FORMAT,
        "status": "ACTIVATED",
        "id": normalized,
        "version": normalized_version,
        "kind": manifest.get("kind"),
        "path": str(path),
    }


def remove_capability(
    pack_id: str,
    *,
    version: str | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    normalized = str(pack_id or "").strip()
    if not _ID_PATTERN.fullmatch(normalized):
        raise CapabilityError("capability id is invalid")
    capability_dir = capability_root(root)
    removed: list[str] = []
    touched_pack_dirs: list[Path] = []
    for kind in CAPABILITY_KINDS:
        pack_dir = capability_dir / kind / normalized
        for path in _pack_version_paths(pack_dir):
            manifest = _manifest_at(path)
            if manifest is None or manifest.get("id") != normalized:
                continue
            if version is not None and str(manifest.get("version")) != str(version):
                continue
            removed.append(str(path))
            shutil.rmtree(path)
            touched_pack_dirs.append(pack_dir)
    if not removed:
        raise CapabilityError(
            f"installed capability not found: {normalized}"
            + (f"@{version}" if version else "")
        )
    active_after: dict[str, str | None] = {}
    for pack_dir in sorted(set(touched_pack_dirs)):
        remaining = _pack_version_paths(pack_dir)
        if not remaining:
            try:
                _active_pointer(pack_dir).unlink()
            except OSError:
                pass
            active_after[str(pack_dir.parent.name)] = None
            try:
                pack_dir.rmdir()
            except OSError:
                pass
            continue
        selected = _active_version_path(pack_dir) or remaining[-1]
        manifest = _manifest_at(selected) or {}
        _write_active_version(
            pack_dir,
            kind=str(manifest.get("kind") or pack_dir.parent.name),
            pack_id=normalized,
            version=str(manifest.get("version") or selected.name),
        )
        active_after[str(pack_dir.parent.name)] = str(manifest.get("version"))
    return {
        "format": CAPABILITY_FORMAT,
        "status": "REMOVED",
        "id": normalized,
        "version": version,
        "removed": removed,
        "activeVersionByKind": active_after,
    }


def _factor_entries(root: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    entries: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in _installed_pack_paths(root):
        manifest = _manifest_at(path)
        if manifest is None or manifest.get("kind") != "factor-pack":
            continue
        factors = manifest.get("factors")
        if not isinstance(factors, list):
            continue
        for factor in factors:
            if not isinstance(factor, dict) or not factor.get("name"):
                continue
            name = str(factor["name"])
            previous = entries.get(name)
            if previous is not None:
                raise CapabilityError(
                    f"duplicate installed factor {name}: "
                    f"{previous[1]} and {path}"
                )
            entries[name] = (factor, path)
    return entries


def installed_factor_specs(
    *,
    root: str | Path | None = None,
    fields: set[str] | None = None,
) -> list[dict[str, Any]]:
    entries = _factor_entries(capability_root(root))
    result: list[dict[str, Any]] = []
    for name, (factor, path) in sorted(entries.items()):
        required = list(factor.get("requiredFields") or [])
        if fields is not None and not set(required).issubset(fields):
            continue
        manifest = _manifest_at(path) or {}
        result.append({
            "name": name,
            "family": str(
                factor.get("family")
                or manifest.get("id")
                or "factor-pack"
            ),
            "source": f"qcap:{manifest.get('id')}@{manifest.get('version')}",
            "library": str(manifest.get("displayName") or manifest.get("id") or "Factor Pack"),
            "license": str(manifest.get("license") or "declared-by-pack"),
            "inputs": required,
            "requiredColumns": required,
            "output": "factor_value",
            "outputs": ["factor_value"],
            "implementation": "installed_factor_pack",
            "minimumHistory": int(factor.get("minimumHistory") or 1),
            "pack": {
                "id": manifest.get("id"),
                "version": manifest.get("version"),
                "kind": manifest.get("kind"),
                "contentSha256": manifest.get("contentSha256"),
            },
        })
    return result


def installed_factor_names(
    *,
    root: str | Path | None = None,
    fields: set[str] | None = None,
) -> list[str]:
    return [item["name"] for item in installed_factor_specs(root=root, fields=fields)]


def _operator_entries(root: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    entries: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in _installed_pack_paths(root):
        manifest = _manifest_at(path)
        if manifest is None or manifest.get("kind") != "operator-pack":
            continue
        operators = manifest.get("operators")
        if not isinstance(operators, list):
            continue
        for operator in operators:
            if not isinstance(operator, dict) or not operator.get("name"):
                continue
            name = str(operator["name"]).lower()
            previous = entries.get(name)
            if previous is not None:
                raise CapabilityError(
                    f"duplicate installed operator {name}: "
                    f"{previous[1]} and {path}"
                )
            entries[name] = (operator, path)
    return entries


def installed_operator_specs(
    *, root: str | Path | None = None,
) -> list[dict[str, Any]]:
    root_path = capability_root(root)
    result: list[dict[str, Any]] = []
    for name, (operator, path) in sorted(_operator_entries(root_path).items()):
        manifest = _manifest_at(path) or {}
        result.append({
            "name": name,
            "category": str(operator.get("category") or "extension"),
            "description": str(operator.get("description") or "installed operator"),
            "signature": str(operator.get("signature") or name),
            "causal": bool(operator.get("causal", True)),
            "implementation": "installed_operator_pack",
            "pack": {
                "id": manifest.get("id"),
                "version": manifest.get("version"),
                "contentSha256": manifest.get("contentSha256"),
            },
        })
    return result


def _connector_entries(root: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    entries: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in _installed_pack_paths(root):
        manifest = _manifest_at(path)
        if manifest is None or manifest.get("kind") != "connector-pack":
            continue
        connectors = manifest.get("connectors")
        if not isinstance(connectors, list):
            continue
        for connector in connectors:
            if not isinstance(connector, dict) or not connector.get("id"):
                continue
            connector_id = str(connector["id"]).lower()
            previous = entries.get(connector_id)
            if previous is not None:
                raise CapabilityError(
                    f"duplicate installed connector {connector_id}: "
                    f"{previous[1]} and {path}"
                )
            entries[connector_id] = (connector, path)
    return entries


def installed_connector_specs(
    *, root: str | Path | None = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for _connector_id, (connector, path) in sorted(
        _connector_entries(capability_root(root)).items()
    ):
        manifest = _manifest_at(path) or {}
        result.append({
            **json.loads(json.dumps(connector)),
            "pack": {
                "id": manifest.get("id"),
                "version": manifest.get("version"),
                "kind": manifest.get("kind"),
                "contentSha256": manifest.get("contentSha256"),
            },
        })
    return result


def installed_connector_callable(
    connector_id: str,
    *, root: str | Path | None = None,
) -> Any | None:
    entry = _connector_entries(capability_root(root)).get(
        str(connector_id).lower()
    )
    if entry is None:
        return None
    connector, pack_path = entry
    return _load_pack_callable(pack_path, str(connector["entry"]))


def installed_benchmark_specs(
    *, root: str | Path | None = None,
) -> list[dict[str, Any]]:
    entries: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in _installed_pack_paths(capability_root(root)):
        manifest = _manifest_at(path)
        if manifest is None or manifest.get("kind") != "benchmark-pack":
            continue
        for benchmark in manifest.get("benchmarks") or []:
            if not isinstance(benchmark, dict) or not benchmark.get("id"):
                continue
            benchmark_id = str(benchmark["id"])
            previous = entries.get(benchmark_id)
            if previous is not None:
                raise CapabilityError(
                    f"duplicate installed benchmark {benchmark_id}: "
                    f"{previous[1]} and {path}"
                )
            entries[benchmark_id] = (benchmark, path)
    result: list[dict[str, Any]] = []
    for _benchmark_id, (benchmark, path) in sorted(entries.items()):
        manifest = _manifest_at(path) or {}
        result.append({
            **json.loads(json.dumps(benchmark)),
            "pack": {
                "id": manifest.get("id"),
                "version": manifest.get("version"),
                "kind": manifest.get("kind"),
                "contentSha256": manifest.get("contentSha256"),
            },
        })
    return result


def _verified_pack_member(path: Path, relative: str) -> bytes:
    manifest = _manifest_at(path)
    if manifest is None:
        raise CapabilityError(f"installed capability manifest is unavailable: {path}")
    _signature_state(manifest)
    members = manifest.get("files")
    if not isinstance(members, list):
        raise CapabilityError("installed capability has no files")
    expected = next(
        (
            str(item.get("sha256") or "")
            for item in members
            if isinstance(item, dict) and item.get("path") == relative
        ),
        None,
    )
    if expected is None:
        raise CapabilityError(f"installed capability member is undeclared: {relative}")
    target = path.joinpath(*PurePosixPath(relative).parts)
    content = target.read_bytes()
    if _sha256_bytes(content) != expected:
        raise CapabilityError(f"installed capability member changed: {relative}")
    return content


def _load_pack_callable(pack_path: Path, entry: str) -> Any:
    relative, function_name = str(entry).split(":", 1)
    content = _verified_pack_member(pack_path, relative)
    source = content.decode("utf-8")
    from quant_sdk.local import _safe_builtins, audit_factor_source

    audit_factor_source(source)
    environment = {
        "__builtins__": _safe_builtins(),
        "__name__": f"__qcap_{pack_path.parent.parent.name.replace('.', '_')}__",
        "pd": pd,
        "pandas": pd,
        "np": np,
        "numpy": np,
    }
    exec(compile(source, str(pack_path / relative), "exec"), environment)
    function = environment.get(function_name)
    if not callable(function):
        raise CapabilityError(f"capability entry is not callable: {entry}")
    return function


def installed_operator_callable(
    name: str,
    *,
    root: str | Path | None = None,
) -> Any | None:
    entry = _operator_entries(capability_root(root)).get(str(name).lower())
    if entry is None:
        return None
    operator, pack_path = entry
    return _load_pack_callable(pack_path, str(operator["entry"]))


def compute_pack_factor(
    name: str,
    frame: pd.DataFrame,
    *,
    root: str | Path | None = None,
) -> pd.Series:
    stack = _COMPUTE_STACK.get()
    if name in stack:
        chain = " -> ".join((*stack, name))
        raise CapabilityError(f"cyclic capability factor dependency: {chain}")
    token = _COMPUTE_STACK.set((*stack, name))
    try:
        entry = _factor_entries(capability_root(root)).get(name)
        if entry is None:
            raise CapabilityError(f"installed factor not found: {name}")
        factor, pack_path = entry
        required = set(factor.get("requiredFields") or [])
        missing = required.difference(frame.columns)
        if missing:
            raise CapabilityError(
                f"factor {name} is missing fields: {', '.join(sorted(missing))}"
            )
        source_entry = factor.get("entry")
        if source_entry:
            function = _load_pack_callable(pack_path, str(source_entry))
            values = function(frame.copy())
        else:
            from quant_sdk.alpha_graph import evaluate_graph

            graph = factor.get("graph")
            if not isinstance(graph, dict):
                raise CapabilityError(f"factor {name} graph is unavailable")
            output, _ = evaluate_graph(
                frame,
                graph,
                horizon=int(factor.get("horizon") or 1),
                min_names=int(factor.get("minNames") or 5),
            )
            column = str(factor.get("output") or graph["outputs"][0])
            if column not in output.columns:
                raise CapabilityError(f"factor {name} graph output is unavailable")
            values = output[column]

        if isinstance(values, pd.DataFrame):
            if "factor_value" in values.columns:
                values = values["factor_value"]
            elif values.shape[1] == 1:
                values = values.iloc[:, 0]
        if not isinstance(values, pd.Series):
            raise CapabilityError(f"factor {name} must return pandas.Series")
        if len(values) != len(frame):
            raise CapabilityError(
                f"factor {name} returned {len(values)} rows for {len(frame)} inputs"
            )
        series = pd.to_numeric(values.reset_index(drop=True), errors="coerce")
        series.index = frame.index
        series.name = name
        if not np.isfinite(series.to_numpy(dtype=float)).any():
            raise CapabilityError(f"factor {name} has no finite output")
        return series.astype(float)
    finally:
        _COMPUTE_STACK.reset(token)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def sign_capability(
    manifest_path: str | Path,
    *,
    private_key: str | Path,
) -> dict[str, Any]:
    """Sign one local qcap manifest over its immutable contents."""
    target = Path(manifest_path).expanduser().resolve()
    if target.is_dir():
        target = target / "qcap.json"
    if target.name != "qcap.json" or not target.is_file():
        raise CapabilityError("capability sign requires a qcap.json file")
    files, _modes = _read_directory(target.parent)
    manifest = _load_manifest(files)
    _checked_manifest(manifest, files)
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )

        key = serialization.load_pem_private_key(
            Path(private_key).expanduser().read_bytes(),
            password=None,
        )
        if not isinstance(key, Ed25519PrivateKey):
            raise TypeError("signing key is not Ed25519")
        public_key = key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        body = _signable_manifest(manifest)
        signature = key.sign(_canonical_json(body))
    except Exception as error:
        raise CapabilityError(
            f"capability signing key is invalid: {error}"
        ) from error
    signed = {
        **body,
        "signature": {
            "algorithm": "ed25519",
            "publicKey": base64.b64encode(public_key).decode("ascii"),
            "value": base64.b64encode(signature).decode("ascii"),
        },
    }
    temporary = target.with_suffix(".json.next")
    temporary.write_text(
        json.dumps(signed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return {
        "format": CAPABILITY_FORMAT,
        "status": "SIGNED",
        "manifest": str(target),
        "id": manifest["id"],
        "version": manifest["version"],
        "kind": manifest["kind"],
        **_signature_state(signed),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    listing = subparsers.add_parser("list")
    listing.add_argument("--json", action="store_true")
    install = subparsers.add_parser("install")
    install.add_argument("--file", required=True)
    install.add_argument("--upgrade", action="store_true")
    install.add_argument("--trusted-key")
    install.add_argument("--json", action="store_true")
    signing = subparsers.add_parser("sign")
    signing.add_argument("--file", required=True)
    signing.add_argument("--key", required=True)
    signing.add_argument("--json", action="store_true")
    remove = subparsers.add_parser("remove")
    remove.add_argument("--id", required=True)
    remove.add_argument("--version")
    remove.add_argument("--json", action="store_true")
    activate = subparsers.add_parser("activate")
    activate.add_argument("--id", required=True)
    activate.add_argument("--version", required=True)
    activate.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "list":
        result = list_capabilities(args.root)
    elif args.command == "install":
        result = install_capability(
            args.file, root=args.root, upgrade=bool(args.upgrade),
            trusted_key=args.trusted_key,
        )
    elif args.command == "sign":
        result = sign_capability(args.file, private_key=args.key)
    elif args.command == "activate":
        result = activate_capability(
            args.id, args.version, root=args.root,
        )
    else:
        result = remove_capability(
            args.id, version=args.version, root=args.root,
        )
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
