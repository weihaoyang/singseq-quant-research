"""Independent Python reference host for the Quant Research Protocol.

This is a transport adapter over the public ``quant_sdk`` modules. It does not
replace the desktop client; it proves that the protocol can be implemented
without Electron or Node and gives macOS/Linux users a headless local research
node before a packaged desktop build is available.

The host exposes MCP over stdio or loopback Streamable HTTP. Every research
operation runs in this Python process on the user's machine. Tool calls never
submit, publish, rank, purchase, or change permissions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TextIO

PROTOCOL_NAME = "quant-research-protocol"
PROTOCOL_VERSION = "2.0"
TOOL_SURFACE_PROFILE = "core"
RUNTIME_VERSION = "1.0"
MCP_PROTOCOL_VERSION = "2025-06-18"
SEMANTIC_FORMAT = "qprotocol-semantic/1"
SEMANTIC_VECTOR = "alpha-graph-operator-vector/1"
TOOL_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["ok"],
    "additionalProperties": False,
    "properties": {
        "ok": {"type": "boolean"},
        "data": {},
        "error": {
            "type": "object",
            "required": ["code", "message"],
            "additionalProperties": False,
            "properties": {
                "code": {"type": "string"},
                "message": {"type": "string"},
                "stage": {"type": "string"},
                "status": {"type": "integer"},
                "details": {},
            },
        },
    },
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _structured(value: Any) -> dict[str, Any]:
    return {"ok": True, "data": _jsonable(value)}


def _read_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError("expected a string or string array")


def _schema(
    properties: dict[str, Any] | None = None,
    *,
    required: list[str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": properties or {},
    }
    if required:
        result["required"] = required
    return result


def _confirm_local_code_change(description: str) -> None:
    """Require a visible native approval before executable pack changes.

    A model-supplied confirmation string is not user authorization. The
    reference host therefore asks the local desktop user directly. On a truly
    headless machine the operation fails closed; use the desktop client or a
    different host that provides an equivalent human approval surface.
    """
    try:
        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        try:
            approved = bool(
                messagebox.askyesno(
                    "Quant Research Protocol",
                    description + "\n\n只批准这一次本地变更。",
                    parent=root,
                )
            )
        finally:
            root.destroy()
    except Exception as error:
        raise RuntimeError(
            "无法显示本机用户授权对话框；可执行能力包变更已拒绝。"
            "请在桌面客户端中批准此操作。"
        ) from error
    if not approved:
        raise PermissionError("用户拒绝执行本地代码变更。")


STRING = {"type": "string"}
INTEGER = {"type": "integer"}
NUMBER = {"type": "number"}
BOOLEAN = {"type": "boolean"}
OBJECT = {"type": "object"}
STRING_ARRAY = {"type": "array", "items": STRING}


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "quant_app_info",
        "description": "Discover the local Python protocol runtime and execution boundary.",
        "inputSchema": _schema(),
        "readOnly": True,
        "openWorld": False,
    },
    {
        "name": "quant_agent_handshake",
        "description": "Read one machine-readable local-runtime readiness and security snapshot.",
        "inputSchema": _schema(),
        "readOnly": True,
        "openWorld": False,
    },
    {
        "name": "quant_capabilities",
        "description": "List the protocol host's stable tool and command surface.",
        "inputSchema": _schema(),
        "readOnly": True,
        "openWorld": False,
    },
    {
        "name": "quant_data_inspect",
        "description": "Inspect a local CSV, Parquet, DuckDB or DB research panel.",
        "inputSchema": _schema(
            {
                "dataPath": STRING,
                "startDate": STRING,
                "endDate": STRING,
                "benchmarkCode": STRING,
            },
            required=["dataPath"],
        ),
        "readOnly": True,
        "openWorld": False,
    },
    {
        "name": "quant_local_capabilities",
        "description": "Read the portable host's Python, CUDA, factor, operator, capability and connector readiness snapshot.",
        "inputSchema": _schema(),
        "readOnly": True,
        "openWorld": False,
    },
    {
        "name": "quant_factor_operators",
        "description": "List the causal operator catalogue accepted by Alpha Graph.",
        "inputSchema": _schema(),
        "readOnly": True,
        "openWorld": False,
    },
    {
        "name": "quant_capability_list",
        "description": "List installed local Factor Packs without executing their code.",
        "inputSchema": _schema(
            {"root": STRING},
        ),
        "readOnly": True,
        "openWorld": False,
    },
    {
        "name": "quant_capability_install",
        "description": "Validate and install executable local capability code after a native user approval dialog.",
        "inputSchema": _schema(
            {
                "file": STRING,
                "root": STRING,
                "upgrade": BOOLEAN,
                "trustedKey": STRING,
            },
            required=["file"],
        ),
        "readOnly": False,
        "destructive": True,
        "openWorld": False,
    },
    {
        "name": "quant_capability_remove",
        "description": "Remove an installed capability version after a native user approval dialog.",
        "inputSchema": _schema(
            {
                "id": STRING,
                "version": STRING,
                "root": STRING,
            },
            required=["id"],
        ),
        "readOnly": False,
        "destructive": True,
        "openWorld": False,
    },
    {
        "name": "quant_capability_activate",
        "description": "Activate one retained capability version after a native user approval dialog.",
        "inputSchema": _schema(
            {
                "id": STRING,
                "version": STRING,
                "root": STRING,
            },
            required=["id", "version"],
        ),
        "readOnly": False,
        "destructive": True,
        "openWorld": False,
    },
    {
        "name": "quant_capability_sign",
        "description": "Sign a local qcap manifest with an Ed25519 publisher key.",
        "inputSchema": _schema(
            {
                "file": STRING,
                "signingKey": STRING,
            },
            required=["file", "signingKey"],
        ),
        "readOnly": False,
        "openWorld": False,
    },
    {
        "name": "quant_alpha_graph",
        "description": "Evaluate a named-factor set or Alpha Graph DAG on a local panel.",
        "inputSchema": _schema(
            {
                "dataPath": STRING,
                "startDate": STRING,
                "endDate": STRING,
                "factorNames": STRING_ARRAY,
                "specPath": STRING,
                "outputDir": STRING,
                "benchmarkCode": STRING,
                "horizon": INTEGER,
                "minNames": INTEGER,
                "backend": {
                    "type": "string",
                    "enum": ["auto", "cpu", "gpu"],
                },
            },
            required=["dataPath", "startDate", "endDate"],
        ),
        "readOnly": False,
        "openWorld": False,
    },
    {
        "name": "quant_factor_screen",
        "description": "Screen many local factor candidates and reject highly correlated names.",
        "inputSchema": _schema(
            {
                "dataPath": STRING,
                "startDate": STRING,
                "endDate": STRING,
                "factorNames": STRING_ARRAY,
                "families": STRING_ARRAY,
                "query": STRING,
                "scanLimit": INTEGER,
                "top": INTEGER,
                "chunkSize": INTEGER,
                "maxCorrelation": NUMBER,
                "minimumCoverage": NUMBER,
                "minimumIcDays": INTEGER,
                "minNames": INTEGER,
                "horizon": INTEGER,
                "workers": INTEGER,
                "benchmarkCode": STRING,
                "output": STRING,
            },
            required=["dataPath", "startDate", "endDate"],
        ),
        "readOnly": False,
        "openWorld": False,
    },
    {
        "name": "quant_evidence_pack",
        "description": "Create a portable content-addressed .qalpha Evidence Pack.",
        "inputSchema": _schema(
            {
                "output": STRING,
                "name": STRING,
                "version": STRING,
                "sourceFile": STRING,
                "graphFile": STRING,
                "factorName": STRING,
                "dataPath": STRING,
                "startDate": STRING,
                "endDate": STRING,
                "benchmarkCode": STRING,
                "resultFile": STRING,
                "auditFile": STRING,
                "familyAuditFile": STRING,
                "metadata": OBJECT,
                "signingKey": STRING,
                "signer": STRING,
            },
            required=["output", "name"],
        ),
        "readOnly": False,
        "openWorld": False,
    },
    {
        "name": "quant_evidence_verify",
        "description": "Verify every digest and optional signature in a .qalpha pack.",
        "inputSchema": _schema(
            {
                "file": STRING,
                "trustedKey": STRING,
                "dataPath": STRING,
                "startDate": STRING,
                "endDate": STRING,
                "benchmarkCode": STRING,
            },
            required=["file"],
        ),
        "readOnly": True,
        "openWorld": False,
    },
    {
        "name": "quant_evidence_reproduce",
        "description": "Re-run a graph or catalog pack locally and compare canonical values.",
        "inputSchema": _schema(
            {
                "file": STRING,
                "dataPath": STRING,
                "startDate": STRING,
                "endDate": STRING,
                "benchmarkCode": STRING,
            },
            required=["file", "dataPath"],
        ),
        "readOnly": False,
        "openWorld": False,
    },
    {
        "name": "quant_evidence_audit",
        "description": "Audit leakage, purged holdout, costs, benchmark and capacity locally.",
        "inputSchema": _schema(
            {
                "dataPath": STRING,
                "startDate": STRING,
                "endDate": STRING,
                "factorName": STRING,
                "sourceFile": STRING,
                "name": STRING,
                "selectionRatio": NUMBER,
                "purgeDays": INTEGER,
                "embargoDays": INTEGER,
                "horizon": INTEGER,
                "minNames": INTEGER,
                "groups": INTEGER,
                "costs": BOOLEAN,
                "allowShort": BOOLEAN,
                "trials": INTEGER,
                "direction": INTEGER,
                "benchmarkCode": STRING,
                "output": STRING,
            },
            required=["dataPath", "startDate", "endDate"],
        ),
        "readOnly": False,
        "openWorld": False,
    },
    {
        "name": "quant_family_audit",
        "description": "Seal local aligned candidate-family PBO/CSCV, Deflated Sharpe and multiple-testing evidence.",
        "inputSchema": _schema(
            {
                "dataPath": STRING,
                "candidatesFile": STRING,
                "startDate": STRING,
                "endDate": STRING,
                "familyId": STRING,
                "selectionRatio": NUMBER,
                "purgeDays": INTEGER,
                "embargoDays": INTEGER,
                "horizon": INTEGER,
                "minNames": INTEGER,
                "groups": INTEGER,
                "costs": BOOLEAN,
                "allowShort": BOOLEAN,
                "benchmarkCode": STRING,
                "pboSplits": INTEGER,
                "alpha": NUMBER,
                "output": STRING,
            },
            required=["dataPath", "candidatesFile", "startDate", "endDate"],
        ),
        "readOnly": False,
        "openWorld": False,
    },
    {
        "name": "quant_connector_list",
        "description": "Discover local, warehouse and market-data connectors without secrets.",
        "inputSchema": _schema(),
        "readOnly": True,
        "openWorld": False,
    },
    {
        "name": "quant_connector_pull",
        "description": "Query one provider locally and write a normalized research panel.",
        "inputSchema": _schema(
            {
                "provider": STRING,
                "output": STRING,
                "startDate": STRING,
                "endDate": STRING,
                "inputPath": STRING,
                "symbols": STRING_ARRAY,
                "query": STRING,
                "options": OBJECT,
                "tableKind": {
                    "type": "string",
                    "enum": [
                        "bars",
                        "reference",
                        "fundamental",
                        "event",
                        "flow",
                    ],
                },
                "identity": STRING_ARRAY,
                "eventTime": STRING,
                "availableAt": STRING,
                "force": BOOLEAN,
            },
            required=["provider", "output", "startDate", "endDate"],
        ),
        "readOnly": False,
        "destructive": True,
        "openWorld": True,
    },
    {
        "name": "quant_connector_join",
        "description": "PIT as-of join two local qpanel tables by availability time.",
        "inputSchema": _schema(
            {
                "left": STRING,
                "right": STRING,
                "output": STRING,
                "identity": STRING_ARRAY,
                "leftTime": STRING,
                "rightEventTime": STRING,
                "rightAvailableAt": STRING,
                "rightKind": {
                    "type": "string",
                    "enum": [
                        "reference",
                        "fundamental",
                        "event",
                        "flow",
                    ],
                },
                "columns": STRING_ARRAY,
                "force": BOOLEAN,
            },
            required=["left", "right", "output"],
        ),
        "readOnly": False,
        "destructive": True,
        "openWorld": False,
    },
    {
        "name": "quant_benchmark_list",
        "description": "List public benchmark rules without evaluating or submitting anything.",
        "inputSchema": _schema(),
        "readOnly": True,
        "openWorld": False,
    },
    {
        "name": "quant_benchmark_inspect",
        "description": "Read one public benchmark data, evaluation and privacy contract.",
        "inputSchema": _schema({"benchmark": STRING}, required=["benchmark"]),
        "readOnly": True,
        "openWorld": False,
    },
    {
        "name": "quant_benchmark_evaluate",
        "description": "Evaluate a graph or factor locally under a public benchmark rule.",
        "inputSchema": _schema(
            {
                "benchmark": STRING,
                "dataPath": STRING,
                "startDate": STRING,
                "endDate": STRING,
                "factorNames": STRING_ARRAY,
                "specPath": STRING,
                "evidencePackPath": STRING,
                "benchmarkCode": STRING,
                "minimumNames": INTEGER,
                "minimumCoverage": NUMBER,
                "selectionRatio": NUMBER,
            },
            required=["benchmark", "dataPath", "startDate", "endDate"],
        ),
        "readOnly": False,
        "openWorld": False,
    },
]

TOOL_MAP = {item["name"]: item for item in TOOL_SPECS}

RESOURCE_SPECS = [
    {
        "uri": "quant://handshake",
        "name": "handshake",
        "description": "Live local-runtime readiness and security snapshot.",
        "mimeType": "application/json",
    },
    {
        "uri": "quant://playbook",
        "name": "playbook",
        "description": "Canonical local research workflow.",
        "mimeType": "text/markdown",
    },
    {
        "uri": "quant://security",
        "name": "security",
        "description": "Execution, disclosure and publication invariants.",
        "mimeType": "application/json",
    },
    {
        "uri": "quant://capabilities",
        "name": "capabilities",
        "description": "Current protocol host command and tool surface.",
        "mimeType": "application/json",
    },
    {
        "uri": "quant://data/status",
        "name": "data-status",
        "description": "Selected local panel binding, if any.",
        "mimeType": "application/json",
    },
    {
        "uri": "quant://operators",
        "name": "operators",
        "description": "Causal operators accepted by Alpha Graph.",
        "mimeType": "application/json",
    },
]

PLAYBOOK = """# Quant Research Protocol

This is an independent Python local research host. Start with
`quant_agent_handshake` or read `quant://handshake`. Read `quant://security`
before disclosure, submission, publication or deployment.

1. Inspect the local data panel with `quant_data_inspect`.
2. Discover installed Factor Packs with `quant_capability_list`.
3. Read the actual CPU/GPU, package and connector readiness with
   `quant_local_capabilities`.
4. Discover causal operators with `quant_factor_operators`.
5. Run a local factor, factor screen or composed Alpha Graph.
6. Audit causal leakage, purged holdout, costs, benchmark and capacity.
7. Create a portable Evidence Pack and verify or reproduce it.

Every factor, backtest, screen, audit and training operation runs on the
user's computer. A factor run never submits, publishes, ranks, purchases or
changes permissions. Credentials stay in the local process environment and
must never appear in tool output.
"""

SECURITY = {
    "format": "quant-agent-security/1",
    "executionBoundary": "local",
    "privateByDefault": True,
    "factorRunDoesNotSubmitOrPublish": True,
    "sourceUploadRequiresExplicitRequest": True,
    "marketDataUploadRequiresExplicitRequest": True,
    "invariants": [
        "Research, factor mining, screens, audits and backtests run on the user device.",
        "Production servers must not start research jobs for this runtime.",
        "A factor run never submits, publishes, ranks, purchases or changes permissions.",
        "Submission, publication, ranking and disclosure are separate explicit actions.",
        "Credentials are secret and must not appear in tool output or logs.",
        "A .qalpha archive is data and is never executed as a program.",
        "Evidence Packs bind causal diagnostics, benchmark, costs, out-of-sample and capacity; signature validity is not publisher trust.",
    ],
}


@dataclass(frozen=True)
class RuntimeContext:
    data_path: str | None = None
    start: str | None = None
    end: str | None = None
    benchmark_code: str | None = None


class ProtocolHost:
    def __init__(self, context: RuntimeContext | None = None) -> None:
        self.context = context or RuntimeContext()

    def app_info(self) -> dict[str, Any]:
        return {
            "shell": "python-protocol-host",
            "runtimeVersion": RUNTIME_VERSION,
            "python": platform.python_version(),
            "platform": sys.platform,
            "executionBoundary": "local",
            "agent": {
                "protocolName": PROTOCOL_NAME,
                "protocolVersion": PROTOCOL_VERSION,
                "clientProductCode": "quant-research-client",
                "competitionProductCode": "quant-arena",
                "toolSurfaceProfile": TOOL_SURFACE_PROFILE,
                "mcpProtocol": MCP_PROTOCOL_VERSION,
                "semanticConformanceFormat": SEMANTIC_FORMAT,
                "semanticVector": SEMANTIC_VECTOR,
                "toolCount": len(TOOL_SPECS),
                "resourceCount": len(RESOURCE_SPECS),
                "promptCount": 3,
            },
            "installPath": str(Path(__file__).resolve()),
            "note": "Headless runtime only; no submission or publication surface.",
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "shell": "python-protocol-host",
            "runtimeVersion": RUNTIME_VERSION,
            "commands": [
                "app.info",
                "agent.handshake",
                "mcp.serve",
                "protocol.serve",
                "data.inspect",
                "local.capabilities",
                "capability.list",
                "capability.install",
                "capability.remove",
                "capability.activate",
                "factor.operators",
                "factor.graph",
                "factor.screen",
                "evidence.pack",
                "evidence.verify",
                "evidence.reproduce",
                "evidence.audit",
                "family.audit",
                "connector.list",
                "connector.pull",
                "connector.join",
                "benchmark.list",
                "benchmark.inspect",
                "benchmark.evaluate",
            ],
            "tools": [item["name"] for item in TOOL_SPECS],
            "toolSurfaceProfile": TOOL_SURFACE_PROFILE,
            "executionBoundary": "local",
        }

    def data_status(self) -> dict[str, Any]:
        if not self.context.data_path:
            return {
                "format": "qdata-local/1",
                "status": "NOT_BOUND",
                "executionLocation": "local",
                "note": "Pass dataPath to research tools or start the host with --data.",
            }
        path = Path(self.context.data_path).expanduser().resolve()
        from quant_sdk.benchmark_resolver import resolve_benchmark, sample_symbols

        benchmark = resolve_benchmark(
            sample_symbols(
                path, start=self.context.start, end=self.context.end,
            ),
            explicit=self.context.benchmark_code,
        )
        return {
            "format": "qdata-local/1",
            "status": "BOUND" if path.is_file() else "MISSING",
            "path": str(path),
            "exists": path.is_file(),
            "window": {"start": self.context.start, "end": self.context.end},
            "benchmarkCode": benchmark.get("code"),
            "benchmarkResolution": benchmark,
            "executionLocation": "local",
        }

    def local_capabilities(self) -> dict[str, Any]:
        import importlib.util
        import os
        import shutil
        from importlib.metadata import version

        def package_version(name: str) -> str | None:
            try:
                return version(name)
            except Exception:
                return None

        torch_installed = importlib.util.find_spec("torch") is not None
        cuda = {
            "torchInstalled": torch_installed,
            "cudaAvailable": False,
            "deviceCount": 0,
        }
        if torch_installed:
            try:
                import torch

                cuda["torchVersion"] = str(torch.__version__)
                cuda["cudaAvailable"] = bool(torch.cuda.is_available())
                cuda["deviceCount"] = int(torch.cuda.device_count())
            except Exception as error:  # noqa: BLE001
                cuda["error"] = str(error)
        try:
            from quant_sdk.capabilities import (
                installed_factor_names,
                list_capabilities,
            )

            capability_catalog = list_capabilities()
            installed_count = len(installed_factor_names())
            capabilities = {
                "count": int(capability_catalog.get("count") or 0),
                "packs": capability_catalog.get("packs") or [],
            }
            factors = {"installedCount": installed_count}
        except Exception as error:  # noqa: BLE001
            capabilities = {"count": 0, "packs": [], "error": str(error)}
            factors = {"error": str(error)}
        try:
            from quant_sdk.operators import catalog

            operator_counts = catalog().get("counts") or {}
            operators = {
                "canonicalCount": operator_counts.get("canonical"),
                "aliasCount": operator_counts.get("aliases"),
                "entryCount": operator_counts.get("entries"),
            }
        except Exception as error:  # noqa: BLE001
            operators = {"error": str(error)}
        try:
            from quant_sdk.data_connectors import list_connectors

            connectors = list_connectors().get("connectors") or []
            connector_status = {
                "total": len(connectors),
                "ready": sum(1 for item in connectors if item.get("ready")),
                "needsAction": sum(
                    1 for item in connectors if not item.get("ready")
                ),
            }
        except Exception as error:  # noqa: BLE001
            connector_status = {"error": str(error)}
        try:
            disk_free = int(shutil.disk_usage(Path.cwd()).free)
        except OSError:
            disk_free = None
        return {
            "format": "quant-local-capabilities/1",
            "executionLocation": "local",
            "python": {
                "version": platform.python_version(),
                "executable": sys.executable,
            },
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "logicalCpus": os.cpu_count(),
                "diskFreeBytes": disk_free,
            },
            "cuda": cuda,
            "packages": {
                name: package_version(name)
                for name in (
                    "numpy", "pandas", "duckdb", "pyarrow", "scipy", "torch",
                )
            },
            "capabilities": capabilities,
            "operators": operators,
            "factors": factors,
            "connectors": connector_status,
        }

    def handshake(self) -> dict[str, Any]:
        return {
            "format": "quant-agent-handshake/1",
            "protocol": {
                "name": PROTOCOL_NAME,
                "version": PROTOCOL_VERSION,
                "clientProductCode": "quant-research-client",
                "competitionProductCode": "quant-arena",
                "toolSurfaceProfile": TOOL_SURFACE_PROFILE,
                "mcpProtocol": MCP_PROTOCOL_VERSION,
                "executionBoundary": "local",
                "semanticConformanceFormat": SEMANTIC_FORMAT,
                "semanticVector": SEMANTIC_VECTOR,
                "toolCount": len(TOOL_SPECS),
                "resourceUris": [item["uri"] for item in RESOURCE_SPECS],
                "promptNames": [
                    "quant_research",
                    "quant_reproduce_alpha",
                    "quant_private_evaluate",
                ],
            },
            "client": {
                "runtime": "python-protocol-host",
                "version": RUNTIME_VERSION,
                "platform": sys.platform,
                "python": platform.python_version(),
                "installationPath": str(Path(__file__).resolve()),
            },
            "executionBoundary": "local",
            "readyForResearch": bool(
                self.context.data_path
                and Path(self.context.data_path).expanduser().is_file()
            ),
            "readiness": {
                "localInterface": True,
                "compute": True,
                "data": self.data_status()["status"] == "BOUND",
                "persistentNode": False,
            },
            "state": {
                "data": self.data_status(),
                "daemon": {
                    "running": False,
                    "transport": "stdio",
                    "executionBoundary": "local",
                },
            },
            "security": {
                **SECURITY,
                "contractResource": "quant://security",
            },
            "nextActions": self._next_actions(),
        }

    def _next_actions(self) -> list[dict[str, str]]:
        actions: list[dict[str, str]] = []
        if self.data_status()["status"] != "BOUND":
            actions.append({
                "id": "select-data",
                "command": "data.inspect",
                "reason": "Bind a local CSV, Parquet, DuckDB or DB panel.",
            })
        else:
            actions.extend([
                {
                    "id": "inspect-data",
                    "command": "data.inspect",
                    "reason": "Confirm fields, coverage and point-in-time semantics.",
                },
                {
                    "id": "audit-candidate",
                    "command": "evidence.audit",
                    "reason": "Test leakage, holdout, costs, benchmark and capacity.",
                },
            ])
        return actions

    def read_resource(self, uri: str) -> tuple[str, str]:
        spec = next((item for item in RESOURCE_SPECS if item["uri"] == uri), None)
        if spec is None:
            raise KeyError(uri)
        if uri == "quant://handshake":
            text = json.dumps(self.handshake(), ensure_ascii=False, indent=2)
        elif uri == "quant://playbook":
            text = PLAYBOOK
        elif uri == "quant://security":
            text = json.dumps(SECURITY, ensure_ascii=False, indent=2)
        elif uri == "quant://capabilities":
            text = json.dumps(self.capabilities(), ensure_ascii=False, indent=2)
        elif uri == "quant://data/status":
            text = json.dumps(self.data_status(), ensure_ascii=False, indent=2)
        elif uri == "quant://operators":
            from quant_sdk.operators import catalog

            text = json.dumps(catalog(), ensure_ascii=False, indent=2)
        else:
            raise KeyError(uri)
        return spec["mimeType"], text

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in TOOL_MAP:
            raise ValueError(f"unknown MCP tool: {name}")
        arguments = dict(arguments or {})
        if name == "quant_app_info":
            return self.app_info()
        if name == "quant_agent_handshake":
            return self.handshake()
        if name == "quant_capabilities":
            return self.capabilities()
        if name == "quant_data_inspect":
            from quant_sdk.research_workflow import inspect_data

            return inspect_data(
                arguments["dataPath"],
                start=str(arguments.get("startDate") or "19000101"),
                end=str(arguments.get("endDate") or "29991231"),
                benchmark_code=str(
                    arguments.get("benchmarkCode") or "000300.SH"
                ),
            )
        if name == "quant_local_capabilities":
            return self.local_capabilities()
        if name == "quant_factor_operators":
            from quant_sdk.operators import catalog

            return catalog()
        if name == "quant_capability_list":
            from quant_sdk.capabilities import list_capabilities

            return list_capabilities(arguments.get("root") or None)
        if name == "quant_capability_install":
            from quant_sdk.capabilities import install_capability

            _confirm_local_code_change(
                "安装可执行的本地因子或算子能力包？\n\n"
                f"{arguments['file']}"
            )
            return install_capability(
                arguments["file"],
                root=arguments.get("root") or None,
                upgrade=bool(arguments.get("upgrade", False)),
                trusted_key=arguments.get("trustedKey") or None,
            )
        if name == "quant_capability_remove":
            from quant_sdk.capabilities import remove_capability

            _confirm_local_code_change(
                "删除已安装的本地能力包？\n\n"
                f"{arguments['id']} "
                f"{arguments.get('version') or 'all versions'}"
            )
            return remove_capability(
                str(arguments["id"]),
                version=arguments.get("version") or None,
                root=arguments.get("root") or None,
            )
        if name == "quant_capability_activate":
            from quant_sdk.capabilities import activate_capability

            _confirm_local_code_change(
                "切换到另一个已保留的本地能力包版本？\n\n"
                f"{arguments['id']}@{arguments['version']}"
            )
            return activate_capability(
                str(arguments["id"]),
                str(arguments["version"]),
                root=arguments.get("root") or None,
            )
        if name == "quant_capability_sign":
            from quant_sdk.capabilities import sign_capability

            return sign_capability(
                arguments["file"],
                private_key=arguments["signingKey"],
            )
        if name == "quant_alpha_graph":
            from quant_sdk.alpha_graph import evaluate_graph_file

            spec = _read_json(arguments.get("specPath"))
            return evaluate_graph_file(
                data_path=str(arguments["dataPath"]),
                start=str(arguments["startDate"]),
                end=str(arguments["endDate"]),
                graph=spec,
                factor_names=_string_list(arguments.get("factorNames")) or None,
                output_dir=(
                    str(arguments["outputDir"])
                    if arguments.get("outputDir") else None
                ),
                benchmark_code=str(
                    arguments.get("benchmarkCode") or "000300.SH"
                ),
                horizon=int(arguments.get("horizon") or 1),
                min_names=int(arguments.get("minNames") or 5),
                backend=str(arguments.get("backend") or "auto"),
            )
        if name == "quant_factor_screen":
            from quant_sdk.alpha_screen import screen_factors

            return screen_factors(
                data_path=str(arguments["dataPath"]),
                start=str(arguments["startDate"]),
                end=str(arguments["endDate"]),
                factor_names=_string_list(arguments.get("factorNames")) or None,
                families=_string_list(arguments.get("families")) or None,
                query=str(arguments.get("query") or ""),
                scan_limit=int(arguments.get("scanLimit") or 0),
                top=int(arguments.get("top") or 30),
                chunk_size=int(arguments.get("chunkSize") or 64),
                max_correlation=float(
                    arguments.get("maxCorrelation", 0.85)
                ),
                minimum_coverage=float(
                    arguments.get("minimumCoverage", 0.5)
                ),
                minimum_ic_days=int(arguments.get("minimumIcDays") or 10),
                min_names=int(arguments.get("minNames") or 5),
                horizon=int(arguments.get("horizon") or 1),
                workers=(
                    int(arguments["workers"])
                    if arguments.get("workers") is not None else None
                ),
                benchmark_code=str(
                    arguments.get("benchmarkCode") or "000300.SH"
                ),
                output=arguments.get("output") or None,
            )
        if name == "quant_evidence_pack":
            from quant_sdk.evidence_pack import create_pack

            return create_pack(
                arguments["output"],
                name=str(arguments["name"]),
                version=str(arguments.get("version") or "1"),
                source_file=arguments.get("sourceFile") or None,
                graph=_read_json(arguments.get("graphFile")),
                factor_name=arguments.get("factorName") or None,
                data_path=arguments.get("dataPath") or None,
                data_start=arguments.get("startDate") or None,
                data_end=arguments.get("endDate") or None,
                benchmark_code=str(
                    arguments.get("benchmarkCode") or "000300.SH"
                ),
                result_file=arguments.get("resultFile") or None,
                audit_file=arguments.get("auditFile") or None,
                family_audit_file=arguments.get("familyAuditFile") or None,
                metadata=arguments.get("metadata") or None,
                signing_key_file=arguments.get("signingKey") or None,
                signer=arguments.get("signer") or None,
            )
        if name == "quant_evidence_verify":
            from quant_sdk.evidence_pack import verify_pack

            return verify_pack(
                arguments["file"],
                trusted_key=arguments.get("trustedKey") or None,
                data_path=arguments.get("dataPath") or None,
                start=arguments.get("startDate") or None,
                end=arguments.get("endDate") or None,
                benchmark_code=str(
                    arguments.get("benchmarkCode") or "000300.SH"
                ),
            )
        if name == "quant_evidence_reproduce":
            from quant_sdk.evidence_pack import reproduce_pack

            return reproduce_pack(
                arguments["file"],
                data_path=arguments["dataPath"],
                start=arguments.get("startDate") or None,
                end=arguments.get("endDate") or None,
                benchmark_code=str(
                    arguments.get("benchmarkCode") or "000300.SH"
                ),
            )
        if name == "quant_evidence_audit":
            from quant_sdk.evidence_audit import audit_factor

            source = ""
            if arguments.get("sourceFile"):
                source = Path(arguments["sourceFile"]).expanduser().read_text(
                    encoding="utf-8"
                )
            return audit_factor(
                data_path=arguments["dataPath"],
                start=str(arguments["startDate"]),
                end=str(arguments["endDate"]),
                factor_name=arguments.get("factorName") or None,
                source=source,
                name=str(
                    arguments.get("name")
                    or arguments.get("factorName")
                    or "factor"
                ),
                selection_ratio=float(arguments.get("selectionRatio", 0.7)),
                purge_days=int(arguments.get("purgeDays", 1)),
                embargo_days=int(arguments.get("embargoDays", 1)),
                horizon=int(arguments.get("horizon", 1)),
                min_names=int(arguments.get("minNames", 5)),
                n_groups=int(arguments.get("groups", 5)),
                costs=bool(arguments.get("costs", True)),
                allow_short=bool(arguments.get("allowShort", False)),
                n_trials=int(arguments.get("trials", 1)),
                direction=int(arguments.get("direction", 1)),
                benchmark_code=str(
                    arguments.get("benchmarkCode") or "000300.SH"
                ),
                output=arguments.get("output") or None,
            )
        if name == "quant_family_audit":
            from quant_sdk.family_audit import audit_family, load_manifest

            manifest = load_manifest(arguments["candidatesFile"])
            return audit_family(
                data_path=arguments["dataPath"],
                start=str(arguments["startDate"]),
                end=str(arguments["endDate"]),
                candidates=manifest["candidates"],
                family_id=str(
                    arguments.get("familyId")
                    or manifest.get("familyId")
                    or "candidate-family"
                ),
                selection_ratio=float(arguments.get("selectionRatio", 0.7)),
                purge_days=int(arguments.get("purgeDays", 1)),
                embargo_days=int(arguments.get("embargoDays", 1)),
                horizon=int(arguments.get("horizon", 1)),
                min_names=int(arguments.get("minNames", 5)),
                n_groups=int(arguments.get("groups", 5)),
                costs=bool(arguments.get("costs", True)),
                allow_short=bool(arguments.get("allowShort", False)),
                benchmark_code=str(
                    arguments.get("benchmarkCode") or "000300.SH"
                ),
                pbo_splits=int(arguments.get("pboSplits", 16)),
                alpha=float(arguments.get("alpha", 0.05)),
                output=arguments.get("output") or None,
            )
        if name == "quant_connector_list":
            from quant_sdk.data_connectors import list_connectors

            return list_connectors()
        if name == "quant_connector_pull":
            from quant_sdk.data_connectors import pull_connector

            return pull_connector(
                str(arguments["provider"]),
                output=arguments["output"],
                start=str(arguments["startDate"]),
                end=str(arguments["endDate"]),
                input_path=arguments.get("inputPath") or None,
                symbols=_string_list(arguments.get("symbols")) or None,
                query=arguments.get("query") or None,
                options=arguments.get("options") or None,
                force=bool(arguments.get("force", False)),
                table_kind=str(arguments.get("tableKind") or "bars"),
                identity=_string_list(arguments.get("identity")) or None,
                event_time=arguments.get("eventTime") or None,
                available_at=arguments.get("availableAt") or None,
            )
        if name == "quant_connector_join":
            from quant_sdk.data_connectors import join_connector_tables

            return join_connector_tables(
                left=arguments["left"],
                right=arguments["right"],
                output=arguments["output"],
                identity=_string_list(arguments.get("identity")) or ("ts_code",),
                left_time=str(arguments.get("leftTime") or "trade_date"),
                right_event_time=str(
                    arguments.get("rightEventTime") or "event_time"
                ),
                right_available_at=str(
                    arguments.get("rightAvailableAt") or "available_at"
                ),
                right_kind=str(arguments.get("rightKind") or "reference"),
                columns=_string_list(arguments.get("columns")) or None,
                force=bool(arguments.get("force", False)),
            )
        if name == "quant_benchmark_list":
            from quant_sdk.benchmarks import list_benchmarks

            return list_benchmarks()
        if name == "quant_benchmark_inspect":
            from quant_sdk.benchmarks import benchmark_manifest

            return benchmark_manifest(str(arguments["benchmark"]))
        if name == "quant_benchmark_evaluate":
            from quant_sdk.benchmarks import evaluate_benchmark

            return evaluate_benchmark(
                benchmark_id=str(arguments["benchmark"]),
                data_path=arguments["dataPath"],
                start=str(arguments["startDate"]),
                end=str(arguments["endDate"]),
                graph=_read_json(arguments.get("specPath")),
                factor_names=_string_list(arguments.get("factorNames")) or None,
                evidence_pack_path=arguments.get("evidencePackPath") or None,
                benchmark_code=str(
                    arguments.get("benchmarkCode") or "000300.SH"
                ),
                minimum_names=(
                    int(arguments["minimumNames"])
                    if arguments.get("minimumNames") is not None else None
                ),
                minimum_coverage=(
                    float(arguments["minimumCoverage"])
                    if arguments.get("minimumCoverage") is not None else None
                ),
                selection_ratio=(
                    float(arguments["selectionRatio"])
                    if arguments.get("selectionRatio") is not None else None
                ),
            )
        raise ValueError(f"unimplemented MCP tool: {name}")

    def prompt(self, name: str, arguments: dict[str, Any]) -> str:
        goal = str(arguments.get("goal") or "Run a local quant research study.")
        if name == "quant_research":
            data = self.data_status()
            return (
                "Use this local Python Quant Research Protocol host. "
                f"Goal: {goal}. Start with quant_agent_handshake, then inspect "
                f"the selected local panel ({data.get('status')}). All factor, "
                "backtest, screen, audit and computation work must stay local. "
                "A factor run must not submit or publish. Finish with an "
                "audited portable .qalpha artifact when the user asks to publish."
            )
        if name == "quant_reproduce_alpha":
            return (
                "Verify the supplied .qalpha archive first, then reproduce "
                "canonical factor values against the user's local data slice. "
                "Report integrity, signature validity, publisher trust, data "
                "matching and reproduction separately. Do not execute the "
                "archive as a program and do not submit or publish."
            )
        if name == "quant_private_evaluate":
            return (
                "Prepare a private evaluation only at the data custodian's "
                "boundary. Keep source disclosure explicit, bind the artifact "
                "and data commitments, and state that software isolation is "
                "not hardware attestation."
            )
        raise KeyError(name)


class McpDispatcher:
    def __init__(self, host: ProtocolHost) -> None:
        self.host = host

    @staticmethod
    def error(
        request_id: Any,
        code: int,
        message: str,
        data: Any = None,
    ) -> dict[str, Any]:
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return {"jsonrpc": "2.0", "id": request_id, "error": error}

    def handle(self, message: Any) -> dict[str, Any] | None:
        if (
            not isinstance(message, dict)
            or message.get("jsonrpc") != "2.0"
            or not isinstance(message.get("method"), str)
        ):
            request_id = message.get("id") if isinstance(message, dict) else None
            return self.error(request_id, -32600, "Invalid Request")
        request_id = message.get("id")
        notification = request_id is None
        method = message["method"]
        params = message.get("params") or {}

        def respond(result: Any) -> dict[str, Any] | None:
            return None if notification else {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result,
            }

        if method == "initialize":
            requested_protocol = str(
                params.get("protocolVersion") or ""
            )
            return respond({
                "protocolVersion": (
                    requested_protocol
                    if requested_protocol == MCP_PROTOCOL_VERSION
                    else MCP_PROTOCOL_VERSION
                ),
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                    "prompts": {"listChanged": False},
                },
                "serverInfo": {
                    "name": "quant-python-protocol-host",
                    "version": RUNTIME_VERSION,
                },
                "instructions": (
                    "This independent runtime executes quant research on the "
                    "user's computer. Read quant://handshake and "
                    "quant://security before work."
                ),
            })
        if method in {"notifications/initialized", "notifications/cancelled"}:
            return None
        if method == "ping":
            return respond({})
        if method == "tools/list":
            return respond({
                "tools": [
                    {
                        "name": item["name"],
                        "description": item["description"],
                        "inputSchema": item["inputSchema"],
                        "outputSchema": TOOL_RESPONSE_SCHEMA,
                        "annotations": {
                            "readOnlyHint": bool(item["readOnly"]),
                            "destructiveHint": bool(item.get("destructive", False)),
                            "idempotentHint": bool(item["readOnly"]),
                            "openWorldHint": bool(item["openWorld"]),
                        },
                    }
                    for item in TOOL_SPECS
                ],
            })
        if method == "tools/call":
            name = str(params.get("name") or "")
            arguments = params.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            try:
                data = self.host.call_tool(name, arguments)
            except Exception as error:  # noqa: BLE001 - MCP error envelope
                envelope = {
                    "ok": False,
                    "error": {
                        "code": "TOOL_EXECUTION_FAILED",
                        "message": f"{type(error).__name__}: {error}",
                    },
                }
                return respond({
                    "content": [{
                        "type": "text",
                        "text": json.dumps(envelope, ensure_ascii=False),
                    }],
                    "structuredContent": envelope,
                    "isError": True,
                })
            envelope = _structured(data)
            return respond({
                "content": [{
                    "type": "text",
                    "text": json.dumps(
                        envelope, ensure_ascii=False, allow_nan=False
                    ),
                }],
                "structuredContent": envelope,
                "isError": False,
            })
        if method == "resources/list":
            return respond({"resources": RESOURCE_SPECS})
        if method == "resources/read":
            uri = str(params.get("uri") or "")
            try:
                mime_type, text = self.host.read_resource(uri)
            except KeyError:
                return self.error(
                    request_id, -32602, f"Unknown resource: {uri}"
                )
            except Exception as error:  # noqa: BLE001
                return self.error(
                    request_id, -32603, f"{type(error).__name__}: {error}"
                )
            return respond({
                "contents": [{
                    "uri": uri,
                    "mimeType": mime_type,
                    "text": text,
                }],
            })
        if method == "prompts/list":
            return respond({
                "prompts": [
                    {
                        "name": "quant_research",
                        "title": "Local Alpha research",
                        "description": "Bootstrap a local Alpha study.",
                        "arguments": [{"name": "goal", "required": False}],
                    },
                    {
                        "name": "quant_reproduce_alpha",
                        "title": "Reproduce an Alpha artifact",
                        "description": "Verify and reproduce a .qalpha asset.",
                        "arguments": [{"name": "artifact", "required": False}],
                    },
                    {
                        "name": "quant_private_evaluate",
                        "title": "Private evaluation",
                        "description": "Evaluate at a data-custodian boundary.",
                        "arguments": [{"name": "benchmark", "required": False}],
                    },
                ],
            })
        if method == "prompts/get":
            name = str(params.get("name") or "")
            arguments = params.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            try:
                text = self.host.prompt(name, arguments)
            except KeyError:
                return self.error(request_id, -32602, f"Unknown prompt: {name}")
            return respond({
                "description": f"Quant protocol prompt: {name}",
                "messages": [{
                    "role": "user",
                    "content": {"type": "text", "text": text},
                }],
            })
        if method == "shutdown":
            return respond(None)
        if notification:
            return None
        return self.error(request_id, -32601, f"Method not found: {method}")


def serve_stdio(
    host: ProtocolHost,
    *,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> None:
    dispatcher = McpDispatcher(host)
    for line in input_stream:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            response = dispatcher.error(None, -32700, "Parse error")
        else:
            response = dispatcher.handle(message)
        if response is not None:
            output_stream.write(
                json.dumps(
                    response, ensure_ascii=False, separators=(",", ":")
                )
                + "\n"
            )
            output_stream.flush()


class LoopbackHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve_http(
    host: ProtocolHost,
    *,
    address: tuple[str, int],
    token: str,
    stderr: TextIO = sys.stderr,
) -> None:
    dispatcher = McpDispatcher(host)

    class Handler(BaseHTTPRequestHandler):
        server_version = "QuantProtocolHost/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            return None

        def _write(
            self,
            status: int,
            payload: dict[str, Any] | None,
        ) -> None:
            body = (
                json.dumps(
                    payload, ensure_ascii=False, allow_nan=False
                ).encode("utf-8")
                if payload is not None else b""
            )
            self.send_response(status)
            if body:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/mcp":
                self._write(404, {"error": "not_found"})
                return
            if self.headers.get("Authorization") != f"Bearer {token}":
                self._write(401, {"error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._write(400, {"error": "invalid_content_length"})
                return
            if length < 1 or length > 16 * 1024 * 1024:
                self._write(413, {"error": "invalid_request_size"})
                return
            try:
                message = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._write(
                    400, dispatcher.error(None, -32700, "Parse error")
                )
                return
            response = dispatcher.handle(message)
            self._write(202 if response is None else 200, response)

        def do_GET(self) -> None:
            self._write(405, {"error": "method_not_allowed"})

    server = LoopbackHttpServer(address, Handler)
    hostname, port = server.server_address[:2]
    stderr.write(
        f"QUANT_PROTOCOL_HTTP http://{hostname}:{port}/mcp\n"
    )
    stderr.flush()
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token")
    parser.add_argument("--token-env", default="QUANT_AGENT_TOKEN")
    parser.add_argument("--data")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--benchmark-code", default=None)
    parser.add_argument("--info", action="store_true")
    args = parser.parse_args(argv)
    context = RuntimeContext(
        data_path=args.data,
        start=args.start,
        end=args.end,
        benchmark_code=args.benchmark_code,
    )
    host = ProtocolHost(context)
    if args.info:
        print(json.dumps(host.app_info(), ensure_ascii=False, indent=2))
        return 0
    if args.transport == "stdio":
        serve_stdio(host)
        return 0
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        parser.error("HTTP transport must bind to a loopback address")
    token = str(args.token or os.environ.get(args.token_env, "")).strip()
    if not token:
        parser.error(
            f"HTTP transport requires --token or {args.token_env}"
        )
    serve_http(host, address=(args.host, args.port), token=token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
