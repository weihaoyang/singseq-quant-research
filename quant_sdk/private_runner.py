"""Compute-to-data runner for private benchmark evaluation.

The runner executes the published benchmark against data that remains on the
custodian machine, redacts the result according to the request, and signs an
evaluator receipt. It uses an isolated Python subprocess with a minimal
environment and ephemeral scratch space.

This is process isolation, not a hardware-enclosed confidential VM. Receipts
state that limitation explicitly so a verifier does not mistake it for an
enclave attestation.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_sdk.benchmarks import evaluate_benchmark
from quant_sdk.private_evaluation import (
    _canonical_json,
    _sha256,
    create_receipt,
)

RUNNER_FORMAT = "qprivate-runner/1"
RUNNER_VERSION = "1"
WORKER_COMMAND = "__worker__"


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: str | Path, value: dict[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=lambda item: item.item() if hasattr(item, "item") else str(item),
        ) + "\n",
        encoding="utf-8",
    )
    return target


def _validate_request(request: dict[str, Any]) -> None:
    if request.get("format") != "qprivate-request/1":
        raise ValueError("request format must be qprivate-request/1")
    body = dict(request)
    digest = body.pop("requestSha256", None)
    if digest != _sha256(_canonical_json(body)):
        raise ValueError("request hash mismatch")
    expires = str(request.get("expiresAt") or "")
    if not expires:
        raise ValueError("request is missing expiresAt")
    expires_at = datetime.fromisoformat(expires.replace("Z", "+00:00"))
    if datetime.now(UTC) > expires_at:
        raise ValueError("private evaluation request has expired")
    artifact = str(request.get("artifactSha256") or "")
    if len(artifact) != 64 or any(ch not in "0123456789abcdef" for ch in artifact):
        raise ValueError("request artifactSha256 is invalid")
    commitments = request.get("dataCommitments")
    if not isinstance(commitments, list) or not commitments:
        raise ValueError("request has no data commitments")
    privacy = request.get("privacy") or {}
    if privacy.get("secureComputeRequired") is not True:
        raise ValueError("request does not require secure compute")


def runner_capabilities() -> dict[str, Any]:
    docker = shutil.which("docker")
    docker_version = None
    if docker:
        try:
            checked = subprocess.run(
                [docker, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=5,
                check=False,
            )
            if checked.returncode == 0:
                docker_version = checked.stdout.strip() or "available"
        except (OSError, subprocess.TimeoutExpired):
            docker_version = None
    return {
        "format": RUNNER_FORMAT,
        "version": RUNNER_VERSION,
        "status": "READY",
        "defaultBackend": "process",
        "isolation": {
            "type": "ISOLATED_SUBPROCESS",
            "hardwareConfidential": False,
            "networkIsolation": False,
            "trustedCodeOnly": True,
            "ephemeralScratch": True,
            "credentialEnvironment": "MINIMAL_ALLOWLIST",
        },
        "backends": {
            "process": {
                "available": True,
                "networkIsolation": False,
                "hardwareConfidential": False,
            },
            "oci": {
                "available": bool(docker_version),
                "runtime": "docker",
                "runtimeVersion": docker_version,
                "networkIsolation": True,
                "readOnlyRoot": True,
                "capabilities": "DROP_ALL",
                "noNewPrivileges": True,
                "hardwareConfidential": False,
                "requiresImage": True,
            },
        },
        "privacy": {
            "dataLeavesCustodian": False,
            "submittedArtifactIsReadableByEvaluatorOperator": True,
            "sourceIsExecuted": False,
            "factorIdentitySupport": ["alpha-graph-1", "catalogName"],
            "sourceIsReturnedToRequester": False,
            "automaticSubmission": False,
        },
        "limitations": [
            "Process isolation does not protect inputs from the machine operator.",
            "Hardware attestation requires a separate confidential VM or enclave backend.",
            "Source-only Evidence Packs must be converted to an Alpha Graph before private evaluation.",
            "The OCI backend enforces no network, read-only root, dropped capabilities and resource limits, but is still not hardware-attested.",
        ],
    }


def _minimal_environment() -> dict[str, str]:
    allowed = {
        "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE",
        "LOCALAPPDATA", "APPDATA", "LANG", "LC_ALL",
    }
    environment = {
        key: value for key, value in os.environ.items()
        if key in allowed and value
    }
    package_root = str(Path(__file__).resolve().parents[1])
    # Never inherit an ambient PYTHONPATH: a custodian machine can have another
    # packaged runtime from a previous product installation on that path.
    environment["PYTHONPATH"] = package_root
    environment.update({
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    return environment


def _worker(config_file: Path) -> int:
    config = _read_json(config_file)
    window = config["evaluationWindow"]
    result = evaluate_benchmark(
        benchmark_id=str(config["benchmarkId"]),
        data_path=str(config["dataPath"]),
        start=str(window["start"]),
        end=str(window["end"]),
        evidence_pack_path=str(config["artifactPath"]),
        benchmark_code=str(config.get("benchmarkCode") or "000300.SH"),
        minimum_names=config.get("minimumNames"),
        minimum_coverage=config.get("minimumCoverage"),
        selection_ratio=config.get("selectionRatio"),
    )
    _write_json(config["resultPath"], result)
    return 0


def _run_worker(config_file: Path, *, timeout_seconds: int, scratch: Path) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "quant_sdk.private_runner",
            WORKER_COMMAND,
            "--config",
            str(config_file),
        ],
        cwd=scratch,
        env=_minimal_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"private evaluation worker failed with exit code {completed.returncode}"
            + (f": {detail[-4000:]}" if detail else "")
        )
    result_file = scratch / "worker-result.json"
    return {
        "result": _read_json(result_file),
        "elapsedSeconds": round(time.monotonic() - started, 3),
        "workerPidFinished": True,
    }


def _docker_image_id(docker: str, image: str) -> str:
    inspected = subprocess.run(
        [docker, "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        check=False,
    )
    if inspected.returncode != 0:
        raise RuntimeError(
            f"OCI runner image is unavailable: {image}: "
            + (inspected.stderr or inspected.stdout or "docker image inspect failed").strip()
        )
    image_id = inspected.stdout.strip()
    if not image_id:
        raise RuntimeError(f"OCI runner image has no immutable id: {image}")
    return image_id


def oci_run_command(
    *,
    docker: str,
    scratch: Path,
    artifact_path: Path,
    data_path: Path,
    toolkit_root: Path,
    image: str,
    cpus: float,
    memory_mb: int,
    pids_limit: int,
) -> list[str]:
    data_target = f"/input/data{data_path.suffix.lower() or '.bin'}"
    return [
        docker, "run", "--rm", "--init",
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        f"--pids-limit={pids_limit}",
        f"--cpus={cpus}",
        f"--memory={memory_mb}m",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m",
        "--mount", f"type=bind,source={scratch},target=/work",
        "--mount", f"type=bind,source={artifact_path},target=/input/artifact.qalpha,readonly",
        "--mount", f"type=bind,source={data_path},target={data_target},readonly",
        "--mount", f"type=bind,source={toolkit_root},target=/opt/quant,readonly",
        "-e", "PYTHONPATH=/opt/quant",
        "-e", "PYTHONHASHSEED=0",
        "-e", "OMP_NUM_THREADS=1",
        "-e", "OPENBLAS_NUM_THREADS=1",
        "-e", "MKL_NUM_THREADS=1",
        "-e", "NUMEXPR_NUM_THREADS=1",
        "-w", "/work",
        image,
        "python", "-m", "quant_sdk.private_runner",
        WORKER_COMMAND, "--config", "/work/worker-config.json",
    ]


def _run_oci_worker(
    config_file: Path,
    *,
    scratch: Path,
    artifact_path: Path,
    data_path: Path,
    toolkit_root: Path,
    image: str,
    cpus: float,
    memory_mb: int,
    pids_limit: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("OCI backend requires Docker on this machine")
    image_id = _docker_image_id(docker, image)
    started = time.monotonic()
    command = oci_run_command(
        docker=docker,
        scratch=scratch,
        artifact_path=artifact_path,
        data_path=data_path,
        toolkit_root=toolkit_root,
        image=image,
        cpus=cpus,
        memory_mb=memory_mb,
        pids_limit=pids_limit,
    )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"OCI private evaluation worker exceeded {timeout_seconds} seconds"
        ) from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"OCI private evaluation worker failed with exit code {completed.returncode}"
            + (f": {detail[-4000:]}" if detail else "")
        )
    result_file = scratch / "worker-result.json"
    return {
        "result": _read_json(result_file),
        "elapsedSeconds": round(time.monotonic() - started, 3),
        "workerPidFinished": True,
        "imageRef": image,
        "imageId": image_id,
        "dockerVersion": subprocess.run(
            [docker, "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
            check=False,
        ).stdout.strip() or "unknown",
    }


def _window(request: dict[str, Any], start: str | None, end: str | None) -> dict[str, str]:
    declared = request.get("evaluationWindow")
    if start or end:
        if not start or not end:
            raise ValueError("both start and end are required when overriding the request window")
        return {"start": str(start), "end": str(end)}
    if not isinstance(declared, dict) or not declared.get("start") or not declared.get("end"):
        raise ValueError(
            "request has no evaluationWindow; create it with --start/--end or pass an override"
        )
    return {"start": str(declared["start"]), "end": str(declared["end"])}


def _visible_result(result: dict[str, Any], disclosure: str) -> dict[str, Any]:
    if disclosure == "full":
        return result
    visible = {
        "format": result.get("format"),
        "version": result.get("version"),
        "status": result.get("status"),
        "official": False,
        "benchmark": {
            "id": (result.get("benchmark") or {}).get("id"),
            "version": (result.get("benchmark") or {}).get("version"),
        },
        "data": {
            "window": (result.get("data") or {}).get("window"),
            "identity": (result.get("data") or {}).get("identity"),
        },
        "split": result.get("split"),
        "outputs": result.get("outputs"),
        "submitted": False,
        "published": False,
        "note": "Compute-to-data result; official ranking and publication require a separate explicit action.",
    }
    if disclosure == "evidence":
        visible["graph"] = result.get("graph")
        visible["artifact"] = {
            key: value
            for key, value in (result.get("artifact") or {}).items()
            if key != "path"
        }
    return visible


def run_request(
    *,
    request_file: str | Path,
    artifact_file: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    signing_key_file: str | Path,
    evaluator: str,
    start: str | None = None,
    end: str | None = None,
    benchmark_code: str = "000300.SH",
    minimum_names: int | None = None,
    minimum_coverage: float | None = None,
    selection_ratio: float | None = None,
    timeout_seconds: int = 3600,
    backend: str = "process",
    container_image: str | None = None,
    cpus: float = 2.0,
    memory_mb: int = 2048,
    pids_limit: int = 128,
    toolkit_root: str | Path | None = None,
) -> dict[str, Any]:
    request_path = Path(request_file).expanduser().resolve()
    artifact_path = Path(artifact_file).expanduser().resolve()
    source_data = Path(data_path).expanduser().resolve()
    key_path = Path(signing_key_file).expanduser().resolve()
    request = _read_json(request_path)
    _validate_request(request)
    if not source_data.is_file():
        raise ValueError("dataPath must be a local file")
    expected_artifact = str(request["artifactSha256"])
    if _sha256(artifact_path.read_bytes()) != expected_artifact:
        raise ValueError("artifact file does not match request artifactSha256")
    disclosure = str((request.get("privacy") or {}).get("resultDisclosure") or "summary")
    if disclosure not in {"summary", "evidence", "full"}:
        raise ValueError("request resultDisclosure is invalid")
    window = _window(request, start, end)
    if timeout_seconds < 1 or timeout_seconds > 24 * 60 * 60:
        raise ValueError("timeoutSeconds must be between 1 and 86400")
    backend = str(backend or "process").strip().lower()
    if backend not in {"process", "oci"}:
        raise ValueError("backend must be process or oci")
    if backend == "oci" and not container_image:
        raise ValueError("OCI backend requires containerImage")
    if cpus <= 0 or cpus > 64:
        raise ValueError("cpus must be between 0 and 64")
    if memory_mb < 128 or memory_mb > 1024 * 1024:
        raise ValueError("memoryMb must be between 128 and 1048576")
    if pids_limit < 16 or pids_limit > 65536:
        raise ValueError("pidsLimit must be between 16 and 65536")
    package_root = Path(toolkit_root).expanduser().resolve() if toolkit_root else (
        Path(__file__).resolve().parents[1]
    )
    if backend == "oci" and not package_root.is_dir():
        raise ValueError("toolkitRoot must be a local directory for the OCI backend")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="qprivate-runner-") as directory:
        scratch = Path(directory)
        isolated_artifact = scratch / "artifact.qalpha"
        shutil.copyfile(artifact_path, isolated_artifact)
        worker_result = scratch / "worker-result.json"
        config_file = scratch / "worker-config.json"
        if backend == "oci":
            artifact_config = "/input/artifact.qalpha"
            data_config = f"/input/data{source_data.suffix.lower() or '.bin'}"
            result_config = "/work/worker-result.json"
        else:
            artifact_config = str(isolated_artifact)
            data_config = str(source_data)
            result_config = str(worker_result)
        config = {
            "benchmarkId": request.get("benchmarkId"),
            "artifactPath": artifact_config,
            "dataPath": data_config,
            "evaluationWindow": window,
            "benchmarkCode": benchmark_code,
            "minimumNames": minimum_names,
            "minimumCoverage": minimum_coverage,
            "selectionRatio": selection_ratio,
            "resultPath": result_config,
        }
        _write_json(config_file, config)
        if backend == "oci":
            execution = _run_oci_worker(
                config_file,
                scratch=scratch,
                artifact_path=isolated_artifact,
                data_path=source_data,
                toolkit_root=package_root,
                image=str(container_image),
                cpus=cpus,
                memory_mb=memory_mb,
                pids_limit=pids_limit,
                timeout_seconds=timeout_seconds,
            )
        else:
            execution = _run_worker(
                config_file,
                timeout_seconds=timeout_seconds,
                scratch=scratch,
            )

    result = execution["result"]
    actual_data_digest = (
        ((result.get("data") or {}).get("identity") or {}).get("sha256")
    )
    committed = {
        str(item.get("sha256")) for item in request.get("dataCommitments") or []
        if isinstance(item, dict) and item.get("sha256")
    }
    if actual_data_digest not in committed:
        raise ValueError(
            "evaluated data slice does not match any request data commitment"
        )
    visible = _visible_result(result, disclosure)
    result_path = _write_json(destination / "result.json", visible)
    bundled_request = destination / "request.json"
    if bundled_request != request_path:
        shutil.copyfile(request_path, bundled_request)
    if backend == "oci":
        secure_environment = {
            "type": "OCI_CONTAINER",
            "backend": "oci",
            "runnerFormat": RUNNER_FORMAT,
            "runnerVersion": RUNNER_VERSION,
            "networkIsolation": True,
            "readOnlyRoot": True,
            "capabilitiesDropped": ["ALL"],
            "noNewPrivileges": True,
            "tmpfs": "/tmp",
            "wallClockTimeoutSeconds": timeout_seconds,
            "cpuLimit": cpus,
            "memoryLimitMb": memory_mb,
            "pidLimit": pids_limit,
            "imageRef": execution["imageRef"],
            "imageId": execution["imageId"],
            "runtimeVersion": execution["dockerVersion"],
            "elapsedSeconds": execution["elapsedSeconds"],
            "operatorCanReadInputs": True,
            "hardwareConfidential": False,
            "attestation": "NOT_HARDWARE_ATTESTED",
        }
    else:
        secure_environment = {
            **runner_capabilities()["isolation"],
            "backend": "process",
            "runnerFormat": RUNNER_FORMAT,
            "runnerVersion": RUNNER_VERSION,
            "runtime": f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "elapsedSeconds": execution["elapsedSeconds"],
            "timeoutSeconds": timeout_seconds,
            "operatorCanReadInputs": True,
            "attestation": "NOT_HARDWARE_ATTESTED",
        }
    receipt_path = destination / "receipt.json"
    receipt = create_receipt(
        request_file=request_path,
        result_file=result_path,
        output=receipt_path,
        signing_key_file=key_path,
        evaluator=evaluator,
        secure_environment=secure_environment,
    )
    return {
        "format": RUNNER_FORMAT,
        "version": RUNNER_VERSION,
        "status": "PRIVATE_EVALUATION_COMPLETE",
        "requestId": request.get("requestId"),
        "benchmarkId": request.get("benchmarkId"),
        "artifactSha256": expected_artifact,
        "dataSha256": actual_data_digest,
        "backend": backend,
        "resultDisclosure": disclosure,
        "resultPath": str(result_path),
        "receiptPath": str(receipt_path),
        "receiptSha256": receipt.get("receiptSha256"),
        "secureEnvironment": secure_environment,
        "submitted": False,
        "published": False,
        "note": "Compute-to-data evaluation completed locally; no submission or publication was performed.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Private compute-to-data runner")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("capabilities")
    run = sub.add_parser("execute")
    run.add_argument("--request", required=True)
    run.add_argument("--artifact", required=True)
    run.add_argument("--data", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--signing-key", required=True)
    run.add_argument("--evaluator", required=True)
    run.add_argument("--start")
    run.add_argument("--end")
    run.add_argument("--benchmark-code", default="000300.SH")
    run.add_argument("--minimum-names", type=int)
    run.add_argument("--minimum-coverage", type=float)
    run.add_argument("--selection-ratio", type=float)
    run.add_argument("--timeout-seconds", type=int, default=3600)
    run.add_argument("--backend", choices=["process", "oci"], default="process")
    run.add_argument("--container-image")
    run.add_argument("--cpus", type=float, default=2.0)
    run.add_argument("--memory-mb", type=int, default=2048)
    run.add_argument("--pids-limit", type=int, default=128)
    run.add_argument("--toolkit-root")
    worker = sub.add_parser(WORKER_COMMAND, help=argparse.SUPPRESS)
    worker.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if args.command == WORKER_COMMAND:
        return _worker(Path(args.config))
    if args.command == "capabilities":
        result = runner_capabilities()
    else:
        result = run_request(
            request_file=args.request,
            artifact_file=args.artifact,
            data_path=args.data,
            output_dir=args.output,
            signing_key_file=args.signing_key,
            evaluator=args.evaluator,
            start=args.start,
            end=args.end,
            benchmark_code=args.benchmark_code,
            minimum_names=args.minimum_names,
            minimum_coverage=args.minimum_coverage,
            selection_ratio=args.selection_ratio,
            timeout_seconds=args.timeout_seconds,
            backend=args.backend,
            container_image=args.container_image,
            cpus=args.cpus,
            memory_mb=args.memory_mb,
            pids_limit=args.pids_limit,
            toolkit_root=args.toolkit_root,
        )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
