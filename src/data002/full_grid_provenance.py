"""Fail-closed provenance and authorization controls for full-grid execution."""

from __future__ import annotations

import json
import os
import socket
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

import psutil

from data002.full_grid_implementation import (
    IMPLEMENTATION_MANIFEST_PATH,
    validate_full_grid_implementation_manifest,
)
from data002.pilot_provenance import verify_design_review
from data002.replay_provenance import (
    ConcurrentRunError,
    file_sha256,
    verify_active_environment,
)
from data002.replication_design import (
    validate_key_manifest,
    validate_replication_implementation_manifest,
)

FULL_OUTPUT_ROOT = "results/replication/full_v1"
PILOT_OUTPUT_ROOT = "results/replication/pilot_v1"
FULL_LOCK_NAME = ".replication_full.lock"
ACCEPTED_PILOT_REPORT_SHA256 = (
    "f08bfbd03456dc8556f388799e827e572c40563a2cde76d2e5c345c27b415c7d"
)
PATHS = {
    "environment_inventory": "results/provenance/environment_inventory_python313_v1.json",
    "design_review": "results/provenance/replication_design_review_v1.json",
    "study_contract": "config/replication_study_v1.json",
    "pilot_contract": "config/replication_pilot_v1.json",
    "analysis_contract": "config/replication_analysis_v1.json",
    "full_key_manifest": "results/provenance/replication_full_grid_keys_v1.json",
    "pilot_key_manifest": "results/provenance/replication_pilot_keys_v1.json",
    "design_implementation_manifest": "results/provenance/replication_design_implementation_manifest_v1.json",
    "pilot_implementation_manifest": "results/provenance/replication_pilot_implementation_manifest_v1.json",
    "pilot_launch": "config/replication_pilot_launch_v1.json",
    "reconstruction": "src/data002/reconstruction.py",
    "replication_constructor": "src/data002/replication_constructor.py",
    "pilot_operational_report": "results/replication/pilot_v1/operational_report.json",
}
EXECUTION_POLICY = {
    "resume_valid_terminal_checkpoints": True,
    "atomic_condition_artifacts": True,
    "all_expected_conditions_must_succeed": True,
    "exclusive_run_lock": True,
    "child_process_isolation": True,
    "metric_blind": True,
    "pilot_reuse_required": True,
    "silent_pilot_refit_forbidden": True,
}
RESOURCE_POLICY = {
    "max_parallel_conditions": 1,
    "condition_timeout_seconds": 900,
    "minimum_free_disk_bytes": 2_147_483_648,
}
REUSE_POLICY = {
    "accepted_operational_report_sha256": ACCEPTED_PILOT_REPORT_SHA256,
    "expected_successful_conditions": 12,
    "copy_mode": "byte_preserving",
}
LAUNCH_KEYS = {
    "schema_version", "record", "status", "authorization", "authorized_by",
    "authorized_at", "scope", "output_root", "full_key_manifest",
    "pilot_key_manifest", "pilot_output_root", "implementation_manifest",
    "resources", "execution", "pilot_reuse", "bindings",
}
BINDING_KEYS = {
    "environment_inventory_sha256", "design_review_sha256",
    "study_contract_sha256", "pilot_contract_sha256",
    "analysis_contract_sha256", "full_key_manifest_sha256",
    "pilot_key_manifest_sha256", "design_implementation_manifest_sha256",
    "pilot_implementation_manifest_sha256", "pilot_launch_sha256",
    "reconstruction_sha256", "replication_constructor_sha256",
    "pilot_operational_report_sha256", "reviewed_base_commit",
    "full_grid_implementation_manifest_sha256",
}


class FullGridProvenanceError(ValueError):
    """Full-grid provenance, policy, or authorization is not exact."""


@dataclass(frozen=True)
class FullGridStartupPolicy:
    timeout_seconds: int
    minimum_free_disk_bytes: int
    output_root: Path
    pilot_output_root: Path


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FullGridProvenanceError(f"{path} must contain an object")
    return value


def expected_bindings(
    project_root: Path,
    implementation_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bindings: dict[str, Any] = {
        f"{name}_sha256": file_sha256(project_root / relative)
        for name, relative in PATHS.items()
    }
    bindings["reviewed_base_commit"] = (
        implementation_report["reviewed_base_commit"]
        if implementation_report is not None else None
    )
    bindings["full_grid_implementation_manifest_sha256"] = (
        implementation_report["manifest_sha256"]
        if implementation_report is not None else None
    )
    return bindings


def _timezone_aware(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def validate_authorized_full_grid_launch(
    launch: Mapping[str, Any], *, bindings: Mapping[str, Any]
) -> None:
    if set(launch) != LAUNCH_KEYS:
        raise FullGridProvenanceError("full-grid launch fields changed")
    if launch.get("schema_version") != 1 or launch.get("record") != (
        "data002_replication_full_launch_v1"
    ):
        raise FullGridProvenanceError("full-grid launch identity changed")
    if launch.get("authorization") != "authorized" or launch.get("status") != "authorized":
        raise PermissionError("full-grid launch is not authorized")
    if not isinstance(launch.get("authorized_by"), str) or not launch["authorized_by"].strip():
        raise PermissionError("full-grid authorized_by must be nonempty")
    if not _timezone_aware(launch.get("authorized_at")):
        raise PermissionError("full-grid authorized_at must be timezone-aware")
    if (
        launch.get("scope") != "fixed_540_condition_metric_blind_replication_full_grid"
        or launch.get("output_root") != FULL_OUTPUT_ROOT
        or launch.get("pilot_output_root") != PILOT_OUTPUT_ROOT
        or launch.get("implementation_manifest") != IMPLEMENTATION_MANIFEST_PATH
        or launch.get("full_key_manifest") != PATHS["full_key_manifest"]
        or launch.get("pilot_key_manifest") != PATHS["pilot_key_manifest"]
        or launch.get("resources") != RESOURCE_POLICY
        or launch.get("execution") != EXECUTION_POLICY
        or launch.get("pilot_reuse") != REUSE_POLICY
    ):
        raise FullGridProvenanceError("full-grid launch policy or path changed")
    observed = launch.get("bindings")
    if not isinstance(observed, Mapping) or set(observed) != BINDING_KEYS:
        raise FullGridProvenanceError("full-grid launch bindings changed")
    if dict(observed) != dict(bindings):
        raise FullGridProvenanceError("full-grid source or evidence binding drift")


def verify_full_grid_startup(
    project_root: Path, launch_path: Path
) -> FullGridStartupPolicy:
    project_root = project_root.resolve()
    launch = _load(launch_path)
    if launch.get("authorization") != "authorized" or launch.get("status") != "authorized":
        validate_authorized_full_grid_launch(launch, bindings={})
    verify_design_review(project_root)
    design_manifest = _load(project_root / PATHS["design_implementation_manifest"])
    validate_replication_implementation_manifest(project_root, design_manifest)
    full_manifest = _load(project_root / PATHS["full_key_manifest"])
    pilot_manifest = _load(project_root / PATHS["pilot_key_manifest"])
    validate_key_manifest(full_manifest, scope="full")
    validate_key_manifest(pilot_manifest, scope="pilot")
    verify_active_environment(
        project_root / PATHS["environment_inventory"],
        project_root / "requirements-lock.txt",
    )
    preliminary_bindings = expected_bindings(project_root)
    if preliminary_bindings["pilot_operational_report_sha256"] != ACCEPTED_PILOT_REPORT_SHA256:
        raise FullGridProvenanceError("accepted pilot report hash changed")
    implementation = validate_full_grid_implementation_manifest(
        project_root,
        _load(project_root / IMPLEMENTATION_MANIFEST_PATH),
    )
    bindings = expected_bindings(project_root, implementation)
    validate_authorized_full_grid_launch(launch, bindings=bindings)
    return FullGridStartupPolicy(
        900,
        2_147_483_648,
        (project_root / FULL_OUTPUT_ROOT).resolve(),
        (project_root / PILOT_OUTPUT_ROOT).resolve(),
    )


def validate_full_grid_lock_record(record: Mapping[str, Any]) -> None:
    if set(record) != {
        "schema_version", "record", "pid", "process_create_time",
        "hostname", "created_at",
    }:
        raise FullGridProvenanceError("full-grid lock fields changed")
    if (
        record.get("schema_version") != 1
        or record.get("record") != "data002_replication_full_run_lock_v1"
        or isinstance(record.get("pid"), bool) or not isinstance(record.get("pid"), int)
        or record["pid"] < 0
        or isinstance(record.get("process_create_time"), bool)
        or not isinstance(record.get("process_create_time"), (int, float))
        or record["process_create_time"] < 0
        or not isinstance(record.get("hostname"), str) or not record["hostname"]
        or not _timezone_aware(record.get("created_at"))
    ):
        raise FullGridProvenanceError("full-grid lock record is invalid")


def _owner_live(record: Mapping[str, Any]) -> bool:
    if record.get("hostname") != socket.gethostname():
        return True
    try:
        process = psutil.Process(record["pid"])
        return abs(process.create_time() - float(record["process_create_time"])) < 1e-3
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except psutil.Error:
        return True


@contextmanager
def exclusive_full_grid_run_lock(output_root: Path) -> Iterator[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / FULL_LOCK_NAME
    record = {
        "schema_version": 1,
        "record": "data002_replication_full_run_lock_v1",
        "pid": os.getpid(),
        "process_create_time": psutil.Process(os.getpid()).create_time(),
        "hostname": socket.gethostname(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    encoded = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                existing = _load(lock_path)
                validate_full_grid_lock_record(existing)
            except Exception as exc:
                raise ConcurrentRunError(
                    f"existing full-grid lock cannot be proven stale: {exc}"
                ) from exc
            if _owner_live(existing):
                raise ConcurrentRunError("another live process owns the full-grid lock")
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            os.replace(lock_path, output_root / (
                f".replication_full.stale.{stamp}.{existing['pid']}.json"
            ))
            continue
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        break
    try:
        yield record
    finally:
        try:
            current = _load(lock_path)
        except Exception:
            current = None
        if current == record:
            lock_path.unlink(missing_ok=True)
