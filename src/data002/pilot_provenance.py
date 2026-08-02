"""Fail-closed provenance and authorization controls for the replication pilot."""

from __future__ import annotations

import json
import os
import socket
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Mapping

import psutil

from data002.replay_provenance import ConcurrentRunError

from data002.replay_provenance import file_sha256, verify_active_environment
from data002.replication_design import (
    validate_key_manifest,
    validate_replication_implementation_manifest,
)

AUTHORIZED = "authorized"
PILOT_OUTPUT_ROOT = "results/replication/pilot_v1"
PILOT_LOCK_NAME = ".replication_pilot.lock"
DESIGN_COMMIT = "e375303602729ffd1aa12ebfbdc4fa9714c1c1a7"
PATHS = {
    "design_review": "results/provenance/replication_design_review_v1.json",
    "study_contract": "config/replication_study_v1.json",
    "pilot_contract": "config/replication_pilot_v1.json",
    "analysis_contract": "config/replication_analysis_v1.json",
    "pilot_key_manifest": "results/provenance/replication_pilot_keys_v1.json",
    "design_implementation_manifest": (
        "results/provenance/replication_design_implementation_manifest_v1.json"
    ),
    "pilot_implementation_manifest": (
        "results/provenance/replication_pilot_implementation_manifest_v1.json"
    ),
    "environment_inventory": (
        "results/provenance/environment_inventory_python313_v1.json"
    ),
}
ACCEPTED_MANIFEST_HASHES = {
    "replication_full_grid_keys_v1": (
        "428846efb67d036975a159b2370855da9c42de9e2a93b53fec0d916e8f0b605b"
    ),
    "replication_pilot_keys_v1": (
        "f8f5126cf489c407701ea8c71adc8c31ac5de4cf3b3811bec0196b288835c6e9"
    ),
    "data001_metric_bundle_manifest_v1": (
        "f2fa5363cab80c5e97c6bb94463c73ef767e5a1d2f413190be5123361afa1c0c"
    ),
    "replication_design_implementation_manifest_v1": (
        "42708b56e62b60c5ad2bc8f8db147a406c31ab46c255a1d517e83542db15cd05"
    ),
}
LAUNCH_KEYS = {
    "schema_version",
    "record",
    "status",
    "authorization",
    "authorized_by",
    "authorized_at",
    "scope",
    "output_root",
    *PATHS,
    "resources",
    "execution",
    "bindings",
}
BINDING_KEYS = {
    "reviewed_git_commit",
    "environment_inventory_sha256",
    "design_review_sha256",
    "study_contract_sha256",
    "pilot_contract_sha256",
    "analysis_contract_sha256",
    "pilot_key_manifest_sha256",
    "design_implementation_manifest_sha256",
    "pilot_implementation_manifest_sha256",
    "critical_files_sha256",
}
EXECUTION_POLICY = {
    "resume_valid_terminal_checkpoints": True,
    "atomic_condition_artifacts": True,
    "all_expected_conditions_must_succeed": True,
    "exclusive_run_lock": True,
    "child_process_isolation": True,
    "metric_blind": True,
}


class PilotProvenanceError(ValueError):
    """Pilot provenance or authorization is not exact."""


@dataclass(frozen=True)
class PilotStartupPolicy:
    timeout_seconds: int
    minimum_free_disk_bytes: int
    output_root: Path


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PilotProvenanceError(f"{path} must contain an object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise PilotProvenanceError(f"{label} keys changed")


def verify_design_review(project_root: Path) -> dict[str, Any]:
    review_path = project_root / PATHS["design_review"]
    review = _load(review_path)
    _exact(
        review,
        {
            "schema_version",
            "record",
            "status",
            "accepted_design_commit",
            "accepted_by",
            "accepted_at",
            "accepted_manifest_sha256",
            "execution_authorization",
            "analysis_authorization",
            "scope",
        },
        "design review",
    )
    if (
        review.get("schema_version") != 1
        or review.get("record") != "data002_replication_design_review_v1"
        or review.get("status") != "scientifically_accepted"
        or review.get("accepted_design_commit") != DESIGN_COMMIT
        or review.get("accepted_manifest_sha256") != ACCEPTED_MANIFEST_HASHES
        or review.get("execution_authorization") is not None
        or review.get("analysis_authorization") is not None
    ):
        raise PilotProvenanceError("design review identity or boundary changed")
    accepted_paths = {
        "replication_full_grid_keys_v1": (
            "results/provenance/replication_full_grid_keys_v1.json"
        ),
        "replication_pilot_keys_v1": (
            "results/provenance/replication_pilot_keys_v1.json"
        ),
        "data001_metric_bundle_manifest_v1": (
            "results/provenance/data001_metric_bundle_manifest_v1.json"
        ),
        "replication_design_implementation_manifest_v1": (
            "results/provenance/replication_design_implementation_manifest_v1.json"
        ),
    }
    for name, relative in accepted_paths.items():
        if file_sha256(project_root / relative) != ACCEPTED_MANIFEST_HASHES[name]:
            raise PilotProvenanceError(f"accepted manifest drift: {name}")
    return {"status": "pass", "sha256": file_sha256(review_path)}


def verify_pilot_implementation_manifest(
    project_root: Path, manifest_path: Path
) -> dict[str, Any]:
    manifest = _load(manifest_path)
    _exact(
        manifest,
        {
            "schema_version",
            "record",
            "status",
            "authorization",
            "base_commit",
            "file_count",
            "files",
        },
        "pilot implementation manifest",
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("record") != "data002_replication_pilot_implementation_v1"
        or manifest.get("status") != "implementation_frozen_pending_authorization"
        or manifest.get("authorization") is not None
    ):
        raise PilotProvenanceError("pilot implementation identity changed")
    base = manifest.get("base_commit")
    files = manifest.get("files")
    if not isinstance(base, str) or len(base) != 40 or not isinstance(files, Mapping):
        raise PilotProvenanceError("pilot implementation manifest malformed")
    if manifest.get("file_count") != len(files):
        raise PilotProvenanceError("pilot implementation file count changed")
    observed: dict[str, str] = {}
    for relative, entry in files.items():
        if not isinstance(entry, Mapping) or set(entry) != {"bytes", "sha256"}:
            raise PilotProvenanceError("pilot critical-file entry malformed")
        path = project_root / relative
        if (
            not path.is_file()
            or path.stat().st_size != entry.get("bytes")
            or file_sha256(path) != entry.get("sha256")
        ):
            raise PilotProvenanceError(f"pilot critical-file drift: {relative}")
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={project_root.resolve().as_posix()}",
                "show",
                f"{base}:{relative}",
            ],
            cwd=project_root,
            capture_output=True,
            check=False,
        )
        if (
            result.returncode
            or len(result.stdout) != entry["bytes"]
            or sha256(result.stdout).hexdigest() != entry["sha256"]
        ):
            raise PilotProvenanceError(f"pilot base-commit drift: {relative}")
        observed[relative] = entry["sha256"]
    return {
        "status": "pass",
        "base_commit": base,
        "manifest_sha256": file_sha256(manifest_path),
        "critical_files_sha256": observed,
        "file_count": len(files),
    }


def _timezone_aware(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def validate_authorized_pilot_launch(
    launch: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, Any],
    implementation_report: Mapping[str, Any],
) -> tuple[int, int]:
    _exact(launch, LAUNCH_KEYS, "pilot launch")
    if launch.get("schema_version") != 1 or launch.get(
        "record"
    ) != "data002_replication_pilot_launch_v1":
        raise PilotProvenanceError("pilot launch identity changed")
    if launch.get("status") != AUTHORIZED or launch.get("authorization") != AUTHORIZED:
        raise PermissionError("pilot launch is not authorized")
    if not isinstance(launch.get("authorized_by"), str) or not launch[
        "authorized_by"
    ].strip():
        raise PermissionError("pilot authorized_by must be nonempty")
    if not _timezone_aware(launch.get("authorized_at")):
        raise PermissionError("pilot authorized_at must be timezone-aware")
    if launch.get("scope") != "fixed_12_condition_metric_blind_replication_pilot":
        raise PilotProvenanceError("pilot launch scope changed")
    if launch.get("output_root") != PILOT_OUTPUT_ROOT:
        raise PilotProvenanceError("pilot output root changed")
    for field, relative in PATHS.items():
        if launch.get(field) != relative:
            raise PilotProvenanceError(f"pilot launch path changed: {field}")
    if launch.get("resources") != {
        "max_parallel_conditions": 1,
        "condition_timeout_seconds": 900,
        "minimum_free_disk_bytes": 2_147_483_648,
    }:
        raise PilotProvenanceError("pilot resources changed")
    if launch.get("execution") != EXECUTION_POLICY:
        raise PilotProvenanceError("pilot execution policy changed")
    bindings = launch.get("bindings")
    if not isinstance(bindings, Mapping):
        raise PilotProvenanceError("pilot bindings missing")
    _exact(bindings, BINDING_KEYS, "pilot bindings")
    if bindings.get("reviewed_git_commit") != implementation_report.get("base_commit"):
        raise PilotProvenanceError("reviewed commit does not equal pilot base commit")
    if dict(bindings) != dict(expected_bindings):
        raise PilotProvenanceError("pilot authorization bindings changed")
    return 900, 2_147_483_648


def verify_pilot_startup(
    project_root: Path, launch_path: Path
) -> PilotStartupPolicy:
    review = verify_design_review(project_root)
    design_manifest = _load(project_root / PATHS["design_implementation_manifest"])
    validate_replication_implementation_manifest(project_root, design_manifest)
    pilot_keys = _load(project_root / PATHS["pilot_key_manifest"])
    validate_key_manifest(pilot_keys, scope="pilot")
    environment_path = project_root / PATHS["environment_inventory"]
    verify_active_environment(environment_path, project_root / "requirements-lock.txt")
    pilot_manifest_path = project_root / PATHS["pilot_implementation_manifest"]
    implementation = verify_pilot_implementation_manifest(
        project_root, pilot_manifest_path
    )
    expected = {
        "reviewed_git_commit": implementation["base_commit"],
        "environment_inventory_sha256": file_sha256(environment_path),
        "design_review_sha256": review["sha256"],
        "study_contract_sha256": file_sha256(
            project_root / PATHS["study_contract"]
        ),
        "pilot_contract_sha256": file_sha256(
            project_root / PATHS["pilot_contract"]
        ),
        "analysis_contract_sha256": file_sha256(
            project_root / PATHS["analysis_contract"]
        ),
        "pilot_key_manifest_sha256": file_sha256(
            project_root / PATHS["pilot_key_manifest"]
        ),
        "design_implementation_manifest_sha256": file_sha256(
            project_root / PATHS["design_implementation_manifest"]
        ),
        "pilot_implementation_manifest_sha256": implementation["manifest_sha256"],
        "critical_files_sha256": implementation["critical_files_sha256"],
    }
    timeout, minimum_disk = validate_authorized_pilot_launch(
        _load(launch_path),
        expected_bindings=expected,
        implementation_report=implementation,
    )
    return PilotStartupPolicy(
        timeout_seconds=timeout,
        minimum_free_disk_bytes=minimum_disk,
        output_root=(project_root / PILOT_OUTPUT_ROOT).resolve(),
    )


def _pilot_lock_owner_is_live(record: Mapping[str, Any]) -> bool:
    if record.get("hostname") != socket.gethostname():
        return True
    pid = record.get("pid")
    create_time = record.get("process_create_time")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or isinstance(create_time, bool)
        or not isinstance(create_time, (int, float))
    ):
        return True
    try:
        process = psutil.Process(pid)
        return abs(process.create_time() - float(create_time)) < 1e-3
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except psutil.Error:
        return True


def validate_pilot_lock_record(record: Mapping[str, Any]) -> None:
    if set(record) != {
        "schema_version",
        "record",
        "pid",
        "process_create_time",
        "hostname",
        "created_at",
    }:
        raise PilotProvenanceError("pilot lock fields changed")
    if (
        record.get("schema_version") != 1
        or record.get("record") != "data002_replication_pilot_run_lock_v1"
        or isinstance(record.get("pid"), bool)
        or not isinstance(record.get("pid"), int)
        or record["pid"] < 0
        or isinstance(record.get("process_create_time"), bool)
        or not isinstance(record.get("process_create_time"), (int, float))
        or record["process_create_time"] < 0
        or not isinstance(record.get("hostname"), str)
        or not record["hostname"]
        or not _timezone_aware(record.get("created_at"))
    ):
        raise PilotProvenanceError("pilot lock record is invalid")


@contextmanager
def exclusive_pilot_run_lock(output_root: Path) -> Iterator[dict[str, Any]]:
    """Acquire a pilot-specific PID/create-time lock with retained stale records."""

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / PILOT_LOCK_NAME
    record = {
        "schema_version": 1,
        "record": "data002_replication_pilot_run_lock_v1",
        "pid": os.getpid(),
        "process_create_time": psutil.Process(os.getpid()).create_time(),
        "hostname": socket.gethostname(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    encoded = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    while True:
        try:
            descriptor = os.open(
                lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
        except FileExistsError:
            try:
                existing = _load(lock_path)
                validate_pilot_lock_record(existing)
            except Exception as exc:
                raise ConcurrentRunError(
                    f"existing pilot lock cannot be proven stale: {exc}"
                ) from exc
            if _pilot_lock_owner_is_live(existing):
                raise ConcurrentRunError(
                    "another live process owns the pilot run lock"
                )
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            stale_path = output_root / (
                f".replication_pilot.stale.{stamp}.{existing['pid']}.json"
            )
            try:
                os.replace(lock_path, stale_path)
            except FileNotFoundError:
                continue
            continue
        else:
            with os.fdopen(descriptor, "wb") as stream:
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
