"""Strict startup provenance and exclusive-run controls for compatibility replay."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import socket
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Mapping

import psutil

AUTHORIZED_VALUE = "authorized"
LAUNCH_SCOPE = "prospectively_fixed_16_condition_baseline_compatibility_replay"
CONTRACT_PATH = "config/compatibility_replay_v1.json"
ENVIRONMENT_INVENTORY_PATH = (
    "results/provenance/environment_inventory_python313_v1.json"
)
IMPLEMENTATION_MANIFEST_PATH = (
    "results/provenance/compatibility_implementation_manifest_v1.json"
)
LOCK_PATH = "requirements-lock.txt"
RUNNER_PATH = "src/data002/compatibility_runner.py"
LAUNCH_KEYS = {
    "schema_version",
    "record",
    "status",
    "authorization",
    "authorized_by",
    "authorized_at",
    "scope",
    "contract",
    "environment_inventory",
    "implementation_manifest",
    "resources",
    "execution",
    "bindings",
}
BINDING_KEYS = {
    "reviewed_git_commit",
    "contract_sha256",
    "environment_inventory_sha256",
    "implementation_manifest_sha256",
    "replay_critical_files_sha256",
}
RESOURCE_KEYS = {
    "max_parallel_conditions",
    "condition_timeout_seconds",
    "minimum_free_disk_bytes",
}
EXECUTION_POLICY = {
    "resume_valid_terminal_checkpoints": True,
    "atomic_condition_artifacts": True,
    "all_expected_conditions_must_pass": True,
}


class ProvenanceError(ValueError):
    """A launch, environment, or implementation identity is not exact."""


class ConcurrentRunError(RuntimeError):
    """Another live process owns the replay output root."""


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ProvenanceError(
            f"{label} keys must be exactly {sorted(expected)}, found {sorted(value)}"
        )


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"{label} cannot be loaded: {type(exc).__name__}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProvenanceError(f"{label} must contain a JSON object")
    return value


def _parse_lock(lock_path: Path) -> dict[str, str]:
    locked: dict[str, str] = {}
    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise ProvenanceError(f"lock entry is not exact: {line!r}")
        name, version = line.split("==")
        normalized = name.casefold()
        if not name or not version or normalized in locked:
            raise ProvenanceError(f"invalid or duplicate lock entry: {line!r}")
        locked[normalized] = version
    return locked


def verify_active_environment(inventory_path: Path, lock_path: Path) -> dict[str, Any]:
    """Require exact Python and distribution agreement with inventory and lock."""

    inventory = _load_object(inventory_path, "environment inventory")
    _exact_keys(
        inventory,
        {
            "schema_version",
            "record",
            "python",
            "platform",
            "requirements_lock",
            "packages",
            "package_count",
        },
        "environment inventory",
    )
    if (
        inventory.get("schema_version") != 1
        or inventory.get("record") != "data002_python_environment_inventory_v1"
    ):
        raise ProvenanceError("environment inventory identity is invalid")
    python_record = inventory.get("python")
    if not isinstance(python_record, Mapping):
        raise ProvenanceError("environment inventory Python record is invalid")
    if python_record.get("version") != platform.python_version():
        raise ProvenanceError(
            f"Python version drift: expected {python_record.get('version')}, "
            f"found {platform.python_version()}"
        )
    if python_record.get("implementation") != platform.python_implementation():
        raise ProvenanceError("Python implementation drift")

    lock_record = inventory.get("requirements_lock")
    if not isinstance(lock_record, Mapping):
        raise ProvenanceError("environment inventory lock record is invalid")
    if lock_record.get("path") != LOCK_PATH:
        raise ProvenanceError("environment inventory lock path changed")
    observed_lock_hash = file_sha256(lock_path)
    if lock_record.get("sha256") != observed_lock_hash:
        raise ProvenanceError("requirements lock SHA-256 drift")

    packages = inventory.get("packages")
    if not isinstance(packages, list) or inventory.get("package_count") != len(packages):
        raise ProvenanceError("environment package count is invalid")
    expected: dict[str, str] = {}
    for package in packages:
        if not isinstance(package, Mapping) or set(package) != {"name", "version"}:
            raise ProvenanceError("environment package entry is invalid")
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise ProvenanceError("environment package name or version is invalid")
        normalized = name.casefold()
        if normalized in expected:
            raise ProvenanceError(f"duplicate environment package: {name}")
        expected[normalized] = version

    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            normalized = name.casefold()
            if normalized in installed:
                raise ProvenanceError(f"duplicate installed distribution: {name}")
            installed[normalized] = distribution.version
    if installed != expected:
        missing = sorted(set(expected) - set(installed))
        unexpected = sorted(set(installed) - set(expected))
        changed = sorted(
            name
            for name in set(expected) & set(installed)
            if expected[name] != installed[name]
        )
        raise ProvenanceError(
            "installed distribution drift: "
            f"missing={missing}, unexpected={unexpected}, changed={changed}"
        )

    locked = _parse_lock(lock_path)
    inventory_without_pip = dict(expected)
    inventory_without_pip.pop("pip", None)
    if inventory_without_pip != locked:
        raise ProvenanceError("environment inventory does not exactly match lock plus pip")
    return {
        "status": "pass",
        "python_version": platform.python_version(),
        "package_count": len(installed),
        "inventory_sha256": file_sha256(inventory_path),
        "lock_sha256": observed_lock_hash,
    }


def verify_implementation_manifest(
    project_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    manifest = _load_object(manifest_path, "implementation manifest")
    _exact_keys(
        manifest,
        {"schema_version", "record", "base_commit", "file_count", "files"},
        "implementation manifest",
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("record") != "data002_compatibility_implementation_v1"
    ):
        raise ProvenanceError("implementation manifest identity is invalid")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or manifest.get("file_count") != len(files):
        raise ProvenanceError("implementation manifest file count is invalid")
    observed: dict[str, str] = {}
    for relative, entry in files.items():
        if not isinstance(relative, str) or not isinstance(entry, Mapping):
            raise ProvenanceError("implementation manifest file entry is invalid")
        _exact_keys(entry, {"bytes", "sha256"}, f"implementation entry {relative}")
        path = (project_root / relative).resolve()
        if not path.is_relative_to(project_root.resolve()) or not path.is_file():
            raise ProvenanceError(f"implementation file missing or unsafe: {relative}")
        if entry.get("bytes") != path.stat().st_size:
            raise ProvenanceError(f"implementation file size drift: {relative}")
        observed_hash = file_sha256(path)
        if entry.get("sha256") != observed_hash:
            raise ProvenanceError(f"implementation file SHA-256 drift: {relative}")
        observed[relative] = observed_hash
    if RUNNER_PATH not in observed:
        raise ProvenanceError("implementation manifest omits the runner")
    base_commit = manifest.get("base_commit")
    if not isinstance(base_commit, str) or not _git_commit_is_available_ancestor(
        project_root, base_commit
    ):
        raise ProvenanceError(
            "implementation manifest base commit is invalid, unavailable, "
            "or not an ancestor"
        )
    command_prefix = [
        "git",
        "-c",
        f"safe.directory={project_root.as_posix()}",
    ]
    for relative, entry in files.items():
        result = subprocess.run(
            [*command_prefix, "show", f"{base_commit}:{relative}"],
            cwd=project_root,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise ProvenanceError(
                f"implementation file is absent from base commit: {relative}"
            )
        if len(result.stdout) != entry["bytes"]:
            raise ProvenanceError(
                f"base-commit implementation file size drift: {relative}"
            )
        if sha256(result.stdout).hexdigest() != entry["sha256"]:
            raise ProvenanceError(
                f"base-commit implementation file SHA-256 drift: {relative}"
            )
    return {
        "status": "pass",
        "base_commit": base_commit,
        "manifest_sha256": file_sha256(manifest_path),
        "files_sha256": observed,
    }


def _valid_authorized_at(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _git_commit_is_available_ancestor(project_root: Path, commit: str) -> bool:
    if (
        len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        return False
    command_prefix = [
        "git",
        "-c",
        f"safe.directory={project_root.as_posix()}",
    ]
    result = subprocess.run(
        [*command_prefix, "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=project_root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    ancestor = subprocess.run(
        [*command_prefix, "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=project_root,
        capture_output=True,
        check=False,
    )
    return ancestor.returncode == 0


def validate_authorized_launch(
    launch: Mapping[str, Any],
    *,
    project_root: Path,
    contract_path: Path,
    inventory_path: Path,
    manifest_path: Path,
    implementation_report: Mapping[str, Any],
) -> tuple[int, int]:
    """Validate the exact launch schema and every reviewed binding."""

    _exact_keys(launch, LAUNCH_KEYS, "launch record")
    if launch.get("schema_version") != 1:
        raise ProvenanceError("launch schema version must equal 1")
    if launch.get("record") != "data002_compatibility_launch_v1":
        raise ProvenanceError("launch record identity is invalid")
    if launch.get("status") != AUTHORIZED_VALUE:
        raise PermissionError("launch status must equal 'authorized'")
    if launch.get("authorization") != AUTHORIZED_VALUE:
        raise PermissionError("launch authorization must equal 'authorized'")
    authorized_by = launch.get("authorized_by")
    if not isinstance(authorized_by, str) or not authorized_by.strip():
        raise PermissionError("authorized_by must be nonempty")
    if not _valid_authorized_at(launch.get("authorized_at")):
        raise PermissionError("authorized_at must be a timezone-aware ISO-8601 value")
    if launch.get("scope") != LAUNCH_SCOPE:
        raise ProvenanceError("launch scope changed")
    if launch.get("contract") != CONTRACT_PATH:
        raise ProvenanceError("launch contract path changed")
    if launch.get("environment_inventory") != ENVIRONMENT_INVENTORY_PATH:
        raise ProvenanceError("launch environment inventory path changed")
    if launch.get("implementation_manifest") != IMPLEMENTATION_MANIFEST_PATH:
        raise ProvenanceError("launch implementation manifest path changed")

    resources = launch.get("resources")
    if not isinstance(resources, Mapping):
        raise ProvenanceError("launch resources are missing")
    _exact_keys(resources, RESOURCE_KEYS, "launch resources")
    if resources.get("max_parallel_conditions") != 1:
        raise ProvenanceError("runner permits exactly one condition at a time")
    timeout_seconds = resources.get("condition_timeout_seconds")
    minimum_free_disk_bytes = resources.get("minimum_free_disk_bytes")
    if (
        not isinstance(timeout_seconds, int)
        or isinstance(timeout_seconds, bool)
        or timeout_seconds <= 0
    ):
        raise ProvenanceError("condition timeout must be a positive integer")
    if (
        not isinstance(minimum_free_disk_bytes, int)
        or isinstance(minimum_free_disk_bytes, bool)
        or minimum_free_disk_bytes < 0
    ):
        raise ProvenanceError("minimum free disk must be a nonnegative integer")
    execution = launch.get("execution")
    if not isinstance(execution, Mapping) or dict(execution) != EXECUTION_POLICY:
        raise ProvenanceError("launch execution policy changed")

    bindings = launch.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ProvenanceError("launch bindings are missing")
    _exact_keys(bindings, BINDING_KEYS, "launch bindings")
    reviewed_commit = bindings.get("reviewed_git_commit")
    if reviewed_commit != implementation_report.get("base_commit"):
        raise ProvenanceError(
            "reviewed Git commit must exactly equal the implementation "
            "manifest base commit"
        )
    if not isinstance(reviewed_commit, str) or not _git_commit_is_available_ancestor(
        project_root, reviewed_commit
    ):
        raise ProvenanceError(
            "reviewed Git commit is invalid, unavailable, or not an ancestor"
        )
    expected_bindings = {
        "contract_sha256": file_sha256(contract_path),
        "environment_inventory_sha256": file_sha256(inventory_path),
        "implementation_manifest_sha256": file_sha256(manifest_path),
        "replay_critical_files_sha256": implementation_report["files_sha256"],
    }
    for name, expected in expected_bindings.items():
        if bindings.get(name) != expected:
            raise ProvenanceError(f"authorization binding mismatch: {name}")
    return timeout_seconds, minimum_free_disk_bytes


def _lock_owner_is_live(record: Mapping[str, Any]) -> bool:
    if record.get("hostname") != socket.gethostname():
        return True
    pid = record.get("pid")
    create_time = record.get("process_create_time")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or not isinstance(create_time, (int, float))
        or isinstance(create_time, bool)
    ):
        return True
    try:
        process = psutil.Process(pid)
        return abs(process.create_time() - float(create_time)) < 1e-3
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except psutil.Error:
        return True


@contextmanager
def exclusive_run_lock(output_root: Path) -> Iterator[dict[str, Any]]:
    """Acquire an exclusive lock; retain only demonstrably stale lock records."""

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".compatibility_replay.lock"
    record = {
        "schema_version": 1,
        "record": "data002_compatibility_run_lock_v1",
        "pid": os.getpid(),
        "process_create_time": psutil.Process(os.getpid()).create_time(),
        "hostname": socket.gethostname(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    encoded = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    while True:
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError:
            try:
                existing = _load_object(lock_path, "run lock")
            except ProvenanceError as exc:
                raise ConcurrentRunError(
                    f"existing run lock cannot be proven stale: {exc}"
                ) from exc
            if _lock_owner_is_live(existing):
                raise ConcurrentRunError("another live process owns the replay run lock")
            stale_path = output_root / (
                f".compatibility_replay.stale.{time_safe_stamp()}.{existing.get('pid')}.json"
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
            current = _load_object(lock_path, "run lock")
        except ProvenanceError:
            current = None
        if current == record:
            lock_path.unlink(missing_ok=True)


def time_safe_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
