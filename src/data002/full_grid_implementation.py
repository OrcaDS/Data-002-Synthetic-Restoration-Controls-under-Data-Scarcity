"""Prospective full-grid implementation freeze manifest tooling."""

from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from data002.replay_provenance import file_sha256

IMPLEMENTATION_MANIFEST_PATH = (
    "results/provenance/replication_full_grid_implementation_manifest_v1.json"
)
MUTABLE_LAUNCH_PATH = "config/replication_full_launch_v1.json"
CRITICAL_FILES = (
    "results/provenance/replication_full_grid_implementation_draft_v1.json",
    "scripts/run_replication_full_grid.py",
    "scripts/write_full_grid_implementation_manifest.py",
    "src/data002/full_grid_implementation.py",
    "src/data002/full_grid_provenance.py",
    "src/data002/full_grid_runner.py",
    "tests/test_full_grid_runner.py",
)


class FullGridImplementationError(ValueError):
    """The prospective implementation freeze is malformed or has drifted."""


def _git_blob(project_root: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={project_root.resolve().as_posix()}",
            "show",
            f"{commit}:{relative}",
        ],
        cwd=project_root,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise FullGridImplementationError(
            f"reviewed commit does not contain critical file: {relative}"
        )
    return result.stdout


def build_full_grid_implementation_manifest(
    project_root: Path, reviewed_base_commit: str
) -> dict[str, Any]:
    project_root = project_root.resolve()
    if (
        not isinstance(reviewed_base_commit, str)
        or len(reviewed_base_commit) != 40
        or any(character not in "0123456789abcdef" for character in reviewed_base_commit)
    ):
        raise FullGridImplementationError("reviewed base commit must be lowercase SHA-1")
    if MUTABLE_LAUNCH_PATH in CRITICAL_FILES:
        raise FullGridImplementationError("mutable launch must not enter the freeze")
    files: dict[str, dict[str, Any]] = {}
    for relative in CRITICAL_FILES:
        path = project_root / relative
        if not path.is_file():
            raise FullGridImplementationError(f"critical file missing: {relative}")
        content = path.read_bytes()
        blob = _git_blob(project_root, reviewed_base_commit, relative)
        if blob != content:
            raise FullGridImplementationError(
                f"critical file differs from reviewed commit: {relative}"
            )
        files[relative] = {
            "bytes": len(content),
            "sha256": sha256(content).hexdigest(),
        }
    return {
        "schema_version": 1,
        "record": "data002_replication_full_grid_implementation_v1",
        "status": "implementation_frozen_pending_authorization",
        "authorization": None,
        "reviewed_base_commit": reviewed_base_commit,
        "mutable_launch_excluded": MUTABLE_LAUNCH_PATH,
        "file_count": len(files),
        "files": files,
    }


def validate_full_grid_implementation_manifest(
    project_root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    expected_fields = {
        "schema_version", "record", "status", "authorization",
        "reviewed_base_commit", "mutable_launch_excluded", "file_count", "files",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_fields:
        raise FullGridImplementationError("implementation manifest fields changed")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or manifest.get("record") != "data002_replication_full_grid_implementation_v1"
        or manifest.get("status") != "implementation_frozen_pending_authorization"
        or manifest.get("authorization") is not None
        or manifest.get("mutable_launch_excluded") != MUTABLE_LAUNCH_PATH
        or MUTABLE_LAUNCH_PATH in manifest.get("files", {})
    ):
        raise FullGridImplementationError("implementation manifest identity changed")
    files = manifest.get("files")
    if (
        not isinstance(files, Mapping)
        or tuple(files) != CRITICAL_FILES
        or type(manifest.get("file_count")) is not int
        or manifest.get("file_count") != len(CRITICAL_FILES)
    ):
        raise FullGridImplementationError("implementation critical files changed")
    base = manifest.get("reviewed_base_commit")
    if (
        not isinstance(base, str)
        or len(base) != 40
        or any(character not in "0123456789abcdef" for character in base)
    ):
        raise FullGridImplementationError("reviewed base commit is invalid")
    for relative, entry in files.items():
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"bytes", "sha256"}
            or type(entry.get("bytes")) is not int
            or entry["bytes"] < 0
            or not isinstance(entry.get("sha256"), str)
            or len(entry["sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in entry["sha256"]
            )
        ):
            raise FullGridImplementationError("critical-file entry malformed")
        path = project_root / relative
        blob = _git_blob(project_root, base, relative)
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or file_sha256(path) != entry["sha256"]
            or len(blob) != entry["bytes"]
            or sha256(blob).hexdigest() != entry["sha256"]
        ):
            raise FullGridImplementationError(f"critical-file drift: {relative}")
    manifest_path = project_root / IMPLEMENTATION_MANIFEST_PATH
    return {
        "status": "pass",
        "reviewed_base_commit": base,
        "manifest_sha256": (
            file_sha256(manifest_path) if manifest_path.is_file() else None
        ),
        "critical_files_sha256": {
            relative: entry["sha256"] for relative, entry in files.items()
        },
    }
