"""Prospective implementation-freeze manifest tooling for Data 002 analysis."""

from __future__ import annotations

import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

IMPLEMENTATION_MANIFEST_PATH = (
    "results/provenance/replication_analysis_implementation_manifest_v1.json"
)
MUTABLE_LAUNCH_PATH = "config/replication_analysis_launch_v1.json"
CRITICAL_FILES = (
    "results/provenance/replication_analysis_implementation_draft_v1.json",
    "scripts/run_replication_analysis.py",
    "scripts/write_replication_analysis_implementation_manifest.py",
    "src/data002/analysis_core.py",
    "src/data002/analysis_implementation.py",
    "src/data002/analysis_provenance.py",
    "src/data002/analysis_runner.py",
    "tests/test_replication_analysis.py",
)


class AnalysisImplementationError(ValueError):
    """The prospective analysis implementation freeze is invalid."""


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob(project_root: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(
        [
            "git", "-c",
            f"safe.directory={project_root.resolve().as_posix()}",
            "show", f"{commit}:{relative}",
        ],
        cwd=project_root,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise AnalysisImplementationError(
            f"reviewed commit lacks analysis file: {relative}"
        )
    return result.stdout


def build_analysis_implementation_manifest(
    project_root: Path, reviewed_base_commit: str
) -> dict[str, Any]:
    if (
        not isinstance(reviewed_base_commit, str)
        or len(reviewed_base_commit) != 40
        or any(character not in "0123456789abcdef" for character in reviewed_base_commit)
    ):
        raise AnalysisImplementationError(
            "reviewed base commit must be a lowercase 40-character SHA"
        )
    if MUTABLE_LAUNCH_PATH in CRITICAL_FILES:
        raise AnalysisImplementationError("mutable launch entered analysis freeze")
    files: dict[str, dict[str, Any]] = {}
    for relative in CRITICAL_FILES:
        path = project_root / relative
        if not path.is_file():
            raise AnalysisImplementationError(f"analysis file missing: {relative}")
        content = path.read_bytes()
        if _git_blob(project_root, reviewed_base_commit, relative) != content:
            raise AnalysisImplementationError(
                f"analysis file differs from reviewed commit: {relative}"
            )
        files[relative] = {
            "bytes": len(content),
            "sha256": sha256(content).hexdigest(),
        }
    return {
        "schema_version": 1,
        "record": "data002_replication_analysis_implementation_v1",
        "status": "implementation_frozen_pending_authorization",
        "authorization": None,
        "reviewed_base_commit": reviewed_base_commit,
        "mutable_launch_excluded": MUTABLE_LAUNCH_PATH,
        "file_count": len(files),
        "files": files,
    }


def validate_analysis_implementation_manifest(
    project_root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    fields = {
        "schema_version", "record", "status", "authorization",
        "reviewed_base_commit", "mutable_launch_excluded", "file_count", "files",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != fields:
        raise AnalysisImplementationError(
            "analysis implementation manifest fields changed"
        )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("record")
        != "data002_replication_analysis_implementation_v1"
        or manifest.get("status")
        != "implementation_frozen_pending_authorization"
        or manifest.get("authorization") is not None
        or manifest.get("mutable_launch_excluded") != MUTABLE_LAUNCH_PATH
        or MUTABLE_LAUNCH_PATH in manifest.get("files", {})
    ):
        raise AnalysisImplementationError(
            "analysis implementation identity changed"
        )
    files = manifest.get("files")
    if (
        not isinstance(files, Mapping)
        or tuple(files) != CRITICAL_FILES
        or manifest.get("file_count") != len(CRITICAL_FILES)
    ):
        raise AnalysisImplementationError(
            "analysis implementation files changed"
        )
    base = manifest.get("reviewed_base_commit")
    if not isinstance(base, str) or len(base) != 40:
        raise AnalysisImplementationError("reviewed commit is invalid")
    for relative, entry in files.items():
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"bytes", "sha256"}
            or type(entry.get("bytes")) is not int
            or entry["bytes"] < 0
            or not isinstance(entry.get("sha256"), str)
            or len(entry["sha256"]) != 64
        ):
            raise AnalysisImplementationError("analysis file entry is invalid")
        path = project_root / relative
        blob = _git_blob(project_root, base, relative)
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or _sha256(path) != entry["sha256"]
            or len(blob) != entry["bytes"]
            or sha256(blob).hexdigest() != entry["sha256"]
        ):
            raise AnalysisImplementationError(
                f"analysis implementation drift: {relative}"
            )
    return {
        "status": "pass",
        "reviewed_base_commit": base,
        "critical_file_count": len(files),
    }
