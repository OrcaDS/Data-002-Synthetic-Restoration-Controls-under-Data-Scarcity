"""Generate commit-bound prospective replication manifests."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data002.replication_design import (
    build_key_manifest,
    build_upstream_evidence_manifest,
    file_sha256,
    validate_key_manifest,
    validate_upstream_evidence_manifest,
)

OUTPUTS = {
    "full": Path("results/provenance/replication_full_grid_keys_v1.json"),
    "pilot": Path("results/provenance/replication_pilot_keys_v1.json"),
    "upstream": Path("results/provenance/data001_metric_bundle_manifest_v1.json"),
}
CRITICAL_PATHS = (
    "SCOPE.md",
    "docs/protocol.md",
    "pyproject.toml",
    "requirements-lock.txt",
    "config/replication_study_v1.json",
    "config/replication_pilot_v1.json",
    "config/replication_analysis_v1.json",
    "config/compatibility_replay_v1.json",
    "results/provenance/environment_inventory_python313_v1.json",
    "results/provenance/compatibility_implementation_manifest_v1.json",
    "src/data002/reconstruction.py",
    "src/data002/replication_design.py",
    "scripts/import_data001_metric_bundle.py",
    "scripts/write_replication_design_manifests.py",
    "scripts/verify_replication_design.py",
    "tests/test_replication_design.py",
    "evidence/data001_metric_bundle_v1/baseline_per_seed.csv",
    "evidence/data001_metric_bundle_v1/baseline_protocol_v1.json",
    "evidence/data001_metric_bundle_v1/baseline_reconciliation.json",
    "evidence/data001_metric_bundle_v1/gaussian_copula_per_seed.csv",
    "evidence/data001_metric_bundle_v1/gaussian_predictive_execution_v1.json",
    "evidence/data001_metric_bundle_v1/gaussian_reconciliation_v1.json",
)
IMPLEMENTATION_OUTPUT = Path(
    "results/provenance/replication_design_implementation_manifest_v1.json"
)


def git_show(project_root: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={project_root.as_posix()}",
            "show",
            f"{commit}:{relative}",
        ],
        cwd=project_root,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    base_commit = args.base_commit
    if len(base_commit) != 40:
        raise RuntimeError("base commit must be a full 40-character identity")

    full = build_key_manifest("full", base_commit)
    pilot = build_key_manifest("pilot", base_commit)
    upstream = build_upstream_evidence_manifest(root, base_commit)
    validate_key_manifest(full, scope="full")
    validate_key_manifest(pilot, scope="pilot")
    validate_upstream_evidence_manifest(root, upstream)
    write_json(root / OUTPUTS["full"], full)
    write_json(root / OUTPUTS["pilot"], pilot)
    write_json(root / OUTPUTS["upstream"], upstream)

    files: dict[str, dict[str, object]] = {}
    for relative in CRITICAL_PATHS:
        content = git_show(root, base_commit, relative)
        working = root / relative
        if not working.is_file() or working.read_bytes() != content:
            raise RuntimeError(f"working tree differs from base commit: {relative}")
        files[relative] = {
            "bytes": len(content),
            "sha256": sha256(content).hexdigest(),
        }
    implementation = {
        "schema_version": 1,
        "record": "data002_replication_design_implementation_v1",
        "status": "design_frozen_pending_review",
        "authorization": None,
        "base_commit": base_commit,
        "file_count": len(files),
        "files": files,
        "generated_manifests": {
            relative.as_posix(): {
                "bytes": (root / relative).stat().st_size,
                "sha256": file_sha256(root / relative),
            }
            for relative in OUTPUTS.values()
        },
    }
    write_json(root / IMPLEMENTATION_OUTPUT, implementation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
