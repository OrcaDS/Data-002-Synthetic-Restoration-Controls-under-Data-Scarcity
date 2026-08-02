"""Generate the pilot implementation manifest from an exact source commit."""

from __future__ import annotations

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(
    "results/provenance/replication_pilot_implementation_manifest_v1.json"
)
CRITICAL_PATHS = (
    "requirements-lock.txt",
    "config/replication_study_v1.json",
    "config/replication_pilot_v1.json",
    "config/replication_analysis_v1.json",
    "results/provenance/environment_inventory_python313_v1.json",
    "results/provenance/replication_design_review_v1.json",
    "results/provenance/replication_full_grid_keys_v1.json",
    "results/provenance/replication_pilot_keys_v1.json",
    "results/provenance/data001_metric_bundle_manifest_v1.json",
    "results/provenance/replication_design_implementation_manifest_v1.json",
    "src/data002/reconstruction.py",
    "src/data002/replication_design.py",
    "src/data002/replication_constructor.py",
    "src/data002/pilot_provenance.py",
    "src/data002/pilot_runner.py",
    "src/data002/replay_provenance.py",
    "scripts/run_replication_pilot.py",
    "scripts/write_pilot_implementation_manifest.py",
    "tests/test_replication_constructor.py",
    "tests/test_pilot_runner.py",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-commit", required=True)
    args = parser.parse_args()
    base = args.base_commit
    if len(base) != 40:
        raise RuntimeError("base commit must be full length")
    files = {}
    for relative in CRITICAL_PATHS:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={PROJECT_ROOT.as_posix()}",
                "show",
                f"{base}:{relative}",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"missing from base commit: {relative}")
        if (PROJECT_ROOT / relative).read_bytes() != result.stdout:
            raise RuntimeError(f"working tree drift: {relative}")
        files[relative] = {
            "bytes": len(result.stdout),
            "sha256": sha256(result.stdout).hexdigest(),
        }
    value = {
        "schema_version": 1,
        "record": "data002_replication_pilot_implementation_v1",
        "status": "implementation_frozen_pending_authorization",
        "authorization": None,
        "base_commit": base,
        "file_count": len(files),
        "files": files,
    }
    path = PROJECT_ROOT / OUTPUT
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
