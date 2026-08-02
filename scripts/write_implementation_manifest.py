"""Generate the deterministic replay-critical Data 002 implementation manifest."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "provenance"
    / "compatibility_implementation_manifest_v1.json"
)
CRITICAL_FILES = (
    "config/compatibility_replay_v1.json",
    "pyproject.toml",
    "requirements-lock.txt",
    "results/provenance/environment_inventory_python313_v1.json",
    "scripts/run_compatibility_replay.py",
    "scripts/write_environment_inventory.py",
    "scripts/write_implementation_manifest.py",
    "src/data002/compatibility_runner.py",
    "src/data002/reconstruction.py",
    "src/data002/reference_bundle.py",
    "src/data002/replay_provenance.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    base_commit = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    files = {}
    for relative in CRITICAL_FILES:
        path = PROJECT_ROOT / relative
        files[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest = {
        "schema_version": 1,
        "record": "data002_compatibility_implementation_v1",
        "base_commit": base_commit,
        "file_count": len(files),
        "files": files,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(OUTPUT.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, OUTPUT)
    finally:
        temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
