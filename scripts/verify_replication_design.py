"""Read-only verification of the prospective replication design package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data002.replication_design import (  # noqa: E402
    UPSTREAM_BUNDLE_ROOT,
    UPSTREAM_COMMIT,
    UPSTREAM_FILES,
    file_sha256,
    validate_design_contracts,
    validate_key_manifest,
    validate_replication_implementation_manifest,
    validate_upstream_evidence_manifest,
)


def load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def upstream_blob(repository: Path, source_path: str) -> bytes:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.as_posix()}",
            "-C",
            str(repository),
            "show",
            f"{UPSTREAM_COMMIT}:{source_path}",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--upstream-repository", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()

    contracts = [
        load_object(root / "config" / name)
        for name in (
            "replication_study_v1.json",
            "replication_pilot_v1.json",
            "replication_analysis_v1.json",
        )
    ]
    contract_report = validate_design_contracts(*contracts)
    full = load_object(
        root / "results/provenance/replication_full_grid_keys_v1.json"
    )
    pilot = load_object(
        root / "results/provenance/replication_pilot_keys_v1.json"
    )
    upstream_manifest_path = (
        root / "results/provenance/data001_metric_bundle_manifest_v1.json"
    )
    upstream_manifest = load_object(upstream_manifest_path)
    implementation_manifest_path = (
        root
        / "results/provenance/replication_design_implementation_manifest_v1.json"
    )
    implementation = load_object(implementation_manifest_path)

    full_report = validate_key_manifest(full, scope="full")
    pilot_report = validate_key_manifest(pilot, scope="pilot")
    upstream_report = validate_upstream_evidence_manifest(root, upstream_manifest)
    implementation_report = validate_replication_implementation_manifest(
        root, implementation
    )

    source_comparison = None
    if args.upstream_repository is not None:
        repository = args.upstream_repository.resolve()
        files = []
        for filename, specification in UPSTREAM_FILES.items():
            local = (root / UPSTREAM_BUNDLE_ROOT / filename).read_bytes()
            source = upstream_blob(repository, specification["source_path"])
            files.append(
                {
                    "file": filename,
                    "bytes": len(local),
                    "sha256": sha256(local).hexdigest(),
                    "exact_upstream_blob_bytes": local == source,
                }
            )
        if not all(item["exact_upstream_blob_bytes"] for item in files):
            raise RuntimeError("one or more imported files differ from upstream Git")
        source_comparison = {
            "file_count": len(files),
            "total_bytes": sum(item["bytes"] for item in files),
            "all_exact": True,
            "files": files,
        }

    print(
        json.dumps(
            {
                "status": "pass",
                "contracts": contract_report,
                "full_grid": full_report,
                "pilot": pilot_report,
                "upstream_bundle": upstream_report,
                "implementation": implementation_report,
                "manifest_sha256": {
                    "full_grid": file_sha256(
                        root
                        / "results/provenance/replication_full_grid_keys_v1.json"
                    ),
                    "pilot": file_sha256(
                        root / "results/provenance/replication_pilot_keys_v1.json"
                    ),
                    "upstream_bundle": file_sha256(upstream_manifest_path),
                    "implementation": file_sha256(implementation_manifest_path),
                },
                "upstream_git_comparison": source_comparison,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
