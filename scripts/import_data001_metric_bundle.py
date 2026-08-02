"""Copy the six approved upstream blobs without parsing their metric values."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data002.replication_design import (
    UPSTREAM_BUNDLE_ROOT,
    UPSTREAM_COMMIT,
    UPSTREAM_FILES,
    UPSTREAM_RELEASE_TAG,
)


def git(repository: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.as_posix()}",
            "-C",
            str(repository),
            *arguments,
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-repository", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    upstream = args.upstream_repository.resolve()
    project_root = args.project_root.resolve()

    tag_commit = git(upstream, "rev-list", "-n", "1", UPSTREAM_RELEASE_TAG).decode().strip()
    if tag_commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"{UPSTREAM_RELEASE_TAG} resolves to {tag_commit}, expected {UPSTREAM_COMMIT}"
        )
    destination = project_root / UPSTREAM_BUNDLE_ROOT
    destination.mkdir(parents=True, exist_ok=True)
    expected_names = set(UPSTREAM_FILES)
    unexpected = sorted(
        path.name for path in destination.iterdir() if path.name not in expected_names
    )
    if unexpected:
        raise RuntimeError(f"unexpected bundle files already exist: {unexpected}")

    for filename, specification in UPSTREAM_FILES.items():
        source_path = specification["source_path"]
        observed_blob = git(
            upstream, "rev-parse", f"{UPSTREAM_COMMIT}:{source_path}"
        ).decode().strip()
        if observed_blob != specification["git_blob"]:
            raise RuntimeError(f"upstream Git blob changed for {source_path}")
        content = git(upstream, "show", f"{UPSTREAM_COMMIT}:{source_path}")
        target = destination / filename
        if target.exists() and target.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite differing bundle file: {target}")
        target.write_bytes(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
