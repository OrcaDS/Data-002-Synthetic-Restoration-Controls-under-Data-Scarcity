"""Write the reviewed prospective analysis implementation manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data002.analysis_implementation import (  # noqa: E402
    IMPLEMENTATION_MANIFEST_PATH,
    build_analysis_implementation_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed-base-commit", required=True)
    args = parser.parse_args()
    manifest = build_analysis_implementation_manifest(
        PROJECT_ROOT, args.reviewed_base_commit
    )
    destination = PROJECT_ROOT / IMPLEMENTATION_MANIFEST_PATH
    if destination.exists():
        raise FileExistsError("analysis implementation manifest already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(
                (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
