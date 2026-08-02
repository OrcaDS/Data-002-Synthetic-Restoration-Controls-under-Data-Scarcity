"""Write the prospective full-grid implementation freeze after review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data002.full_grid_implementation import (  # noqa: E402
    IMPLEMENTATION_MANIFEST_PATH,
    build_full_grid_implementation_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed-base-commit", required=True)
    args = parser.parse_args()
    manifest = build_full_grid_implementation_manifest(
        PROJECT_ROOT, args.reviewed_base_commit
    )
    destination = PROJECT_ROOT / IMPLEMENTATION_MANIFEST_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(destination)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
