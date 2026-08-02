"""Write the proposed metric-blind full-grid execution-evidence manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data002.full_grid_evidence import (  # noqa: E402
    EVIDENCE_MANIFEST_PATH,
    build_production_evidence_manifest,
    write_evidence_manifest_once,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / EVIDENCE_MANIFEST_PATH,
    )
    args = parser.parse_args()
    destination = args.output.resolve()
    expected = (PROJECT_ROOT / EVIDENCE_MANIFEST_PATH).resolve()
    if destination != expected:
        raise ValueError("production evidence manifest output path is fixed")
    manifest = build_production_evidence_manifest(PROJECT_ROOT)
    disposition = write_evidence_manifest_once(destination, manifest)
    print(f"{destination} ({disposition})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
