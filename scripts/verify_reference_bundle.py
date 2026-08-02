"""CLI for the read-only Data 001 replay-reference verifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data002.reference_bundle import (  # noqa: E402
    FROZEN_MANIFEST_SHA256,
    verify_reference_bundle,
)

DEFAULT_BUNDLE = PROJECT_ROOT / "evidence" / "data001_baseline_replay_v1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the frozen replay-reference bundle without computing metrics."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE,
        help="Bundle directory containing manifest.json (default: project evidence bundle).",
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        default=FROZEN_MANIFEST_SHA256,
        help="External trust anchor for manifest.json.",
    )
    args = parser.parse_args()
    report = verify_reference_bundle(
        args.bundle,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
