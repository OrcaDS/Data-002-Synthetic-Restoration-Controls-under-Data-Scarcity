"""Run the separately authorized frozen Data 002 analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data002.analysis_runner import run_analysis  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--launch",
        type=Path,
        default=PROJECT_ROOT / "config/replication_analysis_launch_v1.json",
    )
    args = parser.parse_args()
    run_analysis(project_root=PROJECT_ROOT, launch_path=args.launch.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
