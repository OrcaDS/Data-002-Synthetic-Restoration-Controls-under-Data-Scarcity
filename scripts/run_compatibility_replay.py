"""Authorization-gated CLI for the fixed 16-condition compatibility replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data002.compatibility_runner import (  # noqa: E402
    RealCompatibilityExecutor,
    run_compatibility_replay,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run or resume the fixed compatibility replay after authorization."
    )
    parser.add_argument("--diabetes-path", type=Path, required=True)
    parser.add_argument("--cleveland-path", type=Path, required=True)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "evidence" / "data001_baseline_replay_v1",
    )
    parser.add_argument(
        "--launch",
        type=Path,
        default=PROJECT_ROOT / "config" / "compatibility_launch_v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "compatibility" / "replay_v1",
    )
    args = parser.parse_args()

    executor = RealCompatibilityExecutor(
        args.diabetes_path.resolve(),
        args.cleveland_path.resolve(),
    )
    report = run_compatibility_replay(
        bundle_root=args.bundle,
        diabetes_path=args.diabetes_path,
        cleveland_path=args.cleveland_path,
        output_root=args.output_root,
        launch_path=args.launch,
        executor=executor,
        project_root=PROJECT_ROOT,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
