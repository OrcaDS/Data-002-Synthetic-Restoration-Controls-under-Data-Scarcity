"""Authorization-gated CLI for the exact metric-blind 12-condition pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data002.pilot_runner import RealPilotExecutor, run_pilot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diabetes-path", type=Path, required=True)
    parser.add_argument("--cleveland-path", type=Path, required=True)
    parser.add_argument(
        "--launch",
        type=Path,
        default=PROJECT_ROOT / "config/replication_pilot_launch_v1.json",
    )
    args = parser.parse_args()
    report = run_pilot(
        launch_path=args.launch,
        executor=RealPilotExecutor(
            args.diabetes_path.resolve(), args.cleveland_path.resolve()
        ),
        project_root=PROJECT_ROOT,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
