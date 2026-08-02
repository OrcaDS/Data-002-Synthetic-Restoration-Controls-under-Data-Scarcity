"""Hash and split-identity preflight for the Data 002 compatibility replay."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data002.reconstruction import (  # noqa: E402
    compatibility_conditions,
    file_sha256,
    index_sha256,
    load_datasets,
    nested_scarcity_indices,
    reference_archive_path,
    SOURCE_COMMIT,
    split_dataset,
)
from data002.reference_bundle import verify_reference_bundle  # noqa: E402

DEFAULT_BUNDLE = PROJECT_ROOT / "evidence" / "data001_baseline_replay_v1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify raw hashes and selected split identities without fitting models "
            "or computing predictive metrics."
        )
    )
    parser.add_argument("--diabetes-path", type=Path, required=True)
    parser.add_argument("--cleveland-path", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path; written atomically outside the evidence bundle.",
    )
    args = parser.parse_args()

    bundle_report = verify_reference_bundle(args.bundle)
    report: dict[str, object] = {
        "schema_version": 1,
        "record": "data002_compatibility_input_preflight_v1",
        "status": "fail",
        "source_commit": SOURCE_COMMIT,
        "model_fits_performed": 0,
        "predictive_metrics_computed": False,
        "probability_values_used_for_split_identity": False,
        "probability_values_structurally_validated_by_bundle_verifier": True,
        "bundle_status": bundle_report["status"],
        "bundle_manifest_sha256": bundle_report["manifest_sha256"],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "datasets": {},
        "conditions": [],
    }
    if bundle_report["status"] != "pass":
        _emit_report(report, args.output)
        return 1

    datasets = load_datasets(args.diabetes_path, args.cleveland_path)
    report["datasets"] = {
        name: {
            "sha256": file_sha256(spec.source_path),
            "rows": len(spec.frame),
            "columns_including_target": len(spec.frame.columns),
        }
        for name, spec in datasets.items()
    }

    split_cache = {}
    subset_cache = {}
    condition_reports = report["conditions"]
    for condition in compatibility_conditions():
        cache_key = (condition.dataset, condition.seed)
        if cache_key not in split_cache:
            split_cache[cache_key] = split_dataset(
                datasets[condition.dataset],
                condition.seed,
            )
            subset_cache[cache_key] = nested_scarcity_indices(
                split_cache[cache_key].y_train,
                datasets[condition.dataset].scarcity_levels,
                condition.seed,
            )
        split = split_cache[cache_key]
        retained_index = subset_cache[cache_key][condition.scarcity_fraction]
        retained_target = datasets[condition.dataset].frame.loc[
            retained_index,
            datasets[condition.dataset].target,
        ]
        path = reference_archive_path(args.bundle, condition)
        with np.load(path, allow_pickle=False) as archive:
            test_index_exact = np.array_equal(
                archive["test_index"],
                split.X_test.index.to_numpy(),
            )
            y_true_exact = np.array_equal(
                archive["y_true"],
                split.y_test.to_numpy(dtype=np.int8),
            )
        condition_reports.append(
            {
                "dataset": condition.dataset,
                "scarcity_fraction": condition.scarcity_fraction,
                "seed": condition.seed,
                "model": condition.model,
                "retained_size": len(retained_index),
                "retained_index_sha256": index_sha256(retained_index),
                "retained_target_counts": {
                    str(int(label)): int(count)
                    for label, count in retained_target.value_counts().sort_index().items()
                },
                "test_index_exact": bool(test_index_exact),
                "y_true_exact": bool(y_true_exact),
                "status": "pass" if test_index_exact and y_true_exact else "fail",
            }
        )

    report["status"] = (
        "pass"
        if condition_reports
        and all(condition["status"] == "pass" for condition in condition_reports)
        else "fail"
    )
    _emit_report(report, args.output)
    return 0 if report["status"] == "pass" else 1


def _emit_report(report: dict[str, object], output: Path | None) -> None:
    serialized = json.dumps(report, indent=2) + "\n"
    if output is not None:
        destination = output.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(destination)
    print(serialized, end="")


if __name__ == "__main__":
    raise SystemExit(main())
