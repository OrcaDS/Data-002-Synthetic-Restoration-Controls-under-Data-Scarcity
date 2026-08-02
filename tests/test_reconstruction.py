from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data002.reconstruction import (
    CompatibilityCondition,
    DatasetSpec,
    allocate_class_counts,
    compare_prediction_archives,
    compatibility_conditions,
    index_sha256,
    nested_scarcity_indices,
    retained_training_frame,
    split_dataset,
)


def _toy_spec() -> DatasetSpec:
    rows = 80
    frame = pd.DataFrame(
        {
            "numeric": np.linspace(0, 10, rows),
            "nominal": np.tile([1.0, 2.0, 3.0, 1.0], rows // 4),
            "binary": np.tile([0.0, 1.0], rows // 2),
            "target": np.tile([0, 1], rows // 2).astype(np.int8),
        }
    )
    return DatasetSpec(
        name="toy",
        frame=frame,
        target="target",
        numeric_features=("numeric",),
        nominal_features=("nominal",),
        binary_features=("binary",),
        scarcity_levels=(1.0, 0.5, 0.25),
        source_path=Path("toy.csv"),
    )


def test_compatibility_matrix_is_exactly_16_conditions() -> None:
    conditions = compatibility_conditions()

    assert len(conditions) == 16
    assert len(set(conditions)) == 16
    assert {condition.dataset for condition in conditions} == {
        "diabetes",
        "cleveland",
    }
    assert {condition.seed for condition in conditions} == {0, 29}
    assert {condition.model for condition in conditions} == {
        "logistic_regression",
        "random_forest",
    }
    fractions = {
        dataset: {
            condition.scarcity_fraction
            for condition in conditions
            if condition.dataset == dataset
        }
        for dataset in ("diabetes", "cleveland")
    }
    assert fractions == {
        "diabetes": {0.01, 0.5},
        "cleveland": {0.05, 0.5},
    }


def test_machine_contract_matches_implemented_matrix_and_tolerances() -> None:
    contract = json.loads(
        Path("config/compatibility_replay_v1.json").read_text(encoding="utf-8")
    )

    implemented = compatibility_conditions()
    for dataset, dataset_contract in contract["matrix"].items():
        observed = {
            (condition.scarcity_fraction, condition.seed, condition.model)
            for condition in implemented
            if condition.dataset == dataset
        }
        expected = {
            (fraction, seed, model)
            for fraction in dataset_contract["scarcity_fractions"]
            for seed in dataset_contract["seeds"]
            for model in dataset_contract["models"]
        }
        assert observed == expected
    assert contract["acceptance"]["probability_rtol"] == 0.0
    assert contract["acceptance"]["probability_atol"] == 1e-6
    assert contract["acceptance"]["threshold"] == 0.5
    assert contract["acceptance"]["roc_auc_absolute_tolerance"] == 1e-12
    assert contract["acceptance"]["average_precision_absolute_tolerance"] == 1e-12


def test_class_allocation_uses_stable_largest_remainders() -> None:
    y = pd.Series([0] * 7 + [1] * 3)

    allocation = allocate_class_counts(y, total=5)

    assert allocation == {0: 3, 1: 2}


def test_nested_subsets_are_deterministic_and_nested() -> None:
    spec = _toy_spec()
    split = split_dataset(spec, seed=7)

    first = nested_scarcity_indices(split.y_train, spec.scarcity_levels, seed=7)
    second = nested_scarcity_indices(split.y_train, spec.scarcity_levels, seed=7)

    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert set(first[0.25]).issubset(first[0.5])
    assert set(first[0.5]).issubset(first[1.0])
    assert np.array_equal(first[0.25], np.sort(first[0.25]))
    assert index_sha256(first[0.25]) == index_sha256(second[0.25])


def test_retained_training_frame_uses_sorted_original_rows() -> None:
    spec = _toy_spec()
    split = split_dataset(spec, seed=4)

    retained = retained_training_frame(spec, split, 0.25, seed=4)

    assert retained.index.is_monotonic_increasing
    assert len(retained) == 16
    assert retained["target"].nunique() == 2


def test_archive_comparison_accepts_identical_toy_replay() -> None:
    arrays = {
        "test_index": np.array([4, 7, 9, 12]),
        "y_true": np.array([0, 1, 0, 1], dtype=np.int8),
        "probability": np.array([0.1, 0.8, 0.3, 0.7], dtype=np.float32),
    }

    report = compare_prediction_archives(arrays, {key: value.copy() for key, value in arrays.items()})

    assert report["status"] == "pass"
    assert report["maximum_absolute_probability_difference"] == 0.0
    assert all(report["checks"].values())
    assert all(value == 0.0 for value in report["absolute_metric_differences"].values())


def test_archive_comparison_rejects_probability_drift() -> None:
    reference = {
        "test_index": np.array([4, 7, 9, 12]),
        "y_true": np.array([0, 1, 0, 1], dtype=np.int8),
        "probability": np.array([0.1, 0.8, 0.3, 0.7], dtype=np.float32),
    }
    candidate = {key: value.copy() for key, value in reference.items()}
    candidate["probability"][0] += np.float32(2e-6)

    report = compare_prediction_archives(reference, candidate)

    assert report["status"] == "fail"
    assert not report["checks"]["probability_within_tolerance"]


def test_archive_comparison_reports_nonfinite_candidate_without_crashing() -> None:
    reference = {
        "test_index": np.array([4, 7, 9, 12]),
        "y_true": np.array([0, 1, 0, 1], dtype=np.int8),
        "probability": np.array([0.1, 0.8, 0.3, 0.7], dtype=np.float32),
    }
    candidate = {key: value.copy() for key, value in reference.items()}
    candidate["probability"][1] = np.nan

    report = compare_prediction_archives(reference, candidate)

    assert report["status"] == "fail"
    assert not report["checks"]["probability_finite_and_bounded"]
    assert not report["checks"]["metric_roc_auc_within_tolerance"]
    assert report["maximum_absolute_probability_difference"] is None


def test_condition_order_is_stable() -> None:
    condition = CompatibilityCondition("diabetes", 0.01, 0, "logistic_regression")

    assert condition in compatibility_conditions()
