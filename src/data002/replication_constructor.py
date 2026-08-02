"""Deterministic, metric-free construction of the replication control."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from data002.reconstruction import allocate_class_counts
from data002.replication_design import (
    ORDER_NAMESPACE,
    canonical_json_bytes,
    fraction_token,
    validate_replication_order_payload,
)


def _integer_source_indices(frame: pd.DataFrame) -> list[int]:
    values = frame.index.to_numpy()
    if (
        values.ndim != 1
        or not np.issubdtype(values.dtype, np.integer)
        or any(isinstance(value, (bool, np.bool_)) for value in values)
    ):
        raise ValueError("source indices must be integers")
    indices = [int(value) for value in values]
    if len(indices) != len(set(indices)):
        raise ValueError("source indices must be unique")
    if any(value < 0 for value in indices):
        raise ValueError("source indices must be nonnegative")
    return indices


def _sequence_sha256(values: list[int]) -> str:
    return sha256((",".join(map(str, values))).encode("ascii")).hexdigest()


def construct_replication_control(
    retained: pd.DataFrame,
    *,
    target_column: str,
    final_size: int,
    dataset: str,
    retained_fraction: float,
    split_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Restore a retained table without generating information or metrics."""

    if target_column not in retained:
        raise ValueError("target column is missing")
    if retained.empty:
        raise ValueError("retained table must not be empty")
    indices = _integer_source_indices(retained)
    labels = retained[target_column]
    if (
        any(isinstance(value, (bool, np.bool_)) for value in labels)
        or not np.issubdtype(labels.dtype, np.integer)
        or {int(value) for value in labels.unique()} != {0, 1}
    ):
        raise ValueError("target labels must be exactly integer {0, 1}")
    if isinstance(final_size, bool) or not isinstance(final_size, int):
        raise ValueError("final size must be an integer")
    if final_size < len(retained):
        raise ValueError("final size cannot be smaller than retained size")

    additional = final_size - len(retained)
    duplicate_counts = allocate_class_counts(
        labels.astype(int), additional, minimum_per_class=0
    )
    token = fraction_token(retained_fraction)
    duplicate_indices: list[int] = []
    class_records: list[dict[str, Any]] = []
    for class_label in (0, 1):
        class_indices = [
            index
            for index, label in zip(indices, labels, strict=True)
            if int(label) == class_label
        ]
        ranked: list[tuple[str, int]] = []
        for source_index in class_indices:
            payload = {
                "namespace": ORDER_NAMESPACE,
                "dataset": dataset,
                "retained_fraction_token": token,
                "split_seed": split_seed,
                "class_label": class_label,
                "original_source_row_index": source_index,
            }
            encoded = validate_replication_order_payload(payload)
            ranked.append((sha256(encoded).hexdigest(), source_index))
        ranked.sort(key=lambda item: (item[0], item[1]))
        ranked_indices = [item[1] for item in ranked]
        required = duplicate_counts[class_label]
        block = [
            ranked_indices[position % len(ranked_indices)]
            for position in range(required)
        ]
        duplicate_indices.extend(block)
        repetitions = Counter(block)
        observed = [repetitions[index] for index in ranked_indices]
        floor_count = required // len(ranked_indices)
        ceiling_count = floor_count + bool(required % len(ranked_indices))
        if any(value not in {floor_count, ceiling_count} for value in observed):
            raise AssertionError("duplicate allocation is not floor/ceiling balanced")
        class_records.append(
            {
                "class_label": class_label,
                "retained_source_count": len(ranked_indices),
                "duplicate_count": required,
                "minimum_duplicates_per_source": min(observed),
                "maximum_duplicates_per_source": max(observed),
                "ranked_source_indices_sha256": _sequence_sha256(ranked_indices),
                "duplicate_source_indices_sha256": _sequence_sha256(block),
            }
        )

    duplicates = retained.loc[duplicate_indices].copy()
    reconstructed = pd.concat([retained.copy(), duplicates], axis=0)
    reconstructed_labels = [int(value) for value in reconstructed[target_column]]
    final_counts = Counter(reconstructed_labels)
    expected_counts = {
        label: int((labels == label).sum()) + duplicate_counts[label]
        for label in (0, 1)
    }
    if len(reconstructed) != final_size:
        raise AssertionError("reconstructed table has wrong final size")
    if dict(final_counts) != expected_counts:
        raise AssertionError("reconstructed table has wrong target counts")
    if not set(reconstructed.index).issubset(set(indices)):
        raise AssertionError("reconstructed table contains a foreign source row")
    if list(reconstructed.index[: len(retained)]) != indices:
        raise AssertionError("retained originals are not first")
    duplicate_labels = reconstructed_labels[len(retained) :]
    if duplicate_labels != sorted(duplicate_labels):
        raise AssertionError("duplicate class blocks are not ordered 0 then 1")

    record = {
        "schema_version": 1,
        "record": "data002_replication_allocation_v1",
        "dataset": dataset,
        "retained_fraction_token": token,
        "split_seed": split_seed,
        "original_count": len(retained),
        "duplicate_count": additional,
        "final_count": final_size,
        "retained_target_counts": {
            str(label): int((labels == label).sum()) for label in (0, 1)
        },
        "duplicate_target_counts": {
            str(label): duplicate_counts[label] for label in (0, 1)
        },
        "final_target_counts": {
            str(label): expected_counts[label] for label in (0, 1)
        },
        "original_source_indices_sha256": _sequence_sha256(indices),
        "duplicate_source_indices_sha256": _sequence_sha256(duplicate_indices),
        "final_source_indices_sha256": _sequence_sha256(
            [int(value) for value in reconstructed.index]
        ),
        "final_target_labels_sha256": _sequence_sha256(reconstructed_labels),
        "class_blocks": class_records,
        "canonical_contract_sha256": sha256(
            canonical_json_bytes(
                {
                    "namespace": ORDER_NAMESPACE,
                    "dataset": dataset,
                    "retained_fraction_token": token,
                    "split_seed": split_seed,
                    "class_labels": [0, 1],
                }
            )
        ).hexdigest(),
    }
    return reconstructed, record
