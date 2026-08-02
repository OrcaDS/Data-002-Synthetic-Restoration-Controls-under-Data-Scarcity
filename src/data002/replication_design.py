"""Read-only validation for the prospective Data 002 replication design.

This module defines condition identities and validates frozen design artifacts.
It does not construct replicated training data, fit models, or analyze outcomes.
"""

from __future__ import annotations

import csv
import json
import subprocess
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
DESIGN_STATUS = "design_frozen_pending_review"
MODELS = ("logistic_regression", "random_forest")
SEEDS = tuple(range(30))
FULL_FRACTIONS = {
    "diabetes": (0.01, 0.05, 0.10, 0.25, 0.50),
    "cleveland": (0.05, 0.10, 0.25, 0.50),
}
BASELINE_FRACTIONS = {
    "diabetes": (1.0, 0.50, 0.25, 0.10, 0.05, 0.01),
    "cleveland": (1.0, 0.50, 0.25, 0.10, 0.05),
}
PILOT_FRACTIONS = {
    "diabetes": (0.01, 0.50),
    "cleveland": (0.05,),
}
PILOT_SEEDS = (0, 29)
FRACTION_TOKENS = {
    Decimal("0.01"): "001pct",
    Decimal("0.05"): "005pct",
    Decimal("0.1"): "010pct",
    Decimal("0.25"): "025pct",
    Decimal("0.5"): "050pct",
    Decimal("1"): "100pct",
}
ORDER_NAMESPACE = "data002.replication-order.v1"
ANALYSIS_SEED_NAMESPACE = "data002.paired-bootstrap-seed.v1"
ORDER_PAYLOAD_KEYS = {
    "namespace",
    "dataset",
    "retained_fraction_token",
    "split_seed",
    "class_label",
    "original_source_row_index",
}
ANALYSIS_SEED_PAYLOAD_KEYS = {
    "namespace",
    "dataset",
    "retained_fraction_token",
    "model",
    "contrast",
    "resamples",
}
FULL_GRID_RECORD = "data002_replication_full_grid_keys_v1"
PILOT_GRID_RECORD = "data002_replication_pilot_keys_v1"
KEY_MANIFEST_KEYS = {
    "schema_version",
    "record",
    "status",
    "authorization",
    "base_commit",
    "condition_count",
    "condition_keys_sha256",
    "conditions",
}
CONDITION_KEYS = {
    "key",
    "dataset",
    "retained_fraction",
    "retained_fraction_token",
    "seed",
    "model",
}

UPSTREAM_RELEASE_TAG = "gaussian-v1.0"
UPSTREAM_COMMIT = "ecc2b222eca86c47acdf12efd3b8f779b6a29ef9"
UPSTREAM_BUNDLE_ROOT = Path("evidence/data001_metric_bundle_v1")
UPSTREAM_FILES = {
    "baseline_per_seed.csv": {
        "source_path": "results/baseline_per_seed.csv",
        "git_blob": "af745100804951fcc4eaa5cf022ef4fb676c1204",
        "columns": (
            "dataset",
            "model",
            "seed",
            "scarcity_fraction",
            "training_pool_size",
            "retained_size",
            "retained_negative",
            "retained_positive",
            "test_size",
            "fit_seconds",
            "roc_auc",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "average_precision",
        ),
        "coverage": "baseline_660",
    },
    "baseline_protocol_v1.json": {
        "source_path": "results/baseline_protocol_v1.json",
        "git_blob": "7f3656b8b485094a79c73f292fbc9a4565b27903",
        "top_level_keys": (
            "dataset_sha256",
            "models",
            "package_versions",
            "protocol",
            "repeat_seeds",
            "scarcity_levels",
            "test_size",
        ),
    },
    "baseline_reconciliation.json": {
        "source_path": "results/baseline_reconciliation.json",
        "git_blob": "f39b45b87529bff046f7205774a515a271f2b1d7",
        "top_level_keys": (
            "changed_metric_cells",
            "figure",
            "manifest_status",
            "maximum_absolute_metric_difference",
            "metric_rows",
            "model_fits_performed",
            "summary_rows",
            "validated_archives",
        ),
    },
    "gaussian_copula_per_seed.csv": {
        "source_path": "results/gaussian_copula_per_seed.csv",
        "git_blob": "3b086ef637d0cd72e90a056e0c0126a68c472b64",
        "columns": (
            "dataset",
            "model",
            "seed",
            "scarcity_fraction",
            "training_pool_size",
            "retained_size",
            "synthetic_size",
            "generator_seconds",
            "fit_seconds",
            "roc_auc",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "average_precision",
        ),
        "coverage": "gaussian_540",
    },
    "gaussian_predictive_execution_v1.json": {
        "source_path": "results/provenance/gaussian_predictive_execution_v1.json",
        "git_blob": "270625ca6bb3561bc587114f299178a3b4454c8a",
        "top_level_keys": (
            "artifacts",
            "conditional_allocation_example_function",
            "created_at_utc",
            "dataset_hashes",
            "environment",
            "generator",
            "historical_claim",
            "historical_file_hashes",
            "metadata",
            "record_id",
            "record_type",
            "related_records",
            "schema_version",
            "split_and_scarcity",
        ),
    },
    "gaussian_reconciliation_v1.json": {
        "source_path": "results/provenance/gaussian_reconciliation_v1.json",
        "git_blob": "20b36ff8de01288d3474adf156ea2c7b61f3e78d",
        "top_level_keys": (
            "changed_metric_cells_above_tolerance",
            "duplicate_condition_keys",
            "expected_condition_keys",
            "extra_archives",
            "historical_artifacts_modified",
            "input_hashes",
            "invalid_size_rows",
            "maximum_absolute_metric_difference",
            "metric_rows",
            "metric_tolerance",
            "missing_archives",
            "missing_condition_keys",
            "mode",
            "model_fits_performed",
            "observed_at_utc",
            "passed",
            "record_type",
            "schema_version",
            "unexpected_condition_keys",
            "unique_condition_keys",
            "validated_archives",
        ),
    },
}


class ReplicationDesignError(ValueError):
    """A prospective replication artifact differs from the frozen design."""


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ReplicationDesignError(
            f"{label} keys must be exactly {sorted(expected)}, found {sorted(value)}"
        )


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReplicationDesignError(
            f"{label} cannot be loaded: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ReplicationDesignError(f"{label} must be a JSON object")
    return value


def _fraction_decimal(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ReplicationDesignError(f"invalid retained fraction: {value!r}")
    try:
        fraction = Decimal(str(value))
    except InvalidOperation as exc:
        raise ReplicationDesignError(f"invalid retained fraction: {value!r}") from exc
    if fraction not in FRACTION_TOKENS:
        raise ReplicationDesignError(f"unsupported retained fraction: {value!r}")
    return fraction


def fraction_token(value: Any) -> str:
    return FRACTION_TOKENS[_fraction_decimal(value)]


def condition_key(
    dataset: str,
    retained_fraction: Any,
    seed: int,
    model: str,
) -> str:
    if dataset not in FULL_FRACTIONS:
        raise ReplicationDesignError(f"unsupported dataset: {dataset!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ReplicationDesignError("seed must be an integer")
    if model not in MODELS:
        raise ReplicationDesignError(f"unsupported model: {model!r}")
    return f"{dataset}__{fraction_token(retained_fraction)}__seed_{seed:02d}__{model}"


def _condition_records(
    fractions: Mapping[str, Sequence[float]],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dataset in ("diabetes", "cleveland"):
        for retained_fraction in fractions[dataset]:
            for seed in seeds:
                for model in MODELS:
                    records.append(
                        {
                            "key": condition_key(
                                dataset, retained_fraction, seed, model
                            ),
                            "dataset": dataset,
                            "retained_fraction": retained_fraction,
                            "retained_fraction_token": fraction_token(
                                retained_fraction
                            ),
                            "seed": seed,
                            "model": model,
                        }
                    )
    return records


def full_grid_records() -> list[dict[str, Any]]:
    return _condition_records(FULL_FRACTIONS, SEEDS)


def pilot_records() -> list[dict[str, Any]]:
    return _condition_records(PILOT_FRACTIONS, PILOT_SEEDS)


def condition_keys_sha256(keys: Iterable[str]) -> str:
    encoded = ("\n".join(sorted(keys)) + "\n").encode("utf-8")
    return sha256(encoded).hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReplicationDesignError(f"value is not canonical-JSON-safe: {exc}") from exc


def validate_replication_order_payload(payload: Mapping[str, Any]) -> bytes:
    """Validate and encode one ranking preimage; no ranking or allocation is run."""

    _exact_keys(payload, ORDER_PAYLOAD_KEYS, "replication-order payload")
    if payload.get("namespace") != ORDER_NAMESPACE:
        raise ReplicationDesignError("replication-order namespace changed")
    dataset = payload.get("dataset")
    if dataset not in FULL_FRACTIONS:
        raise ReplicationDesignError("replication-order dataset is invalid")
    token = payload.get("retained_fraction_token")
    allowed_tokens = {fraction_token(value) for value in FULL_FRACTIONS[dataset]}
    if token not in allowed_tokens:
        raise ReplicationDesignError(
            "replication-order retained-fraction token is invalid"
        )
    for field in ("split_seed", "class_label", "original_source_row_index"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ReplicationDesignError(
                f"replication-order {field} must be an integer"
            )
    if payload["split_seed"] not in SEEDS:
        raise ReplicationDesignError("replication-order split seed is invalid")
    if payload["class_label"] not in (0, 1):
        raise ReplicationDesignError(
            "replication-order class label must be exactly integer 0 or 1"
        )
    if payload["original_source_row_index"] < 0:
        raise ReplicationDesignError(
            "replication-order source-row index must be nonnegative"
        )
    return canonical_json_bytes(payload)


def validate_analysis_seed_payload(payload: Mapping[str, Any]) -> int:
    """Validate one bootstrap-seed preimage and derive its frozen uint64 seed."""

    _exact_keys(payload, ANALYSIS_SEED_PAYLOAD_KEYS, "analysis-seed payload")
    if payload.get("namespace") != ANALYSIS_SEED_NAMESPACE:
        raise ReplicationDesignError("analysis-seed namespace changed")
    dataset = payload.get("dataset")
    if dataset not in FULL_FRACTIONS:
        raise ReplicationDesignError("analysis-seed dataset is invalid")
    if payload.get("retained_fraction_token") not in {
        fraction_token(value) for value in FULL_FRACTIONS[dataset]
    }:
        raise ReplicationDesignError("analysis-seed retained-fraction token is invalid")
    if payload.get("model") not in MODELS:
        raise ReplicationDesignError("analysis-seed model is invalid")
    if payload.get("contrast") != "gaussian_minus_replication":
        raise ReplicationDesignError("analysis-seed contrast changed")
    if payload.get("resamples") != 10_000:
        raise ReplicationDesignError("analysis-seed resample count changed")
    digest = sha256(canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _validate_condition_record(record: Any, label: str) -> str:
    if not isinstance(record, Mapping):
        raise ReplicationDesignError(f"{label} must be an object")
    _exact_keys(record, CONDITION_KEYS, label)
    expected_key = condition_key(
        record.get("dataset"),
        record.get("retained_fraction"),
        record.get("seed"),
        record.get("model"),
    )
    if record.get("retained_fraction_token") != fraction_token(
        record.get("retained_fraction")
    ):
        raise ReplicationDesignError(f"{label} fraction token mismatch")
    if record.get("key") != expected_key:
        raise ReplicationDesignError(f"{label} key mismatch")
    return expected_key


def validate_key_manifest(
    manifest: Mapping[str, Any],
    *,
    scope: str,
) -> dict[str, Any]:
    _exact_keys(manifest, KEY_MANIFEST_KEYS, "condition manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ReplicationDesignError("condition manifest schema changed")
    if scope not in {"full", "pilot"}:
        raise ReplicationDesignError(f"unknown manifest scope: {scope}")
    expected_records = full_grid_records() if scope == "full" else pilot_records()
    expected_record_name = (
        FULL_GRID_RECORD if scope == "full" else PILOT_GRID_RECORD
    )
    if manifest.get("record") != expected_record_name:
        raise ReplicationDesignError("condition manifest identity changed")
    if manifest.get("status") != DESIGN_STATUS or manifest.get("authorization") is not None:
        raise ReplicationDesignError("condition manifest must remain unauthorized")
    base_commit = manifest.get("base_commit")
    if (
        not isinstance(base_commit, str)
        or len(base_commit) != 40
        or any(character not in "0123456789abcdef" for character in base_commit)
    ):
        raise ReplicationDesignError("condition manifest base commit is invalid")
    conditions = manifest.get("conditions")
    if not isinstance(conditions, list):
        raise ReplicationDesignError("condition manifest conditions must be a list")
    observed_keys = [
        _validate_condition_record(record, f"condition {index}")
        for index, record in enumerate(conditions)
    ]
    duplicate_keys = sorted(
        key for key in set(observed_keys) if observed_keys.count(key) > 1
    )
    expected_keys = {record["key"] for record in expected_records}
    observed_set = set(observed_keys)
    missing_keys = sorted(expected_keys - observed_set)
    unexpected_keys = sorted(observed_set - expected_keys)
    if duplicate_keys or missing_keys or unexpected_keys:
        raise ReplicationDesignError(
            "condition key reconciliation failed: "
            f"duplicate={duplicate_keys}, missing={missing_keys}, "
            f"unexpected={unexpected_keys}"
        )
    if conditions != expected_records:
        raise ReplicationDesignError("condition manifest ordering or fields changed")
    if manifest.get("condition_count") != len(expected_records):
        raise ReplicationDesignError("condition manifest count changed")
    expected_digest = condition_keys_sha256(expected_keys)
    if manifest.get("condition_keys_sha256") != expected_digest:
        raise ReplicationDesignError("condition-key digest changed")
    return {
        "status": "pass",
        "scope": scope,
        "condition_count": len(expected_records),
        "condition_keys_sha256": expected_digest,
    }


def build_key_manifest(scope: str, base_commit: str) -> dict[str, Any]:
    records = full_grid_records() if scope == "full" else pilot_records()
    record_name = FULL_GRID_RECORD if scope == "full" else PILOT_GRID_RECORD
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "record": record_name,
        "status": DESIGN_STATUS,
        "authorization": None,
        "base_commit": base_commit,
        "condition_count": len(records),
        "condition_keys_sha256": condition_keys_sha256(
            record["key"] for record in records
        ),
        "conditions": records,
    }
    validate_key_manifest(manifest, scope=scope)
    return manifest


def validate_design_contracts(
    study: Mapping[str, Any],
    pilot: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    for label, contract, record in (
        ("study", study, "data002_replication_study_contract_v1"),
        ("pilot", pilot, "data002_replication_pilot_contract_v1"),
        ("analysis", analysis, "data002_replication_analysis_contract_v1"),
    ):
        if contract.get("schema_version") != 1 or contract.get("record") != record:
            raise ReplicationDesignError(f"{label} contract identity changed")
        if contract.get("status") != DESIGN_STATUS or contract.get("authorization") is not None:
            raise ReplicationDesignError(f"{label} contract must remain unauthorized")

    expected_study_keys = {
        "schema_version",
        "record",
        "status",
        "authorization",
        "scope",
        "source",
        "grid",
        "restoration",
        "accepted_reconstruction",
        "condition_manifest",
        "metric_blinding",
        "resources",
        "execution",
    }
    expected_pilot_keys = {
        "schema_version",
        "record",
        "status",
        "authorization",
        "scope",
        "study_contract",
        "matrix",
        "condition_manifest",
        "inspection_boundary",
        "sealed_outcome_evidence",
        "artifact_reuse",
        "execution_authorization_required",
    }
    expected_analysis_keys = {
        "schema_version",
        "record",
        "status",
        "authorization",
        "scope",
        "study_contract",
        "upstream_evidence_manifest",
        "primary_contrast",
        "interval",
        "practical_threshold",
        "multiplicity",
        "analysis_gate",
        "analysis_stage",
        "execution_authorization_required",
    }
    _exact_keys(study, expected_study_keys, "study contract")
    _exact_keys(pilot, expected_pilot_keys, "pilot contract")
    _exact_keys(analysis, expected_analysis_keys, "analysis contract")
    if (
        study.get("scope") != "fixed_540_condition_replication_control_grid"
        or study.get("condition_manifest")
        != "results/provenance/replication_full_grid_keys_v1.json"
    ):
        raise ReplicationDesignError("study scope or manifest path changed")
    if (
        pilot.get("scope") != "fixed_12_condition_metric_blind_replication_pilot"
        or pilot.get("study_contract") != "config/replication_study_v1.json"
        or pilot.get("condition_manifest")
        != "results/provenance/replication_pilot_keys_v1.json"
    ):
        raise ReplicationDesignError("pilot scope or contract path changed")
    if (
        analysis.get("scope")
        != "prespecified_18_stratum_replication_treatment_analysis"
        or analysis.get("study_contract") != "config/replication_study_v1.json"
        or analysis.get("upstream_evidence_manifest")
        != "results/provenance/data001_metric_bundle_manifest_v1.json"
    ):
        raise ReplicationDesignError("analysis scope or contract path changed")

    grid = study.get("grid")
    if grid != {
        "diabetes": {"retained_fractions": list(FULL_FRACTIONS["diabetes"])},
        "cleveland": {"retained_fractions": list(FULL_FRACTIONS["cleveland"])},
        "seeds": {"first": 0, "last": 29, "inclusive": True},
        "models": list(MODELS),
        "condition_count": 540,
    }:
        raise ReplicationDesignError("study grid changed")
    pilot_matrix = pilot.get("matrix")
    if pilot_matrix != {
        "diabetes": {"retained_fractions": list(PILOT_FRACTIONS["diabetes"])},
        "cleveland": {"retained_fractions": list(PILOT_FRACTIONS["cleveland"])},
        "seeds": list(PILOT_SEEDS),
        "models": list(MODELS),
        "condition_count": 12,
    }:
        raise ReplicationDesignError("pilot grid changed")
    source = study.get("source")
    if not isinstance(source, Mapping) or source != {
        "release_tag": UPSTREAM_RELEASE_TAG,
        "commit": UPSTREAM_COMMIT,
        "scarcity_source_path": "src/data/scarcity.py",
        "scarcity_source_blob": "7554d7aa2be9ad6892362f5c926d2b7244a722a9",
        "allocate_class_counts_minimum_per_class": 0,
    }:
        raise ReplicationDesignError("restoration-count source changed")
    restoration = study.get("restoration")
    if restoration != {
        "target_total": "original training-pool row count",
        "required_additional_rows": "target total minus retained row count",
        "class_allocation": (
            "Data 001 allocate_class_counts(retained_target, "
            "required_additional_rows, minimum_per_class=0)"
        ),
        "order_namespace": ORDER_NAMESPACE,
        "order_payload_fields": [
            "namespace",
            "dataset",
            "retained_fraction_token",
            "split_seed",
            "class_label",
            "original_source_row_index",
        ],
        "order_payload_encoding": {
            "format": "canonical_utf8_json",
            "sort_keys": True,
            "separators": [",", ":"],
            "ensure_ascii": False,
            "allow_nan": False,
        },
        "ranking": (
            "ascending_sha256_digest_then_ascending_original_source_row_index"
        ),
        "class_labels": [0, 1],
        "duplicate_class_block_order": "ascending_integer_class_label",
        "allocation": (
            "cycle through ranked retained rows independently within each class"
        ),
        "balance_invariant": (
            "each retained source row receives floor or ceiling of its class "
            "duplicate count"
        ),
        "row_order": (
            "all retained originals first in accepted reconstruction order; "
            "duplicate class block 0 follows, then duplicate class block 1; "
            "each class block preserves its SHA-256-ranked cycle order"
        ),
    }:
        raise ReplicationDesignError("replication ordering or allocation changed")
    if study.get("accepted_reconstruction") != {
        "module": "src/data002/reconstruction.py",
        "module_sha256": (
            "0ca9c701c41d41c9adc418c91fc452a5052230b7e482e35cd02331b21caf1b60"
        ),
        "compatibility_report_sha256": (
            "9741f7e13b7faec0b27c4ad2404ecfb890c756c605a1f53f1259ca3d124eb19b"
        ),
        "compatibility_implementation_manifest": (
            "results/provenance/compatibility_implementation_manifest_v1.json"
        ),
        "compatibility_implementation_manifest_sha256": (
            "6273e56d2b5b34c0349946f40c0a1fd72e4d42bcb5cb03a9b71cfc44a46691fa"
        ),
        "environment_inventory": (
            "results/provenance/environment_inventory_python313_v1.json"
        ),
        "environment_inventory_sha256": (
            "d6e8c4444d5d1ffe63f5866b2da7405e67cdbd7c586499ec5804526b38c78589"
        ),
        "requirements_lock": "requirements-lock.txt",
        "requirements_lock_sha256": (
            "a4b12e0682dbc8308ed3619922f560b966349416d3e95ccee171940fc4d0ee15"
        ),
        "must_reuse_without_modification": [
            "model",
            "preprocessing",
            "split",
            "scarcity_subset",
            "prediction",
            "metric",
        ],
    }:
        raise ReplicationDesignError("accepted reconstruction binding changed")
    if study.get("metric_blinding") != {
        "applies_to": ["replication_execution", "metric_blind_pilot"],
        "predictive_metrics": {
            "compute": False,
            "store": False,
            "print": False,
            "summarize": False,
            "join": False,
        },
        "prediction_artifact": {
            "format": "npz",
            "allowed_arrays": ["test_index", "y_true", "probability"],
        },
        "operational_validation_may_return": [
            "schema_booleans",
            "range_booleans",
            "finiteness_booleans",
            "hashes",
            "counts",
            "warnings",
            "runtime",
            "memory",
            "disk",
            "terminal_status",
        ],
        "operational_validation_must_not_return": [
            "probability_values",
            "predictive_metric_values",
        ],
        "upstream_metric_tables_may_be_loaded_or_joined_by_pilot": False,
        "replication_metrics_stage": (
            "separately_authorized_analysis_after_exact_540_key_"
            "checkpoint_prediction_reconciliation"
        ),
    }:
        raise ReplicationDesignError("metric-blinding boundary changed")
    resources = study.get("resources")
    if resources != {
        "max_parallel_conditions": 1,
        "condition_timeout_seconds": 900,
        "minimum_free_disk_bytes": 2_147_483_648,
    }:
        raise ReplicationDesignError("proposed resource envelope changed")
    if study.get("execution") != {
        "atomic_condition_artifacts": True,
        "verified_resume": True,
        "exclusive_run_lock": True,
        "all_expected_conditions_must_succeed": True,
        "design_values_are_authorization": False,
    }:
        raise ReplicationDesignError("study execution policy changed")
    if {record["key"] for record in pilot_records()} - {
        record["key"] for record in full_grid_records()
    }:
        raise ReplicationDesignError("pilot conditions are not a full-grid subset")
    if pilot.get("inspection_boundary") != {
        "allowed": [
            "runtime",
            "memory",
            "disk",
            "warnings",
            "row_counts",
            "target_counts",
            "deterministic_reconstruction_hashes",
            "prediction_npz_schema_range_finiteness_booleans",
            "prediction_artifact_hashes",
            "expected_key_identity",
            "atomic_persistence",
            "resume_behavior",
            "failure_accounting",
        ],
        "forbidden": [
            "treatment_favorability",
            "aggregate_predictive_metrics",
            "primary_contrast",
            "compute_predictive_metrics",
            "store_predictive_metrics",
            "print_predictive_metrics",
            "summarize_predictive_metrics",
            "join_predictive_metrics",
            "return_probability_values",
            "load_or_join_upstream_metric_tables",
        ],
        "upstream_metric_tables_access": "forbidden",
    }:
        raise ReplicationDesignError("pilot inspection boundary changed")
    if pilot.get("sealed_outcome_evidence") != {
        "prediction_artifacts_are_outcome_evidence": True,
        "retention": "retain sealed for possible final-grid reuse",
        "operational_review_interpretation": "forbidden",
    }:
        raise ReplicationDesignError("pilot sealed-evidence policy changed")
    if pilot.get("artifact_reuse") != {
        "eligible_for_final_grid": True,
        "required_byte_identical_bindings": [
            "protocol",
            "implementation",
            "environment",
            "condition_contracts",
        ],
        "binding_sources": {
            "protocol": [
                "config/replication_study_v1.json",
                "docs/protocol.md",
                "SCOPE.md",
            ],
            "implementation": (
                "results/provenance/"
                "replication_design_implementation_manifest_v1.json"
            ),
            "environment": (
                "results/provenance/environment_inventory_python313_v1.json"
            ),
            "condition_contracts": (
                "results/provenance/replication_full_grid_keys_v1.json"
            ),
        },
        "otherwise": (
            "pilot artifacts are ineligible and must not be silently imported"
        ),
    }:
        raise ReplicationDesignError("pilot artifact-reuse policy changed")

    primary = analysis.get("primary_contrast")
    interval = analysis.get("interval")
    threshold = analysis.get("practical_threshold")
    multiplicity = analysis.get("multiplicity")
    gate = analysis.get("analysis_gate")
    analysis_stage = analysis.get("analysis_stage")
    if primary != {
        "name": "gaussian_minus_replication",
        "metric": "roc_auc",
        "pairing_unit": "seed",
        "strata": ["dataset", "retained_fraction", "model"],
        "stratum_count": 18,
        "seeds_per_stratum": 30,
    }:
        raise ReplicationDesignError("primary contrast changed")
    if interval != {
        "method": "paired_seed_level_bootstrap",
        "resamples": 10_000,
        "confidence_level": 0.95,
        "endpoints": [0.025, 0.975],
        "quantile_method": "linear",
        "resample_size": 30,
        "sampling": "paired seed differences with replacement",
        "bit_generator": "numpy.random.PCG64",
        "analysis_seed": {
            "namespace": ANALYSIS_SEED_NAMESPACE,
            "payload_fields": [
                "namespace",
                "dataset",
                "retained_fraction_token",
                "model",
                "contrast",
                "resamples",
            ],
            "payload_encoding": {
                "format": "canonical_utf8_json",
                "sort_keys": True,
                "separators": [",", ":"],
                "ensure_ascii": False,
                "allow_nan": False,
            },
            "derivation": (
                "first 8 SHA-256 digest bytes interpreted as unsigned "
                "big-endian uint64"
            ),
        },
    }:
        raise ReplicationDesignError("primary interval changed")
    if not isinstance(threshold, Mapping) or threshold != {
        "absolute_roc_auc_difference": 0.01,
        "helpful": "entire 95% interval exceeds +0.01",
        "harmful": "entire 95% interval is below -0.01",
        "otherwise": "no clear practically meaningful difference",
    }:
        raise ReplicationDesignError("practical threshold or classification changed")
    if multiplicity != {
        "p_values": False,
        "adjustment": "none",
        "all_primary_strata_prespecified": True,
        "aggregate_counts": "descriptive",
        "heterogeneity_views": "descriptive",
    }:
        raise ReplicationDesignError("multiplicity policy changed")
    if gate != {
        "expected_replication_keys": 540,
        "all_replication_keys_must_be_successful": True,
        "checkpoint_prediction_reconciliation": "exact",
        "replication_metric_outputs_before_gate": "forbidden",
        "silent_dropping": False,
        "complete_case_analysis": False,
        "blocked_on_any_missing_failed_timeout_unexpected_or_mismatched_item": True,
    }:
        raise ReplicationDesignError("analysis gate changed")
    if analysis_stage != {
        "replication_metric_computation": (
            "only after the pre-analysis gate passes under separate analysis "
            "authorization"
        ),
        "metric_output_reconciliation": "exact",
        "upstream_metric_table_join": (
            "only during separately authorized analysis"
        ),
    }:
        raise ReplicationDesignError("analysis-stage boundary changed")
    if pilot.get("execution_authorization_required") is not True or analysis.get(
        "execution_authorization_required"
    ) is not True:
        raise ReplicationDesignError("pilot and analysis require separate authorization")
    return {
        "status": "pass",
        "full_grid_conditions": len(full_grid_records()),
        "pilot_conditions": len(pilot_records()),
        "primary_strata": 18,
        "authorization": None,
    }


def _metric_table_key_report(
    path: Path,
    columns: Sequence[str],
    coverage: str,
) -> dict[str, Any]:
    expected_records = (
        _condition_records(BASELINE_FRACTIONS, SEEDS)
        if coverage == "baseline_660"
        else full_grid_records()
    )
    expected_keys = {record["key"] for record in expected_records}
    observed_keys: list[str] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != tuple(columns):
                raise ReplicationDesignError(f"CSV schema changed: {path.name}")
            for row in reader:
                observed_keys.append(
                    condition_key(
                        row["dataset"],
                        row["scarcity_fraction"],
                        int(row["seed"]),
                        row["model"],
                    )
                )
    except (OSError, UnicodeError, csv.Error, KeyError, ValueError) as exc:
        if isinstance(exc, ReplicationDesignError):
            raise
        raise ReplicationDesignError(
            f"metric table key validation failed for {path.name}: {exc}"
        ) from exc
    duplicates = sorted(
        key for key in set(observed_keys) if observed_keys.count(key) > 1
    )
    observed = set(observed_keys)
    missing = sorted(expected_keys - observed)
    unexpected = sorted(observed - expected_keys)
    if duplicates or missing or unexpected or len(observed_keys) != len(expected_keys):
        raise ReplicationDesignError(
            f"metric table key coverage failed for {path.name}: "
            f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}"
        )
    return {
        "scope": coverage,
        "condition_key_count": len(observed_keys),
        "condition_keys_sha256": condition_keys_sha256(observed_keys),
    }


def build_upstream_evidence_manifest(
    project_root: Path,
    base_commit: str,
) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    total_bytes = 0
    for filename, specification in UPSTREAM_FILES.items():
        path = project_root / UPSTREAM_BUNDLE_ROOT / filename
        if not path.is_file():
            raise ReplicationDesignError(f"upstream bundle file is missing: {filename}")
        size = path.stat().st_size
        total_bytes += size
        if "columns" in specification:
            expected_schema = {
                "format": "csv",
                "columns": list(specification["columns"]),
            }
            coverage = _metric_table_key_report(
                path,
                specification["columns"],
                specification["coverage"],
            )
        else:
            value = _load_object(path, filename)
            expected_keys = list(specification["top_level_keys"])
            if sorted(value) != expected_keys:
                raise ReplicationDesignError(
                    f"JSON top-level schema changed: {filename}"
                )
            expected_schema = {
                "format": "json",
                "top_level_keys": expected_keys,
            }
            coverage = None
        entries[filename] = {
            "upstream_git_path": specification["source_path"],
            "upstream_git_blob": specification["git_blob"],
            "bytes": size,
            "sha256": file_sha256(path),
            "expected_schema": expected_schema,
            "exact_key_coverage": coverage,
        }
    return {
        "schema_version": 1,
        "record": "data002_data001_metric_bundle_manifest_v1",
        "status": DESIGN_STATUS,
        "authorization": None,
        "source_release_tag": UPSTREAM_RELEASE_TAG,
        "source_commit": UPSTREAM_COMMIT,
        "base_commit": base_commit,
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
    }


def validate_upstream_evidence_manifest(
    project_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_upstream_evidence_manifest(
        project_root, str(manifest.get("base_commit"))
    )
    _exact_keys(
        manifest,
        {
            "schema_version",
            "record",
            "status",
            "authorization",
            "source_release_tag",
            "source_commit",
            "base_commit",
            "file_count",
            "total_bytes",
            "files",
        },
        "upstream evidence manifest",
    )
    if manifest != expected:
        raise ReplicationDesignError("upstream evidence manifest drift")
    return {
        "status": "pass",
        "file_count": expected["file_count"],
        "total_bytes": expected["total_bytes"],
        "baseline_key_count": expected["files"]["baseline_per_seed.csv"][
            "exact_key_coverage"
        ]["condition_key_count"],
        "gaussian_key_count": expected["files"]["gaussian_copula_per_seed.csv"][
            "exact_key_coverage"
        ]["condition_key_count"],
    }


def validate_replication_implementation_manifest(
    project_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_keys(
        manifest,
        {
            "schema_version",
            "record",
            "status",
            "authorization",
            "base_commit",
            "file_count",
            "files",
            "generated_manifests",
        },
        "replication implementation manifest",
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("record")
        != "data002_replication_design_implementation_v1"
        or manifest.get("status") != DESIGN_STATUS
        or manifest.get("authorization") is not None
    ):
        raise ReplicationDesignError(
            "replication implementation manifest identity changed"
        )
    base_commit = manifest.get("base_commit")
    if (
        not isinstance(base_commit, str)
        or len(base_commit) != 40
        or any(character not in "0123456789abcdef" for character in base_commit)
    ):
        raise ReplicationDesignError("replication implementation base commit invalid")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or manifest.get("file_count") != len(files):
        raise ReplicationDesignError("replication implementation file count changed")
    command_prefix = [
        "git",
        "-c",
        f"safe.directory={project_root.resolve().as_posix()}",
        "show",
    ]
    for relative, entry in files.items():
        if not isinstance(relative, str) or not isinstance(entry, Mapping):
            raise ReplicationDesignError("invalid replication implementation entry")
        _exact_keys(entry, {"bytes", "sha256"}, f"implementation entry {relative}")
        path = (project_root / relative).resolve()
        if (
            not path.is_relative_to(project_root.resolve())
            or not path.is_file()
            or path.stat().st_size != entry.get("bytes")
            or file_sha256(path) != entry.get("sha256")
        ):
            raise ReplicationDesignError(
                f"replication implementation working-tree drift: {relative}"
            )
        result = subprocess.run(
            [*command_prefix, f"{base_commit}:{relative}"],
            cwd=project_root,
            capture_output=True,
            check=False,
        )
        if (
            result.returncode != 0
            or len(result.stdout) != entry.get("bytes")
            or sha256(result.stdout).hexdigest() != entry.get("sha256")
        ):
            raise ReplicationDesignError(
                f"replication implementation Git-tree drift: {relative}"
            )
    generated = manifest.get("generated_manifests")
    if not isinstance(generated, Mapping) or not generated:
        raise ReplicationDesignError("generated manifest bindings are missing")
    for relative, entry in generated.items():
        if not isinstance(relative, str) or not isinstance(entry, Mapping):
            raise ReplicationDesignError("invalid generated manifest entry")
        _exact_keys(entry, {"bytes", "sha256"}, f"generated entry {relative}")
        path = project_root / relative
        if (
            not path.is_file()
            or path.stat().st_size != entry.get("bytes")
            or file_sha256(path) != entry.get("sha256")
        ):
            raise ReplicationDesignError(f"generated manifest drift: {relative}")
    return {
        "status": "pass",
        "base_commit": base_commit,
        "critical_file_count": len(files),
        "generated_manifest_count": len(generated),
    }
