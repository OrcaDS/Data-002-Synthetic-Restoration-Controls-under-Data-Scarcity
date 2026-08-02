from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from data002.replication_design import (
    ORDER_NAMESPACE,
    ReplicationDesignError,
    build_key_manifest,
    build_upstream_evidence_manifest,
    canonical_json_bytes,
    condition_keys_sha256,
    full_grid_records,
    pilot_records,
    validate_design_contracts,
    validate_analysis_seed_payload,
    validate_key_manifest,
    validate_replication_order_payload,
    validate_upstream_evidence_manifest,
)

PROJECT_ROOT = Path.cwd().resolve()
TOY_COMMIT = "a" * 40


def load_contract(name: str) -> dict:
    return json.loads(
        (PROJECT_ROOT / "config" / name).read_text(encoding="utf-8")
    )


def contracts() -> tuple[dict, dict, dict]:
    return (
        load_contract("replication_study_v1.json"),
        load_contract("replication_pilot_v1.json"),
        load_contract("replication_analysis_v1.json"),
    )


def test_repository_contracts_freeze_counts_and_remain_unauthorized() -> None:
    report = validate_design_contracts(*contracts())

    assert report == {
        "status": "pass",
        "full_grid_conditions": 540,
        "pilot_conditions": 12,
        "primary_strata": 18,
        "authorization": None,
    }


def test_full_grid_has_exact_unique_coverage() -> None:
    records = full_grid_records()

    assert len(records) == 540
    assert len({record["key"] for record in records}) == 540
    assert {record["seed"] for record in records} == set(range(30))
    assert {
        (record["dataset"], record["retained_fraction_token"])
        for record in records
    } == {
        ("diabetes", "001pct"),
        ("diabetes", "005pct"),
        ("diabetes", "010pct"),
        ("diabetes", "025pct"),
        ("diabetes", "050pct"),
        ("cleveland", "005pct"),
        ("cleveland", "010pct"),
        ("cleveland", "025pct"),
        ("cleveland", "050pct"),
    }


def test_pilot_has_exact_unique_coverage_and_is_full_grid_subset() -> None:
    pilot = pilot_records()
    full_keys = {record["key"] for record in full_grid_records()}

    assert len(pilot) == 12
    assert len({record["key"] for record in pilot}) == 12
    assert {record["key"] for record in pilot} <= full_keys
    assert {record["seed"] for record in pilot} == {0, 29}


def test_key_manifests_validate() -> None:
    full = build_key_manifest("full", TOY_COMMIT)
    pilot = build_key_manifest("pilot", TOY_COMMIT)

    assert validate_key_manifest(full, scope="full")["condition_count"] == 540
    assert validate_key_manifest(pilot, scope="pilot")["condition_count"] == 12
    assert full["condition_keys_sha256"] == condition_keys_sha256(
        record["key"] for record in full_grid_records()
    )


@pytest.mark.parametrize("scope", ["full", "pilot"])
def test_duplicate_condition_key_is_rejected(scope: str) -> None:
    manifest = build_key_manifest(scope, TOY_COMMIT)
    manifest["conditions"][-1] = copy.deepcopy(manifest["conditions"][0])

    with pytest.raises(ReplicationDesignError, match="duplicate="):
        validate_key_manifest(manifest, scope=scope)


@pytest.mark.parametrize("scope", ["full", "pilot"])
def test_missing_condition_key_is_rejected(scope: str) -> None:
    manifest = build_key_manifest(scope, TOY_COMMIT)
    manifest["conditions"].pop()

    with pytest.raises(ReplicationDesignError, match="missing="):
        validate_key_manifest(manifest, scope=scope)


def test_unexpected_condition_key_is_rejected() -> None:
    manifest = build_key_manifest("full", TOY_COMMIT)
    manifest["conditions"][-1]["seed"] = 30
    manifest["conditions"][-1]["key"] = manifest["conditions"][-1]["key"].replace(
        "seed_29", "seed_30"
    )

    with pytest.raises(ReplicationDesignError, match="unexpected="):
        validate_key_manifest(manifest, scope="full")


def test_pilot_full_grid_manifest_mismatch_is_rejected() -> None:
    pilot = build_key_manifest("pilot", TOY_COMMIT)

    with pytest.raises(ReplicationDesignError, match="identity changed"):
        validate_key_manifest(pilot, scope="full")


def test_replication_order_payload_has_exact_canonical_utf8_json() -> None:
    payload = {
        "namespace": ORDER_NAMESPACE,
        "dataset": "diabetes",
        "retained_fraction_token": "001pct",
        "split_seed": 0,
        "class_label": 1,
        "original_source_row_index": 42,
    }

    encoded = validate_replication_order_payload(payload)

    assert encoded == (
        b'{"class_label":1,"dataset":"diabetes",'
        b'"namespace":"data002.replication-order.v1",'
        b'"original_source_row_index":42,'
        b'"retained_fraction_token":"001pct","split_seed":0}'
    )
    assert canonical_json_bytes(payload) == encoded


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("namespace", "other", "namespace"),
        ("dataset", "other", "dataset"),
        ("retained_fraction_token", "100pct", "fraction token"),
        ("split_seed", 30, "split seed"),
        ("class_label", "1", "must be an integer"),
        ("original_source_row_index", -1, "must be nonnegative"),
    ],
)
def test_malformed_canonical_payload_fields_are_rejected(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = {
        "namespace": ORDER_NAMESPACE,
        "dataset": "diabetes",
        "retained_fraction_token": "001pct",
        "split_seed": 0,
        "class_label": 1,
        "original_source_row_index": 42,
    }
    payload[field] = value

    with pytest.raises(ReplicationDesignError, match=message):
        validate_replication_order_payload(payload)


def test_unexpected_canonical_payload_field_is_rejected() -> None:
    payload = {
        "namespace": ORDER_NAMESPACE,
        "dataset": "diabetes",
        "retained_fraction_token": "001pct",
        "split_seed": 0,
        "class_label": 1,
        "original_source_row_index": 42,
        "outcome": 1,
    }

    with pytest.raises(ReplicationDesignError, match="keys must be exactly"):
        validate_replication_order_payload(payload)


@pytest.mark.parametrize("class_label", [-1, 2, 3])
def test_nonbinary_integer_class_label_is_rejected(class_label: int) -> None:
    payload = {
        "namespace": ORDER_NAMESPACE,
        "dataset": "diabetes",
        "retained_fraction_token": "001pct",
        "split_seed": 0,
        "class_label": class_label,
        "original_source_row_index": 42,
    }

    with pytest.raises(ReplicationDesignError, match="exactly integer 0 or 1"):
        validate_replication_order_payload(payload)


@pytest.mark.parametrize("class_label", [0, 1])
def test_exact_binary_integer_class_labels_are_accepted(class_label: int) -> None:
    payload = {
        "namespace": ORDER_NAMESPACE,
        "dataset": "diabetes",
        "retained_fraction_token": "001pct",
        "split_seed": 0,
        "class_label": class_label,
        "original_source_row_index": 42,
    }

    assert validate_replication_order_payload(payload)


def test_analysis_seed_payload_is_canonical_and_bounded() -> None:
    payload = {
        "namespace": "data002.paired-bootstrap-seed.v1",
        "dataset": "cleveland",
        "retained_fraction_token": "005pct",
        "model": "random_forest",
        "contrast": "gaussian_minus_replication",
        "resamples": 10000,
    }

    seed = validate_analysis_seed_payload(payload)

    assert 0 <= seed < 2**64


def test_malformed_analysis_seed_payload_is_rejected() -> None:
    payload = {
        "namespace": "data002.paired-bootstrap-seed.v1",
        "dataset": "cleveland",
        "retained_fraction_token": "005pct",
        "model": "random_forest",
        "contrast": "gaussian_minus_replication",
        "resamples": 9999,
    }

    with pytest.raises(ReplicationDesignError, match="resample count"):
        validate_analysis_seed_payload(payload)


@pytest.mark.parametrize("contract_index", [0, 1, 2])
@pytest.mark.parametrize("authorization", [False, "", "authorized", {}])
def test_design_contract_authorization_must_be_null(
    contract_index: int,
    authorization: object,
) -> None:
    values = list(contracts())
    values[contract_index]["authorization"] = authorization

    with pytest.raises(ReplicationDesignError, match="must remain unauthorized"):
        validate_design_contracts(*values)


def test_study_grid_drift_is_rejected() -> None:
    study, pilot, analysis = contracts()
    study["grid"]["diabetes"]["retained_fractions"].pop()

    with pytest.raises(ReplicationDesignError, match="study grid changed"):
        validate_design_contracts(study, pilot, analysis)


def test_pilot_grid_drift_is_rejected() -> None:
    study, pilot, analysis = contracts()
    pilot["matrix"]["seeds"] = [0, 1]

    with pytest.raises(ReplicationDesignError, match="pilot grid changed"):
        validate_design_contracts(study, pilot, analysis)


def test_cross_class_duplicate_order_drift_is_rejected() -> None:
    study, pilot, analysis = contracts()
    study["restoration"]["duplicate_class_block_order"] = "descending"

    with pytest.raises(
        ReplicationDesignError, match="replication ordering or allocation changed"
    ):
        validate_design_contracts(study, pilot, analysis)


@pytest.mark.parametrize(
    "operation",
    ["compute", "store", "print", "summarize", "join"],
)
def test_replication_metric_blinding_drift_is_rejected(operation: str) -> None:
    study, pilot, analysis = contracts()
    study["metric_blinding"]["predictive_metrics"][operation] = True

    with pytest.raises(ReplicationDesignError, match="metric-blinding boundary"):
        validate_design_contracts(study, pilot, analysis)


def test_operational_validation_probability_return_is_rejected() -> None:
    study, pilot, analysis = contracts()
    study["metric_blinding"]["operational_validation_may_return"].append(
        "probability_values"
    )

    with pytest.raises(ReplicationDesignError, match="metric-blinding boundary"):
        validate_design_contracts(study, pilot, analysis)


def test_pilot_upstream_metric_access_drift_is_rejected() -> None:
    study, pilot, analysis = contracts()
    pilot["inspection_boundary"]["upstream_metric_tables_access"] = "allowed"

    with pytest.raises(ReplicationDesignError, match="inspection boundary"):
        validate_design_contracts(study, pilot, analysis)


def test_pilot_sealed_evidence_interpretation_is_rejected() -> None:
    study, pilot, analysis = contracts()
    pilot["sealed_outcome_evidence"]["operational_review_interpretation"] = "allowed"

    with pytest.raises(ReplicationDesignError, match="sealed-evidence policy"):
        validate_design_contracts(study, pilot, analysis)


def test_analysis_gate_drift_is_rejected() -> None:
    study, pilot, analysis = contracts()
    analysis["analysis_gate"]["complete_case_analysis"] = True

    with pytest.raises(ReplicationDesignError, match="analysis gate changed"):
        validate_design_contracts(study, pilot, analysis)


def test_preanalysis_metric_reconciliation_is_rejected() -> None:
    study, pilot, analysis = contracts()
    analysis["analysis_gate"]["checkpoint_prediction_metric_reconciliation"] = (
        analysis["analysis_gate"].pop("checkpoint_prediction_reconciliation")
    )

    with pytest.raises(ReplicationDesignError, match="analysis gate changed"):
        validate_design_contracts(study, pilot, analysis)


def test_later_metric_output_reconciliation_drift_is_rejected() -> None:
    study, pilot, analysis = contracts()
    analysis["analysis_stage"]["metric_output_reconciliation"] = "best_effort"

    with pytest.raises(ReplicationDesignError, match="analysis-stage boundary"):
        validate_design_contracts(study, pilot, analysis)


def test_upstream_bundle_schema_and_key_coverage() -> None:
    manifest = build_upstream_evidence_manifest(PROJECT_ROOT, TOY_COMMIT)
    report = validate_upstream_evidence_manifest(PROJECT_ROOT, manifest)

    assert report["file_count"] == 6
    assert report["baseline_key_count"] == 660
    assert report["gaussian_key_count"] == 540


def test_upstream_manifest_drift_is_rejected() -> None:
    manifest = build_upstream_evidence_manifest(PROJECT_ROOT, TOY_COMMIT)
    manifest["files"]["baseline_per_seed.csv"]["sha256"] = "0" * 64

    with pytest.raises(ReplicationDesignError, match="manifest drift"):
        validate_upstream_evidence_manifest(PROJECT_ROOT, manifest)
