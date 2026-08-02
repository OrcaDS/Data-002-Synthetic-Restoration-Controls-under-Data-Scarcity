"""Immutable, authorization-gated runner for the frozen Data 002 analysis."""

from __future__ import annotations

import csv
import json
import os
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from data002.analysis_core import (
    ACCOUNTING,
    CONTRASTS,
    PRACTICAL_THRESHOLD,
    RECONCILIATION,
    RESAMPLES,
    analyze_metric_rows,
    baseline_records,
    replication_metrics_from_predictions,
    validate_analysis_outputs,
    validate_metric_rows,
)
from data002.analysis_provenance import (
    REPORT_PROVENANCE_FIELDS,
    AnalysisStartupPolicy,
    verify_analysis_startup,
)
from data002.replication_design import condition_key, fraction_token
from data002.replication_design import UPSTREAM_FILES, full_grid_records

PredictionLoader = Callable[
    [Path], Mapping[str, tuple[Sequence[Any], Sequence[Any]]]
]
MetricTableLoader = Callable[
    [Path], tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]
]


class AnalysisRunnerError(RuntimeError):
    """The authorized immutable analysis run cannot proceed."""


REPORT_POLICY = {
    "primary_contrast": "gaussian_minus_replication",
    "secondary_contrasts": [
        "replication_minus_real_only",
        "gaussian_minus_real_only",
    ],
    "bootstrap_method": "paired_seed_level_bootstrap",
    "bootstrap_resamples": 10_000,
    "quantiles": [0.025, 0.975],
    "quantile_method": "linear",
    "practical_threshold": 0.01,
    "inferential_tests_performed": False,
    "multiplicity_adjustment": "none",
    "silent_key_dropping": False,
    "complete_case_analysis": False,
}


def _contains_forbidden_output_name(value: Any) -> bool:
    forbidden = ("p_value", "pvalue", "probability", "y_true", "test_index")
    if isinstance(value, Mapping):
        return any(
            any(token in str(key).lower() for token in forbidden)
            or _contains_forbidden_output_name(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_output_name(item) for item in value)
    return False


def validate_output_documents(
    seed_metrics: Mapping[str, Any],
    summaries: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    expected_provenance: Mapping[str, str],
) -> None:
    if set(seed_metrics) != {
        "schema_version", "record", "metric", "condition_count", "rows"
    } or (
        seed_metrics.get("schema_version") != 1
        or seed_metrics.get("record") != "data002_replication_seed_metrics_v1"
        or seed_metrics.get("metric") != "roc_auc"
        or seed_metrics.get("condition_count") != 540
        or not isinstance(seed_metrics.get("rows"), list)
        or len(seed_metrics["rows"]) != 540
    ):
        raise AnalysisRunnerError("replication seed-metric schema changed")
    core_outputs = {
        "replication_metrics": seed_metrics["rows"],
        "summaries": summaries.get("rows"),
        "accounting": report.get("accounting"),
        "reconciliation": report.get("reconciliation"),
    }
    try:
        validate_analysis_outputs(core_outputs)
    except Exception as exc:
        raise AnalysisRunnerError(
            f"persisted analysis content validation failed: {exc}"
        ) from exc
    if set(summaries) != {
        "schema_version", "record", "stratum_count",
        "paired_seeds_per_stratum", "primary_contrast",
        "secondary_contrasts", "bootstrap", "practical_threshold",
        "interpretation_labels", "rows",
    } or (
        summaries.get("schema_version") != 1
        or summaries.get("record")
        != "data002_replication_stratum_summaries_v1"
        or summaries.get("stratum_count") != 18
        or summaries.get("paired_seeds_per_stratum") != 30
        or summaries.get("primary_contrast") != CONTRASTS[0]
        or summaries.get("secondary_contrasts") != list(CONTRASTS[1:])
        or summaries.get("bootstrap") != {
            "method": "paired_seed_level_bootstrap",
            "resamples": RESAMPLES,
            "bit_generator": "numpy.random.PCG64",
            "quantiles": [0.025, 0.975],
            "quantile_method": "linear",
        }
        or summaries.get("practical_threshold") != PRACTICAL_THRESHOLD
        or summaries.get("interpretation_labels") != [
            "helpful", "harmful",
            "no_clear_practically_meaningful_difference",
        ]
        or not isinstance(summaries.get("rows"), list)
        or len(summaries["rows"]) != 18
    ):
        raise AnalysisRunnerError("stratum-summary schema changed")
    if set(report) != {
        "schema_version", "record", "status", "scientific_scope",
        "accounting", "reconciliation", "allocation_invariants_required",
        "inferential_tests_performed", "multiplicity_adjustment",
        "raw_prediction_arrays_persisted", "policy", "provenance", "outputs",
    } or (
        report.get("schema_version") != 1
        or report.get("record") != "data002_replication_analysis_report_v1"
        or report.get("status") != "complete"
        or report.get("scientific_scope")
        != "prespecified_replication_treatment_analysis"
        or report.get("accounting") != ACCOUNTING
        or report.get("reconciliation") != RECONCILIATION
        or report.get("allocation_invariants_required") is not True
        or report.get("inferential_tests_performed") is not False
        or report.get("multiplicity_adjustment") != "none"
        or report.get("raw_prediction_arrays_persisted") is not False
        or report.get("policy") != REPORT_POLICY
        or set(expected_provenance) != REPORT_PROVENANCE_FIELDS
        or not isinstance(report.get("provenance"), Mapping)
        or set(report["provenance"]) != REPORT_PROVENANCE_FIELDS
        or report.get("provenance") != dict(expected_provenance)
    ):
        raise AnalysisRunnerError("analysis-report schema changed")
    outputs = report.get("outputs")
    if (
        not isinstance(outputs, Mapping)
        or tuple(outputs) != (
            "replication_seed_metrics.json", "stratum_summaries.json"
        )
    ):
        raise AnalysisRunnerError("analysis output filenames changed")
    for filename, entry in outputs.items():
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"bytes", "sha256"}
            or type(entry.get("bytes")) is not int
            or entry["bytes"] <= 0
            or not isinstance(entry.get("sha256"), str)
            or len(entry["sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in entry["sha256"]
            )
        ):
            raise AnalysisRunnerError(
                f"analysis output identity is invalid: {filename}"
            )
    expected_output_entries = {
        "replication_seed_metrics.json": {
            "bytes": len(_json_bytes(seed_metrics)),
            "sha256": sha256(_json_bytes(seed_metrics)).hexdigest(),
        },
        "stratum_summaries.json": {
            "bytes": len(_json_bytes(summaries)),
            "sha256": sha256(_json_bytes(summaries)).hexdigest(),
        },
    }
    if dict(outputs) != expected_output_entries:
        raise AnalysisRunnerError("analysis output byte or SHA-256 binding changed")
    if _contains_forbidden_output_name([seed_metrics, summaries, report]):
        raise AnalysisRunnerError("persisted analysis output contains forbidden fields")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_new(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    encoded = _json_bytes(value)
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return {"bytes": len(encoded), "sha256": sha256(encoded).hexdigest()}


def load_production_predictions(
    project_root: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load sealed arrays only after the separately authorized startup gate."""
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    root = project_root / "results/replication/full_v1/predictions"
    expected = [
        root / f"{record['key']}.npz" for record in full_grid_records()
    ]
    observed = sorted(path for path in root.rglob("*") if path.is_file())
    if observed != sorted(expected):
        raise AnalysisRunnerError("production prediction file reconciliation failed")
    for path in expected:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"test_index", "y_true", "probability"}:
                raise AnalysisRunnerError(f"prediction schema changed: {path.name}")
            test_index = np.asarray(archive["test_index"])
            y_true = np.asarray(archive["y_true"])
            probability = np.asarray(archive["probability"])
            if (
                test_index.ndim != 1
                or y_true.ndim != 1
                or probability.ndim != 1
                or not (test_index.shape == y_true.shape == probability.shape)
                or test_index.size == 0
                or not np.issubdtype(test_index.dtype, np.integer)
                or not np.issubdtype(y_true.dtype, np.integer)
                or probability.dtype != np.dtype(np.float32)
                or np.any(test_index < 0)
                or len(np.unique(test_index)) != test_index.size
                or not np.isin(y_true, (0, 1)).all()
                or not np.isfinite(probability).all()
                or np.any((probability < 0.0) | (probability > 1.0))
            ):
                raise AnalysisRunnerError(
                    f"prediction dtype, range, or shape changed: {path.name}"
                )
            predictions[path.stem] = (
                y_true.copy(),
                probability.copy(),
            )
    return predictions


def _metric_rows(
    path: Path, expected_columns: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != tuple(expected_columns):
                raise AnalysisRunnerError(
                    f"metric table header changed: {path.name}"
                )
            for row in reader:
                retained_fraction = float(row["scarcity_fraction"])
                seed = int(row["seed"])
                rows.append(
                    {
                        "key": condition_key(
                            row["dataset"], retained_fraction, seed, row["model"]
                        ),
                        "dataset": row["dataset"],
                        "retained_fraction": retained_fraction,
                        "retained_fraction_token": fraction_token(
                            retained_fraction
                        ),
                        "seed": seed,
                        "model": row["model"],
                        "roc_auc": float(row["roc_auc"]),
                    }
                )
    except (OSError, UnicodeError, csv.Error, KeyError, ValueError) as exc:
        if isinstance(exc, AnalysisRunnerError):
            raise
        raise AnalysisRunnerError(
            f"metric table parsing failed: {path.name}"
        ) from exc
    return rows


def load_production_metric_tables(
    project_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load frozen upstream values only after separate analysis authorization."""
    root = project_root / "evidence/data001_metric_bundle_v1"
    gaussian_raw = _metric_rows(
        root / "gaussian_copula_per_seed.csv",
        UPSTREAM_FILES["gaussian_copula_per_seed.csv"]["columns"],
    )
    baseline_raw = _metric_rows(
        root / "baseline_per_seed.csv",
        UPSTREAM_FILES["baseline_per_seed.csv"]["columns"],
    )
    gaussian = validate_metric_rows(
        gaussian_raw, full_grid_records(), label="gaussian adapter"
    )
    baseline = validate_metric_rows(
        baseline_raw, baseline_records(), label="baseline adapter"
    )
    return list(gaussian.values()), list(baseline.values())


def run_analysis(
    *,
    project_root: Path,
    launch_path: Path,
    prediction_loader: PredictionLoader = load_production_predictions,
    metric_table_loader: MetricTableLoader = load_production_metric_tables,
    startup_verifier: Callable[
        [Path, Path], AnalysisStartupPolicy
    ] = verify_analysis_startup,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    policy = startup_verifier(project_root, launch_path)
    if not isinstance(policy, AnalysisStartupPolicy):
        raise AnalysisRunnerError("startup verifier returned invalid policy")
    output_root = policy.output_root.resolve()
    expected_root = (project_root / "results/analysis/replication_v1").resolve()
    if output_root != expected_root:
        raise AnalysisRunnerError("analysis output path is not fixed")
    if output_root.exists():
        raise FileExistsError("immutable analysis output already exists")

    predictions = prediction_loader(project_root)
    replication_rows = replication_metrics_from_predictions(predictions)
    gaussian_rows, baseline_rows = metric_table_loader(project_root)
    results = analyze_metric_rows(
        replication_rows, gaussian_rows, baseline_rows
    )
    validate_analysis_outputs(results)

    seed_metrics = {
        "schema_version": 1,
        "record": "data002_replication_seed_metrics_v1",
        "metric": "roc_auc",
        "condition_count": 540,
        "rows": results["replication_metrics"],
    }
    summaries = {
        "schema_version": 1,
        "record": "data002_replication_stratum_summaries_v1",
        "stratum_count": 18,
        "paired_seeds_per_stratum": 30,
        "primary_contrast": CONTRASTS[0],
        "secondary_contrasts": list(CONTRASTS[1:]),
        "bootstrap": {
            "method": "paired_seed_level_bootstrap",
            "resamples": RESAMPLES,
            "bit_generator": "numpy.random.PCG64",
            "quantiles": [0.025, 0.975],
            "quantile_method": "linear",
        },
        "practical_threshold": PRACTICAL_THRESHOLD,
        "interpretation_labels": [
            "helpful",
            "harmful",
            "no_clear_practically_meaningful_difference",
        ],
        "rows": results["summaries"],
    }
    staging = output_root.with_name(
        f".{output_root.name}.{os.getpid()}.tmp"
    )
    if staging.exists():
        raise AnalysisRunnerError("analysis staging path already exists")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        seed_entry = _write_new(
            staging / "replication_seed_metrics.json", seed_metrics
        )
        summary_entry = _write_new(
            staging / "stratum_summaries.json", summaries
        )
        report = {
            "schema_version": 1,
            "record": "data002_replication_analysis_report_v1",
            "status": "complete",
            "scientific_scope": "prespecified_replication_treatment_analysis",
            "accounting": results["accounting"],
            "reconciliation": results["reconciliation"],
            "allocation_invariants_required": True,
            "inferential_tests_performed": False,
            "multiplicity_adjustment": "none",
            "raw_prediction_arrays_persisted": False,
            "policy": dict(REPORT_POLICY),
            "provenance": dict(policy.report_provenance),
            "outputs": {
                "replication_seed_metrics.json": seed_entry,
                "stratum_summaries.json": summary_entry,
            },
        }
        validate_output_documents(
            seed_metrics,
            summaries,
            report,
            expected_provenance=policy.report_provenance,
        )
        _write_new(staging / "analysis_report.json", report)
        if output_root.exists():
            raise FileExistsError("immutable analysis output appeared concurrently")
        staging.rename(output_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return report
