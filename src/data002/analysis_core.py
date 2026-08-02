"""Frozen statistical core for the prospective Data 002 analysis."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from data002.replication_design import (
    ANALYSIS_SEED_NAMESPACE,
    BASELINE_FRACTIONS,
    FULL_FRACTIONS,
    MODELS,
    SEEDS,
    canonical_json_bytes,
    condition_key,
    fraction_token,
    full_grid_records,
)

CONTRASTS = (
    "gaussian_minus_replication",
    "replication_minus_real_only",
    "gaussian_minus_real_only",
)
RESAMPLES = 10_000
PRACTICAL_THRESHOLD = 0.01
INTERPRETATION_HELPFUL = "helpful"
INTERPRETATION_HARMFUL = "harmful"
INTERPRETATION_NO_CLEAR = "no_clear_practically_meaningful_difference"
METRIC_ROW_FIELDS = {
    "key", "dataset", "retained_fraction", "retained_fraction_token",
    "seed", "model", "roc_auc",
}
CONTRAST_SUMMARY_FIELDS = {
    "estimate", "ci_lower", "ci_upper", "interpretation",
    "bootstrap_seed", "bootstrap_resamples", "method", "quantiles",
    "quantile_method",
}
ACCOUNTING = {
    "replication_keys": 540,
    "gaussian_keys": 540,
    "baseline_source_keys": 660,
    "baseline_matched_keys": 540,
    "strata": 18,
    "paired_seeds_per_stratum": 30,
}
RECONCILIATION = {
    "missing_replication_keys": [],
    "unexpected_replication_keys": [],
    "missing_gaussian_keys": [],
    "unexpected_gaussian_keys": [],
    "missing_baseline_keys": [],
    "unexpected_baseline_keys": [],
    "incomplete_strata": [],
}


class AnalysisError(ValueError):
    """Analysis input, pairing, output, or policy is not exact."""


def baseline_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dataset in ("diabetes", "cleveland"):
        for retained_fraction in BASELINE_FRACTIONS[dataset]:
            for seed in SEEDS:
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


def _finite_roc_auc(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError("ROC-AUC must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise AnalysisError("ROC-AUC must be finite and within [0, 1]")
    return result


def validate_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_records: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    expected = {record["key"]: dict(record) for record in expected_records}
    observed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != METRIC_ROW_FIELDS:
            raise AnalysisError(f"{label} metric row {index} schema changed")
        key = row.get("key")
        if not isinstance(key, str):
            raise AnalysisError(f"{label} metric key must be a string")
        if key in observed:
            duplicates.append(key)
            continue
        record = expected.get(key)
        if record is None:
            observed[key] = dict(row)
            continue
        if any(row.get(field) != record[field] for field in (
            "dataset", "retained_fraction", "retained_fraction_token",
            "seed", "model",
        )):
            raise AnalysisError(f"{label} metric identity mismatch: {key}")
        normalized = dict(row)
        normalized["roc_auc"] = _finite_roc_auc(row["roc_auc"])
        observed[key] = normalized
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    if duplicates or missing or unexpected or len(rows) != len(expected):
        raise AnalysisError(
            f"{label} key reconciliation failed: duplicates={sorted(duplicates)}, "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {key: observed[key] for key in expected}


def replication_metrics_from_predictions(
    predictions: Mapping[str, tuple[Sequence[Any], Sequence[Any]]],
) -> list[dict[str, Any]]:
    records = full_grid_records()
    expected = [record["key"] for record in records]
    missing = sorted(set(expected) - set(predictions))
    unexpected = sorted(set(predictions) - set(expected))
    if missing or unexpected or len(predictions) != len(expected):
        raise AnalysisError(
            "replication prediction reconciliation failed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    output: list[dict[str, Any]] = []
    for record in records:
        y_true_raw, probability_raw = predictions[record["key"]]
        y_true = np.asarray(y_true_raw)
        probability = np.asarray(probability_raw)
        if (
            y_true.ndim != 1
            or probability.ndim != 1
            or y_true.shape != probability.shape
            or y_true.size == 0
            or not np.issubdtype(y_true.dtype, np.integer)
            or probability.dtype != np.dtype(np.float32)
            or not np.isfinite(probability).all()
            or np.any((probability < 0.0) | (probability > 1.0))
            or not np.isin(y_true, (0, 1)).all()
            or len(np.unique(y_true)) != 2
        ):
            raise AnalysisError(
                f"invalid sealed prediction arrays: {record['key']}"
            )
        metric = _finite_roc_auc(float(roc_auc_score(y_true, probability)))
        output.append({**record, "roc_auc": metric})
    return output


def bootstrap_seed(
    dataset: str,
    retained_fraction_token: str,
    model: str,
    contrast: str,
) -> int:
    if contrast not in CONTRASTS:
        raise AnalysisError("unsupported contrast")
    payload = {
        "namespace": ANALYSIS_SEED_NAMESPACE,
        "dataset": dataset,
        "retained_fraction_token": retained_fraction_token,
        "model": model,
        "contrast": contrast,
        "resamples": RESAMPLES,
    }
    digest = sha256(canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def interpretation_label(lower: float, upper: float) -> str:
    if lower > PRACTICAL_THRESHOLD:
        return INTERPRETATION_HELPFUL
    if upper < -PRACTICAL_THRESHOLD:
        return INTERPRETATION_HARMFUL
    return INTERPRETATION_NO_CLEAR


def _bootstrap_summary(
    differences: np.ndarray,
    *,
    dataset: str,
    retained_fraction_token: str,
    model: str,
    contrast: str,
) -> dict[str, Any]:
    if differences.shape != (30,) or not np.isfinite(differences).all():
        raise AnalysisError("each paired stratum must contain 30 finite differences")
    seed = bootstrap_seed(
        dataset, retained_fraction_token, model, contrast
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, 30, size=(RESAMPLES, 30))
    bootstrap_means = differences[indices].mean(axis=1)
    lower, upper = np.quantile(
        bootstrap_means, [0.025, 0.975], method="linear"
    )
    estimate = float(differences.mean())
    lower_float = float(lower)
    upper_float = float(upper)
    return {
        "estimate": estimate,
        "ci_lower": lower_float,
        "ci_upper": upper_float,
        "interpretation": interpretation_label(lower_float, upper_float),
        "bootstrap_seed": seed,
        "bootstrap_resamples": RESAMPLES,
        "method": "paired_seed_level_bootstrap",
        "quantiles": [0.025, 0.975],
        "quantile_method": "linear",
    }


def analyze_metric_rows(
    replication_rows: Sequence[Mapping[str, Any]],
    gaussian_rows: Sequence[Mapping[str, Any]],
    baseline_rows_input: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    full_records = full_grid_records()
    replication = validate_metric_rows(
        replication_rows, full_records, label="replication"
    )
    gaussian = validate_metric_rows(
        gaussian_rows, full_records, label="gaussian"
    )
    baseline_all = validate_metric_rows(
        baseline_rows_input, baseline_records(), label="baseline"
    )
    full_keys = [record["key"] for record in full_records]
    baseline = {key: baseline_all[key] for key in full_keys}
    summaries: list[dict[str, Any]] = []
    for dataset in ("diabetes", "cleveland"):
        for retained_fraction in FULL_FRACTIONS[dataset]:
            token = fraction_token(retained_fraction)
            for model in MODELS:
                keys = [
                    condition_key(dataset, retained_fraction, seed, model)
                    for seed in SEEDS
                ]
                if len(keys) != 30 or any(key not in replication for key in keys):
                    raise AnalysisError("stratum pairing is incomplete")
                replication_values = np.array(
                    [replication[key]["roc_auc"] for key in keys], dtype=float
                )
                gaussian_values = np.array(
                    [gaussian[key]["roc_auc"] for key in keys], dtype=float
                )
                baseline_values = np.array(
                    [baseline[key]["roc_auc"] for key in keys], dtype=float
                )
                differences = {
                    "gaussian_minus_replication": (
                        gaussian_values - replication_values
                    ),
                    "replication_minus_real_only": (
                        replication_values - baseline_values
                    ),
                    "gaussian_minus_real_only": (
                        gaussian_values - baseline_values
                    ),
                }
                summaries.append(
                    {
                        "dataset": dataset,
                        "retained_fraction": retained_fraction,
                        "retained_fraction_token": token,
                        "model": model,
                        "paired_seed_count": 30,
                        "contrasts": {
                            contrast: _bootstrap_summary(
                                values,
                                dataset=dataset,
                                retained_fraction_token=token,
                                model=model,
                                contrast=contrast,
                            )
                            for contrast, values in differences.items()
                        },
                    }
                )
    if len(summaries) != 18:
        raise AnalysisError("analysis must contain exactly 18 strata")
    return {
        "replication_metrics": [replication[key] for key in full_keys],
        "summaries": summaries,
        "accounting": dict(ACCOUNTING),
        "reconciliation": {key: list(value) for key, value in RECONCILIATION.items()},
    }


def _forbidden_output_name(value: Any) -> bool:
    forbidden = ("p_value", "pvalue", "probability", "y_true", "test_index")
    if isinstance(value, Mapping):
        return any(
            any(token in str(key).lower() for token in forbidden)
            or _forbidden_output_name(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_forbidden_output_name(item) for item in value)
    return False


def validate_analysis_outputs(outputs: Mapping[str, Any]) -> None:
    if _forbidden_output_name(outputs):
        raise AnalysisError("analysis output contains forbidden fields")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "replication_metrics", "summaries", "accounting", "reconciliation"
    }:
        raise AnalysisError("analysis output fields changed")
    try:
        json.dumps(outputs, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("analysis output is not finite JSON") from exc
    metric_rows = outputs.get("replication_metrics")
    expected_metrics = full_grid_records()
    if (
        not isinstance(metric_rows, list)
        or len(metric_rows) != 540
        or any(
            not isinstance(row, Mapping)
            or set(row) != METRIC_ROW_FIELDS
            for row in metric_rows
        )
    ):
        raise AnalysisError("replication metric output schema changed")
    for row, expected in zip(metric_rows, expected_metrics, strict=True):
        if any(row.get(field) != expected[field] for field in (
            "key", "dataset", "retained_fraction", "retained_fraction_token",
            "seed", "model",
        )):
            raise AnalysisError("replication metric identity or order changed")
        _finite_roc_auc(row.get("roc_auc"))

    summaries = outputs.get("summaries")
    expected_strata = [
        (dataset, retained_fraction, fraction_token(retained_fraction), model)
        for dataset in ("diabetes", "cleveland")
        for retained_fraction in FULL_FRACTIONS[dataset]
        for model in MODELS
    ]
    if not isinstance(summaries, list) or len(summaries) != 18:
        raise AnalysisError("stratum summary output count changed")
    for row, identity in zip(summaries, expected_strata, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row) != {
                "dataset", "retained_fraction", "retained_fraction_token",
                "model", "paired_seed_count", "contrasts",
            }
            or (
                row.get("dataset"), row.get("retained_fraction"),
                row.get("retained_fraction_token"), row.get("model"),
            ) != identity
            or row.get("paired_seed_count") != 30
            or not isinstance(row.get("contrasts"), Mapping)
            or tuple(row["contrasts"]) != CONTRASTS
        ):
            raise AnalysisError("stratum identity, order, or schema changed")
        for contrast, summary in row["contrasts"].items():
            if (
                not isinstance(summary, Mapping)
                or set(summary) != CONTRAST_SUMMARY_FIELDS
            ):
                raise AnalysisError("contrast summary schema changed")
            values = [
                summary.get("estimate"),
                summary.get("ci_lower"),
                summary.get("ci_upper"),
            ]
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not -1.0 <= float(value) <= 1.0
                for value in values
            ):
                raise AnalysisError("contrast values must be finite within [-1, 1]")
            lower = float(summary["ci_lower"])
            upper = float(summary["ci_upper"])
            if lower > upper:
                raise AnalysisError("contrast interval endpoints are reversed")
            if (
                summary.get("bootstrap_seed") != bootstrap_seed(
                    row["dataset"], row["retained_fraction_token"],
                    row["model"], contrast,
                )
                or summary.get("bootstrap_resamples") != RESAMPLES
                or summary.get("method") != "paired_seed_level_bootstrap"
                or summary.get("quantiles") != [0.025, 0.975]
                or summary.get("quantile_method") != "linear"
                or summary.get("interpretation")
                != interpretation_label(lower, upper)
            ):
                raise AnalysisError("contrast bootstrap or interpretation drift")
    if outputs.get("accounting") != ACCOUNTING:
        raise AnalysisError("analysis output accounting changed")
    if outputs.get("reconciliation") != RECONCILIATION:
        raise AnalysisError("analysis output reconciliation is incomplete")
