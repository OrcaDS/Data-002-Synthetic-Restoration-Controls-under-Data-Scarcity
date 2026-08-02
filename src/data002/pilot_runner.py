"""Authorization-gated, metric-blind runner for the exact 12-condition pilot."""

from __future__ import annotations

import io
import json
import math
import multiprocessing
import os
import queue
import shutil
import time
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import psutil

from data002.pilot_provenance import (
    PILOT_LOCK_NAME,
    PilotStartupPolicy,
    exclusive_pilot_run_lock,
    validate_pilot_lock_record,
    verify_pilot_startup,
)
from data002.reconstruction import (
    MODEL_BUILDERS,
    load_datasets,
    retained_training_frame,
    split_dataset,
)
from data002.replication_constructor import construct_replication_control
from data002.replication_design import (
    fraction_token,
    pilot_records,
    validate_key_manifest,
)

ARRAY_KEYS = {"test_index", "y_true", "probability"}
EXECUTOR_KEYS = {
    "prediction_arrays",
    "allocation_record",
    "warnings",
    "resource_observations",
}
TERMINAL_STATES = {"success", "failure", "timeout"}
PREDICTIVE_NAMES = {
    "roc_auc",
    "average_precision",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "predictive_metric",
    "predictive_metrics",
    "probability_values",
}
HASH_KEYS = {
    "original_source_indices_sha256",
    "duplicate_source_indices_sha256",
    "final_source_indices_sha256",
    "final_target_labels_sha256",
    "canonical_contract_sha256",
}
ALLOCATION_KEYS = {
    "schema_version",
    "record",
    "dataset",
    "retained_fraction_token",
    "split_seed",
    "original_count",
    "duplicate_count",
    "final_count",
    "retained_target_counts",
    "duplicate_target_counts",
    "final_target_counts",
    *HASH_KEYS,
    "class_blocks",
}
CLASS_BLOCK_KEYS = {
    "class_label",
    "retained_source_count",
    "duplicate_count",
    "minimum_duplicates_per_source",
    "maximum_duplicates_per_source",
    "ranked_source_indices_sha256",
    "duplicate_source_indices_sha256",
}
OPERATIONAL_VALIDATION_KEYS = {
    "schema_exact",
    "array_count",
    "row_count",
    "test_index_integer",
    "y_true_binary_integer",
    "probability_float32",
    "probability_finite",
    "probability_in_unit_interval",
}
CAPTURED_OUTPUT_KEYS = {
    "stdout_bytes",
    "stderr_bytes",
    "stdout_sha256",
    "stderr_sha256",
}
EMPTY_CAPTURED_OUTPUT = {
    "stdout_bytes": 0,
    "stderr_bytes": 0,
    "stdout_sha256": sha256(b"").hexdigest(),
    "stderr_sha256": sha256(b"").hexdigest(),
}


class PilotRunnerError(ValueError):
    """Pilot execution or persistence violates the frozen operational contract."""


@dataclass(frozen=True, order=True)
class PilotCondition:
    dataset: str
    retained_fraction: float
    seed: int
    model: str


class PilotExecutor(Protocol):
    def __call__(self, condition: PilotCondition) -> Mapping[str, Any]: ...


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def condition_key(condition: PilotCondition) -> str:
    percentage = round(condition.retained_fraction * 100)
    return (
        f"{condition.dataset}__{percentage:03d}pct__seed_{condition.seed:02d}"
        f"__{condition.model}"
    )


def expected_conditions() -> dict[str, PilotCondition]:
    return {
        record["key"]: PilotCondition(
            dataset=record["dataset"],
            retained_fraction=record["retained_fraction"],
            seed=record["seed"],
            model=record["model"],
        )
        for record in pilot_records()
    }


@dataclass(frozen=True)
class RealPilotExecutor:
    """Picklable production executor; usable only after launch authorization."""

    diabetes_path: Path
    cleveland_path: Path

    def __call__(self, condition: PilotCondition) -> Mapping[str, Any]:
        datasets = load_datasets(self.diabetes_path, self.cleveland_path)
        spec = datasets[condition.dataset]
        split = split_dataset(spec, condition.seed)
        retained = retained_training_frame(
            spec, split, condition.retained_fraction, condition.seed
        )
        reconstructed, allocation = construct_replication_control(
            retained,
            target_column=spec.target,
            final_size=len(split.y_train),
            dataset=condition.dataset,
            retained_fraction=condition.retained_fraction,
            split_seed=condition.seed,
        )
        pipeline = MODEL_BUILDERS[condition.model](spec, condition.seed)
        with warnings.catch_warnings(record=True) as observed_warnings:
            warnings.simplefilter("always")
            pipeline.fit(
                reconstructed.drop(columns=spec.target),
                reconstructed[spec.target],
            )
            probability = pipeline.predict_proba(split.X_test)[:, 1].astype(
                np.float32
            )
        return {
            "prediction_arrays": {
                "test_index": split.X_test.index.to_numpy(),
                "y_true": split.y_test.to_numpy(dtype=np.int8),
                "probability": probability,
            },
            "allocation_record": allocation,
            "warnings": [
                {
                    "category": item.category.__name__,
                    "message_sha256": sha256(
                        str(item.message).encode("utf-8")
                    ).hexdigest(),
                }
                for item in observed_warnings
            ],
            "resource_observations": {
                "process_rss_bytes": psutil.Process(os.getpid()).memory_info().rss
            },
        }


def _contains_forbidden_name(value: Any) -> bool:
    if isinstance(value, Mapping):
        for nested in value.values():
            if _contains_forbidden_name(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_forbidden_name(item) for item in value)
    elif isinstance(value, str):
        if _valid_sha256(value):
            return False
        normalized = value.casefold()
        return any(name in normalized for name in PREDICTIVE_NAMES)
    return False


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _nonnegative_integer(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _validate_allocation_record(
    record: Any, condition: PilotCondition
) -> dict[str, Any]:
    if not isinstance(record, Mapping) or set(record) != ALLOCATION_KEYS:
        raise PilotRunnerError("allocation record fields changed")
    if _contains_forbidden_name(record):
        raise PilotRunnerError("allocation record contains forbidden leakage")
    if (
        record.get("schema_version") != 1
        or record.get("record") != "data002_replication_allocation_v1"
        or record.get("dataset") != condition.dataset
        or record.get("retained_fraction_token")
        != fraction_token(condition.retained_fraction)
        or record.get("split_seed") != condition.seed
    ):
        raise PilotRunnerError("allocation condition identity changed")
    for field in ("original_count", "duplicate_count", "final_count"):
        if not _nonnegative_integer(record.get(field)):
            raise PilotRunnerError("allocation counts must be nonnegative integers")
    if record["original_count"] + record["duplicate_count"] != record["final_count"]:
        raise PilotRunnerError("allocation total-count invariant failed")
    count_fields = (
        "retained_target_counts",
        "duplicate_target_counts",
        "final_target_counts",
    )
    for field in count_fields:
        counts = record.get(field)
        if (
            not isinstance(counts, Mapping)
            or set(counts) != {"0", "1"}
            or not all(_nonnegative_integer(value) for value in counts.values())
        ):
            raise PilotRunnerError("allocation target-count schema changed")
    for label in ("0", "1"):
        if (
            record["retained_target_counts"][label]
            + record["duplicate_target_counts"][label]
            != record["final_target_counts"][label]
        ):
            raise PilotRunnerError("allocation class-count invariant failed")
    if sum(record["retained_target_counts"].values()) != record["original_count"]:
        raise PilotRunnerError("allocation retained counts do not sum")
    if sum(record["duplicate_target_counts"].values()) != record["duplicate_count"]:
        raise PilotRunnerError("allocation duplicate counts do not sum")
    if sum(record["final_target_counts"].values()) != record["final_count"]:
        raise PilotRunnerError("allocation final counts do not sum")
    if not all(_valid_sha256(record.get(field)) for field in HASH_KEYS):
        raise PilotRunnerError("allocation hash field is invalid")
    blocks = record.get("class_blocks")
    if not isinstance(blocks, list) or len(blocks) != 2:
        raise PilotRunnerError("allocation class blocks changed")
    for expected_label, block in enumerate(blocks):
        if not isinstance(block, Mapping) or set(block) != CLASS_BLOCK_KEYS:
            raise PilotRunnerError("allocation class-block fields changed")
        if block.get("class_label") != expected_label:
            raise PilotRunnerError("allocation class-block order changed")
        for field in (
            "retained_source_count",
            "duplicate_count",
            "minimum_duplicates_per_source",
            "maximum_duplicates_per_source",
        ):
            if not _nonnegative_integer(block.get(field)):
                raise PilotRunnerError("allocation class-block count is invalid")
        retained_count = block["retained_source_count"]
        duplicate_count = block["duplicate_count"]
        if retained_count <= 0:
            raise PilotRunnerError("allocation class block has no source rows")
        floor_count = duplicate_count // retained_count
        ceiling_count = floor_count + int(duplicate_count % retained_count != 0)
        if (
            block["minimum_duplicates_per_source"] != floor_count
            or block["maximum_duplicates_per_source"] != ceiling_count
            or duplicate_count
            != record["duplicate_target_counts"][str(expected_label)]
            or retained_count
            != record["retained_target_counts"][str(expected_label)]
            or not _valid_sha256(block["ranked_source_indices_sha256"])
            or not _valid_sha256(block["duplicate_source_indices_sha256"])
        ):
            raise PilotRunnerError("allocation floor/ceiling invariant failed")
    return dict(record)


def _validate_warnings(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise PilotRunnerError("warnings must be a list")
    normalized: list[dict[str, str]] = []
    for entry in value:
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"category", "message_sha256"}
            or not isinstance(entry.get("category"), str)
            or not entry["category"]
            or _contains_forbidden_name(entry["category"])
            or not _valid_sha256(entry.get("message_sha256"))
        ):
            raise PilotRunnerError("warning entry schema changed")
        normalized.append(dict(entry))
    return normalized


def _validate_resources(value: Any) -> dict[str, int]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"process_rss_bytes"}
        or not _nonnegative_integer(value.get("process_rss_bytes"))
    ):
        raise PilotRunnerError("resource observation schema changed")
    return {"process_rss_bytes": value["process_rss_bytes"]}


def _validate_captured_output(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != CAPTURED_OUTPUT_KEYS:
        raise PilotRunnerError("captured-output fields changed")
    if (
        not _nonnegative_integer(value.get("stdout_bytes"))
        or not _nonnegative_integer(value.get("stderr_bytes"))
        or not _valid_sha256(value.get("stdout_sha256"))
        or not _valid_sha256(value.get("stderr_sha256"))
    ):
        raise PilotRunnerError("captured-output values are invalid")
    return dict(value)


def _captured_output_record(stdout: str, stderr: str) -> dict[str, Any]:
    combined = f"{stdout}\n{stderr}".casefold()
    if any(name in combined for name in PREDICTIVE_NAMES):
        raise PilotRunnerError("executor output contains a forbidden predictive name")
    return {
        "stdout_bytes": len(stdout.encode("utf-8")),
        "stderr_bytes": len(stderr.encode("utf-8")),
        "stdout_sha256": sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": sha256(stderr.encode("utf-8")).hexdigest(),
    }


def _worker(
    result_queue: multiprocessing.Queue[Any],
    executor: PilotExecutor,
    condition: PilotCondition,
) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = executor(condition)
        result_queue.put(
            (
                "success",
                {
                    "_worker_result": result,
                    "_captured_output": _captured_output_record(
                        stdout.getvalue(), stderr.getvalue()
                    ),
                },
            )
        )
    except BaseException as exc:
        result_queue.put(
            (
                "failure",
                {
                    "_worker_error": {
                        "type": type(exc).__name__,
                        "message_sha256": sha256(
                            str(exc).encode("utf-8")
                        ).hexdigest(),
                        "traceback_sha256": sha256(
                            traceback.format_exc().encode("utf-8")
                        ).hexdigest(),
                    },
                    "_captured_output": _captured_output_record(
                        stdout.getvalue(), stderr.getvalue()
                    ),
                },
            )
        )


def execute_with_timeout(
    executor: PilotExecutor,
    condition: PilotCondition,
    timeout_seconds: int,
) -> tuple[str, Any, float]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=_worker, args=(result_queue, executor, condition))
    started = time.monotonic()
    process.start()
    try:
        state, payload = result_queue.get(timeout=timeout_seconds)
    except queue.Empty:
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        return "timeout", {"reason": "condition_timeout"}, time.monotonic() - started
    finally:
        result_queue.close()
    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    return state, payload, time.monotonic() - started


def _validate_executor_result(
    result: Mapping[str, Any],
    condition: PilotCondition,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if set(result) != EXECUTOR_KEYS:
        raise PilotRunnerError(
            f"executor fields must be exactly {sorted(EXECUTOR_KEYS)}"
        )
    if _contains_forbidden_name(result):
        raise PilotRunnerError("executor result contains a forbidden predictive field")
    arrays = result["prediction_arrays"]
    if not isinstance(arrays, Mapping) or set(arrays) != ARRAY_KEYS:
        raise PilotRunnerError("prediction arrays have unexpected fields")
    normalized = {name: np.asarray(arrays[name]) for name in ARRAY_KEYS}
    if any(value.ndim != 1 for value in normalized.values()):
        raise PilotRunnerError("prediction arrays must be one-dimensional")
    lengths = {len(value) for value in normalized.values()}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise PilotRunnerError("prediction arrays must be nonempty and equal length")
    if not np.issubdtype(normalized["test_index"].dtype, np.integer):
        raise PilotRunnerError("test_index must be integer")
    if not np.issubdtype(normalized["y_true"].dtype, np.integer) or not np.isin(
        normalized["y_true"], [0, 1]
    ).all():
        raise PilotRunnerError("y_true must be binary integer")
    probability = normalized["probability"]
    if probability.dtype != np.float32:
        raise PilotRunnerError("probability must be float32")
    finite = bool(np.isfinite(probability).all())
    bounded = bool(((probability >= 0) & (probability <= 1)).all())
    if not finite or not bounded:
        raise PilotRunnerError("probability array is not finite and bounded")
    allocation = _validate_allocation_record(result["allocation_record"], condition)
    warning_records = _validate_warnings(result["warnings"])
    resources = _validate_resources(result["resource_observations"])
    operational = {
        "schema_exact": True,
        "array_count": 3,
        "row_count": len(probability),
        "test_index_integer": True,
        "y_true_binary_integer": True,
        "probability_float32": True,
        "probability_finite": finite,
        "probability_in_unit_interval": bounded,
    }
    metadata = {
        "allocation_record": allocation,
        "warnings": warning_records,
        "resource_observations": resources,
        "operational_validation": operational,
    }
    return normalized, metadata


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **{name: arrays[name] for name in ARRAY_KEYS})
            stream.flush()
            os.fsync(stream.fileno())
        _read_npz(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != ARRAY_KEYS:
            raise PilotRunnerError("sealed NPZ schema changed")
        return {name: archive[name] for name in ARRAY_KEYS}


def _checkpoint(
    condition: PilotCondition,
    state: str,
    elapsed: float,
    *,
    artifact: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    captured_output: Mapping[str, Any] | None,
    error: Mapping[str, Any] | None,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "record": "data002_replication_pilot_condition_v1",
        "key": condition_key(condition),
        "condition": asdict(condition),
        "terminal_state": state,
        "elapsed_seconds": elapsed,
        "completed_at": utc_now(),
        "artifact": artifact,
        "operational_metadata": metadata,
        "captured_output": captured_output,
        "error": error,
    }
    if _contains_forbidden_name(value):
        raise PilotRunnerError("checkpoint contains a forbidden predictive field")
    return value


def validate_checkpoint(
    checkpoint_path: Path,
    prediction_path: Path,
    condition: PilotCondition,
) -> dict[str, Any]:
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "record",
        "key",
        "condition",
        "terminal_state",
        "elapsed_seconds",
        "completed_at",
        "artifact",
        "operational_metadata",
        "captured_output",
        "error",
    }
    if not isinstance(checkpoint, dict) or set(checkpoint) != expected_fields:
        raise PilotRunnerError("checkpoint fields changed")
    if _contains_forbidden_name(checkpoint):
        raise PilotRunnerError("checkpoint contains a forbidden predictive field")
    if (
        checkpoint["schema_version"] != 1
        or checkpoint["record"] != "data002_replication_pilot_condition_v1"
        or checkpoint["key"] != condition_key(condition)
        or checkpoint["condition"] != asdict(condition)
        or checkpoint["terminal_state"] not in TERMINAL_STATES
    ):
        raise PilotRunnerError("checkpoint identity changed")
    elapsed = checkpoint.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or elapsed < 0
    ):
        raise PilotRunnerError("checkpoint elapsed time is invalid")
    completed_at = checkpoint.get("completed_at")
    if not isinstance(completed_at, str):
        raise PilotRunnerError("checkpoint completion time is invalid")
    try:
        parsed_time = datetime.fromisoformat(completed_at)
    except ValueError as exc:
        raise PilotRunnerError("checkpoint completion time is invalid") from exc
    if parsed_time.tzinfo is None or parsed_time.utcoffset() is None:
        raise PilotRunnerError("checkpoint completion time must be timezone-aware")
    _validate_captured_output(checkpoint.get("captured_output"))
    if checkpoint["terminal_state"] == "success":
        artifact = checkpoint["artifact"]
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != {"path", "bytes", "sha256"}
            or artifact.get("path") != prediction_path.name
            or not prediction_path.is_file()
            or not _nonnegative_integer(artifact.get("bytes"))
            or artifact.get("bytes") != prediction_path.stat().st_size
            or not _valid_sha256(artifact.get("sha256"))
            or artifact.get("sha256") != _sha256(prediction_path)
        ):
            raise PilotRunnerError("checkpoint artifact mismatch")
        metadata = checkpoint.get("operational_metadata")
        if not isinstance(metadata, Mapping) or set(metadata) != {
            "allocation_record",
            "warnings",
            "resource_observations",
            "operational_validation",
        }:
            raise PilotRunnerError("operational metadata fields changed")
        validation = metadata.get("operational_validation")
        if (
            not isinstance(validation, Mapping)
            or set(validation) != OPERATIONAL_VALIDATION_KEYS
        ):
            raise PilotRunnerError("operational validation fields changed")
        arrays = _read_npz(prediction_path)
        _, observed = _validate_executor_result(
            {
                "prediction_arrays": arrays,
                "allocation_record": metadata["allocation_record"],
                "warnings": metadata["warnings"],
                "resource_observations": metadata["resource_observations"],
            },
            condition,
        )
        if observed != metadata:
            raise PilotRunnerError("checkpoint operational validation mismatch")
        if checkpoint["error"] is not None:
            raise PilotRunnerError("successful checkpoint contains error")
    else:
        if checkpoint["artifact"] is not None or checkpoint[
            "operational_metadata"
        ] is not None:
            raise PilotRunnerError("unsuccessful checkpoint claims artifacts")
        error = checkpoint.get("error")
        if checkpoint["terminal_state"] == "failure":
            if (
                not isinstance(error, Mapping)
                or set(error) != {"type", "message_sha256", "traceback_sha256"}
                or not isinstance(error.get("type"), str)
                or not error["type"]
                or _contains_forbidden_name(error["type"])
                or not _valid_sha256(error.get("message_sha256"))
                or not _valid_sha256(error.get("traceback_sha256"))
            ):
                raise PilotRunnerError("failure error schema changed")
        elif error != {"reason": "condition_timeout"}:
            raise PilotRunnerError("timeout error schema changed")
    return checkpoint


def reconcile_artifacts(
    output_root: Path,
    terminal: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    checkpoints_dir = output_root / "checkpoints"
    predictions_dir = output_root / "predictions"
    expected = expected_conditions()
    checkpoint_files = {
        path.relative_to(checkpoints_dir).as_posix()
        for path in checkpoints_dir.rglob("*")
        if path.is_file()
    }
    prediction_files = {
        path.relative_to(predictions_dir).as_posix()
        for path in predictions_dir.rglob("*")
        if path.is_file()
    }
    expected_checkpoints = {f"{key}.json" for key in expected}
    successful = {
        key for key, value in terminal.items() if value["terminal_state"] == "success"
    }
    expected_predictions = {f"{key}.npz" for key in successful}
    mismatched: list[dict[str, str]] = []
    for key in sorted(terminal):
        try:
            validate_checkpoint(
                checkpoints_dir / f"{key}.json",
                predictions_dir / f"{key}.npz",
                expected[key],
            )
        except Exception as exc:
            mismatched.append({"key": key, "error_type": type(exc).__name__})
    all_root_files = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    allowed_root_files = {
        *(f"checkpoints/{name}" for name in expected_checkpoints),
        *(f"predictions/{name}" for name in expected_predictions),
        "operational_report.json",
        PILOT_LOCK_NAME,
    }
    retained_lock_records = sorted(
        relative
        for relative in all_root_files
        if "/" not in relative
        and relative.startswith(".replication_pilot.stale.")
        and relative.endswith(".json")
    )
    lock_record_errors: list[str] = []
    for relative in [PILOT_LOCK_NAME, *retained_lock_records]:
        path = output_root / relative
        if not path.exists():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping):
                raise PilotRunnerError("lock record is not an object")
            validate_pilot_lock_record(value)
        except Exception:
            lock_record_errors.append(relative)
    allowed_root_files.update(retained_lock_records)
    return {
        "unexpected_checkpoint_artifacts": sorted(
            checkpoint_files - expected_checkpoints
        ),
        "missing_checkpoint_artifacts": sorted(
            expected_checkpoints - checkpoint_files
        ),
        "unexpected_prediction_artifacts": sorted(
            prediction_files - expected_predictions
        ),
        "missing_prediction_artifacts": sorted(
            expected_predictions - prediction_files
        ),
        "orphan_prediction_artifacts": sorted(
            prediction_files - {f"{key}.npz" for key in successful}
        ),
        "mismatched_artifacts": mismatched,
        "unexpected_root_artifacts": sorted(all_root_files - allowed_root_files),
        "invalid_lock_records": lock_record_errors,
    }


REPORT_KEYS = {
    "schema_version",
    "record",
    "scientific_status",
    "status",
    "expected_key_count",
    "terminal_key_count",
    "successful_key_count",
    "reused_keys",
    "executed_keys",
    "reuse_count",
    "execution_count",
    "terminal_states",
    "reconciliation",
    "preflight",
    "runtime",
    "resources",
    "persistence",
    "allocation_invariants_pass",
    "scientific_measures_exposed",
    "sealed_array_values_exposed",
}
RECONCILIATION_KEYS = {
    "unexpected_checkpoint_artifacts",
    "missing_checkpoint_artifacts",
    "unexpected_prediction_artifacts",
    "missing_prediction_artifacts",
    "orphan_prediction_artifacts",
    "mismatched_artifacts",
    "unexpected_root_artifacts",
    "invalid_lock_records",
}


def validate_operational_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or set(report) != REPORT_KEYS:
        raise PilotRunnerError("operational report fields changed")
    if _contains_forbidden_name(report):
        raise PilotRunnerError("operational report contains forbidden leakage")
    if (
        report["schema_version"] != 1
        or report["record"] != "data002_replication_pilot_operational_report_v1"
        or report["scientific_status"]
        != "non_scientific_metric_blind_operational_only"
        or report["status"] not in {"pass", "fail"}
        or report["expected_key_count"] != 12
        or report["scientific_measures_exposed"] is not False
        or report["sealed_array_values_exposed"] is not False
        or not isinstance(report["allocation_invariants_pass"], bool)
    ):
        raise PilotRunnerError("operational report identity changed")
    for field in (
        "terminal_key_count",
        "successful_key_count",
        "reuse_count",
        "execution_count",
    ):
        if not _nonnegative_integer(report.get(field)):
            raise PilotRunnerError("operational report count is invalid")
    if (
        not isinstance(report.get("reused_keys"), list)
        or not isinstance(report.get("executed_keys"), list)
        or report["reuse_count"] != len(report["reused_keys"])
        or report["execution_count"] != len(report["executed_keys"])
        or not isinstance(report.get("terminal_states"), Mapping)
        or set(report["terminal_states"]) != set(expected_conditions())
        or any(
            state not in TERMINAL_STATES
            for state in report["terminal_states"].values()
        )
        or not isinstance(report.get("reconciliation"), Mapping)
        or set(report["reconciliation"]) != RECONCILIATION_KEYS
    ):
        raise PilotRunnerError("operational report accounting changed")
    expected_keys = set(expected_conditions())
    reused = report["reused_keys"]
    executed = report["executed_keys"]
    if (
        len(set(reused)) != len(reused)
        or len(set(executed)) != len(executed)
        or set(reused) & set(executed)
        or set(reused) | set(executed) != expected_keys
        or report["terminal_key_count"] != len(report["terminal_states"])
        or report["successful_key_count"]
        != sum(
            state == "success" for state in report["terminal_states"].values()
        )
    ):
        raise PilotRunnerError("operational report key accounting changed")
    reconciliation = report["reconciliation"]
    if any(
        not isinstance(value, list) for value in reconciliation.values()
    ):
        raise PilotRunnerError("operational report reconciliation changed")
    clean = not any(reconciliation.values())
    should_pass = (
        report["terminal_key_count"] == 12
        and report["successful_key_count"] == 12
        and clean
        and report["allocation_invariants_pass"]
    )
    if (report["status"] == "pass") != should_pass:
        raise PilotRunnerError("operational report status is inconsistent")
    expected_nested = {
        "preflight": {"free_disk_bytes", "minimum_free_disk_bytes"},
        "runtime": {
            "wall_seconds",
            "summed_condition_seconds",
            "maximum_condition_seconds",
        },
        "resources": {"maximum_observed_rss_bytes", "warning_count"},
        "persistence": {
            "condition_output_file_count",
            "condition_output_bytes",
            "operational_report_file_count",
            "active_lock_file_count",
            "retained_stale_lock_file_count",
            "total_files_before_report_promotion",
            "total_bytes_before_report_promotion",
        },
    }
    for section, keys in expected_nested.items():
        value = report.get(section)
        if not isinstance(value, Mapping) or set(value) != keys:
            raise PilotRunnerError(f"operational report {section} fields changed")
        for nested_value in value.values():
            if (
                isinstance(nested_value, bool)
                or not isinstance(nested_value, (int, float))
                or not math.isfinite(float(nested_value))
                or nested_value < 0
            ):
                raise PilotRunnerError(
                    f"operational report {section} value is invalid"
                )
    return report


def run_pilot(
    *,
    launch_path: Path,
    executor: PilotExecutor,
    project_root: Path = Path.cwd(),
    startup_verifier: Callable[
        [Path, Path], PilotStartupPolicy
    ] = verify_pilot_startup,
    timeout_executor: Callable[
        [PilotExecutor, PilotCondition, int], tuple[str, Any, float]
    ] = execute_with_timeout,
    checkpoint_writer: Callable[[Path, Mapping[str, Any]], None] = _atomic_json,
) -> dict[str, Any]:
    """Run or resume exactly 12 metric-blind pilot conditions."""

    wall_started = time.monotonic()
    project_root = project_root.resolve()
    policy = startup_verifier(project_root, launch_path)
    if not isinstance(policy, PilotStartupPolicy):
        raise PilotRunnerError("startup verifier returned an invalid policy")
    timeout_seconds = policy.timeout_seconds
    minimum_disk = policy.minimum_free_disk_bytes
    output_root = policy.output_root.resolve()
    key_manifest = json.loads(
        (
            project_root
            / "results/provenance/replication_pilot_keys_v1.json"
        ).read_text(encoding="utf-8")
    )
    validate_key_manifest(key_manifest, scope="pilot")
    expected = expected_conditions()
    if set(expected) != {item["key"] for item in key_manifest["conditions"]}:
        raise PilotRunnerError("production pilot keys differ from frozen manifest")
    preflight_free_disk = shutil.disk_usage(project_root).free
    if preflight_free_disk < minimum_disk:
        raise PilotRunnerError("pilot free-disk gate failed")
    checkpoints = output_root / "checkpoints"
    predictions = output_root / "predictions"
    terminal: dict[str, dict[str, Any]] = {}
    reused: list[str] = []
    executed: list[str] = []
    with exclusive_pilot_run_lock(output_root):
        checkpoints.mkdir(parents=True, exist_ok=True)
        predictions.mkdir(parents=True, exist_ok=True)
        existing_report = output_root / "operational_report.json"
        if existing_report.exists():
            validate_operational_report(existing_report)
        for key, condition in expected.items():
            checkpoint_path = checkpoints / f"{key}.json"
            prediction_path = predictions / f"{key}.npz"
            if checkpoint_path.exists():
                checkpoint = validate_checkpoint(
                    checkpoint_path, prediction_path, condition
                )
                terminal[key] = checkpoint
                reused.append(key)
                continue
            if prediction_path.exists():
                raise PilotRunnerError(
                    f"refusing to overwrite orphan prediction artifact: {key}"
                )
            state, payload, elapsed = timeout_executor(
                executor, condition, timeout_seconds
            )
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed))
                or elapsed < 0
            ):
                raise PilotRunnerError("executor elapsed time is invalid")
            executed.append(key)
            if state == "success":
                captured = dict(EMPTY_CAPTURED_OUTPUT)
                result = payload
                if isinstance(payload, Mapping) and "_worker_result" in payload:
                    result = payload["_worker_result"]
                    captured = _validate_captured_output(
                        payload["_captured_output"]
                    )
                arrays, metadata = _validate_executor_result(result, condition)
                _atomic_npz(prediction_path, arrays)
                artifact = {
                    "path": prediction_path.name,
                    "bytes": prediction_path.stat().st_size,
                    "sha256": _sha256(prediction_path),
                }
                checkpoint = _checkpoint(
                    condition,
                    "success",
                    elapsed,
                    artifact=artifact,
                    metadata=metadata,
                    captured_output=captured,
                    error=None,
                )
            elif state == "failure":
                captured = dict(EMPTY_CAPTURED_OUTPUT)
                error = payload
                if isinstance(payload, Mapping) and "_worker_error" in payload:
                    error = payload["_worker_error"]
                    captured = _validate_captured_output(
                        payload["_captured_output"]
                    )
                checkpoint = _checkpoint(
                    condition,
                    "failure",
                    elapsed,
                    artifact=None,
                    metadata=None,
                    captured_output=captured,
                    error=error,
                )
            elif state == "timeout":
                checkpoint = _checkpoint(
                    condition,
                    "timeout",
                    elapsed,
                    artifact=None,
                    metadata=None,
                    captured_output=dict(EMPTY_CAPTURED_OUTPUT),
                    error=payload,
                )
            else:
                raise PilotRunnerError("executor returned an invalid terminal state")
            checkpoint_writer(checkpoint_path, checkpoint)
            validate_checkpoint(checkpoint_path, prediction_path, condition)
            terminal[key] = checkpoint

        reconciliation = reconcile_artifacts(output_root, terminal)
        passing = sum(
            value["terminal_state"] == "success" for value in terminal.values()
        )
        clean = not any(reconciliation.values())
        condition_times = [
            float(value["elapsed_seconds"]) for value in terminal.values()
        ]
        maximum_rss = max(
            (
                value["operational_metadata"]["resource_observations"][
                    "process_rss_bytes"
                ]
                for value in terminal.values()
                if value["terminal_state"] == "success"
            ),
            default=0,
        )
        warning_count = sum(
            len(value["operational_metadata"]["warnings"])
            for value in terminal.values()
            if value["terminal_state"] == "success"
        )
        condition_output_files = [
            path
            for namespace in (checkpoints, predictions)
            for path in namespace.rglob("*")
            if path.is_file()
        ]
        retained_stale_locks = list(
            output_root.glob(".replication_pilot.stale.*.json")
        )
        files_before_report = [
            path
            for path in output_root.rglob("*")
            if path.is_file() and path.name != "operational_report.json"
        ]
        allocation_invariants_pass = bool(
            passing == 12
            and all(
                value["operational_metadata"]["allocation_record"][
                    "final_count"
                ]
                == value["operational_metadata"]["allocation_record"][
                    "original_count"
                ]
                + value["operational_metadata"]["allocation_record"][
                    "duplicate_count"
                ]
                for value in terminal.values()
            )
        )
        report = {
            "schema_version": 1,
            "record": "data002_replication_pilot_operational_report_v1",
            "scientific_status": "non_scientific_metric_blind_operational_only",
            "status": (
                "pass"
                if len(terminal) == 12 and passing == 12 and clean
                else "fail"
            ),
            "expected_key_count": 12,
            "terminal_key_count": len(terminal),
            "successful_key_count": passing,
            "reused_keys": reused,
            "executed_keys": executed,
            "reuse_count": len(reused),
            "execution_count": len(executed),
            "terminal_states": {
                key: value["terminal_state"] for key, value in terminal.items()
            },
            "reconciliation": reconciliation,
            "preflight": {
                "free_disk_bytes": preflight_free_disk,
                "minimum_free_disk_bytes": minimum_disk,
            },
            "runtime": {
                "wall_seconds": time.monotonic() - wall_started,
                "summed_condition_seconds": sum(condition_times),
                "maximum_condition_seconds": max(condition_times, default=0),
            },
            "resources": {
                "maximum_observed_rss_bytes": maximum_rss,
                "warning_count": warning_count,
            },
            "persistence": {
                "condition_output_file_count": len(condition_output_files),
                "condition_output_bytes": sum(
                    path.stat().st_size for path in condition_output_files
                ),
                "operational_report_file_count": 1,
                "active_lock_file_count": int(
                    (output_root / PILOT_LOCK_NAME).is_file()
                ),
                "retained_stale_lock_file_count": len(retained_stale_locks),
                "total_files_before_report_promotion": len(files_before_report),
                "total_bytes_before_report_promotion": sum(
                    path.stat().st_size for path in files_before_report
                ),
            },
            "allocation_invariants_pass": allocation_invariants_pass,
            "scientific_measures_exposed": False,
            "sealed_array_values_exposed": False,
        }
        if _contains_forbidden_name(report):
            raise PilotRunnerError("operational report contains forbidden fields")
        _atomic_json(output_root / "operational_report.json", report)
        return validate_operational_report(
            output_root / "operational_report.json"
        )
