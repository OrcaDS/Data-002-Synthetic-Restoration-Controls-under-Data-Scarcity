"""Checkpointed runner for the fixed Data 002 compatibility replay.

The production entry point is authorization-gated. Tests inject toy inputs and
executors, but the runner's condition universe remains the frozen 16 keys.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import shutil
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from data002.reconstruction import (
    CompatibilityCondition,
    compare_prediction_archives,
    compatibility_conditions,
    file_sha256,
    load_datasets,
    reference_archive_path,
    run_baseline_condition,
    split_dataset,
)
from data002.reference_bundle import verify_reference_bundle
from data002.replay_provenance import (
    CONTRACT_PATH,
    ENVIRONMENT_INVENTORY_PATH,
    IMPLEMENTATION_MANIFEST_PATH,
    exclusive_run_lock,
    validate_authorized_launch,
    verify_active_environment,
    verify_implementation_manifest,
)

SCHEMA_VERSION = 1
TERMINAL_STATES = {"success", "failure", "timeout"}
REQUIRED_ARRAYS = {"test_index", "y_true", "probability"}
LOCKED_ACCEPTANCE = {
    "all_conditions_must_pass": True,
    "ordered_test_indices": "exact",
    "test_labels": "exact",
    "probability_shape": "exact",
    "probability_dtype": "float32",
    "probability_rtol": 0.0,
    "probability_atol": 1e-6,
    "threshold": 0.5,
    "thresholded_predictions": "exact",
    "roc_auc_absolute_tolerance": 1e-12,
    "average_precision_absolute_tolerance": 1e-12,
    "precision_recall_f1_accuracy": "exact",
}


class ConditionExecutor(Protocol):
    def __call__(self, condition: CompatibilityCondition) -> dict[str, np.ndarray]: ...


class InputVerifier(Protocol):
    def __call__(
        self,
        bundle_root: Path,
        diabetes_path: Path,
        cleveland_path: Path,
    ) -> dict[str, Any]: ...


class RunnerConfigurationError(ValueError):
    """The prospective runner configuration is invalid."""


@dataclass(frozen=True)
class RealCompatibilityExecutor:
    """Picklable real executor used only after a launch is authorized."""

    diabetes_path: Path
    cleveland_path: Path

    def __call__(self, condition: CompatibilityCondition) -> dict[str, np.ndarray]:
        datasets = load_datasets(self.diabetes_path, self.cleveland_path)
        return run_baseline_condition(datasets[condition.dataset], condition)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def condition_key(condition: CompatibilityCondition) -> str:
    percentage = round(condition.scarcity_fraction * 100)
    return (
        f"{condition.dataset}__{percentage:03d}pct__seed_{condition.seed:02d}"
        f"__{condition.model}"
    )


def expected_condition_map() -> dict[str, CompatibilityCondition]:
    return {condition_key(condition): condition for condition in compatibility_conditions()}


def account_expected_keys(
    terminal_keys: set[str],
    observed_checkpoint_keys: set[str],
) -> dict[str, Any]:
    """Reconcile exact keys without treating partial or majority success as enough."""

    expected_keys = set(expected_condition_map())
    return {
        "expected_key_count": len(expected_keys),
        "terminal_key_count": len(terminal_keys & expected_keys),
        "missing_keys": sorted(expected_keys - terminal_keys),
        "unexpected_checkpoint_keys": sorted(observed_checkpoint_keys - expected_keys),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RunnerConfigurationError(f"{path} must contain a JSON object")
    return value


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("execution_authorization") != "none":
        raise RunnerConfigurationError(
            "compatibility contract execution_authorization must remain 'none'"
        )
    if contract.get("acceptance") != LOCKED_ACCEPTANCE:
        raise RunnerConfigurationError("compatibility acceptance criteria changed")

    observed: set[tuple[str, float, int, str]] = set()
    matrix = contract.get("matrix")
    if not isinstance(matrix, Mapping):
        raise RunnerConfigurationError("compatibility matrix is missing")
    for dataset, values in matrix.items():
        if not isinstance(values, Mapping):
            raise RunnerConfigurationError(f"invalid matrix entry for {dataset}")
        for fraction in values.get("scarcity_fractions", []):
            for seed in values.get("seeds", []):
                for model in values.get("models", []):
                    observed.add((str(dataset), float(fraction), int(seed), str(model)))
    expected = {
        (item.dataset, item.scarcity_fraction, item.seed, item.model)
        for item in compatibility_conditions()
    }
    if observed != expected or len(observed) != 16:
        raise RunnerConfigurationError("contract matrix is not the fixed 16 conditions")


def verify_replay_inputs(
    bundle_root: Path,
    diabetes_path: Path,
    cleveland_path: Path,
) -> dict[str, Any]:
    """Verify bundle hashes, raw hashes, and every split identity before fitting."""

    bundle_report = verify_reference_bundle(bundle_root)
    report: dict[str, Any] = {
        "status": "fail",
        "bundle_status": bundle_report["status"],
        "bundle_manifest_sha256": bundle_report.get("manifest_sha256"),
        "datasets": {},
        "conditions": [],
    }
    if bundle_report["status"] != "pass":
        return report

    datasets = load_datasets(diabetes_path, cleveland_path)
    report["datasets"] = {
        name: {"path": str(spec.source_path), "sha256": file_sha256(spec.source_path)}
        for name, spec in sorted(datasets.items())
    }
    split_cache: dict[tuple[str, int], Any] = {}
    for condition in compatibility_conditions():
        cache_key = (condition.dataset, condition.seed)
        split = split_cache.setdefault(
            cache_key,
            split_dataset(datasets[condition.dataset], condition.seed),
        )
        with np.load(reference_archive_path(bundle_root, condition), allow_pickle=False) as ref:
            index_exact = bool(
                np.array_equal(ref["test_index"], split.X_test.index.to_numpy())
            )
            labels_exact = bool(
                np.array_equal(ref["y_true"], split.y_test.to_numpy(dtype=np.int8))
            )
        report["conditions"].append(
            {
                "key": condition_key(condition),
                "test_index_exact": index_exact,
                "y_true_exact": labels_exact,
                "status": "pass" if index_exact and labels_exact else "fail",
            }
        )
    report["status"] = (
        "pass"
        if len(report["conditions"]) == 16
        and all(item["status"] == "pass" for item in report["conditions"])
        else "fail"
    )
    return report


def _worker(
    result_queue: multiprocessing.Queue[Any],
    executor: ConditionExecutor,
    condition: CompatibilityCondition,
) -> None:
    try:
        result_queue.put(("success", executor(condition)))
    except BaseException as exc:
        result_queue.put(
            (
                "failure",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        )


def execute_with_timeout(
    executor: ConditionExecutor,
    condition: CompatibilityCondition,
    timeout_seconds: int,
) -> tuple[str, Any, float]:
    """Execute one picklable condition callable in a terminable child process."""

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
        return "timeout", {"message": f"exceeded {timeout_seconds} seconds"}, (
            time.monotonic() - started
        )
    finally:
        result_queue.close()
    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    return state, payload, time.monotonic() - started


def _archive_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != REQUIRED_ARRAYS:
            raise ValueError(
                f"archive keys must be exactly {sorted(REQUIRED_ARRAYS)}, "
                f"found {sorted(archive.files)}"
            )
        return {name: archive[name] for name in sorted(REQUIRED_ARRAYS)}


def _validate_candidate_arrays(arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if set(arrays) != REQUIRED_ARRAYS:
        raise ValueError(
            f"executor keys must be exactly {sorted(REQUIRED_ARRAYS)}, "
            f"found {sorted(arrays)}"
        )
    normalized = {name: np.asarray(arrays[name]) for name in REQUIRED_ARRAYS}
    if any(value.ndim != 1 for value in normalized.values()):
        raise ValueError("executor arrays must be one-dimensional")
    if len({len(value) for value in normalized.values()}) != 1:
        raise ValueError("executor arrays must have equal lengths")
    if not len(normalized["test_index"]):
        raise ValueError("executor arrays must not be empty")
    return normalized


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
        if temporary.exists():
            temporary.unlink()


def _atomic_archive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                test_index=arrays["test_index"],
                y_true=arrays["y_true"],
                probability=arrays["probability"],
            )
            stream.flush()
            os.fsync(stream.fileno())
        _archive_arrays(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _condition_identity(condition: CompatibilityCondition) -> dict[str, Any]:
    return asdict(condition)


def _new_terminal_checkpoint(
    condition: CompatibilityCondition,
    terminal_state: str,
    elapsed_seconds: float,
    *,
    artifact: Mapping[str, Any] | None,
    compatibility_report: Mapping[str, Any] | None,
    error: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record": "data002_compatibility_condition_v1",
        "key": condition_key(condition),
        "condition": _condition_identity(condition),
        "terminal_state": terminal_state,
        "elapsed_seconds": elapsed_seconds,
        "completed_at": utc_now(),
        "artifact": artifact,
        "compatibility_report": compatibility_report,
        "error": error,
    }


def _validate_checkpoint(
    checkpoint_path: Path,
    artifact_path: Path,
    reference_path: Path,
    condition: CompatibilityCondition,
) -> dict[str, Any]:
    checkpoint = _load_json(checkpoint_path)
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint schema version mismatch")
    if checkpoint.get("record") != "data002_compatibility_condition_v1":
        raise ValueError("checkpoint record type mismatch")
    if checkpoint.get("key") != condition_key(condition):
        raise ValueError("checkpoint key mismatch")
    if checkpoint.get("condition") != _condition_identity(condition):
        raise ValueError("checkpoint condition mismatch")
    terminal_state = checkpoint.get("terminal_state")
    if terminal_state not in TERMINAL_STATES:
        raise ValueError("checkpoint terminal state is invalid")
    elapsed_seconds = checkpoint.get("elapsed_seconds")
    if (
        not isinstance(elapsed_seconds, (int, float))
        or isinstance(elapsed_seconds, bool)
        or elapsed_seconds < 0
    ):
        raise ValueError("checkpoint elapsed time is invalid")
    if not isinstance(checkpoint.get("completed_at"), str):
        raise ValueError("checkpoint completion timestamp is invalid")

    if terminal_state == "success":
        artifact = checkpoint.get("artifact")
        if not isinstance(artifact, Mapping):
            raise ValueError("successful checkpoint has no artifact")
        if artifact.get("path") != artifact_path.name:
            raise ValueError("checkpoint artifact path mismatch")
        if not artifact_path.is_file():
            raise ValueError("checkpoint artifact is missing")
        if artifact.get("bytes") != artifact_path.stat().st_size:
            raise ValueError("checkpoint artifact size mismatch")
        if artifact.get("sha256") != _sha256(artifact_path):
            raise ValueError("checkpoint artifact hash mismatch")
        reference = _archive_arrays(reference_path)
        candidate = _archive_arrays(artifact_path)
        comparison = compare_prediction_archives(reference, candidate)
        if checkpoint.get("compatibility_report") != comparison:
            raise ValueError("checkpoint compatibility report does not revalidate")
        if checkpoint.get("error") is not None:
            raise ValueError("successful checkpoint must not contain an error")
    else:
        if checkpoint.get("artifact") is not None:
            raise ValueError("failed or timed-out checkpoint must not claim an artifact")
        if checkpoint.get("compatibility_report") is not None:
            raise ValueError(
                "failed or timed-out checkpoint must not claim a comparison"
            )
        if not isinstance(checkpoint.get("error"), Mapping):
            raise ValueError("failed or timed-out checkpoint has no error")
    return checkpoint


def reconcile_artifacts(
    *,
    checkpoints_dir: Path,
    predictions_dir: Path,
    terminal_by_key: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, CompatibilityCondition],
    bundle_root: Path,
) -> dict[str, Any]:
    """Reconcile both persistence namespaces and revalidate successful pairs."""

    checkpoint_files = {
        path.relative_to(checkpoints_dir).as_posix()
        for path in checkpoints_dir.rglob("*.json")
        if path.is_file()
    }
    prediction_files = {
        path.relative_to(predictions_dir).as_posix()
        for path in predictions_dir.rglob("*.npz")
        if path.is_file()
    }
    nested_checkpoint_artifacts = sorted(
        relative for relative in checkpoint_files if "/" in relative
    )
    nested_prediction_artifacts = sorted(
        relative for relative in prediction_files if "/" in relative
    )
    observed_checkpoint_keys = {
        Path(relative).stem
        for relative in checkpoint_files
        if "/" not in relative
    }
    observed_prediction_keys = {
        Path(relative).stem
        for relative in prediction_files
        if "/" not in relative
    }
    expected_keys = set(expected)
    successful_keys = {
        key
        for key, checkpoint in terminal_by_key.items()
        if checkpoint.get("terminal_state") == "success"
    }
    mismatched: list[dict[str, str]] = []
    for key in sorted(successful_keys & expected_keys):
        condition = expected[key]
        try:
            _validate_checkpoint(
                checkpoints_dir / f"{key}.json",
                predictions_dir / f"{key}.npz",
                reference_archive_path(bundle_root, condition),
                condition,
            )
        except Exception as exc:
            mismatched.append(
                {"key": key, "error": f"{type(exc).__name__}: {exc}"}
            )
    return {
        "observed_checkpoint_key_count": len(observed_checkpoint_keys),
        "observed_prediction_key_count": len(observed_prediction_keys),
        "observed_checkpoint_artifact_count": len(checkpoint_files),
        "observed_prediction_artifact_count": len(prediction_files),
        "unexpected_checkpoint_keys": sorted(observed_checkpoint_keys - expected_keys),
        "unexpected_prediction_keys": sorted(observed_prediction_keys - expected_keys),
        "unexpected_nested_checkpoint_artifacts": nested_checkpoint_artifacts,
        "unexpected_nested_prediction_artifacts": nested_prediction_artifacts,
        "orphan_prediction_keys": sorted(observed_prediction_keys - successful_keys),
        "missing_prediction_keys": sorted(successful_keys - observed_prediction_keys),
        "mismatched_predictions": mismatched,
    }


def run_compatibility_replay(
    *,
    bundle_root: Path,
    diabetes_path: Path,
    cleveland_path: Path,
    output_root: Path,
    launch_path: Path,
    executor: ConditionExecutor,
    project_root: Path = Path.cwd(),
    input_verifier: InputVerifier = verify_replay_inputs,
    timeout_executor: Callable[
        [ConditionExecutor, CompatibilityCondition, int], tuple[str, Any, float]
    ] = execute_with_timeout,
) -> dict[str, Any]:
    """Run or safely resume the exact 16-condition replay sequentially."""

    project_root = project_root.resolve()
    contract_path = project_root / CONTRACT_PATH
    inventory_path = project_root / ENVIRONMENT_INVENTORY_PATH
    manifest_path = project_root / IMPLEMENTATION_MANIFEST_PATH
    contract = _load_json(contract_path)
    launch = _load_json(launch_path.resolve())
    validate_contract(contract)
    implementation_report = verify_implementation_manifest(
        project_root,
        manifest_path,
    )
    timeout_seconds, minimum_free_disk_bytes = validate_authorized_launch(
        launch,
        project_root=project_root,
        contract_path=contract_path,
        inventory_path=inventory_path,
        manifest_path=manifest_path,
        implementation_report=implementation_report,
    )
    environment_report = verify_active_environment(
        inventory_path,
        project_root / "requirements-lock.txt",
    )
    expected = expected_condition_map()
    if len(expected) != 16:
        raise RunnerConfigurationError("implemented expected-key count is not 16")

    output_root = output_root.resolve()
    with exclusive_run_lock(output_root) as lock_record:
        return _run_locked_replay(
            bundle_root=bundle_root,
            diabetes_path=diabetes_path,
            cleveland_path=cleveland_path,
            output_root=output_root,
            executor=executor,
            input_verifier=input_verifier,
            timeout_executor=timeout_executor,
            expected=expected,
            timeout_seconds=timeout_seconds,
            minimum_free_disk_bytes=minimum_free_disk_bytes,
            environment_report=environment_report,
            implementation_report=implementation_report,
            lock_record=lock_record,
        )


def _run_locked_replay(
    *,
    bundle_root: Path,
    diabetes_path: Path,
    cleveland_path: Path,
    output_root: Path,
    executor: ConditionExecutor,
    input_verifier: InputVerifier,
    timeout_executor: Callable[
        [ConditionExecutor, CompatibilityCondition, int], tuple[str, Any, float]
    ],
    expected: Mapping[str, CompatibilityCondition],
    timeout_seconds: int,
    minimum_free_disk_bytes: int,
    environment_report: Mapping[str, Any],
    implementation_report: Mapping[str, Any],
    lock_record: Mapping[str, Any],
) -> dict[str, Any]:
    free_bytes = shutil.disk_usage(output_root).free
    if free_bytes < minimum_free_disk_bytes:
        raise OSError(
            f"free-disk preflight failed: require {minimum_free_disk_bytes}, "
            f"found {free_bytes}"
        )

    input_report = input_verifier(
        bundle_root.resolve(),
        diabetes_path.resolve(),
        cleveland_path.resolve(),
    )
    if input_report.get("status") != "pass":
        raise ValueError("reference-bundle or raw-input verification failed")

    checkpoints_dir = output_root / "checkpoints"
    predictions_dir = output_root / "predictions"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    condition_reports: list[dict[str, Any]] = []
    invalid_checkpoints: list[dict[str, str]] = []
    reused_keys: list[str] = []
    executed_keys: list[str] = []
    for key, condition in expected.items():
        checkpoint_path = checkpoints_dir / f"{key}.json"
        artifact_path = predictions_dir / f"{key}.npz"
        reference_path = reference_archive_path(bundle_root, condition)
        if checkpoint_path.is_file():
            try:
                checkpoint = _validate_checkpoint(
                    checkpoint_path,
                    artifact_path,
                    reference_path,
                    condition,
                )
            except Exception as exc:
                invalid_checkpoints.append(
                    {"key": key, "error": f"{type(exc).__name__}: {exc}"}
                )
            else:
                condition_reports.append(checkpoint)
                reused_keys.append(key)
                continue

        state, payload, elapsed = timeout_executor(executor, condition, timeout_seconds)
        executed_keys.append(key)
        if state == "success":
            try:
                arrays = _validate_candidate_arrays(payload)
                reference = _archive_arrays(reference_path)
                comparison = compare_prediction_archives(reference, arrays)
                _atomic_archive(artifact_path, arrays)
                artifact = {
                    "path": artifact_path.name,
                    "bytes": artifact_path.stat().st_size,
                    "sha256": _sha256(artifact_path),
                }
                checkpoint = _new_terminal_checkpoint(
                    condition,
                    "success",
                    elapsed,
                    artifact=artifact,
                    compatibility_report=comparison,
                    error=None,
                )
            except Exception as exc:
                checkpoint = _new_terminal_checkpoint(
                    condition,
                    "failure",
                    elapsed,
                    artifact=None,
                    compatibility_report=None,
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
        elif state in {"failure", "timeout"}:
            error = payload if isinstance(payload, Mapping) else {"message": str(payload)}
            checkpoint = _new_terminal_checkpoint(
                condition,
                state,
                elapsed,
                artifact=None,
                compatibility_report=None,
                error=error,
            )
        else:
            raise RuntimeError(f"timeout executor returned unknown state {state!r}")
        _atomic_json(checkpoint_path, checkpoint)
        condition_reports.append(checkpoint)

    terminal_by_key = {item["key"]: item for item in condition_reports}
    observed_checkpoint_keys = {path.stem for path in checkpoints_dir.glob("*.json")}
    accounting = account_expected_keys(set(terminal_by_key), observed_checkpoint_keys)
    missing_keys = accounting["missing_keys"]
    passing_keys = sorted(
        key
        for key, item in terminal_by_key.items()
        if item["terminal_state"] == "success"
        and item["compatibility_report"]["status"] == "pass"
    )
    artifact_reconciliation = reconcile_artifacts(
        checkpoints_dir=checkpoints_dir,
        predictions_dir=predictions_dir,
        terminal_by_key=terminal_by_key,
        expected=expected,
        bundle_root=bundle_root,
    )
    persistence_clean = not any(
        artifact_reconciliation[field]
        for field in (
            "unexpected_checkpoint_keys",
            "unexpected_prediction_keys",
            "unexpected_nested_checkpoint_artifacts",
            "unexpected_nested_prediction_artifacts",
            "orphan_prediction_keys",
            "missing_prediction_keys",
            "mismatched_predictions",
        )
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record": "data002_compatibility_replay_report_v1",
        "status": "pass"
        if len(passing_keys) == 16
        and not missing_keys
        and persistence_clean
        else "fail",
        "all_conditions_must_pass": True,
        "acceptance": dict(LOCKED_ACCEPTANCE),
        "expected_key_count": accounting["expected_key_count"],
        "expected_keys": sorted(expected),
        "terminal_key_count": accounting["terminal_key_count"],
        "passing_key_count": len(passing_keys),
        "missing_keys": missing_keys,
        "unexpected_checkpoint_keys": artifact_reconciliation[
            "unexpected_checkpoint_keys"
        ],
        "artifact_reconciliation": artifact_reconciliation,
        "invalid_checkpoints_replaced": invalid_checkpoints,
        "reused_keys": reused_keys,
        "executed_keys": executed_keys,
        "resources": {
            "max_parallel_conditions": 1,
            "condition_timeout_seconds": timeout_seconds,
            "minimum_free_disk_bytes": minimum_free_disk_bytes,
            "free_disk_bytes_at_preflight": free_bytes,
        },
        "input_verification": input_report,
        "environment_verification": dict(environment_report),
        "implementation_verification": dict(implementation_report),
        "run_lock": dict(lock_record),
        "conditions": [terminal_by_key[key] for key in sorted(terminal_by_key)],
        "completed_at": utc_now(),
    }
    _atomic_json(output_root / "compatibility_report.json", report)
    return report
