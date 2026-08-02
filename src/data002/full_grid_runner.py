"""Authorization-gated, metric-blind runner for the frozen 540-condition grid."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping

from data002.full_grid_provenance import (
    ACCEPTED_PILOT_REPORT_SHA256,
    FULL_LOCK_NAME,
    FullGridStartupPolicy,
    exclusive_full_grid_run_lock,
    validate_full_grid_lock_record,
    verify_full_grid_startup,
)
from data002.pilot_runner import (
    EMPTY_CAPTURED_OUTPUT,
    TERMINAL_STATES,
    PilotCondition,
    PilotExecutor,
    PilotRunnerError,
    RealPilotExecutor,
    _atomic_json,
    _atomic_npz,
    _checkpoint,
    _contains_forbidden_name,
    _sha256,
    _validate_captured_output,
    _validate_executor_result,
    execute_with_timeout,
    utc_now,
    validate_checkpoint as validate_pilot_checkpoint,
    validate_operational_report as validate_pilot_report,
)
from data002.replication_design import (
    full_grid_records,
    pilot_records,
    validate_key_manifest,
)

RealFullGridExecutor = RealPilotExecutor

EXPECTED_COUNT = 540
PILOT_REUSE_COUNT = 12
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
REPORT_KEYS = {
    "schema_version", "record", "scientific_status", "status",
    "expected_key_count", "terminal_key_count", "successful_key_count",
    "failure_keys", "timeout_keys", "pilot_reused_keys", "resumed_keys",
    "executed_keys", "pilot_reuse_count", "resume_count", "execution_count",
    "terminal_states", "reconciliation", "preflight", "runtime", "resources",
    "persistence", "allocation_invariants_pass", "scientific_measures_exposed",
    "sealed_array_values_exposed", "upstream_metric_tables_accessed",
}


class FullGridRunnerError(PilotRunnerError):
    """Full-grid execution or persistence violates its operational contract."""


def expected_conditions() -> dict[str, PilotCondition]:
    return {
        record["key"]: PilotCondition(
            record["dataset"], record["retained_fraction"],
            record["seed"], record["model"],
        )
        for record in full_grid_records()
    }


def pilot_keys() -> list[str]:
    return [record["key"] for record in pilot_records()]


def _full_checkpoint(
    condition: PilotCondition,
    state: str,
    elapsed: float,
    *,
    artifact: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    captured_output: Mapping[str, Any] | None,
    error: Mapping[str, Any] | None,
) -> dict[str, Any]:
    value = _checkpoint(
        condition, state, elapsed, artifact=artifact, metadata=metadata,
        captured_output=captured_output, error=error,
    )
    value["record"] = "data002_replication_full_condition_v1"
    if _contains_forbidden_name(value):
        raise FullGridRunnerError("full-grid checkpoint contains forbidden leakage")
    return value


def validate_full_checkpoint(
    checkpoint_path: Path,
    prediction_path: Path,
    condition: PilotCondition,
    *,
    allow_pilot_record: bool = False,
) -> dict[str, Any]:
    value = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if allow_pilot_record and value.get("record") == "data002_replication_pilot_condition_v1":
        return validate_pilot_checkpoint(checkpoint_path, prediction_path, condition)
    if value.get("record") != "data002_replication_full_condition_v1":
        raise FullGridRunnerError("full-grid checkpoint record changed")
    pilot_form = dict(value)
    pilot_form["record"] = "data002_replication_pilot_condition_v1"
    with tempfile.TemporaryDirectory(prefix="data002-full-checkpoint-validation-") as raw:
        temporary = Path(raw) / checkpoint_path.name
        temporary.write_text(
            json.dumps(pilot_form, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        validate_pilot_checkpoint(temporary, prediction_path, condition)
    return value


def validate_pilot_reuse_source(
    project_root: Path, pilot_output_root: Path
) -> dict[str, dict[str, Any]]:
    report_path = pilot_output_root / "operational_report.json"
    if not report_path.is_file() or _sha256(report_path) != ACCEPTED_PILOT_REPORT_SHA256:
        raise FullGridRunnerError("accepted pilot operational-report hash changed")
    report = validate_pilot_report(report_path)
    expected_pilot = pilot_keys()
    expected_full = set(expected_conditions())
    if (
        report["status"] != "pass"
        or report["expected_key_count"] != PILOT_REUSE_COUNT
        or report["terminal_key_count"] != PILOT_REUSE_COUNT
        or report["successful_key_count"] != PILOT_REUSE_COUNT
        or report["reuse_count"] != 0
        or report["reused_keys"] != []
        or set(report["terminal_states"]) != set(expected_pilot)
        or any(state != "success" for state in report["terminal_states"].values())
        or any(report["reconciliation"].values())
        or not report["allocation_invariants_pass"]
        or not set(expected_pilot) < expected_full
    ):
        raise FullGridRunnerError("accepted pilot is ineligible for full-grid reuse")
    validated: dict[str, dict[str, Any]] = {}
    for key in expected_pilot:
        checkpoint = pilot_output_root / "checkpoints" / f"{key}.json"
        prediction = pilot_output_root / "predictions" / f"{key}.npz"
        validated[key] = validate_pilot_checkpoint(
            checkpoint, prediction, expected_conditions()[key]
        )
    all_files = {
        path.relative_to(pilot_output_root).as_posix()
        for path in pilot_output_root.rglob("*") if path.is_file()
    }
    expected_files = {
        "operational_report.json",
        *(f"checkpoints/{key}.json" for key in expected_pilot),
        *(f"predictions/{key}.npz" for key in expected_pilot),
    }
    if all_files != expected_files:
        raise FullGridRunnerError("accepted pilot output root no longer reconciles")
    return validated


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer, 1_048_576)
            writer.flush()
            os.fsync(writer.fileno())
        if temporary.stat().st_size != source.stat().st_size or _sha256(temporary) != _sha256(source):
            raise FullGridRunnerError("pilot artifact copy is not byte-identical")
        os.replace(temporary, destination)
        if _sha256(destination) != _sha256(source):
            raise FullGridRunnerError("promoted pilot artifact hash changed")
    finally:
        temporary.unlink(missing_ok=True)


def _copy_pilot_reuse(
    pilot_root: Path, output_root: Path, validated: Mapping[str, Mapping[str, Any]]
) -> None:
    for key in pilot_keys():
        source_checkpoint = pilot_root / "checkpoints" / f"{key}.json"
        source_prediction = pilot_root / "predictions" / f"{key}.npz"
        destination_checkpoint = output_root / "checkpoints" / f"{key}.json"
        destination_prediction = output_root / "predictions" / f"{key}.npz"
        checkpoint_exists = destination_checkpoint.exists()
        prediction_exists = destination_prediction.exists()
        if checkpoint_exists != prediction_exists:
            raise FullGridRunnerError(
                f"one-sided pre-existing pilot reuse state: {key}"
            )
        if checkpoint_exists:
            for source, destination in (
                (source_checkpoint, destination_checkpoint),
                (source_prediction, destination_prediction),
            ):
                if (
                    destination.stat().st_size != source.stat().st_size
                    or _sha256(destination) != _sha256(source)
                ):
                    raise FullGridRunnerError(
                        f"pre-existing pilot reuse artifact differs: {key}"
                    )
            validate_full_checkpoint(
                destination_checkpoint,
                destination_prediction,
                expected_conditions()[key],
                allow_pilot_record=True,
            )
            continue
        destination_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        destination_prediction.parent.mkdir(parents=True, exist_ok=True)
        staged_checkpoint = destination_checkpoint.with_name(
            destination_checkpoint.name + f".{os.getpid()}.pair.tmp"
        )
        staged_prediction = destination_prediction.with_name(
            destination_prediction.name + f".{os.getpid()}.pair.tmp"
        )
        prediction_promoted = False
        try:
            _atomic_copy(source_checkpoint, staged_checkpoint)
            _atomic_copy(source_prediction, staged_prediction)
            os.replace(staged_prediction, destination_prediction)
            prediction_promoted = True
            os.replace(staged_checkpoint, destination_checkpoint)
        except Exception:
            if prediction_promoted and not destination_checkpoint.exists():
                destination_prediction.unlink(missing_ok=True)
            raise
        finally:
            staged_checkpoint.unlink(missing_ok=True)
            staged_prediction.unlink(missing_ok=True)
        validate_full_checkpoint(
            destination_checkpoint, destination_prediction,
            expected_conditions()[key], allow_pilot_record=True,
        )


def reconcile_artifacts(
    output_root: Path, terminal: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    expected = expected_conditions()
    checkpoints_dir = output_root / "checkpoints"
    predictions_dir = output_root / "predictions"
    checkpoint_files = {
        p.relative_to(checkpoints_dir).as_posix()
        for p in checkpoints_dir.rglob("*") if p.is_file()
    }
    prediction_files = {
        p.relative_to(predictions_dir).as_posix()
        for p in predictions_dir.rglob("*") if p.is_file()
    }
    expected_checkpoints = {f"{key}.json" for key in expected}
    successful = {
        key for key, value in terminal.items() if value["terminal_state"] == "success"
    }
    expected_predictions = {f"{key}.npz" for key in successful}
    mismatched = []
    pilot_set = set(pilot_keys())
    for key in sorted(terminal):
        try:
            validate_full_checkpoint(
                checkpoints_dir / f"{key}.json",
                predictions_dir / f"{key}.npz",
                expected[key],
                allow_pilot_record=key in pilot_set,
            )
        except Exception as exc:
            mismatched.append({"key": key, "error_type": type(exc).__name__})
    all_files = {
        p.relative_to(output_root).as_posix()
        for p in output_root.rglob("*") if p.is_file()
    }
    allowed = {
        *(f"checkpoints/{name}" for name in expected_checkpoints),
        *(f"predictions/{name}" for name in expected_predictions),
        "operational_report.json", FULL_LOCK_NAME,
    }
    stale = sorted(
        relative for relative in all_files
        if "/" not in relative and relative.startswith(".replication_full.stale.")
        and relative.endswith(".json")
    )
    invalid_locks = []
    for relative in [FULL_LOCK_NAME, *stale]:
        path = output_root / relative
        if path.exists():
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                validate_full_grid_lock_record(record)
            except Exception:
                invalid_locks.append(relative)
    allowed.update(stale)
    return {
        "unexpected_checkpoint_artifacts": sorted(checkpoint_files - expected_checkpoints),
        "missing_checkpoint_artifacts": sorted(expected_checkpoints - checkpoint_files),
        "unexpected_prediction_artifacts": sorted(prediction_files - expected_predictions),
        "missing_prediction_artifacts": sorted(expected_predictions - prediction_files),
        "orphan_prediction_artifacts": sorted(
            prediction_files - {f"{key}.npz" for key in successful}
        ),
        "mismatched_artifacts": mismatched,
        "unexpected_root_artifacts": sorted(all_files - allowed),
        "invalid_lock_records": invalid_locks,
    }


def _nonnegative_integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _validate_ordered_key_list(
    value: Any, expected_order: list[str], label: str
) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(key, str) for key in value)
        or len(value) != len(set(value))
        or any(key not in set(expected_order) for key in value)
        or value != [key for key in expected_order if key in set(value)]
    ):
        raise FullGridRunnerError(
            f"full-grid report {label} must be a unique deterministic key list"
        )
    return value


def validate_operational_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or set(report) != REPORT_KEYS:
        raise FullGridRunnerError("full-grid operational report fields changed")
    if _contains_forbidden_name(report):
        raise FullGridRunnerError("full-grid report contains forbidden leakage")
    if (
        type(report["schema_version"]) is not int
        or report["schema_version"] != 1
        or report["record"] != "data002_replication_full_operational_report_v1"
        or report["scientific_status"] != "non_scientific_metric_blind_operational_only"
        or report["status"] not in {"pass", "fail"}
        or report["scientific_measures_exposed"] is not False
        or report["sealed_array_values_exposed"] is not False
        or report["upstream_metric_tables_accessed"] is not False
        or type(report["allocation_invariants_pass"]) is not bool
    ):
        raise FullGridRunnerError("full-grid report identity changed")
    count_fields = (
        "expected_key_count", "terminal_key_count", "successful_key_count",
        "pilot_reuse_count", "resume_count", "execution_count",
    )
    if any(not _nonnegative_integer(report.get(field)) for field in count_fields):
        raise FullGridRunnerError("full-grid report count is invalid")
    if report["expected_key_count"] != EXPECTED_COUNT:
        raise FullGridRunnerError("full-grid expected count changed")
    expected_order = list(expected_conditions())
    expected = set(expected_order)
    states = report.get("terminal_states")
    reused = _validate_ordered_key_list(
        report.get("pilot_reused_keys"), expected_order, "pilot_reused_keys"
    )
    resumed = _validate_ordered_key_list(
        report.get("resumed_keys"), expected_order, "resumed_keys"
    )
    executed = _validate_ordered_key_list(
        report.get("executed_keys"), expected_order, "executed_keys"
    )
    failures = _validate_ordered_key_list(
        report.get("failure_keys"), expected_order, "failure_keys"
    )
    timeouts = _validate_ordered_key_list(
        report.get("timeout_keys"), expected_order, "timeout_keys"
    )
    if (
        not isinstance(states, Mapping)
        or any(not isinstance(key, str) for key in states)
        or set(states) != expected
        or any(not isinstance(state, str) or state not in TERMINAL_STATES
               for state in states.values())
        or len(set(reused + resumed + executed)) != len(reused + resumed + executed)
        or set(reused + resumed + executed) != expected
        or reused != pilot_keys()
        or report["pilot_reuse_count"] != len(reused)
        or report["resume_count"] != len(resumed)
        or report["execution_count"] != len(executed)
        or report["terminal_key_count"] != len(states)
        or report["successful_key_count"] != sum(v == "success" for v in states.values())
        or failures != [key for key in expected_order if states[key] == "failure"]
        or timeouts != [key for key in expected_order if states[key] == "timeout"]
    ):
        raise FullGridRunnerError("full-grid report key accounting changed")
    reconciliation = report.get("reconciliation")
    if (
        not isinstance(reconciliation, Mapping)
        or set(reconciliation) != RECONCILIATION_KEYS
        or any(not isinstance(value, list) for value in reconciliation.values())
    ):
        raise FullGridRunnerError("full-grid reconciliation schema changed")
    should_pass = (
        report["terminal_key_count"] == EXPECTED_COUNT
        and report["successful_key_count"] == EXPECTED_COUNT
        and not any(reconciliation.values())
        and report["allocation_invariants_pass"] is True
    )
    if (report["status"] == "pass") != should_pass:
        raise FullGridRunnerError("full-grid report status is inconsistent")
    integer_nested = {
        "preflight": {"free_disk_bytes", "minimum_free_disk_bytes"},
        "resources": {"maximum_observed_rss_bytes", "warning_count"},
        "persistence": {
            "condition_output_file_count", "condition_output_bytes",
            "operational_report_file_count", "active_lock_file_count",
            "retained_stale_lock_file_count", "total_files_before_report_promotion",
            "total_bytes_before_report_promotion",
        },
    }
    for section, fields in integer_nested.items():
        values = report.get(section)
        if not isinstance(values, Mapping) or set(values) != fields:
            raise FullGridRunnerError(f"full-grid report {section} fields changed")
        if any(not _nonnegative_integer(value) for value in values.values()):
            raise FullGridRunnerError(f"full-grid report {section} value invalid")
    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "wall_seconds", "summed_condition_seconds", "maximum_condition_seconds"
    }:
        raise FullGridRunnerError("full-grid report runtime fields changed")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value)) or value < 0
        for value in runtime.values()
    ):
        raise FullGridRunnerError("full-grid report runtime value invalid")
    return report


def _validate_completed_output(
    output_root: Path,
    pilot_root: Path,
    report: Mapping[str, Any],
) -> None:
    expected = expected_conditions()
    for key in pilot_keys():
        for namespace, suffix in (("checkpoints", "json"), ("predictions", "npz")):
            source = pilot_root / namespace / f"{key}.{suffix}"
            destination = output_root / namespace / f"{key}.{suffix}"
            if (
                not destination.is_file()
                or destination.stat().st_size != source.stat().st_size
                or _sha256(destination) != _sha256(source)
            ):
                raise FullGridRunnerError(
                    f"completed output pilot reuse artifact differs: {key}"
                )
    terminal: dict[str, dict[str, Any]] = {}
    pilot_set = set(pilot_keys())
    for key, condition in expected.items():
        checkpoint = validate_full_checkpoint(
            output_root / "checkpoints" / f"{key}.json",
            output_root / "predictions" / f"{key}.npz",
            condition,
            allow_pilot_record=key in pilot_set,
        )
        if checkpoint["terminal_state"] != report["terminal_states"][key]:
            raise FullGridRunnerError(
                f"completed report terminal state differs from checkpoint: {key}"
            )
        terminal[key] = checkpoint
    observed_reconciliation = reconcile_artifacts(output_root, terminal)
    if observed_reconciliation != report["reconciliation"]:
        raise FullGridRunnerError(
            "completed report reconciliation differs from current output"
        )


def run_full_grid(
    *,
    launch_path: Path,
    executor: PilotExecutor,
    project_root: Path = Path.cwd(),
    startup_verifier: Callable[[Path, Path], FullGridStartupPolicy] = verify_full_grid_startup,
    timeout_executor: Callable[
        [PilotExecutor, PilotCondition, int], tuple[str, Any, float]
    ] = execute_with_timeout,
    checkpoint_writer: Callable[[Path, Mapping[str, Any]], None] = _atomic_json,
) -> dict[str, Any]:
    wall_started = time.monotonic()
    project_root = project_root.resolve()
    policy = startup_verifier(project_root, launch_path)
    if not isinstance(policy, FullGridStartupPolicy):
        raise FullGridRunnerError("startup verifier returned invalid policy")
    output_root = policy.output_root.resolve()
    if output_root != (project_root / "results/replication/full_v1").resolve():
        raise FullGridRunnerError("full-grid output path is not bound")
    manifest = json.loads(
        (project_root / "results/provenance/replication_full_grid_keys_v1.json")
        .read_text(encoding="utf-8")
    )
    validate_key_manifest(manifest, scope="full")
    expected = expected_conditions()
    manifest_keys = [item["key"] for item in manifest["conditions"]]
    if manifest_keys != list(expected) or len(manifest_keys) != len(set(manifest_keys)):
        raise FullGridRunnerError("production full-grid keys or ordering changed")
    existing_report_path = output_root / "operational_report.json"
    if existing_report_path.exists():
        existing_report = validate_operational_report(existing_report_path)
        validate_pilot_reuse_source(project_root, policy.pilot_output_root)
        _validate_completed_output(
            output_root, policy.pilot_output_root, existing_report
        )
        return existing_report
    pilot_source = validate_pilot_reuse_source(project_root, policy.pilot_output_root)
    free_disk = shutil.disk_usage(project_root).free
    if free_disk < policy.minimum_free_disk_bytes:
        raise FullGridRunnerError("full-grid free-disk gate failed")
    checkpoints = output_root / "checkpoints"
    predictions = output_root / "predictions"
    terminal: dict[str, dict[str, Any]] = {}
    pilot_reused: list[str] = []
    resumed: list[str] = []
    executed: list[str] = []
    with exclusive_full_grid_run_lock(output_root):
        checkpoints.mkdir(parents=True, exist_ok=True)
        predictions.mkdir(parents=True, exist_ok=True)
        _copy_pilot_reuse(policy.pilot_output_root, output_root, pilot_source)
        for key, condition in expected.items():
            checkpoint_path = checkpoints / f"{key}.json"
            prediction_path = predictions / f"{key}.npz"
            if checkpoint_path.exists():
                checkpoint = validate_full_checkpoint(
                    checkpoint_path, prediction_path, condition,
                    allow_pilot_record=key in set(pilot_keys()),
                )
                terminal[key] = checkpoint
                (pilot_reused if key in set(pilot_keys()) else resumed).append(key)
                continue
            if key in set(pilot_keys()):
                raise FullGridRunnerError("pilot reuse failed; silent refit forbidden")
            if prediction_path.exists():
                raise FullGridRunnerError(f"refusing orphan prediction: {key}")
            state, payload, elapsed = timeout_executor(executor, condition, policy.timeout_seconds)
            if (
                isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed)) or elapsed < 0
            ):
                raise FullGridRunnerError("executor elapsed time invalid")
            executed.append(key)
            captured = dict(EMPTY_CAPTURED_OUTPUT)
            if state == "success":
                result = payload
                if isinstance(payload, Mapping) and "_worker_result" in payload:
                    result = payload["_worker_result"]
                    captured = _validate_captured_output(payload["_captured_output"])
                arrays, metadata = _validate_executor_result(result, condition)
                _atomic_npz(prediction_path, arrays)
                artifact = {
                    "path": prediction_path.name,
                    "bytes": prediction_path.stat().st_size,
                    "sha256": _sha256(prediction_path),
                }
                checkpoint = _full_checkpoint(
                    condition, state, elapsed, artifact=artifact, metadata=metadata,
                    captured_output=captured, error=None,
                )
            elif state == "failure":
                error = payload
                if isinstance(payload, Mapping) and "_worker_error" in payload:
                    error = payload["_worker_error"]
                    captured = _validate_captured_output(payload["_captured_output"])
                checkpoint = _full_checkpoint(
                    condition, state, elapsed, artifact=None, metadata=None,
                    captured_output=captured, error=error,
                )
            elif state == "timeout":
                checkpoint = _full_checkpoint(
                    condition, state, elapsed, artifact=None, metadata=None,
                    captured_output=captured, error=payload,
                )
            else:
                raise FullGridRunnerError("executor returned invalid terminal state")
            checkpoint_writer(checkpoint_path, checkpoint)
            validate_full_checkpoint(checkpoint_path, prediction_path, condition)
            terminal[key] = checkpoint
        reconciliation = reconcile_artifacts(output_root, terminal)
        success_count = sum(v["terminal_state"] == "success" for v in terminal.values())
        times = [float(v["elapsed_seconds"]) for v in terminal.values()]
        max_rss = max((
            v["operational_metadata"]["resource_observations"]["process_rss_bytes"]
            for v in terminal.values() if v["terminal_state"] == "success"
        ), default=0)
        warning_count = sum(
            len(v["operational_metadata"]["warnings"])
            for v in terminal.values() if v["terminal_state"] == "success"
        )
        condition_files = [
            p for namespace in (checkpoints, predictions)
            for p in namespace.rglob("*") if p.is_file()
        ]
        stale = list(output_root.glob(".replication_full.stale.*.json"))
        before_report = [
            p for p in output_root.rglob("*")
            if p.is_file() and p.name != "operational_report.json"
        ]
        allocation_pass = bool(
            success_count == EXPECTED_COUNT and all(
                v["operational_metadata"]["allocation_record"]["final_count"]
                == v["operational_metadata"]["allocation_record"]["original_count"]
                + v["operational_metadata"]["allocation_record"]["duplicate_count"]
                for v in terminal.values()
            )
        )
        states = {key: value["terminal_state"] for key, value in terminal.items()}
        report = {
            "schema_version": 1,
            "record": "data002_replication_full_operational_report_v1",
            "scientific_status": "non_scientific_metric_blind_operational_only",
            "status": "pass" if (
                len(terminal) == EXPECTED_COUNT and success_count == EXPECTED_COUNT
                and not any(reconciliation.values()) and allocation_pass
            ) else "fail",
            "expected_key_count": EXPECTED_COUNT,
            "terminal_key_count": len(terminal),
            "successful_key_count": success_count,
            "failure_keys": [k for k in expected if states[k] == "failure"],
            "timeout_keys": [k for k in expected if states[k] == "timeout"],
            "pilot_reused_keys": pilot_reused,
            "resumed_keys": resumed,
            "executed_keys": executed,
            "pilot_reuse_count": len(pilot_reused),
            "resume_count": len(resumed),
            "execution_count": len(executed),
            "terminal_states": states,
            "reconciliation": reconciliation,
            "preflight": {
                "free_disk_bytes": free_disk,
                "minimum_free_disk_bytes": policy.minimum_free_disk_bytes,
            },
            "runtime": {
                "wall_seconds": time.monotonic() - wall_started,
                "summed_condition_seconds": sum(times),
                "maximum_condition_seconds": max(times, default=0),
            },
            "resources": {
                "maximum_observed_rss_bytes": max_rss,
                "warning_count": warning_count,
            },
            "persistence": {
                "condition_output_file_count": len(condition_files),
                "condition_output_bytes": sum(p.stat().st_size for p in condition_files),
                "operational_report_file_count": 1,
                "active_lock_file_count": int((output_root / FULL_LOCK_NAME).is_file()),
                "retained_stale_lock_file_count": len(stale),
                "total_files_before_report_promotion": len(before_report),
                "total_bytes_before_report_promotion": sum(
                    p.stat().st_size for p in before_report
                ),
            },
            "allocation_invariants_pass": allocation_pass,
            "scientific_measures_exposed": False,
            "sealed_array_values_exposed": False,
            "upstream_metric_tables_accessed": False,
        }
        if _contains_forbidden_name(report):
            raise FullGridRunnerError("full-grid report contains forbidden fields")
        _atomic_json(output_root / "operational_report.json", report)
        return validate_operational_report(output_root / "operational_report.json")


__all__ = [
    "FullGridRunnerError", "RealFullGridExecutor", "expected_conditions",
    "pilot_keys", "reconcile_artifacts", "run_full_grid",
    "validate_full_checkpoint", "validate_operational_report",
    "validate_pilot_reuse_source",
]
