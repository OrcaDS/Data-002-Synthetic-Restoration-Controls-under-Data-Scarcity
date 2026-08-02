"""Metric-blind execution-evidence manifest tooling for the accepted full grid."""

from __future__ import annotations

import json
import os
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from data002.full_grid_runner import validate_operational_report
from data002.replication_design import validate_key_manifest

EVIDENCE_MANIFEST_PATH = (
    "results/provenance/replication_full_grid_execution_evidence_manifest_v1.json"
)
FULL_OUTPUT_ROOT = "results/replication/full_v1"
PILOT_OUTPUT_ROOT = "results/replication/pilot_v1"
FULL_KEY_MANIFEST_PATH = "results/provenance/replication_full_grid_keys_v1.json"
PILOT_KEY_MANIFEST_PATH = "results/provenance/replication_pilot_keys_v1.json"
LAUNCH_PATH = "config/replication_full_launch_v1.json"
IMPLEMENTATION_MANIFEST_PATH = (
    "results/provenance/replication_full_grid_implementation_manifest_v1.json"
)

AUTHORIZATION_COMMIT = "2b77fb87ecdccd870b881f1d84d2e1c05597a689"
SOURCE_FREEZE_COMMIT = "f0c6e900563c4b81a2b59ca8234468c3a71b4209"
MANIFEST_FREEZE_COMMIT = "f8d4c4a5db95b1a3b882da9248c219ec87c76c02"
LAUNCH_SHA256 = "b064117a72dfc587cfe82f06d218c7225a274b07e88f07cec830dcb4ce9e94ac"
IMPLEMENTATION_MANIFEST_SHA256 = (
    "16c46ee9547f741741f33602fb53940fe084c58fced7fee766d9a3b99077e8b4"
)
OPERATIONAL_REPORT_SHA256 = (
    "176da41109ac85e2693dbbcd1b0a988331fe8f7ef03faaaa6cd8b1a65d80cfb0"
)
PILOT_OPERATIONAL_REPORT_SHA256 = (
    "f08bfbd03456dc8556f388799e827e572c40563a2cde76d2e5c345c27b415c7d"
)

ATTESTATIONS = {
    "predictive_metrics_exposed": False,
    "sealed_array_values_exposed": False,
    "upstream_metric_tables_joined": False,
    "treatment_comparisons_performed": False,
    "scientific_analysis_performed": False,
    "npz_arrays_loaded_by_manifest_writer": False,
    "npz_files_hashed_as_opaque_bytes_only": True,
}


class FullGridEvidenceError(ValueError):
    """The metric-blind execution-evidence package is malformed or has drifted."""


def encoded_manifest(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=False) + "\n").encode("utf-8")


def write_evidence_manifest_once(
    destination: Path, manifest: Mapping[str, Any]
) -> str:
    """Atomically create a manifest, or accept an identical existing file."""
    destination = destination.resolve()
    encoded = encoded_manifest(manifest)
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != encoded:
            raise FullGridEvidenceError(
                "existing execution-evidence manifest differs; refusing overwrite"
            )
        return "unchanged"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if (
                not destination.is_file()
                or destination.read_bytes() != encoded
            ):
                raise FullGridEvidenceError(
                    "concurrent execution-evidence manifest differs; "
                    "refusing overwrite"
                )
            return "unchanged"
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullGridEvidenceError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise FullGridEvidenceError(f"JSON evidence must be an object: {path}")
    return value


def _entry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FullGridEvidenceError(f"evidence file missing: {path}")
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise FullGridEvidenceError("evidence path escaped the project root") from exc


def _enumerated_relative_files(root: Path, project_root: Path) -> list[str]:
    if not root.is_dir():
        raise FullGridEvidenceError(f"evidence root missing: {root}")
    return sorted(
        _relative(path, project_root)
        for path in root.rglob("*")
        if path.is_file()
    )


def _expected_package_paths(
    root_relative: str, expected_keys: Sequence[str]
) -> list[str]:
    return sorted(
        [
            f"{root_relative}/checkpoints/{key}.json"
            for key in expected_keys
        ]
        + [
            f"{root_relative}/predictions/{key}.npz"
            for key in expected_keys
        ]
        + [f"{root_relative}/operational_report.json"]
    )


def _strict_keys(keys: Sequence[str], *, label: str) -> list[str]:
    result = list(keys)
    if (
        not result
        or any(not isinstance(key, str) or not key for key in result)
        or len(result) != len(set(result))
    ):
        raise FullGridEvidenceError(
            f"{label} keys must be nonempty strings with no duplicates"
        )
    return result


def _check_report_accounting(
    report: Mapping[str, Any],
    expected_keys: Sequence[str],
    pilot_keys: Sequence[str],
) -> None:
    required = {
        "status", "expected_key_count", "terminal_key_count",
        "successful_key_count", "pilot_reuse_count", "resume_count",
        "execution_count", "pilot_reused_keys", "resumed_keys", "executed_keys",
        "failure_keys", "timeout_keys", "allocation_invariants_pass",
        "reconciliation", "runtime", "preflight", "resources",
        "scientific_measures_exposed", "sealed_array_values_exposed",
        "upstream_metric_tables_accessed",
    }
    if not required.issubset(report):
        raise FullGridEvidenceError("operational report lacks required evidence")
    expected = list(expected_keys)
    pilot = list(pilot_keys)
    executed = [key for key in expected if key not in set(pilot)]
    if (
        report["status"] != "pass"
        or report["expected_key_count"] != len(expected)
        or report["terminal_key_count"] != len(expected)
        or report["successful_key_count"] != len(expected)
        or report["pilot_reuse_count"] != len(pilot)
        or report["resume_count"] != 0
        or report["execution_count"] != len(executed)
        or report["pilot_reused_keys"] != pilot
        or report["resumed_keys"] != []
        or report["executed_keys"] != executed
        or report["failure_keys"] != []
        or report["timeout_keys"] != []
        or report["allocation_invariants_pass"] is not True
        or not isinstance(report["reconciliation"], Mapping)
        or any(report["reconciliation"].values())
        or report["scientific_measures_exposed"] is not False
        or report["sealed_array_values_exposed"] is not False
        or report["upstream_metric_tables_accessed"] is not False
    ):
        raise FullGridEvidenceError("operational report accounting is not accepted")


def _package_files(
    *,
    project_root: Path,
    root: Path,
    root_relative: str,
    expected_keys: Sequence[str],
) -> dict[str, dict[str, Any]]:
    expected_paths = _expected_package_paths(root_relative, expected_keys)
    observed_paths = _enumerated_relative_files(root, project_root)
    if observed_paths != expected_paths:
        missing = sorted(set(expected_paths) - set(observed_paths))
        unexpected = sorted(set(observed_paths) - set(expected_paths))
        raise FullGridEvidenceError(
            f"evidence package reconciliation failed; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        relative: _entry(project_root / relative)
        for relative in expected_paths
    }


def _package_identity(files: Mapping[str, Mapping[str, Any]]) -> str:
    payload = "".join(
        f"{relative}\t{entry['bytes']}\t{entry['sha256']}\n"
        for relative, entry in files.items()
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def build_evidence_manifest(
    *,
    project_root: Path,
    full_root: Path,
    pilot_root: Path,
    expected_keys: Sequence[str],
    pilot_keys: Sequence[str],
    bindings: Mapping[str, str],
    expected_operational_report_sha256: str,
    expected_pilot_report_sha256: str,
) -> dict[str, Any]:
    """Build a deterministic manifest without loading any sealed NPZ array."""
    project_root = project_root.resolve()
    full_root = full_root.resolve()
    pilot_root = pilot_root.resolve()
    expected = _strict_keys(expected_keys, label="full-grid")
    pilot = _strict_keys(pilot_keys, label="pilot")
    if any(key not in set(expected) for key in pilot):
        raise FullGridEvidenceError("pilot keys are not an exact full-grid subset")
    if (full_root / ".replication_full.lock").exists():
        raise FullGridEvidenceError("active full-grid lock remains")

    report_path = full_root / "operational_report.json"
    pilot_report_path = pilot_root / "operational_report.json"
    if _sha256(report_path) != expected_operational_report_sha256:
        raise FullGridEvidenceError("operational-report SHA-256 changed")
    if _sha256(pilot_report_path) != expected_pilot_report_sha256:
        raise FullGridEvidenceError("pilot operational-report SHA-256 changed")
    report = _load_object(report_path)
    pilot_report = _load_object(pilot_report_path)
    _check_report_accounting(report, expected, pilot)
    if (
        pilot_report.get("status") != "pass"
        or pilot_report.get("expected_key_count") != len(pilot)
        or pilot_report.get("terminal_key_count") != len(pilot)
        or pilot_report.get("successful_key_count") != len(pilot)
    ):
        raise FullGridEvidenceError("pilot source report is not accepted")

    full_relative = _relative(full_root, project_root)
    pilot_relative = _relative(pilot_root, project_root)
    full_files = _package_files(
        project_root=project_root,
        root=full_root,
        root_relative=full_relative,
        expected_keys=expected,
    )
    pilot_files = _package_files(
        project_root=project_root,
        root=pilot_root,
        root_relative=pilot_relative,
        expected_keys=pilot,
    )

    reused_conditions: list[dict[str, Any]] = []
    for key in pilot:
        checkpoint_source = f"{pilot_relative}/checkpoints/{key}.json"
        checkpoint_destination = f"{full_relative}/checkpoints/{key}.json"
        prediction_source = f"{pilot_relative}/predictions/{key}.npz"
        prediction_destination = f"{full_relative}/predictions/{key}.npz"
        if (
            pilot_files[checkpoint_source] != full_files[checkpoint_destination]
            or pilot_files[prediction_source] != full_files[prediction_destination]
        ):
            raise FullGridEvidenceError(
                f"pilot reuse is not byte-identical: {key}"
            )
        reused_conditions.append(
            {
                "key": key,
                "checkpoint_source": checkpoint_source,
                "checkpoint_destination": checkpoint_destination,
                "checkpoint_byte_identical": True,
                "prediction_source": prediction_source,
                "prediction_destination": prediction_destination,
                "prediction_byte_identical": True,
            }
        )

    checkpoint_identities = [
        f"{full_relative}/checkpoints/{key}.json" for key in expected
    ]
    prediction_identities = [
        f"{full_relative}/predictions/{key}.npz" for key in expected
    ]
    condition_accounting = {
        "expected": report["expected_key_count"],
        "terminal": report["terminal_key_count"],
        "successful": report["successful_key_count"],
        "pilot_reused": report["pilot_reuse_count"],
        "resumed": report["resume_count"],
        "freshly_executed": report["execution_count"],
        "failure_keys": report["failure_keys"],
        "timeout_keys": report["timeout_keys"],
        "allocation_invariants_pass": report["allocation_invariants_pass"],
    }
    return {
        "schema_version": 1,
        "record": "data002_replication_full_grid_execution_evidence_v1",
        "status": "proposed_pending_senior_review",
        "scientific_status": "non_scientific_metric_blind_operational_only",
        "bindings": dict(bindings),
        "condition_accounting": condition_accounting,
        "reconciliation": dict(report["reconciliation"]),
        "runtime": dict(report["runtime"]),
        "preflight": dict(report["preflight"]),
        "resources": dict(report["resources"]),
        "warning_count": report["resources"]["warning_count"],
        "checkpoint_identities": checkpoint_identities,
        "prediction_identities": prediction_identities,
        "evidence_files": full_files,
        "package": {
            "file_count": len(full_files),
            "bytes": sum(entry["bytes"] for entry in full_files.values()),
            "identity_sha256": _package_identity(full_files),
        },
        "pilot_source": {
            "operational_report_sha256": expected_pilot_report_sha256,
            "file_count": len(pilot_files),
            "bytes": sum(entry["bytes"] for entry in pilot_files.values()),
            "identity_sha256": _package_identity(pilot_files),
            "files": pilot_files,
            "reused_condition_count": len(reused_conditions),
            "reused_conditions": reused_conditions,
            "all_reused_files_byte_identical": True,
        },
        "metric_blind_attestations": dict(ATTESTATIONS),
    }


def _git_file_sha256(project_root: Path, commit: str, relative: str) -> str:
    result = subprocess.run(
        [
            "git", "-c",
            f"safe.directory={project_root.resolve().as_posix()}",
            "show", f"{commit}:{relative}",
        ],
        cwd=project_root,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise FullGridEvidenceError(
            f"commit does not contain required binding: {commit}:{relative}"
        )
    return sha256(result.stdout).hexdigest()


def production_bindings(project_root: Path) -> dict[str, str]:
    project_root = project_root.resolve()
    if _sha256(project_root / LAUNCH_PATH) != LAUNCH_SHA256:
        raise FullGridEvidenceError("authorized launch SHA-256 changed")
    if (
        _git_file_sha256(project_root, AUTHORIZATION_COMMIT, LAUNCH_PATH)
        != LAUNCH_SHA256
    ):
        raise FullGridEvidenceError("authorization commit does not bind launch")
    if (
        _sha256(project_root / IMPLEMENTATION_MANIFEST_PATH)
        != IMPLEMENTATION_MANIFEST_SHA256
    ):
        raise FullGridEvidenceError("implementation-manifest SHA-256 changed")
    implementation = _load_object(project_root / IMPLEMENTATION_MANIFEST_PATH)
    if implementation.get("reviewed_base_commit") != SOURCE_FREEZE_COMMIT:
        raise FullGridEvidenceError("source-freeze commit binding changed")
    if (
        _git_file_sha256(
            project_root, MANIFEST_FREEZE_COMMIT, IMPLEMENTATION_MANIFEST_PATH
        )
        != IMPLEMENTATION_MANIFEST_SHA256
    ):
        raise FullGridEvidenceError("manifest-freeze commit binding changed")
    return {
        "authorization_commit": AUTHORIZATION_COMMIT,
        "authorized_launch_path": LAUNCH_PATH,
        "authorized_launch_sha256": LAUNCH_SHA256,
        "source_freeze_commit": SOURCE_FREEZE_COMMIT,
        "manifest_freeze_commit": MANIFEST_FREEZE_COMMIT,
        "implementation_manifest_path": IMPLEMENTATION_MANIFEST_PATH,
        "implementation_manifest_sha256": IMPLEMENTATION_MANIFEST_SHA256,
        "operational_report_path": f"{FULL_OUTPUT_ROOT}/operational_report.json",
        "operational_report_sha256": OPERATIONAL_REPORT_SHA256,
    }


def build_production_evidence_manifest(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    full_manifest = _load_object(project_root / FULL_KEY_MANIFEST_PATH)
    pilot_manifest = _load_object(project_root / PILOT_KEY_MANIFEST_PATH)
    validate_key_manifest(full_manifest, scope="full")
    validate_key_manifest(pilot_manifest, scope="pilot")
    expected = [item["key"] for item in full_manifest["conditions"]]
    pilot = [item["key"] for item in pilot_manifest["conditions"]]
    if len(expected) != 540 or len(pilot) != 12:
        raise FullGridEvidenceError("frozen key-manifest cardinality changed")
    validate_operational_report(
        project_root / FULL_OUTPUT_ROOT / "operational_report.json"
    )
    return build_evidence_manifest(
        project_root=project_root,
        full_root=project_root / FULL_OUTPUT_ROOT,
        pilot_root=project_root / PILOT_OUTPUT_ROOT,
        expected_keys=expected,
        pilot_keys=pilot,
        bindings=production_bindings(project_root),
        expected_operational_report_sha256=OPERATIONAL_REPORT_SHA256,
        expected_pilot_report_sha256=PILOT_OPERATIONAL_REPORT_SHA256,
    )


def validate_evidence_manifest(
    project_root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    expected = build_production_evidence_manifest(project_root)
    if not isinstance(manifest, Mapping) or dict(manifest) != expected:
        raise FullGridEvidenceError(
            "execution-evidence manifest differs from current accepted evidence"
        )
    return {
        "status": "pass",
        "evidence_file_count": manifest["package"]["file_count"],
        "evidence_bytes": manifest["package"]["bytes"],
        "checkpoint_count": len(manifest["checkpoint_identities"]),
        "prediction_count": len(manifest["prediction_identities"]),
        "pilot_reuse_count": manifest["pilot_source"]["reused_condition_count"],
        "package_identity_sha256": manifest["package"]["identity_sha256"],
    }
