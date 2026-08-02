"""Read-only verification for the frozen Data 001 replay-reference bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

FROZEN_MANIFEST_SHA256 = (
    "caa360e5d0daf34ce1d28633978a2dcc44d7ef44ac0291a9c2d241bd375c6a0e"
)
MANIFEST_NAME = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def _safe_artifact_path(bundle_root: Path, declared_path: object) -> tuple[str, Path] | None:
    if not isinstance(declared_path, str):
        return None
    pure_path = PurePosixPath(declared_path)
    if (
        pure_path.is_absolute()
        or not pure_path.parts
        or any(part in {"", ".", ".."} for part in pure_path.parts)
        or pure_path.suffix != ".npz"
    ):
        return None
    full_path = bundle_root.joinpath(*pure_path.parts)
    if not full_path.resolve(strict=False).is_relative_to(bundle_root):
        return None
    return pure_path.as_posix(), full_path


def _validate_archive(
    path: Path,
    required_arrays: set[str],
    probability_dtype: str,
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    try:
        with np.load(path, allow_pickle=False) as archive:
            actual_arrays = set(archive.files)
            if actual_arrays != required_arrays:
                errors.append(
                    _error(
                        "archive_arrays_mismatch",
                        "expected "
                        f"{sorted(required_arrays)}, found {sorted(actual_arrays)}",
                    )
                )
                return errors

            test_index = archive["test_index"]
            y_true = archive["y_true"]
            probability = archive["probability"]

            if any(array.ndim != 1 for array in (test_index, y_true, probability)):
                errors.append(_error("arrays_not_1d", "all required arrays must be one-dimensional"))
                return errors
            if not (len(test_index) == len(y_true) == len(probability)):
                errors.append(_error("array_length_mismatch", "required arrays have different lengths"))
            if len(test_index) == 0:
                errors.append(_error("empty_archive", "required arrays must not be empty"))
            if str(probability.dtype) != probability_dtype:
                errors.append(
                    _error(
                        "probability_dtype_mismatch",
                        f"expected {probability_dtype}, found {probability.dtype}",
                    )
                )
            if not np.issubdtype(test_index.dtype, np.integer):
                errors.append(_error("test_index_not_integer", f"found {test_index.dtype}"))
            if len(np.unique(test_index)) != len(test_index):
                errors.append(_error("duplicate_test_index", "test indices must be unique"))
            if not np.isin(y_true, [0, 1]).all():
                errors.append(_error("nonbinary_y_true", "test labels must be binary"))
            if not np.isfinite(probability).all():
                errors.append(_error("nonfinite_probability", "probabilities must be finite"))
            if not ((probability >= 0) & (probability <= 1)).all():
                errors.append(
                    _error("probability_out_of_range", "probabilities must lie in [0, 1]")
                )
    except Exception as exc:  # malformed archives must become evidence, not crashes
        errors.append(_error("archive_load_error", f"{type(exc).__name__}: {exc}"))
    return errors


def verify_reference_bundle(
    bundle_root: str | Path,
    *,
    expected_manifest_sha256: str = FROZEN_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Verify the frozen bundle without calculating predictive metrics.

    The returned dictionary is JSON serializable. The function never writes to
    the bundle or to any other path.
    """

    root = Path(bundle_root).resolve()
    manifest_path = root / MANIFEST_NAME
    report: dict[str, Any] = {
        "verifier": "data002_reference_bundle_v1",
        "status": "fail",
        "expected_manifest_sha256": expected_manifest_sha256,
        "manifest_sha256": None,
        "bundle_errors": [],
        "artifacts": [],
        "summary": {
            "declared_count": None,
            "actual_npz_count": None,
            "passed_count": 0,
            "failed_count": 0,
        },
    }
    bundle_errors: list[dict[str, str]] = report["bundle_errors"]

    if not manifest_path.is_file():
        bundle_errors.append(_error("manifest_missing", str(manifest_path)))
        return report

    actual_manifest_sha256 = _sha256(manifest_path)
    report["manifest_sha256"] = actual_manifest_sha256
    if actual_manifest_sha256 != expected_manifest_sha256:
        bundle_errors.append(
            _error(
                "manifest_hash_mismatch",
                f"expected {expected_manifest_sha256}, found {actual_manifest_sha256}",
            )
        )
        return report

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        bundle_errors.append(_error("manifest_load_error", f"{type(exc).__name__}: {exc}"))
        return report

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        bundle_errors.append(_error("invalid_artifacts", "manifest artifacts must be a list"))
        return report

    declared_count = manifest.get("artifact_count")
    declared_bytes = manifest.get("artifact_bytes")
    report["summary"]["declared_count"] = declared_count
    if not isinstance(declared_count, int) or isinstance(declared_count, bool):
        bundle_errors.append(_error("invalid_artifact_count", repr(declared_count)))
    elif declared_count != len(artifacts):
        bundle_errors.append(
            _error("artifact_count_mismatch", f"declared {declared_count}, listed {len(artifacts)}")
        )

    archive_contract = manifest.get("archive_contract")
    if not isinstance(archive_contract, dict):
        bundle_errors.append(_error("invalid_archive_contract", "missing or non-object contract"))
        return report
    required_arrays_raw = archive_contract.get("required_arrays")
    probability_dtype = archive_contract.get("probability_dtype")
    if (
        not isinstance(required_arrays_raw, list)
        or not all(isinstance(value, str) for value in required_arrays_raw)
        or set(required_arrays_raw) != {"test_index", "y_true", "probability"}
    ):
        bundle_errors.append(_error("invalid_required_arrays", repr(required_arrays_raw)))
        return report
    if probability_dtype != "float32":
        bundle_errors.append(_error("invalid_probability_dtype_contract", repr(probability_dtype)))
        return report
    required_arrays = set(required_arrays_raw)

    declared_paths: set[str] = set()
    valid_entries: list[tuple[dict[str, Any], str, Path]] = []
    listed_bytes = 0
    for position, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            bundle_errors.append(_error("invalid_artifact_entry", f"position {position}"))
            continue
        safe_path = _safe_artifact_path(root, artifact.get("path"))
        if safe_path is None:
            bundle_errors.append(
                _error("unsafe_artifact_path", f"position {position}: {artifact.get('path')!r}")
            )
            continue
        relative_path, full_path = safe_path
        if relative_path in declared_paths:
            bundle_errors.append(_error("duplicate_artifact_path", relative_path))
            continue
        declared_paths.add(relative_path)

        expected_bytes = artifact.get("bytes")
        expected_sha256 = artifact.get("sha256")
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
        ):
            bundle_errors.append(_error("invalid_artifact_bytes", relative_path))
            continue
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            bundle_errors.append(_error("invalid_artifact_sha256", relative_path))
            continue
        listed_bytes += expected_bytes
        valid_entries.append((artifact, relative_path, full_path))

    if not isinstance(declared_bytes, int) or isinstance(declared_bytes, bool):
        bundle_errors.append(_error("invalid_artifact_bytes_total", repr(declared_bytes)))
    elif declared_bytes != listed_bytes:
        bundle_errors.append(
            _error("artifact_bytes_mismatch", f"declared {declared_bytes}, listed {listed_bytes}")
        )

    actual_paths = {
        path.relative_to(root).as_posix() for path in root.rglob("*.npz") if path.is_file()
    }
    report["summary"]["actual_npz_count"] = len(actual_paths)
    for path in sorted(declared_paths - actual_paths):
        bundle_errors.append(_error("artifact_missing", path))
    for path in sorted(actual_paths - declared_paths):
        bundle_errors.append(_error("unexpected_artifact", path))

    artifact_reports: list[dict[str, Any]] = report["artifacts"]
    for artifact, relative_path, full_path in valid_entries:
        errors: list[dict[str, str]] = []
        if full_path.is_file():
            actual_bytes = full_path.stat().st_size
            if actual_bytes != artifact["bytes"]:
                errors.append(
                    _error(
                        "artifact_size_mismatch",
                        f"expected {artifact['bytes']}, found {actual_bytes}",
                    )
                )
            actual_sha256 = _sha256(full_path)
            if actual_sha256 != artifact["sha256"]:
                errors.append(
                    _error(
                        "artifact_hash_mismatch",
                        f"expected {artifact['sha256']}, found {actual_sha256}",
                    )
                )
            if not errors:
                errors.extend(
                    _validate_archive(full_path, required_arrays, probability_dtype)
                )
        else:
            errors.append(_error("artifact_missing", relative_path))
        artifact_reports.append(
            {
                "path": relative_path,
                "status": "pass" if not errors else "fail",
                "errors": errors,
            }
        )

    passed_count = sum(item["status"] == "pass" for item in artifact_reports)
    failed_count = sum(item["status"] != "pass" for item in artifact_reports)
    report["summary"]["passed_count"] = passed_count
    report["summary"]["failed_count"] = failed_count
    if not bundle_errors and failed_count == 0 and passed_count == len(artifacts):
        report["status"] = "pass"
    return report
