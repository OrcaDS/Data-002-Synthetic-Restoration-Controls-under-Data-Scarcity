from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from data002.reference_bundle import verify_reference_bundle


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(bundle: Path, artifacts: list[dict[str, object]]) -> str:
    manifest = {
        "schema_version": 1,
        "record": "test_reference_bundle",
        "archive_contract": {
            "format": "numpy_npz_compressed",
            "required_arrays": ["test_index", "y_true", "probability"],
            "probability_dtype": "float32",
        },
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "artifact_bytes": sum(int(artifact["bytes"]) for artifact in artifacts),
    }
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return _sha256(manifest_path)


def _make_bundle(tmp_path: Path) -> tuple[Path, Path, str]:
    bundle = tmp_path / "bundle"
    archive_path = bundle / "dataset" / "seed_00" / "model_050pct.npz"
    archive_path.parent.mkdir(parents=True)
    np.savez_compressed(
        archive_path,
        test_index=np.array([10, 11, 12], dtype=np.int64),
        y_true=np.array([0, 1, 0], dtype=np.int64),
        probability=np.array([0.2, 0.8, 0.3], dtype=np.float32),
    )
    artifacts = [
        {
            "path": archive_path.relative_to(bundle).as_posix(),
            "bytes": archive_path.stat().st_size,
            "sha256": _sha256(archive_path),
        }
    ]
    manifest_hash = _write_manifest(bundle, artifacts)
    return bundle, archive_path, manifest_hash


def _codes(report: dict[str, object]) -> set[str]:
    bundle_codes = {error["code"] for error in report["bundle_errors"]}
    artifact_codes = {
        error["code"]
        for artifact in report["artifacts"]
        for error in artifact["errors"]
    }
    return bundle_codes | artifact_codes


def test_valid_bundle_passes(tmp_path: Path) -> None:
    bundle, _, manifest_hash = _make_bundle(tmp_path)

    report = verify_reference_bundle(
        bundle, expected_manifest_sha256=manifest_hash
    )

    assert report["status"] == "pass"
    assert report["summary"] == {
        "declared_count": 1,
        "actual_npz_count": 1,
        "passed_count": 1,
        "failed_count": 0,
    }


def test_manifest_hash_is_an_external_trust_anchor(tmp_path: Path) -> None:
    bundle, _, manifest_hash = _make_bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )

    report = verify_reference_bundle(
        bundle, expected_manifest_sha256=manifest_hash
    )

    assert report["status"] == "fail"
    assert _codes(report) == {"manifest_hash_mismatch"}


def test_missing_archive_fails(tmp_path: Path) -> None:
    bundle, archive_path, manifest_hash = _make_bundle(tmp_path)
    archive_path.unlink()

    report = verify_reference_bundle(
        bundle, expected_manifest_sha256=manifest_hash
    )

    assert report["status"] == "fail"
    assert "artifact_missing" in _codes(report)


def test_unexpected_archive_fails(tmp_path: Path) -> None:
    bundle, archive_path, manifest_hash = _make_bundle(tmp_path)
    unexpected = archive_path.with_name("unexpected.npz")
    unexpected.write_bytes(archive_path.read_bytes())

    report = verify_reference_bundle(
        bundle, expected_manifest_sha256=manifest_hash
    )

    assert report["status"] == "fail"
    assert "unexpected_artifact" in _codes(report)


def test_size_mismatch_fails(tmp_path: Path) -> None:
    bundle, archive_path, manifest_hash = _make_bundle(tmp_path)
    archive_path.write_bytes(archive_path.read_bytes() + b"x")

    report = verify_reference_bundle(
        bundle, expected_manifest_sha256=manifest_hash
    )

    assert report["status"] == "fail"
    assert {"artifact_size_mismatch", "artifact_hash_mismatch"} <= _codes(report)


def test_hash_mismatch_with_unchanged_size_fails(tmp_path: Path) -> None:
    bundle, archive_path, manifest_hash = _make_bundle(tmp_path)
    content = bytearray(archive_path.read_bytes())
    content[-1] ^= 1
    archive_path.write_bytes(content)

    report = verify_reference_bundle(
        bundle, expected_manifest_sha256=manifest_hash
    )

    assert report["status"] == "fail"
    assert "artifact_hash_mismatch" in _codes(report)
    assert "artifact_size_mismatch" not in _codes(report)


def test_manifest_bound_malformed_archive_fails(tmp_path: Path) -> None:
    bundle, archive_path, _ = _make_bundle(tmp_path)
    archive_path.write_bytes(b"not a numpy archive")
    artifacts = [
        {
            "path": archive_path.relative_to(bundle).as_posix(),
            "bytes": archive_path.stat().st_size,
            "sha256": _sha256(archive_path),
        }
    ]
    manifest_hash = _write_manifest(bundle, artifacts)

    report = verify_reference_bundle(
        bundle, expected_manifest_sha256=manifest_hash
    )

    assert report["status"] == "fail"
    assert "archive_load_error" in _codes(report)


@pytest.mark.parametrize(
    ("probability", "expected_code"),
    [
        (np.array([0.2, np.nan, 0.3], dtype=np.float32), "nonfinite_probability"),
        (np.array([0.2, 1.1, 0.3], dtype=np.float32), "probability_out_of_range"),
        (np.array([0.2, 0.8, 0.3], dtype=np.float64), "probability_dtype_mismatch"),
    ],
)
def test_invalid_probability_contract_fails(
    tmp_path: Path,
    probability: np.ndarray,
    expected_code: str,
) -> None:
    bundle, archive_path, _ = _make_bundle(tmp_path)
    np.savez_compressed(
        archive_path,
        test_index=np.array([10, 11, 12], dtype=np.int64),
        y_true=np.array([0, 1, 0], dtype=np.int64),
        probability=probability,
    )
    artifacts = [
        {
            "path": archive_path.relative_to(bundle).as_posix(),
            "bytes": archive_path.stat().st_size,
            "sha256": _sha256(archive_path),
        }
    ]
    manifest_hash = _write_manifest(bundle, artifacts)

    report = verify_reference_bundle(
        bundle, expected_manifest_sha256=manifest_hash
    )

    assert report["status"] == "fail"
    assert expected_code in _codes(report)
