from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from data002.full_grid_evidence import (
    ATTESTATIONS,
    FullGridEvidenceError,
    build_evidence_manifest,
    encoded_manifest,
    write_evidence_manifest_once,
)


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8192), b""):
            digest.update(block)
    return digest.hexdigest()


def _keys(count: int) -> list[str]:
    return [f"toy__condition_{index:03d}" for index in range(count)]


def _report(keys: list[str], pilot: list[str]) -> dict:
    executed = [key for key in keys if key not in set(pilot)]
    return {
        "status": "pass",
        "expected_key_count": len(keys),
        "terminal_key_count": len(keys),
        "successful_key_count": len(keys),
        "pilot_reuse_count": len(pilot),
        "resume_count": 0,
        "execution_count": len(executed),
        "pilot_reused_keys": pilot,
        "resumed_keys": [],
        "executed_keys": executed,
        "failure_keys": [],
        "timeout_keys": [],
        "allocation_invariants_pass": True,
        "reconciliation": {
            "invalid_lock_records": [],
            "mismatched_artifacts": [],
            "missing_checkpoint_artifacts": [],
            "missing_prediction_artifacts": [],
            "orphan_prediction_artifacts": [],
            "unexpected_checkpoint_artifacts": [],
            "unexpected_prediction_artifacts": [],
            "unexpected_root_artifacts": [],
        },
        "runtime": {
            "wall_seconds": 1.0,
            "summed_condition_seconds": 2.0,
            "maximum_condition_seconds": 0.5,
        },
        "preflight": {
            "free_disk_bytes": 10_000,
            "minimum_free_disk_bytes": 100,
        },
        "resources": {"maximum_observed_rss_bytes": 123, "warning_count": 0},
        "scientific_measures_exposed": False,
        "sealed_array_values_exposed": False,
        "upstream_metric_tables_accessed": False,
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _toy_package(
    tmp_path: Path, *, count: int = 6, pilot_count: int = 2
) -> tuple[Path, Path, list[str], list[str]]:
    full = tmp_path / "full"
    pilot_root = tmp_path / "pilot"
    keys = _keys(count)
    pilot = keys[:pilot_count]
    for root, selected in ((full, keys), (pilot_root, pilot)):
        (root / "checkpoints").mkdir(parents=True)
        (root / "predictions").mkdir()
        for key in selected:
            checkpoint = f"checkpoint:{key}\n".encode()
            prediction = b"opaque-npz-bytes:" + key.encode()
            (root / "checkpoints" / f"{key}.json").write_bytes(checkpoint)
            (root / "predictions" / f"{key}.npz").write_bytes(prediction)
    _write_json(full / "operational_report.json", _report(keys, pilot))
    _write_json(
        pilot_root / "operational_report.json",
        {
            "status": "pass",
            "expected_key_count": len(pilot),
            "terminal_key_count": len(pilot),
            "successful_key_count": len(pilot),
        },
    )
    return full, pilot_root, keys, pilot


def _build(
    tmp_path: Path,
    full: Path,
    pilot_root: Path,
    keys: list[str],
    pilot: list[str],
) -> dict:
    return build_evidence_manifest(
        project_root=tmp_path,
        full_root=full,
        pilot_root=pilot_root,
        expected_keys=keys,
        pilot_keys=pilot,
        bindings={"authorization_commit": "a" * 40},
        expected_operational_report_sha256=_hash(
            full / "operational_report.json"
        ),
        expected_pilot_report_sha256=_hash(
            pilot_root / "operational_report.json"
        ),
    )


def test_toy_exact_540_coverage_and_opaque_npz_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full, pilot_root, keys, pilot = _toy_package(
        tmp_path, count=540, pilot_count=12
    )
    monkeypatch.setitem(__import__("sys").modules, "numpy", None)
    manifest = _build(tmp_path, full, pilot_root, keys, pilot)
    assert len(manifest["checkpoint_identities"]) == 540
    assert len(manifest["prediction_identities"]) == 540
    assert manifest["package"]["file_count"] == 1081
    assert manifest["pilot_source"]["reused_condition_count"] == 12
    assert manifest["metric_blind_attestations"] == ATTESTATIONS


@pytest.mark.parametrize(
    "mutation",
    ["missing_checkpoint", "missing_prediction", "unexpected", "nested"],
)
def test_package_reconciliation_rejects_artifact_drift(
    tmp_path: Path, mutation: str
) -> None:
    full, pilot_root, keys, pilot = _toy_package(tmp_path)
    if mutation == "missing_checkpoint":
        (full / "checkpoints" / f"{keys[-1]}.json").unlink()
    elif mutation == "missing_prediction":
        (full / "predictions" / f"{keys[-1]}.npz").unlink()
    elif mutation == "unexpected":
        (full / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    else:
        nested = full / "predictions" / "nested"
        nested.mkdir()
        (nested / "unexpected.npz").write_bytes(b"opaque")
    with pytest.raises(FullGridEvidenceError, match="reconciliation"):
        _build(tmp_path, full, pilot_root, keys, pilot)


@pytest.mark.parametrize("namespace,suffix", [("checkpoints", "json"), ("predictions", "npz")])
def test_byte_different_pilot_reuse_is_rejected(
    tmp_path: Path, namespace: str, suffix: str
) -> None:
    full, pilot_root, keys, pilot = _toy_package(tmp_path)
    path = full / namespace / f"{pilot[0]}.{suffix}"
    original = path.read_bytes()
    path.write_bytes(b"X" + original[1:])
    with pytest.raises(FullGridEvidenceError, match="byte-identical"):
        _build(tmp_path, full, pilot_root, keys, pilot)


def test_report_hash_drift_is_rejected(tmp_path: Path) -> None:
    full, pilot_root, keys, pilot = _toy_package(tmp_path)
    accepted = _hash(full / "operational_report.json")
    report = json.loads((full / "operational_report.json").read_text())
    report["resources"]["warning_count"] = 1
    _write_json(full / "operational_report.json", report)
    with pytest.raises(FullGridEvidenceError, match="SHA-256"):
        build_evidence_manifest(
            project_root=tmp_path,
            full_root=full,
            pilot_root=pilot_root,
            expected_keys=keys,
            pilot_keys=pilot,
            bindings={},
            expected_operational_report_sha256=accepted,
            expected_pilot_report_sha256=_hash(
                pilot_root / "operational_report.json"
            ),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "complete"),
        ("successful_key_count", 5),
        ("allocation_invariants_pass", False),
        ("failure_keys", ["toy__condition_000"]),
    ],
)
def test_unaccepted_operational_accounting_is_rejected(
    tmp_path: Path, field: str, value: object
) -> None:
    full, pilot_root, keys, pilot = _toy_package(tmp_path)
    report = json.loads((full / "operational_report.json").read_text())
    report[field] = value
    _write_json(full / "operational_report.json", report)
    with pytest.raises(FullGridEvidenceError, match="accounting"):
        _build(tmp_path, full, pilot_root, keys, pilot)


def test_active_lock_is_rejected(tmp_path: Path) -> None:
    full, pilot_root, keys, pilot = _toy_package(tmp_path)
    (full / ".replication_full.lock").write_text("locked", encoding="utf-8")
    with pytest.raises(FullGridEvidenceError, match="active full-grid lock"):
        _build(tmp_path, full, pilot_root, keys, pilot)


def test_writer_identical_existing_manifest_is_byte_preserving_noop(
    tmp_path: Path
) -> None:
    destination = tmp_path / "provenance" / "manifest.json"
    manifest = {"record": "toy", "files": {"opaque.npz": {"bytes": 4}}}
    assert write_evidence_manifest_once(destination, manifest) == "created"
    before = destination.read_bytes()
    before_stat = destination.stat()
    assert before == encoded_manifest(manifest)
    assert write_evidence_manifest_once(destination, manifest) == "unchanged"
    after_stat = destination.stat()
    assert destination.read_bytes() == before
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    assert after_stat.st_ino == before_stat.st_ino


def test_writer_differing_existing_manifest_is_preserved(
    tmp_path: Path
) -> None:
    destination = tmp_path / "provenance" / "manifest.json"
    destination.parent.mkdir(parents=True)
    original = b'{"record":"senior-reviewed-existing"}\n'
    destination.write_bytes(original)
    before_stat = destination.stat()
    with pytest.raises(FullGridEvidenceError, match="refusing overwrite"):
        write_evidence_manifest_once(destination, {"record": "different"})
    after_stat = destination.stat()
    assert destination.read_bytes() == original
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    assert after_stat.st_ino == before_stat.st_ino
    assert list(destination.parent.glob("*.tmp")) == []


@pytest.mark.parametrize(
    "keys",
    [
        ["toy__condition_000", "toy__condition_000"],
        ["toy__condition_000", 1],
    ],
)
def test_key_order_duplicates_and_types_are_rejected(
    tmp_path: Path, keys: list
) -> None:
    full, pilot_root, _, pilot = _toy_package(tmp_path)
    with pytest.raises(FullGridEvidenceError, match="keys must"):
        build_evidence_manifest(
            project_root=tmp_path,
            full_root=full,
            pilot_root=pilot_root,
            expected_keys=keys,
            pilot_keys=pilot,
            bindings={},
            expected_operational_report_sha256=_hash(
                full / "operational_report.json"
            ),
            expected_pilot_report_sha256=_hash(
                pilot_root / "operational_report.json"
            ),
        )
