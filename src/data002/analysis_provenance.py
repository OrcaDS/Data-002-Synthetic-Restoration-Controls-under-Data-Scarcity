"""Authorization and provenance gate for the prospective Data 002 analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from data002.analysis_implementation import (
    IMPLEMENTATION_MANIFEST_PATH,
    validate_analysis_implementation_manifest,
)
from data002.full_grid_evidence import validate_evidence_manifest
from data002.replication_design import (
    full_grid_records,
    validate_design_contracts,
    validate_key_manifest,
    validate_upstream_evidence_manifest,
)

ANALYSIS_OUTPUT_ROOT = "results/analysis/replication_v1"
MUTABLE_LAUNCH_PATH = "config/replication_analysis_launch_v1.json"
PATHS = {
    "analysis_contract": "config/replication_analysis_v1.json",
    "study_contract": "config/replication_study_v1.json",
    "full_key_manifest": "results/provenance/replication_full_grid_keys_v1.json",
    "upstream_evidence_manifest": (
        "results/provenance/data001_metric_bundle_manifest_v1.json"
    ),
    "execution_evidence_review": (
        "results/provenance/replication_full_grid_execution_review_v1.json"
    ),
    "execution_evidence_manifest": (
        "results/provenance/"
        "replication_full_grid_execution_evidence_manifest_v1.json"
    ),
    "full_operational_report": (
        "results/replication/full_v1/operational_report.json"
    ),
    "environment_inventory": (
        "results/provenance/environment_inventory_python313_v1.json"
    ),
    "requirements_lock": "requirements-lock.txt",
}
FROZEN_BINDINGS = {
    "analysis_contract_sha256": (
        "127e921232c03d98820d49d962be0170da0cf4a933d0b465d92c5d6be37a7158"
    ),
    "study_contract_sha256": (
        "b2a693afb379c473c8d7645913d44429866e1a3180a7c48aa6013fb5bd8a8721"
    ),
    "full_key_manifest_sha256": (
        "428846efb67d036975a159b2370855da9c42de9e2a93b53fec0d916e8f0b605b"
    ),
    "upstream_evidence_manifest_sha256": (
        "f2fa5363cab80c5e97c6bb94463c73ef767e5a1d2f413190be5123361afa1c0c"
    ),
    "execution_evidence_review_sha256": (
        "baf02b52d2b82d0965dc9a80890f1f51d6344b32ec60368958b18e8bee639fa9"
    ),
    "execution_evidence_manifest_sha256": (
        "08dbe77600844485cce503813fcd7298dd6eb87cab3e4d4b8f0fe91043bb27b7"
    ),
    "full_operational_report_sha256": (
        "176da41109ac85e2693dbbcd1b0a988331fe8f7ef03faaaa6cd8b1a65d80cfb0"
    ),
    "execution_evidence_commit": (
        "b288e4869bc2ffdb2fef01b669567c7cd679c2e3"
    ),
    "execution_package_identity_sha256": (
        "d5c0c7dac2ef918fe16343ccfb23f901d00c54c7ebee296ba72bb5c134692448"
    ),
    "pilot_source_identity_sha256": (
        "b4d277ba78e4a4d6dbefec0f3b33ac13c8df692bfbf615ce4da66f078337036a"
    ),
    "environment_inventory_sha256": (
        "d6e8c4444d5d1ffe63f5866b2da7405e67cdbd7c586499ec5804526b38c78589"
    ),
    "requirements_lock_sha256": (
        "a4b12e0682dbc8308ed3619922f560b966349416d3e95ccee171940fc4d0ee15"
    ),
}
LAUNCH_FIELDS = {
    "schema_version", "record", "status", "authorization", "authorized_by",
    "authorized_at", "scope", "output_root", "analysis_contract",
    "full_key_manifest", "upstream_evidence_manifest",
    "execution_evidence_review", "implementation_manifest", "execution",
    "bindings",
}
EXECUTION_POLICY = {
    "immutable_output": True,
    "exact_key_reconciliation": True,
    "silent_key_dropping": False,
    "complete_case_analysis": False,
    "raw_probabilities_in_outputs": False,
}
REPORT_PROVENANCE_FIELDS = {
    "analysis_launch_path", "analysis_launch_sha256",
    "analysis_contract_path", "analysis_contract_sha256",
    "study_contract_path", "study_contract_sha256",
    "full_key_manifest_path", "full_key_manifest_sha256",
    "upstream_evidence_manifest_path", "upstream_evidence_manifest_sha256",
    "execution_evidence_review_path", "execution_evidence_review_sha256",
    "execution_evidence_manifest_path", "execution_evidence_manifest_sha256",
    "full_operational_report_path", "full_operational_report_sha256",
    "execution_evidence_commit", "execution_package_identity_sha256",
    "pilot_source_identity_sha256", "environment_inventory_path",
    "environment_inventory_sha256", "requirements_lock_path",
    "requirements_lock_sha256", "analysis_implementation_manifest_path",
    "analysis_implementation_manifest_sha256", "reviewed_base_commit",
}


class AnalysisProvenanceError(ValueError):
    """Analysis authorization or provenance is invalid."""


@dataclass(frozen=True)
class AnalysisStartupPolicy:
    output_root: Path
    report_provenance: Mapping[str, str]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisProvenanceError(f"invalid analysis provenance: {path}") from exc
    if not isinstance(value, dict):
        raise AnalysisProvenanceError(f"analysis provenance must be an object: {path}")
    return value


def _timezone_aware(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def validate_analysis_launch(
    launch: Mapping[str, Any], *, require_authorized: bool
) -> None:
    if not isinstance(launch, Mapping) or set(launch) != LAUNCH_FIELDS:
        raise AnalysisProvenanceError("analysis launch fields changed")
    if (
        launch.get("schema_version") != 1
        or launch.get("record") != "data002_replication_analysis_launch_v1"
        or launch.get("scope")
        != "fixed_18_stratum_replication_treatment_analysis"
        or launch.get("output_root") != ANALYSIS_OUTPUT_ROOT
        or launch.get("analysis_contract") != PATHS["analysis_contract"]
        or launch.get("full_key_manifest") != PATHS["full_key_manifest"]
        or launch.get("upstream_evidence_manifest")
        != PATHS["upstream_evidence_manifest"]
        or launch.get("execution_evidence_review")
        != PATHS["execution_evidence_review"]
        or launch.get("implementation_manifest")
        != IMPLEMENTATION_MANIFEST_PATH
        or launch.get("execution") != EXECUTION_POLICY
    ):
        raise AnalysisProvenanceError("analysis launch identity or policy changed")
    bindings = launch.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        *FROZEN_BINDINGS,
        "reviewed_base_commit",
        "analysis_implementation_manifest_sha256",
    }:
        raise AnalysisProvenanceError("analysis launch bindings changed")
    if any(bindings.get(key) != value for key, value in FROZEN_BINDINGS.items()):
        raise AnalysisProvenanceError("analysis frozen binding drift")
    if require_authorized:
        if (
            launch.get("status") != "authorized"
            or launch.get("authorization") != "authorized"
            or not isinstance(launch.get("authorized_by"), str)
            or not launch["authorized_by"].strip()
            or not _timezone_aware(launch.get("authorized_at"))
            or not isinstance(bindings.get("reviewed_base_commit"), str)
            or len(bindings["reviewed_base_commit"]) != 40
            or not isinstance(
                bindings.get("analysis_implementation_manifest_sha256"), str
            )
            or len(bindings["analysis_implementation_manifest_sha256"]) != 64
        ):
            raise PermissionError("analysis launch is not authorized")
    elif (
        launch.get("status") != "draft_pending_implementation_review"
        or launch.get("authorization") is not None
        or launch.get("authorized_by") is not None
        or launch.get("authorized_at") is not None
        or bindings.get("reviewed_base_commit") is not None
        or bindings.get("analysis_implementation_manifest_sha256") is not None
    ):
        raise AnalysisProvenanceError("draft analysis launch is not inert")


def validate_execution_evidence_documents(
    review: Mapping[str, Any],
    evidence: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    expected_keys = [record["key"] for record in full_grid_records()]
    expected_checkpoints = [
        f"results/replication/full_v1/checkpoints/{key}.json"
        for key in expected_keys
    ]
    expected_predictions = [
        f"results/replication/full_v1/predictions/{key}.npz"
        for key in expected_keys
    ]
    if (
        review.get("record")
        != "data002_replication_full_grid_execution_review_v1"
        or review.get("status") != "operational_evidence_accepted"
        or review.get("analysis_authorization") is not None
        or review.get("evidence", {}).get("commit")
        != FROZEN_BINDINGS["execution_evidence_commit"]
        or review.get("evidence", {}).get("manifest_sha256")
        != FROZEN_BINDINGS["execution_evidence_manifest_sha256"]
        or review.get("evidence", {}).get("operational_report_sha256")
        != FROZEN_BINDINGS["full_operational_report_sha256"]
        or review.get("evidence", {}).get("package_identity_sha256")
        != FROZEN_BINDINGS["execution_package_identity_sha256"]
        or review.get("evidence", {}).get("pilot_source_identity_sha256")
        != FROZEN_BINDINGS["pilot_source_identity_sha256"]
        or review.get("condition_accounting") != {
            "expected": 540,
            "terminal": 540,
            "successful": 540,
            "pilot_reused": 12,
            "resumed": 0,
            "freshly_executed": 528,
        }
        or review.get("allocation_invariants_pass") is not True
        or not review.get("reconciliation", {}).get("all_lists_empty")
        or any(
            value for key, value in review.get("reconciliation", {}).items()
            if key != "all_lists_empty"
        )
        or review.get("attestations", {}).get("scientific_analysis_performed")
        is not False
    ):
        raise AnalysisProvenanceError("execution review binding drift")
    if (
        evidence.get("record")
        != "data002_replication_full_grid_execution_evidence_v1"
        or evidence.get("package", {}).get("file_count") != 1081
        or evidence.get("package", {}).get("identity_sha256")
        != FROZEN_BINDINGS["execution_package_identity_sha256"]
        or evidence.get("pilot_source", {}).get("identity_sha256")
        != FROZEN_BINDINGS["pilot_source_identity_sha256"]
        or evidence.get("condition_accounting", {}).get("expected") != 540
        or evidence.get("condition_accounting", {}).get("successful") != 540
        or evidence.get("checkpoint_identities") != expected_checkpoints
        or evidence.get("prediction_identities") != expected_predictions
        or len(set(evidence.get("checkpoint_identities", []))) != 540
        or len(set(evidence.get("prediction_identities", []))) != 540
    ):
        raise AnalysisProvenanceError("execution evidence 540-key gate failed")
    if (
        report.get("status") != "pass"
        or report.get("expected_key_count") != 540
        or report.get("terminal_key_count") != 540
        or report.get("successful_key_count") != 540
        or report.get("pilot_reuse_count") != 12
        or report.get("resume_count") != 0
        or report.get("execution_count") != 528
        or report.get("failure_keys") != []
        or report.get("timeout_keys") != []
        or report.get("allocation_invariants_pass") is not True
        or any(report.get("reconciliation", {}).values())
    ):
        raise AnalysisProvenanceError("operational report gate failed")


def report_provenance(
    project_root: Path,
    launch_path: Path,
    launch: Mapping[str, Any],
) -> dict[str, str]:
    bindings = launch["bindings"]
    provenance = {
        "analysis_launch_path": launch_path.resolve().relative_to(
            project_root.resolve()
        ).as_posix(),
        "analysis_launch_sha256": file_sha256(launch_path),
        "analysis_contract_path": PATHS["analysis_contract"],
        "analysis_contract_sha256": bindings["analysis_contract_sha256"],
        "study_contract_path": PATHS["study_contract"],
        "study_contract_sha256": bindings["study_contract_sha256"],
        "full_key_manifest_path": PATHS["full_key_manifest"],
        "full_key_manifest_sha256": bindings["full_key_manifest_sha256"],
        "upstream_evidence_manifest_path": PATHS["upstream_evidence_manifest"],
        "upstream_evidence_manifest_sha256": bindings[
            "upstream_evidence_manifest_sha256"
        ],
        "execution_evidence_review_path": PATHS["execution_evidence_review"],
        "execution_evidence_review_sha256": bindings[
            "execution_evidence_review_sha256"
        ],
        "execution_evidence_manifest_path": PATHS[
            "execution_evidence_manifest"
        ],
        "execution_evidence_manifest_sha256": bindings[
            "execution_evidence_manifest_sha256"
        ],
        "full_operational_report_path": PATHS["full_operational_report"],
        "full_operational_report_sha256": bindings[
            "full_operational_report_sha256"
        ],
        "execution_evidence_commit": bindings["execution_evidence_commit"],
        "execution_package_identity_sha256": bindings[
            "execution_package_identity_sha256"
        ],
        "pilot_source_identity_sha256": bindings[
            "pilot_source_identity_sha256"
        ],
        "environment_inventory_path": PATHS["environment_inventory"],
        "environment_inventory_sha256": bindings[
            "environment_inventory_sha256"
        ],
        "requirements_lock_path": PATHS["requirements_lock"],
        "requirements_lock_sha256": bindings["requirements_lock_sha256"],
        "analysis_implementation_manifest_path": IMPLEMENTATION_MANIFEST_PATH,
        "analysis_implementation_manifest_sha256": bindings[
            "analysis_implementation_manifest_sha256"
        ],
        "reviewed_base_commit": bindings["reviewed_base_commit"],
    }
    if (
        set(provenance) != REPORT_PROVENANCE_FIELDS
        or any(not isinstance(value, str) or not value for value in provenance.values())
        or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for key, value in provenance.items()
            if key.endswith("_sha256")
        )
    ):
        raise AnalysisProvenanceError("report provenance is incomplete")
    return provenance


def verify_analysis_startup(
    project_root: Path, launch_path: Path
) -> AnalysisStartupPolicy:
    project_root = project_root.resolve()
    canonical_launch = (project_root / MUTABLE_LAUNCH_PATH).resolve()
    if launch_path.resolve() != canonical_launch:
        raise AnalysisProvenanceError(
            "analysis launch path is not the canonical mutable launch"
        )
    launch = _load(canonical_launch)
    validate_analysis_launch(launch, require_authorized=True)
    for name, relative in PATHS.items():
        expected = FROZEN_BINDINGS[f"{name}_sha256"]
        if file_sha256(project_root / relative) != expected:
            raise AnalysisProvenanceError(f"analysis provenance drift: {name}")
    review = _load(project_root / PATHS["execution_evidence_review"])
    evidence = _load(project_root / PATHS["execution_evidence_manifest"])
    report = _load(project_root / PATHS["full_operational_report"])
    validate_execution_evidence_documents(review, evidence, report)
    study = _load(project_root / PATHS["study_contract"])
    analysis = _load(project_root / PATHS["analysis_contract"])
    pilot = _load(project_root / "config/replication_pilot_v1.json")
    validate_design_contracts(study, pilot, analysis)
    full_manifest = _load(project_root / PATHS["full_key_manifest"])
    validate_key_manifest(full_manifest, scope="full")
    validate_evidence_manifest(project_root, evidence)
    upstream = _load(project_root / PATHS["upstream_evidence_manifest"])
    validate_upstream_evidence_manifest(project_root, upstream)
    implementation_path = project_root / IMPLEMENTATION_MANIFEST_PATH
    implementation = _load(implementation_path)
    validated = validate_analysis_implementation_manifest(
        project_root, implementation
    )
    bindings = launch["bindings"]
    if (
        bindings["reviewed_base_commit"] != validated["reviewed_base_commit"]
        or bindings["analysis_implementation_manifest_sha256"]
        != file_sha256(implementation_path)
    ):
        raise AnalysisProvenanceError("analysis implementation binding drift")
    output_root = (project_root / ANALYSIS_OUTPUT_ROOT).resolve()
    if output_root != (project_root / "results/analysis/replication_v1").resolve():
        raise AnalysisProvenanceError("analysis output root changed")
    return AnalysisStartupPolicy(
        output_root=output_root,
        report_provenance=report_provenance(project_root, launch_path, launch),
    )
