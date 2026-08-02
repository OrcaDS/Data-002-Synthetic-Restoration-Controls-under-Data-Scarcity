"""Frozen Data 001 baseline reconstruction for the compatibility gate."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import floor
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TEST_SIZE = 0.2
PREDICTION_DTYPE = np.dtype("float32")
SOURCE_COMMIT = "ecc2b222eca86c47acdf12efd3b8f779b6a29ef9"
DATASET_SHA256 = {
    "diabetes": "19f367e3e3350768f0c144c5d73ee5b355f67a57eaaa86ca7bd8aec594d8b1d0",
    "cleveland": "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8",
}
LOGISTIC_PARAMS = {
    "C": 1.0,
    "solver": "liblinear",
    "max_iter": 2000,
    "class_weight": None,
}
RANDOM_FOREST_PARAMS = {
    "n_estimators": 500,
    "max_features": "sqrt",
    "max_depth": None,
    "min_samples_leaf": 1,
    "bootstrap": True,
    "class_weight": None,
    "n_jobs": -1,
}
MODELS = ("logistic_regression", "random_forest")
METRICS = ("roc_auc", "precision", "recall", "f1", "accuracy", "average_precision")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    frame: pd.DataFrame
    target: str
    numeric_features: tuple[str, ...]
    nominal_features: tuple[str, ...]
    binary_features: tuple[str, ...]
    scarcity_levels: tuple[float, ...]
    source_path: Path


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass(frozen=True, order=True)
class CompatibilityCondition:
    dataset: str
    scarcity_fraction: float
    seed: int
    model: str


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_sha256(indices: np.ndarray) -> str:
    """Hash an ordered integer index through a platform-neutral ASCII encoding."""

    values = np.asarray(indices)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Indices must be a one-dimensional integer array")
    canonical = ",".join(str(int(value)) for value in values).encode("ascii")
    return sha256(canonical).hexdigest()


def load_datasets(
    diabetes_path: str | Path,
    cleveland_path: str | Path,
) -> dict[str, DatasetSpec]:
    """Hash-check and load the two frozen raw datasets."""

    paths = {
        "diabetes": Path(diabetes_path).resolve(),
        "cleveland": Path(cleveland_path).resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = file_sha256(path)
        if observed != DATASET_SHA256[name]:
            raise ValueError(
                f"{name} dataset SHA-256 mismatch: "
                f"expected {DATASET_SHA256[name]}, found {observed}"
            )

    diabetes = pd.read_csv(paths["diabetes"])
    diabetes["Diabetes_binary"] = diabetes["Diabetes_binary"].astype("int8")
    diabetes_numeric = (
        "BMI",
        "MentHlth",
        "PhysHlth",
        "GenHlth",
        "Age",
        "Education",
        "Income",
    )
    diabetes_binary = tuple(
        column
        for column in diabetes.columns
        if column not in diabetes_numeric and column != "Diabetes_binary"
    )

    heart_columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "num",
    ]
    heart = pd.read_csv(paths["cleveland"], names=heart_columns, na_values="?")
    heart["target"] = heart.pop("num").gt(0).astype("int8")

    return {
        "diabetes": DatasetSpec(
            name="diabetes",
            frame=diabetes,
            target="Diabetes_binary",
            numeric_features=diabetes_numeric,
            nominal_features=(),
            binary_features=diabetes_binary,
            scarcity_levels=(1.0, 0.5, 0.25, 0.1, 0.05, 0.01),
            source_path=paths["diabetes"],
        ),
        "cleveland": DatasetSpec(
            name="cleveland",
            frame=heart,
            target="target",
            numeric_features=("age", "trestbps", "chol", "thalach", "oldpeak", "ca"),
            nominal_features=("cp", "restecg", "slope", "thal"),
            binary_features=("sex", "fbs", "exang"),
            scarcity_levels=(1.0, 0.5, 0.25, 0.1, 0.05),
            source_path=paths["cleveland"],
        ),
    }


def split_dataset(spec: DatasetSpec, seed: int) -> DatasetSplit:
    X = spec.frame.drop(columns=spec.target)
    y = spec.frame[spec.target]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=seed,
    )
    return DatasetSplit(X_train, X_test, y_train, y_test)


def round_half_up(value: float) -> int:
    return floor(value + 0.5)


def allocate_class_counts(
    y: pd.Series,
    total: int,
    minimum_per_class: int = 2,
) -> dict[int, int]:
    counts = y.value_counts().sort_index()
    if total < minimum_per_class * len(counts):
        raise ValueError("Retained total cannot preserve the minimum per class")
    raw = counts / counts.sum() * total
    allocation = np.floor(raw).astype(int)
    remaining = total - int(allocation.sum())
    remainders = (raw - allocation).sort_values(ascending=False, kind="stable")
    for label in remainders.index[:remaining]:
        allocation.loc[label] += 1
    for label in allocation.index:
        deficit = max(0, minimum_per_class - int(allocation.loc[label]))
        if deficit:
            donor = next(
                candidate
                for candidate in allocation.drop(label).sort_values(ascending=False).index
                if allocation.loc[candidate] - deficit >= minimum_per_class
            )
            allocation.loc[donor] -= deficit
            allocation.loc[label] += deficit
    if int(allocation.sum()) != total:
        raise AssertionError("Class allocation does not sum to the requested total")
    return {int(label): int(count) for label, count in allocation.items()}


def nested_scarcity_indices(
    y_train: pd.Series,
    fractions: tuple[float, ...],
    seed: int,
) -> dict[float, np.ndarray]:
    rng = np.random.default_rng(seed + 10_000)
    order = {
        int(label): rng.permutation(y_train.index[y_train.eq(label)].to_numpy())
        for label in sorted(y_train.unique())
    }
    subsets: dict[float, np.ndarray] = {}
    previous = {label: 0 for label in order}
    for fraction in sorted(fractions):
        retained_total = round_half_up(len(y_train) * fraction)
        allocation = allocate_class_counts(y_train, retained_total)
        if not all(allocation[label] >= previous[label] for label in allocation):
            raise AssertionError("Nested class allocations decreased")
        previous.update(allocation)
        subsets[fraction] = np.sort(
            np.concatenate(
                [order[label][: allocation[label]] for label in sorted(order)]
            )
        )
    ordered = sorted(fractions)
    for smaller, larger in zip(ordered, ordered[1:]):
        if not set(subsets[smaller]).issubset(set(subsets[larger])):
            raise AssertionError("Scarcity subsets are not nested")
    return subsets


def retained_training_frame(
    spec: DatasetSpec,
    split: DatasetSplit,
    scarcity_fraction: float,
    seed: int,
) -> pd.DataFrame:
    if scarcity_fraction not in spec.scarcity_levels:
        raise ValueError(f"Unsupported scarcity fraction for {spec.name}")
    subsets = nested_scarcity_indices(split.y_train, spec.scarcity_levels, seed)
    return spec.frame.loc[subsets[scarcity_fraction]].copy()


def make_preprocessor(spec: DatasetSpec, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    nominal = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), list(spec.numeric_features)),
            ("nominal", nominal, list(spec.nominal_features)),
            (
                "binary",
                SimpleImputer(strategy="most_frequent"),
                list(spec.binary_features),
            ),
        ]
    )


def build_logistic_regression(spec: DatasetSpec, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(spec, scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    **LOGISTIC_PARAMS,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_random_forest(spec: DatasetSpec, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(spec, scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    **RANDOM_FOREST_PARAMS,
                    random_state=seed,
                ),
            ),
        ]
    )


MODEL_BUILDERS: dict[str, Callable[[DatasetSpec, int], Pipeline]] = {
    "logistic_regression": build_logistic_regression,
    "random_forest": build_random_forest,
}


def compatibility_conditions() -> tuple[CompatibilityCondition, ...]:
    conditions = []
    for dataset, fractions in (
        ("diabetes", (0.01, 0.5)),
        ("cleveland", (0.05, 0.5)),
    ):
        for scarcity_fraction in fractions:
            for seed in (0, 29):
                for model in MODELS:
                    conditions.append(
                        CompatibilityCondition(
                            dataset=dataset,
                            scarcity_fraction=scarcity_fraction,
                            seed=seed,
                            model=model,
                        )
                    )
    return tuple(sorted(conditions))


def reference_archive_path(
    bundle_root: str | Path,
    condition: CompatibilityCondition,
) -> Path:
    percentage = round(condition.scarcity_fraction * 100)
    return (
        Path(bundle_root)
        / condition.dataset
        / f"seed_{condition.seed:02d}"
        / f"{condition.model}_{percentage:03d}pct.npz"
    )


def run_baseline_condition(
    spec: DatasetSpec,
    condition: CompatibilityCondition,
) -> dict[str, np.ndarray]:
    """Fit one compatibility condition.

    This function is implemented for prospective replay use. Calling it on the
    real datasets remains unauthorized until the replay launch record exists.
    """

    if spec.name != condition.dataset:
        raise ValueError("Condition and dataset do not match")
    split = split_dataset(spec, condition.seed)
    retained = retained_training_frame(
        spec,
        split,
        condition.scarcity_fraction,
        condition.seed,
    )
    X_retained = retained.drop(columns=spec.target)
    y_retained = retained[spec.target]
    pipeline = MODEL_BUILDERS[condition.model](spec, condition.seed)
    pipeline.fit(X_retained, y_retained)
    probability = pipeline.predict_proba(split.X_test)[:, 1].astype(PREDICTION_DTYPE)
    return {
        "test_index": split.X_test.index.to_numpy(),
        "y_true": split.y_test.to_numpy(dtype=np.int8),
        "probability": probability,
    }


def evaluate_probabilities(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    prediction = (np.asarray(probability) >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, probability),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "accuracy": accuracy_score(y_true, prediction),
        "average_precision": average_precision_score(y_true, probability),
    }


def compare_prediction_archives(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    probability_atol: float = 1e-6,
    metric_atol: float = 1e-12,
) -> dict[str, object]:
    """Compare one candidate with one reference without reporting metric values."""

    required = {"test_index", "y_true", "probability"}
    if set(reference) != required or set(candidate) != required:
        raise ValueError("Reference and candidate must contain exactly the required arrays")

    reference_probability = np.asarray(reference["probability"])
    candidate_probability = np.asarray(candidate["probability"])
    reference_y = np.asarray(reference["y_true"])
    candidate_y = np.asarray(candidate["y_true"])
    same_shape = reference_probability.shape == candidate_probability.shape
    finite_and_bounded = bool(
        np.isfinite(candidate_probability).all()
        and ((candidate_probability >= 0) & (candidate_probability <= 1)).all()
    )
    maximum_probability_difference = None
    probabilities_match = False
    predictions_match = False
    metric_differences: dict[str, float] = {}
    metric_checks: dict[str, bool] = {metric: False for metric in METRICS}

    probabilities_are_finite = bool(
        np.isfinite(reference_probability).all()
        and np.isfinite(candidate_probability).all()
    )
    if same_shape and reference_probability.size and probabilities_are_finite:
        maximum_probability_difference = float(
            np.max(np.abs(reference_probability - candidate_probability))
        )
        probabilities_match = bool(
            np.allclose(
                reference_probability,
                candidate_probability,
                rtol=0,
                atol=probability_atol,
            )
        )
        predictions_match = bool(
            np.array_equal(
                reference_probability >= 0.5,
                candidate_probability >= 0.5,
            )
        )
    metrics_ready = bool(
        same_shape
        and reference_probability.size
        and finite_and_bounded
        and np.isfinite(reference_probability).all()
        and reference_y.shape == reference_probability.shape
        and candidate_y.shape == candidate_probability.shape
        and np.isin(reference_y, [0, 1]).all()
        and np.isin(candidate_y, [0, 1]).all()
    )
    if metrics_ready:
        reference_metrics = evaluate_probabilities(
            reference_y,
            reference_probability,
        )
        candidate_metrics = evaluate_probabilities(
            candidate_y,
            candidate_probability,
        )
        for metric in METRICS:
            difference = abs(reference_metrics[metric] - candidate_metrics[metric])
            metric_differences[metric] = float(difference)
            if metric in {"roc_auc", "average_precision"}:
                metric_checks[metric] = difference <= metric_atol
            else:
                metric_checks[metric] = difference == 0

    checks = {
        "test_index_exact": bool(
            np.array_equal(reference["test_index"], candidate["test_index"])
        ),
        "y_true_exact": bool(np.array_equal(reference["y_true"], candidate["y_true"])),
        "probability_shape_exact": same_shape,
        "probability_dtype_float32": candidate_probability.dtype == PREDICTION_DTYPE,
        "probability_finite_and_bounded": finite_and_bounded,
        "probability_within_tolerance": probabilities_match,
        "thresholded_prediction_exact": predictions_match,
        **{f"metric_{metric}_within_tolerance": passed for metric, passed in metric_checks.items()},
    }
    return {
        "status": "pass" if checks and all(checks.values()) else "fail",
        "checks": checks,
        "maximum_absolute_probability_difference": maximum_probability_difference,
        "absolute_metric_differences": metric_differences,
    }
