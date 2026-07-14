"""Leak-safe training pipeline for the bundled UCI Wine dataset."""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .artifact import save_bundle

FEATURE_NAMES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315_of_diluted_wines",
    "proline",
]


def load_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    dataset = load_wine(as_frame=True)
    frame = dataset.frame.rename(
        columns={"od280/od315_of_diluted_wines": "od280_od315_of_diluted_wines"}
    )
    X = frame[FEATURE_NAMES].copy()
    y = frame["target"].astype(int).copy()
    return X, y, list(dataset.target_names)


def dataset_sha256(X: pd.DataFrame, y: pd.Series) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(X.to_numpy(dtype=np.float64)).tobytes())
    digest.update(np.ascontiguousarray(y.to_numpy(dtype=np.int64)).tobytes())
    return digest.hexdigest()


def bootstrap_accuracy_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        scores.append(float(accuracy_score(y_true[indices], y_pred[indices])))
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def reference_statistics(X_train: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in FEATURE_NAMES:
        values = X_train[name].to_numpy(dtype=float)
        result[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
            "q01": float(np.quantile(values, 0.01)),
            "q99": float(np.quantile(values, 0.99)),
        }
    return result


def train_model(*, random_state: int = 42) -> tuple[Any, dict[str, Any]]:
    X, y, class_names = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=3000, solver="lbfgs", random_state=random_state),
            ),
        ]
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"model__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        refit=True,
        n_jobs=1,
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    ci_low, ci_high = bootstrap_accuracy_interval(
        y_test.to_numpy(), predictions, seed=random_state
    )
    metrics = {
        "baseline_test_accuracy": float(accuracy_score(y_test, baseline_predictions)),
        "baseline_test_macro_f1": float(
            f1_score(y_test, baseline_predictions, average="macro")
        ),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_accuracy_ci95": [ci_low, ci_high],
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "test_macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "test_log_loss": float(log_loss(y_test, probabilities, labels=sorted(y.unique()))),
        "confusion_matrix": confusion_matrix(y_test, predictions).astype(int).tolist(),
        "cv_best_macro_f1": float(search.best_score_),
        "test_accuracy_lift_over_baseline": float(
            accuracy_score(y_test, predictions)
            - accuracy_score(y_test, baseline_predictions)
        ),
    }
    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "model_version": "wine-logreg-v1",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "scikit-learn UCI Wine recognition dataset",
            "samples": int(len(X)),
            "features": int(X.shape[1]),
            "classes": int(y.nunique()),
            "license_note": "Use scikit-learn dataset documentation for provenance and citation.",
        },
        "dataset_sha256": dataset_sha256(X, y),
        "problem_framing": {
            "prediction_unit": "one wine sample represented by 13 chemical measurements",
            "target": "cultivar class in the bundled recognition dataset",
            "supported_decision": "educational laboratory triage and dataset exploration",
            "error_implication": "a sample is routed to the wrong cultivar group",
            "prohibited_decisions": ["quality", "price", "authenticity", "safety"],
        },
        "feature_names": FEATURE_NAMES,
        "class_names": {str(index): name for index, name in enumerate(class_names)},
        "best_parameters": search.best_params_,
        "experiment": {
            "hypothesis": "scaled multinomial logistic regression beats the majority baseline",
            "selection_metric": "stratified five-fold macro F1 on training data only",
            "candidate_model": "StandardScaler plus LogisticRegression",
            "candidate_C_values": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            "minimum_practical_accuracy_lift": 0.20,
        },
        "metrics": metrics,
        "reference_statistics": reference_statistics(X_train),
        "split": {
            "strategy": "stratified holdout with stratified CV inside training data",
            "random_state": random_state,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        },
        "intended_use": "Educational cultivar classification; not a quality, safety, or pricing decision.",
    }
    return model, metadata


def train_and_save(artifact_dir: Path, *, random_state: int = 42) -> dict[str, Any]:
    model, metadata = train_model(random_state=random_state)
    save_bundle(model, metadata, artifact_dir)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    metadata = train_and_save(args.artifact_dir, random_state=args.random_state)
    print(f"saved {metadata['model_version']} to {args.artifact_dir}")
    print(metadata["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
