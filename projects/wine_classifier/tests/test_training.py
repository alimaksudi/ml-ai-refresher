from __future__ import annotations

import json

import numpy as np

from wine_classifier.artifact import load_bundle
from wine_classifier.training import FEATURE_NAMES, load_dataset


def test_artifact_contract_and_metrics(artifact_dir):
    bundle = load_bundle(artifact_dir)
    assert bundle.metadata["feature_names"] == FEATURE_NAMES
    assert bundle.metadata["metrics"]["test_accuracy"] >= 0.85
    assert bundle.metadata["metrics"]["test_macro_f1"] >= 0.85
    assert bundle.metadata["metrics"]["test_accuracy_lift_over_baseline"] >= 0.20
    assert set(bundle.metadata["reference_statistics"]) == set(FEATURE_NAMES)


def test_mastery_checkpoint_records_problem_baseline_and_experiment(artifact_dir):
    metadata = load_bundle(artifact_dir).metadata
    framing = metadata["problem_framing"]
    experiment = metadata["experiment"]
    metrics = metadata["metrics"]

    assert framing["prediction_unit"]
    assert framing["target"]
    assert framing["supported_decision"]
    assert "quality" in framing["prohibited_decisions"]
    assert experiment["hypothesis"]
    assert experiment["selection_metric"] == "stratified five-fold macro F1 on training data only"
    assert len(experiment["candidate_C_values"]) >= 3
    assert metrics["test_accuracy"] > metrics["baseline_test_accuracy"]
    assert metrics["test_macro_f1"] > metrics["baseline_test_macro_f1"]


def test_saved_metadata_is_json_and_predictions_are_probabilities(artifact_dir):
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    X, _, _ = load_dataset()
    bundle = load_bundle(artifact_dir)
    probabilities = bundle.model.predict_proba(X.iloc[:3])
    assert metadata["schema_version"] == "1.0"
    assert probabilities.shape == (3, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
