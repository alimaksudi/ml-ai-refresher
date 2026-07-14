from __future__ import annotations

import json

import numpy as np
import torch

from digit_classifier.training import DigitMLP, load_splits, save_checkpoint


def test_deep_learning_mastery_checkpoint(tmp_path):
    metadata = save_checkpoint(tmp_path)
    assert metadata["mlp_test"]["accuracy"] >= 0.94
    assert metadata["mlp_test"]["macro_f1"] >= 0.94
    assert metadata["cnn_test"]["accuracy"] >= 0.94
    assert metadata["selection"]["best_epoch"] < len(metadata["history"]) + 1
    assert metadata["linear_baseline"]["accuracy"] >= 0.90
    assert metadata["naive_baseline"]["accuracy"] < metadata["linear_baseline"]["accuracy"]
    assert metadata["dataset_sha256"]
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "cnn_model.pt").exists()
    assert (tmp_path / "scaler.joblib").exists()


def test_saved_model_reproduces_reported_metrics(tmp_path):
    save_checkpoint(tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    _, _, _, _, X_test, y_test, _, _ = load_splits(metadata["seed"])
    model = DigitMLP()
    model.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(X_test)).argmax(dim=1).numpy()
    accuracy = float(np.mean(predictions == y_test))
    assert accuracy == metadata["mlp_test"]["accuracy"]


def test_metadata_keeps_test_separate_from_selection(tmp_path):
    metadata = save_checkpoint(tmp_path)
    assert metadata["selection"]["metric"] == "validation log loss"
    assert "test" not in metadata["selection"]["metric"]
    assert metadata["split"]["validation"] > 0
    assert metadata["split"]["test"] > 0
