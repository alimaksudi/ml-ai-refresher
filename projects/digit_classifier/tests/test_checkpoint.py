from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch

from digit_classifier.training import (
    DigitCNN,
    DigitMLP,
    TestData as SealedTestData,
    evaluate,
    file_hash,
    load_splits,
    save_checkpoint,
    select_models,
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """Train once; every test inspects the same frozen experiment."""
    output_dir = tmp_path_factory.mktemp("digit-checkpoint")
    metadata = save_checkpoint(output_dir)
    return output_dir, metadata


def test_checkpoint_meets_learning_contract(checkpoint):
    output_dir, metadata = checkpoint
    final_test = metadata["final_test"]

    assert final_test["mlp"]["accuracy"] >= 0.94
    assert final_test["mlp"]["macro_f1"] >= 0.94
    assert final_test["cnn"]["accuracy"] >= 0.94
    assert final_test["cnn"]["macro_f1"] >= 0.94
    assert final_test["linear_baseline"]["accuracy"] >= 0.90
    assert (
        final_test["naive_baseline"]["accuracy"]
        < final_test["linear_baseline"]["accuracy"]
    )

    for filename in ("model.pt", "cnn_model.pt", "scaler.joblib", "metadata.json"):
        assert (output_dir / filename).is_file()


def test_both_saved_models_reproduce_reported_metrics(checkpoint):
    output_dir, metadata = checkpoint
    _, _, _, _, X_test, y_test, _, _ = load_splits(metadata["split"]["seed"])

    selected_dropout = metadata["selection"]["dropout_ablation"]["selected_dropout"]
    mlp = DigitMLP(dropout=selected_dropout)
    mlp.load_state_dict(torch.load(output_dir / "model.pt", weights_only=True))
    cnn = DigitCNN()
    cnn.load_state_dict(torch.load(output_dir / "cnn_model.pt", weights_only=True))

    for model_name, model in (("mlp", mlp), ("cnn", cnn)):
        reproduced = evaluate(model, X_test, y_test)
        for metric in ("accuracy", "macro_f1", "log_loss"):
            assert reproduced[metric] == pytest.approx(
                metadata["final_test"][model_name][metric], abs=1e-12
            )


def test_selection_is_multi_seed_and_validation_only(checkpoint):
    _, metadata = checkpoint
    selection = metadata["selection"]

    assert selection["training_seeds"] == [11, 22, 33]
    assert len(selection["mlp_runs"]) == 3
    assert len(selection["cnn_runs"]) == 3
    assert len(selection["dropout_ablation"]["candidates"]) == 2
    assert len(selection["shift_augmentation_ablation"]["candidates"]) == 2
    assert {row["dropout"] for row in selection["dropout_ablation"]["candidates"]} == {
        0.0,
        0.15,
    }
    assert {
        row["enabled"]
        for row in selection["shift_augmentation_ablation"]["candidates"]
    } == {False, True}

    # The development-stage API cannot receive a TestData object accidentally.
    assert "test" not in inspect.signature(select_models).parameters
    assert "test" not in json.dumps(selection).lower()
    assert all(
        parameter.annotation is not SealedTestData
        for parameter in inspect.signature(select_models).parameters.values()
    )


def test_split_and_experiment_records_are_complete(checkpoint):
    _, metadata = checkpoint
    split = metadata["split"]

    assert split["seed"] == 42
    assert split["train"] + split["validation"] + split["test"] == 1797
    assert split["train_sha256"] != split["validation_sha256"]
    assert split["validation_sha256"] != split["test_sha256"]
    assert metadata["configuration"]["selection_metric"] == "validation log loss"
    assert metadata["configuration"]["output_initialization"] == "Xavier uniform"
    assert metadata["representative_histories"]["mlp"]
    assert metadata["representative_histories"]["cnn"]
    assert metadata["selection"]["run_variation"]["mlp"]["accuracy"][
        "standard_deviation"
    ] >= 0
    assert metadata["total_elapsed_seconds"] > 0

    for model_family in ("mlp_runs", "cnn_runs"):
        for run in metadata["selection"][model_family]:
            assert run["best_epoch"] <= run["epochs_ran"]
            assert run["elapsed_seconds"] > 0
            assert all(np.isfinite(value) for value in run["validation"].values())


def test_artifact_hashes_detect_file_changes(checkpoint):
    output_dir, metadata = checkpoint
    expected_paths = {
        "mlp": output_dir / "model.pt",
        "cnn": output_dir / "cnn_model.pt",
        "scaler": output_dir / "scaler.joblib",
    }
    assert metadata["artifact_sha256"] == {
        name: file_hash(path) for name, path in expected_paths.items()
    }
