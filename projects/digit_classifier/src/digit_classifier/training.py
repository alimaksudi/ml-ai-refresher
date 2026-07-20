"""Reproducible, leakage-resistant experiments on sklearn's digits dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import sklearn
import torch
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SPLIT_SEED = 42
TRAINING_SEEDS = (11, 22, 33)
NUM_CLASSES = 10


@dataclass(frozen=True)
class DevelopmentData:
    """Only the data allowed during training and model selection."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    scaler: StandardScaler


@dataclass(frozen=True)
class TestData:
    """The sealed split opened once, after all choices have been frozen."""

    X_test: np.ndarray
    y_test: np.ndarray


@dataclass
class TrainingRun:
    model: nn.Module
    seed: int
    best_epoch: int
    validation: dict[str, float]
    history: list[dict[str, float | int]]
    elapsed_seconds: float


class DigitMLP(nn.Module):
    def __init__(self, dropout: float = 0.15):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 2 * 2, NUM_CLASSES)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs.reshape(-1, 1, 8, 8))
        return self.classifier(features.flatten(start_dim=1))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(split_seed: int = SPLIT_SEED) -> tuple[DevelopmentData, TestData, dict]:
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)

    X_train_raw, X_remaining, y_train, y_remaining = train_test_split(
        X, y, test_size=0.30, random_state=split_seed, stratify=y
    )
    X_validation_raw, X_test_raw, y_validation, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.50,
        random_state=split_seed,
        stratify=y_remaining,
    )

    # Fit statistics on training rows only. Validation and test imitate unseen data.
    scaler = StandardScaler().fit(X_train_raw)
    development = DevelopmentData(
        X_train=scaler.transform(X_train_raw).astype(np.float32),
        y_train=y_train,
        X_validation=scaler.transform(X_validation_raw).astype(np.float32),
        y_validation=y_validation,
        scaler=scaler,
    )
    test = TestData(
        X_test=scaler.transform(X_test_raw).astype(np.float32),
        y_test=y_test,
    )
    split_record = {
        "seed": split_seed,
        "train": len(y_train),
        "validation": len(y_validation),
        "test": len(y_test),
        "stratified": True,
        "dataset_sha256": array_hash(X, y),
        "train_sha256": array_hash(X_train_raw, y_train),
        "validation_sha256": array_hash(X_validation_raw, y_validation),
        "test_sha256": array_hash(X_test_raw, y_test),
    }
    return development, test, split_record


def load_splits(seed: int = SPLIT_SEED):
    """Compatibility helper used by notebooks and checkpoint tests."""
    development, test, split_record = load_data(split_seed=seed)
    return (
        development.X_train,
        development.y_train,
        development.X_validation,
        development.y_validation,
        test.X_test,
        test.y_test,
        development.scaler,
        split_record["dataset_sha256"],
    )


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro")),
        "log_loss": float(log_loss(y, probabilities, labels=list(range(NUM_CLASSES)))),
    }


def initialize_model(model: nn.Module) -> None:
    """Use ReLU-friendly hidden weights and neutral output logits."""
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    linear_layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
    for layer in linear_layers[:-1]:
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        nn.init.zeros_(layer.bias)
    nn.init.xavier_uniform_(linear_layers[-1].weight)
    nn.init.zeros_(linear_layers[-1].bias)


def shift_images_once(
    X_train: np.ndarray, y_train: np.ndarray, scaler: StandardScaler
) -> tuple[np.ndarray, np.ndarray]:
    """Add one no-wrap, one-pixel shift per image using only training rows."""
    raw_images = scaler.inverse_transform(X_train).reshape(-1, 8, 8)
    shifted_images = np.zeros_like(raw_images)

    # Cycle through four directions. Empty pixels stay black instead of wrapping.
    for index, image in enumerate(raw_images):
        direction = index % 4
        if direction == 0:  # up
            shifted_images[index, :-1, :] = image[1:, :]
        elif direction == 1:  # down
            shifted_images[index, 1:, :] = image[:-1, :]
        elif direction == 2:  # left
            shifted_images[index, :, :-1] = image[:, 1:]
        else:  # right
            shifted_images[index, :, 1:] = image[:, :-1]

    # A small dose teaches translation tolerance without letting synthetic rows
    # overwhelm the real handwriting distribution.
    selected_indices = np.arange(0, len(shifted_images), 4)
    shifted_scaled = scaler.transform(
        shifted_images[selected_indices].reshape(-1, 64)
    ).astype(np.float32)
    return (
        np.concatenate([X_train, shifted_scaled]),
        np.concatenate([y_train, y_train[selected_indices]]),
    )


def train_neural_model(
    model_factory: Callable[[], nn.Module],
    development: DevelopmentData,
    *,
    seed: int,
    augment: bool = False,
    max_epochs: int = 60,
    patience: int = 8,
) -> TrainingRun:
    seed_everything(seed)
    model = model_factory()
    initialize_model(model)

    X_training = development.X_train
    y_training = development.y_train
    if augment:
        X_training, y_training = shift_images_once(
            X_training, y_training, development.scaler
        )

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_training), torch.from_numpy(y_training)),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Training stopped because a gradient became non-finite.")
            optimizer.step()
            total_loss += loss.item() * len(features)

        validation = evaluate(
            model, development.X_validation, development.y_validation
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(X_training),
                "validation_log_loss": validation["log_loss"],
                "validation_accuracy": validation["accuracy"],
                "validation_macro_f1": validation["macro_f1"],
            }
        )
        if validation["log_loss"] < best_loss - 1e-4:
            best_loss = validation["log_loss"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a usable checkpoint.")
    model.load_state_dict(best_state)
    return TrainingRun(
        model=model,
        seed=seed,
        best_epoch=best_epoch,
        validation=evaluate(
            model, development.X_validation, development.y_validation
        ),
        history=history,
        elapsed_seconds=time.perf_counter() - start_time,
    )


def summarize_run(run: TrainingRun) -> dict:
    return {
        "seed": run.seed,
        "best_epoch": run.best_epoch,
        "epochs_ran": len(run.history),
        "elapsed_seconds": run.elapsed_seconds,
        "validation": run.validation,
    }


def representative_run(runs: list[TrainingRun]) -> TrainingRun:
    """Choose the run nearest median validation loss, never by test score."""
    median_loss = float(np.median([run.validation["log_loss"] for run in runs]))
    return min(
        runs,
        key=lambda run: (abs(run.validation["log_loss"] - median_loss), run.seed),
    )


def variation_summary(runs: list[TrainingRun]) -> dict:
    """Make stability visible instead of asking learners to infer it from rows."""
    return {
        metric: {
            "mean": float(np.mean([run.validation[metric] for run in runs])),
            "standard_deviation": float(
                np.std([run.validation[metric] for run in runs], ddof=0)
            ),
            "minimum": float(min(run.validation[metric] for run in runs)),
            "maximum": float(max(run.validation[metric] for run in runs)),
        }
        for metric in ("accuracy", "macro_f1", "log_loss")
    }


def select_models(
    development: DevelopmentData,
    *,
    training_seeds: tuple[int, ...] = TRAINING_SEEDS,
    max_epochs: int = 60,
    patience: int = 8,
) -> dict:
    """Make every modeling decision using train and validation data only."""
    selection_seed = training_seeds[0]

    dropout_runs = {
        dropout: train_neural_model(
            lambda dropout=dropout: DigitMLP(dropout=dropout),
            development,
            seed=selection_seed,
            max_epochs=max_epochs,
            patience=patience,
        )
        for dropout in (0.0, 0.15)
    }
    selected_dropout = min(
        dropout_runs,
        key=lambda value: dropout_runs[value].validation["log_loss"],
    )

    augmentation_runs = {
        augment: train_neural_model(
            DigitCNN,
            development,
            seed=selection_seed,
            augment=augment,
            max_epochs=max_epochs,
            patience=patience,
        )
        for augment in (False, True)
    }
    selected_augmentation = min(
        augmentation_runs,
        key=lambda value: augmentation_runs[value].validation["log_loss"],
    )

    # Reuse an ablation run when it already has the frozen configuration and seed.
    mlp_runs = []
    cnn_runs = []
    for seed in training_seeds:
        mlp_runs.append(
            dropout_runs[selected_dropout]
            if seed == selection_seed
            else train_neural_model(
                lambda: DigitMLP(dropout=selected_dropout),
                development,
                seed=seed,
                max_epochs=max_epochs,
                patience=patience,
            )
        )
        cnn_runs.append(
            augmentation_runs[selected_augmentation]
            if seed == selection_seed
            else train_neural_model(
                DigitCNN,
                development,
                seed=seed,
                augment=selected_augmentation,
                max_epochs=max_epochs,
                patience=patience,
            )
        )

    naive = DummyClassifier(strategy="most_frequent").fit(
        development.X_train, development.y_train
    )
    logistic = LogisticRegression(max_iter=3000, random_state=SPLIT_SEED).fit(
        development.X_train, development.y_train
    )
    logistic_probabilities = logistic.predict_proba(development.X_validation)
    logistic_predictions = logistic_probabilities.argmax(axis=1)
    linear_validation = {
        "accuracy": float(
            accuracy_score(development.y_validation, logistic_predictions)
        ),
        "macro_f1": float(
            f1_score(development.y_validation, logistic_predictions, average="macro")
        ),
        "log_loss": float(
            log_loss(
                development.y_validation,
                logistic_probabilities,
                labels=list(range(NUM_CLASSES)),
            )
        ),
    }

    median_mlp_accuracy = float(
        np.median([run.validation["accuracy"] for run in mlp_runs])
    )
    median_cnn_accuracy = float(
        np.median([run.validation["accuracy"] for run in cnn_runs])
    )
    best_neural_accuracy = max(median_mlp_accuracy, median_cnn_accuracy)
    complexity_earned = best_neural_accuracy >= linear_validation["accuracy"] + 0.005
    complexity_conclusion = (
        "A neural model earns its extra complexity because its median validation "
        "accuracy improves on logistic regression by at least 0.5 percentage points."
        if complexity_earned
        else "Logistic regression remains the default: the neural models do not improve "
        "median validation accuracy by at least 0.5 percentage points."
    )

    return {
        "mlp": representative_run(mlp_runs),
        "cnn": representative_run(cnn_runs),
        "naive": naive,
        "logistic": logistic,
        "record": {
            "rule": "Choose configurations and representative runs using validation log loss only.",
            "selection_seed": selection_seed,
            "training_seeds": list(training_seeds),
            "dropout_ablation": {
                "selected_dropout": selected_dropout,
                "candidates": [
                    {"dropout": value, **summarize_run(dropout_runs[value])}
                    for value in (0.0, 0.15)
                ],
            },
            "shift_augmentation_ablation": {
                "selected": selected_augmentation,
                "candidates": [
                    {"enabled": value, **summarize_run(augmentation_runs[value])}
                    for value in (False, True)
                ],
            },
            "mlp_runs": [summarize_run(run) for run in mlp_runs],
            "cnn_runs": [summarize_run(run) for run in cnn_runs],
            "run_variation": {
                "mlp": variation_summary(mlp_runs),
                "cnn": variation_summary(cnn_runs),
            },
            "representative_mlp_seed": representative_run(mlp_runs).seed,
            "representative_cnn_seed": representative_run(cnn_runs).seed,
            "linear_validation": linear_validation,
            "complexity_decision": {
                "minimum_accuracy_gain": 0.005,
                "median_mlp_accuracy": median_mlp_accuracy,
                "median_cnn_accuracy": median_cnn_accuracy,
                "linear_accuracy": linear_validation["accuracy"],
                "neural_complexity_earned": complexity_earned,
                "conclusion": complexity_conclusion,
            },
        },
    }


def finalize_on_test(selection: dict, test: TestData) -> dict:
    """Open the sealed split only after the selection record is complete."""
    y_test = test.y_test
    naive_predictions = selection["naive"].predict(test.X_test)
    logistic_probabilities = selection["logistic"].predict_proba(test.X_test)
    logistic_predictions = logistic_probabilities.argmax(axis=1)
    return {
        "naive_baseline": {
            "model": "most frequent class",
            "accuracy": float(accuracy_score(y_test, naive_predictions)),
            "macro_f1": float(f1_score(y_test, naive_predictions, average="macro")),
        },
        "linear_baseline": {
            "model": "scaled logistic regression",
            "accuracy": float(accuracy_score(y_test, logistic_predictions)),
            "macro_f1": float(f1_score(y_test, logistic_predictions, average="macro")),
            "log_loss": float(
                log_loss(
                    y_test,
                    logistic_probabilities,
                    labels=list(range(NUM_CLASSES)),
                )
            ),
        },
        "mlp": evaluate(selection["mlp"].model, test.X_test, y_test),
        "cnn": evaluate(selection["cnn"].model, test.X_test, y_test),
    }


def run_experiment(
    *,
    split_seed: int = SPLIT_SEED,
    training_seeds: tuple[int, ...] = TRAINING_SEEDS,
    max_epochs: int = 60,
    patience: int = 8,
) -> tuple[nn.Module, nn.Module, StandardScaler, dict]:
    if len(training_seeds) < 3:
        raise ValueError("Use at least three training seeds to measure run variation.")
    started = time.perf_counter()
    development, test, split_record = load_data(split_seed)
    selection = select_models(
        development,
        training_seeds=training_seeds,
        max_epochs=max_epochs,
        patience=patience,
    )
    final_test = finalize_on_test(selection, test)

    metadata = {
        "schema_version": "2.0",
        "models": {"mlp": "digits-mlp-v2", "cnn": "digits-cnn-v2"},
        "dataset": "sklearn digits",
        "split": split_record,
        "input_contract": {
            "shape": [64],
            "image_shape": [1, 8, 8],
            "dtype": "float32",
            "classes": list(range(NUM_CLASSES)),
            "preprocessing": "StandardScaler fitted on the training split only",
        },
        "configuration": {
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 64,
            "gradient_clip_norm": 5.0,
            "max_epochs": max_epochs,
            "early_stopping_patience": patience,
            "selection_metric": "validation log loss",
            "hidden_initialization": "Kaiming normal",
            "output_initialization": "Xavier uniform",
        },
        "selection": selection["record"],
        "representative_histories": {
            "mlp": selection["mlp"].history,
            "cnn": selection["cnn"].history,
        },
        "final_test": final_test,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "device": "cpu",
        },
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    return selection["mlp"].model, selection["cnn"].model, development.scaler, metadata


def train(seed: int = SPLIT_SEED, max_epochs: int = 60, patience: int = 8):
    """Backward-compatible entry point; ``seed`` controls the fixed data split."""
    return run_experiment(
        split_seed=seed,
        training_seeds=TRAINING_SEEDS,
        max_epochs=max_epochs,
        patience=patience,
    )


def save_checkpoint(output_dir: Path, seed: int = SPLIT_SEED) -> dict:
    model, cnn, scaler, metadata = train(seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "mlp": output_dir / "model.pt",
        "cnn": output_dir / "cnn_model.pt",
        "scaler": output_dir / "scaler.joblib",
    }
    torch.save(model.state_dict(), paths["mlp"])
    torch.save(cnn.state_dict(), paths["cnn"])
    joblib.dump(scaler, paths["scaler"])
    metadata["artifact_sha256"] = {
        name: file_hash(path) for name, path in paths.items()
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED, help="fixed split seed")
    args = parser.parse_args()
    metadata = save_checkpoint(args.output_dir, args.seed)
    print(json.dumps(metadata["final_test"], indent=2))
    print(metadata["selection"]["complexity_decision"]["conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
