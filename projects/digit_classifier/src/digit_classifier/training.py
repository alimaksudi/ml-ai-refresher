"""Reproducible PyTorch checkpoint on the bundled sklearn digits dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DigitMLP(nn.Module):
    def __init__(self, dropout: float = 0.15):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(16 * 2 * 2, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs.reshape(-1, 1, 8, 8))
        return self.classifier(features.flatten(start_dim=1))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def dataset_hash(X: np.ndarray, y: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(X, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(y, dtype=np.int64).tobytes())
    return digest.hexdigest()


def load_splits(seed: int = 42):
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )
    scaler = StandardScaler().fit(X_train)
    return (
        scaler.transform(X_train).astype(np.float32),
        y_train,
        scaler.transform(X_validation).astype(np.float32),
        y_validation,
        scaler.transform(X_test).astype(np.float32),
        y_test,
        scaler,
        dataset_hash(X, y),
    )


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        probabilities = torch.softmax(logits, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro")),
        "log_loss": float(log_loss(y, probabilities, labels=list(range(10)))),
    }


def train(seed: int = 42, max_epochs: int = 60, patience: int = 8):
    seed_everything(seed)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, fingerprint = load_splits(seed)

    naive = DummyClassifier(strategy="most_frequent")
    naive.fit(X_train, y_train)
    naive_predictions = naive.predict(X_test)
    naive_metrics = {
        "accuracy": float(accuracy_score(y_test, naive_predictions)),
        "macro_f1": float(f1_score(y_test, naive_predictions, average="macro")),
    }

    baseline = LogisticRegression(max_iter=3000, random_state=seed)
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)
    baseline_metrics = {
        "accuracy": float(accuracy_score(y_test, baseline_predictions)),
        "macro_f1": float(f1_score(y_test, baseline_predictions, average="macro")),
    }

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=64,
        shuffle=True,
        generator=generator,
    )
    model = DigitMLP()
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite gradient norm")
            optimizer.step()
            train_loss += loss.item() * len(features)

        validation = evaluate(model, X_val, y_val)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss / len(X_train),
            "validation_log_loss": validation["log_loss"],
            "validation_accuracy": validation["accuracy"],
        })
        if validation["log_loss"] < best_loss - 1e-4:
            best_loss = validation["log_loss"]
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, X_test, y_test)

    # Same split, seed, batch size, optimizer family, and selection rule for the CNN.
    cnn_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    cnn = DigitCNN()
    for layer in cnn.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    cnn_optimizer = torch.optim.AdamW(cnn.parameters(), lr=1e-3, weight_decay=1e-4)
    cnn_best_loss = float("inf")
    cnn_best_state = None
    cnn_best_epoch = 0
    cnn_stale = 0
    for epoch in range(1, max_epochs + 1):
        cnn.train()
        for features, labels in cnn_loader:
            cnn_optimizer.zero_grad(set_to_none=True)
            loss = criterion(cnn(features), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(cnn.parameters(), max_norm=5.0)
            cnn_optimizer.step()
        cnn_validation = evaluate(cnn, X_val, y_val)
        if cnn_validation["log_loss"] < cnn_best_loss - 1e-4:
            cnn_best_loss = cnn_validation["log_loss"]
            cnn_best_epoch = epoch
            cnn_best_state = {
                key: value.detach().clone() for key, value in cnn.state_dict().items()
            }
            cnn_stale = 0
        else:
            cnn_stale += 1
            if cnn_stale >= patience:
                break
    if cnn_best_state is None:
        raise RuntimeError("CNN training produced no checkpoint")
    cnn.load_state_dict(cnn_best_state)
    cnn_test_metrics = evaluate(cnn, X_test, y_test)
    metadata = {
        "schema_version": "1.0",
        "model_version": "digits-mlp-v1",
        "framework": f"torch-{torch.__version__}",
        "seed": seed,
        "dataset": "sklearn digits",
        "dataset_sha256": fingerprint,
        "input_contract": {"shape": [64], "dtype": "float32", "classes": list(range(10))},
        "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test), "stratified": True},
        "selection": {"metric": "validation log loss", "best_epoch": best_epoch, "patience": patience},
        "naive_baseline": {"model": "most frequent class", **naive_metrics},
        "linear_baseline": {"model": "scaled logistic regression", **baseline_metrics},
        "mlp_test": test_metrics,
        "cnn_test": cnn_test_metrics,
        "cnn_selection": {"metric": "validation log loss", "best_epoch": cnn_best_epoch},
        "ablation": {"control": "dropout=0.15", "required_follow_up": "compare dropout=0.0 under identical split and seed"},
        "history": history,
    }
    return model, cnn, scaler, metadata


def save_checkpoint(output_dir: Path, seed: int = 42) -> dict:
    model, cnn, scaler, metadata = train(seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    torch.save(cnn.state_dict(), output_dir / "cnn_model.pt")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metadata = save_checkpoint(args.output_dir, args.seed)
    print(json.dumps({
        "naive": metadata["naive_baseline"],
        "linear": metadata["linear_baseline"],
        "mlp": metadata["mlp_test"],
        "cnn": metadata["cnn_test"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
