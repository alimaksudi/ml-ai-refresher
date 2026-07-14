"""Versioned and validated model-artifact contract."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
REQUIRED_METADATA = {
    "schema_version",
    "model_version",
    "trained_at_utc",
    "dataset",
    "dataset_sha256",
    "feature_names",
    "class_names",
    "metrics",
    "reference_statistics",
}


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    metadata: dict[str, Any]


def validate_metadata(metadata: dict[str, Any]) -> None:
    missing = REQUIRED_METADATA - metadata.keys()
    if missing:
        raise ValueError(f"artifact metadata is missing: {sorted(missing)}")
    features = metadata["feature_names"]
    if not isinstance(features, list) or not features:
        raise ValueError("feature_names must be a non-empty list")
    if set(features) != set(metadata["reference_statistics"]):
        raise ValueError("reference statistics must cover every feature exactly")


def save_bundle(model: Any, metadata: dict[str, Any], artifact_dir: Path) -> None:
    """Write model and metadata atomically within one artifact directory."""
    validate_metadata(metadata)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_tmp = artifact_dir / f".{MODEL_FILE}.tmp"
    metadata_tmp = artifact_dir / f".{METADATA_FILE}.tmp"
    joblib.dump(model, model_tmp)
    metadata_tmp.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(model_tmp, artifact_dir / MODEL_FILE)
    os.replace(metadata_tmp, artifact_dir / METADATA_FILE)


def load_bundle(artifact_dir: Path) -> ModelBundle:
    model_path = artifact_dir / MODEL_FILE
    metadata_path = artifact_dir / METADATA_FILE
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"model artifacts are missing from {artifact_dir}; run the training command"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_metadata(metadata)
    return ModelBundle(model=joblib.load(model_path), metadata=metadata)
