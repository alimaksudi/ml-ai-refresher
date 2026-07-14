"""Inference service independent of the web framework."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .artifact import ModelBundle, load_bundle
from .monitoring import RequestMonitor


@dataclass(frozen=True)
class Prediction:
    predicted_class_id: int
    predicted_class_name: str
    probabilities: dict[str, float]
    model_version: str
    warnings: list[str]


class ModelService:
    def __init__(self, bundle: ModelBundle) -> None:
        self.model = bundle.model
        self.metadata = bundle.metadata
        self.monitor = RequestMonitor(bundle.metadata["reference_statistics"])

    @classmethod
    def from_artifact_dir(cls, artifact_dir: Path) -> "ModelService":
        return cls(load_bundle(artifact_dir))

    @property
    def feature_names(self) -> list[str]:
        return list(self.metadata["feature_names"])

    def predict(self, features: dict[str, float]) -> Prediction:
        missing = set(self.feature_names) - features.keys()
        extra = features.keys() - set(self.feature_names)
        if missing or extra:
            raise ValueError(f"feature mismatch; missing={sorted(missing)}, extra={sorted(extra)}")

        row = pd.DataFrame([[features[name] for name in self.feature_names]], columns=self.feature_names)
        class_id = int(self.model.predict(row)[0])
        raw_probabilities = self.model.predict_proba(row)[0]
        class_names = self.metadata["class_names"]
        probabilities = {
            class_names[str(index)]: float(probability)
            for index, probability in enumerate(raw_probabilities)
        }
        class_name = class_names[str(class_id)]
        warnings = self.monitor.record(features, class_name)
        return Prediction(
            predicted_class_id=class_id,
            predicted_class_name=class_name,
            probabilities=probabilities,
            model_version=self.metadata["model_version"],
            warnings=warnings,
        )

    def public_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.metadata["schema_version"],
            "model_version": self.metadata["model_version"],
            "trained_at_utc": self.metadata["trained_at_utc"],
            "dataset": self.metadata["dataset"],
            "feature_names": self.metadata["feature_names"],
            "class_names": self.metadata["class_names"],
            "metrics": self.metadata["metrics"],
            "intended_use": self.metadata["intended_use"],
        }
