"""FastAPI application factory for the capstone model."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from .service import ModelService

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


class WineFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


class PredictionResponse(BaseModel):
    predicted_class_id: int
    predicted_class_name: str
    probabilities: dict[str, float]
    model_version: str
    warnings: list[str]


def create_app(artifact_dir: Path | None = None) -> FastAPI:
    resolved = artifact_dir or Path(
        os.environ.get("WINE_MODEL_DIR", str(DEFAULT_ARTIFACT_DIR))
    )
    service = ModelService.from_artifact_dir(resolved)
    app = FastAPI(
        title="Wine Cultivar Classifier",
        version=service.metadata["model_version"],
        description="Educational real-data ML serving capstone. Not for commercial decisions.",
    )
    app.state.model_service = service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_version": service.metadata["model_version"]}

    @app.get("/v1/model")
    def model_metadata() -> dict:
        return service.public_metadata()

    @app.post("/v1/predict", response_model=PredictionResponse)
    def predict(payload: WineFeatures) -> PredictionResponse:
        try:
            values = payload.model_dump()
            result = service.predict(values)
            return PredictionResponse(**result.__dict__)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/metrics")
    def metrics() -> dict:
        return service.monitor.snapshot()

    return app


app = create_app()
