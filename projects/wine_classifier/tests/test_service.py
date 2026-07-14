from __future__ import annotations

from fastapi.testclient import TestClient

from wine_classifier.app import create_app
from wine_classifier.training import FEATURE_NAMES, load_dataset


def first_payload():
    X, _, _ = load_dataset()
    return {name: float(X.iloc[0][name]) for name in FEATURE_NAMES}


def test_health_prediction_metadata_and_metrics(artifact_dir):
    client = TestClient(create_app(artifact_dir))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    prediction = client.post("/v1/predict", json=first_payload())
    assert prediction.status_code == 200
    body = prediction.json()
    assert body["predicted_class_name"] in {"class_0", "class_1", "class_2"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-9

    metadata = client.get("/v1/model")
    assert metadata.status_code == 200
    assert metadata.json()["feature_names"] == FEATURE_NAMES

    metrics = client.get("/metrics")
    assert metrics.json()["requests"] == 1


def test_schema_rejects_missing_and_extra_features(artifact_dir):
    client = TestClient(create_app(artifact_dir))
    payload = first_payload()
    payload.pop("alcohol")
    assert client.post("/v1/predict", json=payload).status_code == 422

    payload = first_payload()
    payload["unknown_feature"] = 1.0
    assert client.post("/v1/predict", json=payload).status_code == 422


def test_out_of_range_input_returns_warning(artifact_dir):
    client = TestClient(create_app(artifact_dir))
    payload = first_payload()
    payload["alcohol"] = 100.0
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    assert any("alcohol" in warning for warning in response.json()["warnings"])
