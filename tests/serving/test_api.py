from typing import Any

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
def test_health_endpoint_returns_healthy_status(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.unit
def test_predict_endpoint_returns_prediction_payload(
    client: TestClient,
    sample_json: Any,
) -> None:
    response = client.post("/predict", json=sample_json)

    payload = response.json()

    assert response.status_code == 200
    assert "predicted_class" in payload
    assert "predicted_probability" in payload
    assert "metadata" in payload
    assert payload["predicted_class"] in {0, 1}
    assert 0 <= payload["predicted_probability"] <= 1


@pytest.mark.unit
def test_predict_endpoint_rejects_invalid_payload(client: TestClient) -> None:
    response = client.post("/predict", json={"age": 45, "gender": "XY"})

    assert response.status_code == 422
