from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from churn_mlops.serving import api
from churn_mlops.serving.model_loader import LoadedModel, load_model


@pytest.mark.unit
def test_add_request_id_header_is_preserved(client: TestClient) -> None:
    response = client.get(
        "/health",
        headers={"X-Request-ID": "test-request-id"},
    )

    assert response.headers["X-Request-ID"] == "test-request-id"


@pytest.mark.unit
def test_add_request_id_header_is_generated(client: TestClient) -> None:
    response = client.get("/health")

    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"]


@pytest.mark.unit
def test_health_endpoint_returns_healthy_status(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.unit
def test_ready_endpoint_returns_ready_status(client: TestClient) -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@pytest.mark.unit
def test_predict_endpoint_returns_prediction_payload(
    client: TestClient,
    sample_dict: dict[str, Any],
) -> None:
    response = client.post("/predict", json=sample_dict)

    payload = response.json()

    assert response.status_code == 200
    assert "predicted_class" in payload
    assert "predicted_probability" in payload
    assert "metadata" in payload
    assert payload["predicted_class"] in {0, 1}
    assert 0 <= payload["predicted_probability"] <= 1


@pytest.mark.unit
def test_predict_logs_error_when_prediction_fails(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    sample_dict: dict[str, Any],
):
    class FailingPredictor:
        def predict(self, features):
            raise RuntimeError("prediction failed")

    calls = {}

    def log_error(**kwargs):
        calls.update(kwargs)

    client.app.state.predictor = FailingPredictor()
    monkeypatch.setattr(
        client.app.state.prediction_logger,
        "log_error",
        log_error,
    )

    with pytest.raises(RuntimeError, match="prediction failed"):
        response = client.post("/predict", json=sample_dict)
        assert response.status_code == 500
        assert calls["request_id"]
        assert calls["latency_ms"] >= 0
        assert isinstance(calls["exception"], RuntimeError)


@pytest.mark.unit
def test_predict_handles_log_error_failure(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    sample_dict: dict[str, Any],
):
    class FailingPredictor:
        def predict(self, features):
            raise RuntimeError("prediction failed")

    def failing_log_error(**kwargs):
        raise RuntimeError("cannot write log")

    client.app.state.predictor = FailingPredictor()
    monkeypatch.setattr(
        client.app.state.prediction_logger,
        "log_error",
        failing_log_error,
    )

    with pytest.raises(RuntimeError, match="prediction failed"):
        response = client.post("/predict", json=sample_dict)
        assert response.status_code == 500


@pytest.mark.unit
def test_predictor_loads_once_per_api_lifetime(
    monkeypatch: pytest.MonkeyPatch,
    mock_model_factory: Callable[..., LoadedModel],
    sample_dict: dict[str, Any],
) -> None:
    load_count = 0

    def load_once() -> LoadedModel:
        nonlocal load_count
        load_count += 1
        return mock_model_factory()

    monkeypatch.setattr(api, "load_model", load_once)

    with TestClient(api.app) as test_client:
        first_response = test_client.post("/predict", json=sample_dict)
        second_response = test_client.post("/predict", json=sample_dict)

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert load_count == 1


@pytest.mark.unit
def test_api_reports_not_ready_when_model_loading_fails(
    monkeypatch: pytest.MonkeyPatch,
    sample_dict: dict[str, Any],
) -> None:
    def fail_to_load() -> None:
        raise RuntimeError("model alias is unavailable")

    monkeypatch.setattr(api, "load_model", fail_to_load)

    with TestClient(api.app) as test_client:
        health_response = test_client.get("/health")
        ready_response = test_client.get("/ready")
        predict_response = test_client.post("/predict", json=sample_dict)

    assert health_response.status_code == 200
    assert ready_response.status_code == 503
    assert ready_response.json() == {"status": "not_ready"}
    assert predict_response.status_code == 503


@pytest.mark.unit
def test_predict_endpoint_rejects_invalid_payload(client: TestClient) -> None:
    response = client.post("/predict", json={"age": 45, "gender": "XY"})

    assert response.status_code == 422


@pytest.mark.integration
def test_api_serves_predictions_from_registered_model(
    monkeypatch: pytest.MonkeyPatch,
    registered_model: dict[str, object],
    sample_dict: dict[str, Any],
) -> None:
    def load_registered_model() -> LoadedModel:
        return load_model(
            tracking_uri=str(registered_model["tracking_uri"]),
            model_name=str(registered_model["model_name"]),
            model_alias=str(registered_model["model_alias"]),
        )

    monkeypatch.setattr(api, "load_model", load_registered_model)

    with TestClient(api.app) as test_client:
        health_response = test_client.get("/health")
        ready_response = test_client.get("/ready")
        predict_response = test_client.post("/predict", json=sample_dict)

    assert health_response.status_code == 200
    assert ready_response.status_code == 200
    assert predict_response.status_code == 200
    assert 0 <= predict_response.json()["predicted_probability"] <= 1
