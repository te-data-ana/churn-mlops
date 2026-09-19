import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from churn_mlops.serving.api import app
from churn_mlops.serving.model_loader import LoadedModel, ModelMetadata
from churn_mlops.serving.predictor import PredictionResult, Predictor
from churn_mlops.serving.schemas import InputFeatures

client = TestClient(app)


@pytest.fixture
def sample_input():
    return InputFeatures(
        age=45,
        tenure=24,
        usage_frequency=12,
        support_calls=1,
        payment_delay=0,
        last_interaction=5,
        total_spend=800.0,
        gender="Female",
        subscription_type="Premium",
        contract_length="Annual",
    )


@pytest.fixture
def sample_json(sample_input):
    return sample_input.model_dump()


@pytest.fixture
def mock_model_factory():

    def factory(proba=0.5, threshold=0.5):

        class MockModel:
            def predict_proba(self, X):
                return np.array([[1 - proba, proba]])

        metadata = ModelMetadata(
            model_name="churn-propensity",
            model_alias="champion",
            model_version=5,
            threshold=threshold,
        )

        return LoadedModel(
            model=MockModel(),
            metadata=metadata,
        )

    return factory


@pytest.mark.unit
def test_schemas_input_validity():
    with pytest.raises(ValidationError):
        InputFeatures(
            age=0,
            tenure=0,
            usage_frequency=0,
            support_calls=-1,
            payment_delay=-1,
            last_interaction=0,
            total_spend=0.0,
            gender="XY",
            subscription_type="Free",
            contract_length="undefined",
        )


@pytest.mark.unit
def test_predictor_predict_output(mock_model_factory, sample_input):

    loaded_model = mock_model_factory(proba=0.8, threshold=0.5)

    predictor = Predictor(loaded_model)

    result = predictor.predict(sample_input)

    assert isinstance(result, PredictionResult)
    assert result.predicted_class == 1
    assert result.predicted_probability == 0.8
    assert result.metadata == loaded_model.metadata
    assert result.metadata.model_name == "churn-propensity"
    assert result.metadata.model_alias == "champion"
    assert result.metadata.threshold == 0.5


@pytest.mark.unit
def test_predictor_respects_threshold(mock_model_factory, sample_input):

    loaded_model_0 = mock_model_factory(proba=0.4, threshold=0.5)
    loaded_model_1 = mock_model_factory(proba=0.4, threshold=0.3)

    predictor_0 = Predictor(loaded_model_0)
    predictor_1 = Predictor(loaded_model_1)

    result_0 = predictor_0.predict(sample_input)
    result_1 = predictor_1.predict(sample_input)

    assert result_0.predicted_class == 0
    assert result_1.predicted_class == 1


@pytest.mark.unit
def test_api_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.unit
def test_api_predict_endpoint(sample_json):

    response = client.post("/predict", json=sample_json)

    payload = response.json()

    assert response.status_code == 200

    assert "predicted_class" in payload
    assert "predicted_probability" in payload
    assert "metadata" in payload

    assert payload["predicted_class"] in {0, 1}
    assert 0 <= payload["predicted_probability"] <= 1


@pytest.mark.unit
def test_api_predict_endpoint_invalid_payload():

    response = client.post("/predict", json={"age": 45, "gender": "XY"})

    assert response.status_code == 422
