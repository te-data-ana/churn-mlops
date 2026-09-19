from fastapi import FastAPI

from churn_mlops.serving.model_loader import load_model
from churn_mlops.serving.predictor import PredictionResult, Predictor
from churn_mlops.serving.schemas import InputFeatures

app = FastAPI()


def get_predictor() -> Predictor:
    """Load the configured serving model and construct its predictor."""
    return Predictor(load_model())


@app.get("/health")
def health() -> dict[str, str]:
    """Return the service health status."""
    return {"status": "healthy"}


@app.post("/predict")
def predict(features: InputFeatures) -> PredictionResult:
    """Predict churn for one validated customer feature record.

    Args:
        features: Validated request payload containing customer features.

    Returns:
        Prediction result produced by the configured registered model.
    """
    predictor = get_predictor()
    return predictor.predict(features)
