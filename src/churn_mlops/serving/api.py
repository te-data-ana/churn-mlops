from fastapi import FastAPI

from churn_mlops.serving.model_loader import load_model
from churn_mlops.serving.predictor import Predictor
from churn_mlops.serving.schemas import InputFeatures

app = FastAPI()


def get_predictor() -> Predictor:
    return Predictor(load_model())


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(features: InputFeatures):
    predictor = get_predictor()
    return predictor.predict(features)
