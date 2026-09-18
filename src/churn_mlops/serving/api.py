from fastapi import FastAPI

from churn_mlops.serving.model_loader import load_model
from churn_mlops.serving.predictor import Predictor
from churn_mlops.serving.schemas import InputFeatures

app = FastAPI()

predictor = Predictor(load_model())


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(features: InputFeatures):
    return predictor.predict(features)
