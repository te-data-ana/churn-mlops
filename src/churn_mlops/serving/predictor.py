from dataclasses import dataclass

import pandas as pd

from churn_mlops.serving.model_loader import LoadedModel, ModelMetadata
from churn_mlops.serving.schemas import InputFeatures


@dataclass
class PredictionResult:
    predicted_class: int
    predicted_probability: float
    metadata: ModelMetadata


class Predictor:
    def __init__(self, loaded_model: LoadedModel):
        self.loaded_model = loaded_model

    def predict(self, payload: InputFeatures) -> PredictionResult:
        df = pd.DataFrame([payload.model_dump()])

        model = self.loaded_model.model
        threshold = self.loaded_model.metadata.threshold

        predicted_probability = float(model.predict_proba(df)[0, 1])

        predicted_class = int(predicted_probability >= threshold)

        return PredictionResult(
            predicted_class=predicted_class,
            predicted_probability=predicted_probability,
            metadata=self.loaded_model.metadata,
        )
