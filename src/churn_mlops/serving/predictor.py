from dataclasses import dataclass

import numpy as np
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
        self.model = loaded_model.model
        self.metadata = loaded_model.metadata

    @staticmethod
    def _to_probability_array(probabilities):
        array = np.asarray(probabilities)
        if array.ndim == 1:
            array = np.column_stack([1.0 - array, array])
        if array.ndim != 2 or array.shape[1] < 2:
            raise ValueError(
                "Model probability output must be 2D with at least two columns."
            )
        return array

    def predict(self, payload: InputFeatures) -> PredictionResult:
        """Prediction of one single record of input features."""
        df = pd.DataFrame([payload.model_dump()])
        probabilities = self._to_probability_array(self.model.predict_proba(df))

        predicted_probability = float(probabilities[0, 1])
        predicted_class = int(predicted_probability >= self.metadata.threshold)

        return PredictionResult(
            predicted_class=predicted_class,
            predicted_probability=predicted_probability,
            metadata=self.metadata,
        )

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch prediction for a pandas dataframe containing multiple records."""
        probabilities = self._to_probability_array(self.model.predict_proba(df))
        predicted_probabilities = probabilities[:, 1]

        predicted_classes = (predicted_probabilities >= self.metadata.threshold).astype(
            int
        )

        result = pd.DataFrame(index=df.index.copy())

        result["predicted_probability"] = predicted_probabilities
        result["predicted_class"] = predicted_classes
        result["threshold"] = self.metadata.threshold
        result["model_version"] = self.metadata.model_version

        return result
