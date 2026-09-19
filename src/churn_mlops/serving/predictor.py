from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from churn_mlops.serving.model_loader import LoadedModel, ModelMetadata
from churn_mlops.serving.schemas import InputFeatures


@dataclass
class PredictionResult:
    predicted_class: int
    predicted_probability: float
    metadata: ModelMetadata


class Predictor:
    def __init__(self, loaded_model: LoadedModel) -> None:
        """Initialize a predictor from a loaded model and its metadata.

        Args:
            loaded_model: Model object and serving metadata returned by
                ``load_model``.
        """
        self.model = loaded_model.model
        self.metadata = loaded_model.metadata

    @staticmethod
    def _to_probability_array(probabilities: ArrayLike) -> np.ndarray:
        """Normalize model probabilities to a two-column NumPy array.

        Args:
            probabilities: One-dimensional positive-class probabilities or a
                two-dimensional class-probability array.

        Returns:
            Two-dimensional array with negative-class probabilities in column
            zero and positive-class probabilities in column one.

        Raises:
            ValueError: If the array is not two-dimensional with at least two
                columns after normalization.
        """
        array = np.asarray(probabilities)
        if array.ndim == 1:
            # A one-dimensional output contains positive-class probabilities;
            # complement them to construct the negative-class column.
            array = np.column_stack([1.0 - array, array])
        if array.ndim != 2 or array.shape[1] < 2:
            raise ValueError(
                "Model probability output must be 2D with at least two columns."
            )
        return array

    def predict(self, payload: InputFeatures) -> PredictionResult:
        """Predict churn for a single validated feature payload.

        Args:
            payload: Validated customer feature values.

        Returns:
            Prediction result containing the positive-class probability,
            thresholded class, and model metadata.
        """
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
        """Predict churn for multiple records while preserving their index.

        Args:
            df: DataFrame containing one customer record per row.

        Returns:
            DataFrame indexed like ``df`` with predicted probabilities,
            thresholded classes, the configured threshold, and model version.
        """
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
