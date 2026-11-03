import logging

import pandas as pd

from churn_mlops.data import validate_data
from churn_mlops.serving import Predictor, load_model
from churn_mlops.tracking import TrainingManifest

logger = logging.getLogger(__name__)


def run_batch_prediction(
    df: pd.DataFrame,
    tracking_uri: str | None = None,
    model_name: str | None = None,
    model_alias: str | None = None,
    model_version: int | None = None,
    manifest: TrainingManifest | None = None,
) -> pd.DataFrame:
    """Validate input data and return a DataFrame containing predictions.

    Args:
        df: Customer feature DataFrame to score.
        tracking_uri: Optional MLflow tracking URI.
        model_name: Optional registered model name.
        model_alias: Optional registered model alias.
        model_version: Optional exact registered model version.
        manifest: Optional manifest identifying an exact registered or logged
            model.

    Returns:
        Validated input data with model predictions.
    """
    try:
        logger.info("Starting batch prediction for %d rows.", len(df))
        df = validate_data(df)

        loaded_model = load_model(
            tracking_uri=tracking_uri,
            model_name=model_name,
            model_alias=model_alias,
            model_version=model_version,
            manifest=manifest,
        )

        predictor = Predictor(loaded_model)
        df_pred = predictor.predict_batch(df=df)

        logger.info("Generated %d predictions.", len(df_pred))
        logger.info(
            "Average churn probability: %.4f",
            df_pred["predicted_probability"].mean(),
        )
        return df_pred
    except Exception:
        logger.exception("Batch prediction failed.")
        raise
