import logging

import pandas as pd

from churn_mlops.data.schemas import InferenceDataSchema, TrainingDataSchema

logger = logging.getLogger(__name__)


def validate_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a DataFrame against the training data schema.

    Args:
        df: DataFrame containing inference features and a binary ``churn``
            target column.

    Returns:
        The validated and type-coerced DataFrame.

    Raises:
        pandera.errors.SchemaError: If the DataFrame violates the schema.
    """
    try:
        logger.info("Validating training data schema for %d rows.", len(df))
        validated = TrainingDataSchema.validate(df)
        logger.info("Training data validation passed for %d rows.", len(validated))
        return validated
    except Exception:
        logger.exception("Training data validation failed.")
        raise


def validate_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a DataFrame against the inference data schema.

    Args:
        df: DataFrame containing the required inference feature columns.

    Returns:
        The validated and type-coerced DataFrame.

    Raises:
        pandera.errors.SchemaError: If the DataFrame violates the schema.
    """
    try:
        logger.info("Validating inference data schema for %d rows.", len(df))
        validated = InferenceDataSchema.validate(df)
        logger.info("Inference data validation passed for %d rows.", len(validated))
        return validated
    except Exception:
        logger.exception("Inference data validation failed.")
        raise
