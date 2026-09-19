import pandas as pd

from churn_mlops.data.schemas import InferenceDataSchema, TrainingDataSchema


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
    return TrainingDataSchema.validate(df)


def validate_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a DataFrame against the inference data schema.

    Args:
        df: DataFrame containing the required inference feature columns.

    Returns:
        The validated and type-coerced DataFrame.

    Raises:
        pandera.errors.SchemaError: If the DataFrame violates the schema.
    """
    return InferenceDataSchema.validate(df)
