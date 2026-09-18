import pandas as pd

from churn_mlops.data.schemas import InferenceDataSchema, TrainingDataSchema


def validate_training_data(df: pd.DataFrame) -> pd.DataFrame:
    return TrainingDataSchema.validate(df)


def validate_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    return InferenceDataSchema.validate(df)
