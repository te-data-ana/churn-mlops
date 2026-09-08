import pandas as pd

from churn_mlops.data.schemas import ChurnDataSchema


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    return ChurnDataSchema.validate(df)
