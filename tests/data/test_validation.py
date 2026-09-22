import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from churn_mlops.data import load_raw_data, validate_training_data


@pytest.fixture
def ingested_df(generated_training_csv) -> pd.DataFrame:
    return load_raw_data(
        generated_training_csv,
        index_col="customerid",
    )


@pytest.mark.unit
def test_validate_training_data_accepts_valid_dataframe(
    ingested_df: pd.DataFrame,
) -> None:

    validated_df = validate_training_data(ingested_df)

    assert validated_df.shape == ingested_df.shape


@pytest.mark.unit
def test_validate_training_data_rejects_invalid_dtype(
    ingested_df: pd.DataFrame,
) -> None:
    df = ingested_df.copy()
    # A text value violates the numeric age schema.
    df["age"] = "age"

    with pytest.raises(SchemaErrors):
        validate_training_data(df)


@pytest.mark.unit
def test_validate_training_data_rejects_missing_required_column(
    ingested_df: pd.DataFrame,
) -> None:
    # Removing a required field should fail before row-level checks run.
    ingested_df = ingested_df.drop(columns=["age"])

    with pytest.raises(SchemaError):
        validate_training_data(ingested_df)


@pytest.mark.unit
def test_validate_training_data_rejects_invalid_churn_value(
    ingested_df: pd.DataFrame,
) -> None:
    # Churn is a binary target, so value 2 violates the data contract.
    ingested_df.loc[ingested_df.index[0], "churn"] = 2

    with pytest.raises(SchemaError, match="churn"):
        validate_training_data(ingested_df)


@pytest.mark.unit
def test_validate_training_data_rejects_negative_total_spend(
    ingested_df: pd.DataFrame,
) -> None:
    # Spending cannot be negative.
    ingested_df.loc[ingested_df.index[0], "total_spend"] = -100

    with pytest.raises(SchemaError, match="total_spend"):
        validate_training_data(ingested_df)


@pytest.mark.unit
def test_validate_training_data_rejects_age_below_minimum(
    ingested_df: pd.DataFrame,
) -> None:
    # The schema requires a realistic minimum age of 18.
    ingested_df.loc[ingested_df.index[0], "age"] = 10

    with pytest.raises(SchemaError, match="age"):
        validate_training_data(ingested_df)
