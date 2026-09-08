import pytest
from pandera.errors import SchemaError

from churn_mlops.data import load_raw_data, validate_data


@pytest.fixture
def ingested_df():
    return load_raw_data(
        "customer_churn_dataset-training.csv",
        index_col="customerid",
    )


def test_validation_of_training_data(ingested_df):

    validated_df = validate_data(ingested_df)

    assert validated_df.shape == ingested_df.shape


def test_validation_rejects_invalid_dtype(ingested_df):
    ingested_df.loc[ingested_df.index[0], "age"] = "not_an_int"

    with pytest.raises(SchemaError, match="age"):
        validate_data(ingested_df)


def test_validation_rejects_invalid_churn_value(ingested_df):
    ingested_df.loc[ingested_df.index[0], "churn"] = 2

    with pytest.raises(SchemaError, match="churn"):
        validate_data(ingested_df)


def test_validation_rejects_negative_total_spend(ingested_df):
    ingested_df.loc[ingested_df.index[0], "total_spend"] = -100

    with pytest.raises(SchemaError, match="total_spend"):
        validate_data(ingested_df)


def test_validation_rejects_age_10(ingested_df):
    ingested_df.loc[ingested_df.index[0], "age"] = 10

    with pytest.raises(SchemaError, match="age"):
        validate_data(ingested_df)
