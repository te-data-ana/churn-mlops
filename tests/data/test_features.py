import numpy as np
import pandas as pd
import pytest

from churn_mlops.data import FeatureBuilder


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "tenure": [0, 12, 24],
            "contract_length": ["Monthly", "Quarterly", "Annual"],
            "usage_frequency": [10, 20, 30],
            "last_interaction": [5, 15, 20],
            "total_spend": [0, 400, 1000],
            "support_calls": [10, 6, 1],
            "payment_delay": [25, 10, 0],
        }
    )


def test_engineered_features_exist_and_finite(sample_df):
    transformed_df = FeatureBuilder().transform(sample_df)

    engineered_cols = [
        "contract_commitment",
        "tenure_rel2_commitment",
        "engagement",
        "avg_spend_per_year",
        "avg_s_calls_per_year",
        "late_payer",
        "inactive_customer",
    ]

    assert set(engineered_cols).issubset(set(transformed_df.columns))
    assert not transformed_df[engineered_cols].isna().values.any()
    assert np.isfinite(transformed_df[engineered_cols]).values.all()


def test_input_df_not_modified(sample_df):
    original_df = sample_df.copy(deep=True)

    transformed_df = FeatureBuilder().transform(sample_df)

    pd.testing.assert_frame_equal(
        sample_df,
        original_df,
    )

    pd.testing.assert_frame_equal(
        transformed_df[original_df.columns],
        original_df,
    )


def test_handles_zero_tenure(sample_df):
    transformed_df = FeatureBuilder().transform(sample_df)

    assert transformed_df["avg_spend_per_year"].iloc[0] == 0
    assert transformed_df["avg_s_calls_per_year"].iloc[0] == 0


def test_contract_mapping(sample_df):
    transformed_df = FeatureBuilder().transform(sample_df)

    assert transformed_df["contract_commitment"].iloc[0] == 1
    assert transformed_df["contract_commitment"].iloc[1] == 4
    assert transformed_df["contract_commitment"].iloc[2] == 12


def test_tenure_rel2_commitment(sample_df):
    transformed_df = FeatureBuilder().transform(sample_df)

    assert transformed_df["tenure_rel2_commitment"].iloc[1] == 3
    assert transformed_df["tenure_rel2_commitment"].iloc[2] == 2


def test_engagement(sample_df):
    transformed_df = FeatureBuilder().transform(sample_df)

    assert transformed_df["engagement"].iloc[1] == 10
    assert transformed_df["engagement"].iloc[2] == 10


def test_avg_spend_per_year(sample_df):
    transformed_df = FeatureBuilder().transform(sample_df)

    assert transformed_df["avg_spend_per_year"].iloc[1] == 400
    assert transformed_df["avg_spend_per_year"].iloc[2] == 500


def test_avg_s_calls_per_year(sample_df):
    transformed_df = FeatureBuilder().transform(sample_df)

    assert transformed_df["avg_s_calls_per_year"].iloc[1] == 6
    assert transformed_df["avg_s_calls_per_year"].iloc[2] == 1 / 2


def test_late_payer_flag(sample_df):
    transformed_df = FeatureBuilder().transform(sample_df)

    assert transformed_df["late_payer"].iloc[0] == 1
    assert transformed_df["late_payer"].iloc[1] == 0
    assert transformed_df["late_payer"].iloc[2] == 0


def test_inactive_customer(sample_df):
    transformed_df = FeatureBuilder().transform(sample_df)

    assert transformed_df["inactive_customer"].iloc[0] == 0
    assert transformed_df["inactive_customer"].iloc[1] == 0
    assert transformed_df["inactive_customer"].iloc[2] == 1


def test_get_feature_names_out(sample_df):
    transformer = FeatureBuilder()

    features = transformer.get_feature_names_out(sample_df.columns)

    expected_features = {
        "tenure",
        "contract_length",
        "usage_frequency",
        "last_interaction",
        "total_spend",
        "support_calls",
        "payment_delay",
        "contract_commitment",
        "tenure_rel2_commitment",
        "engagement",
        "avg_spend_per_year",
        "avg_s_calls_per_year",
        "late_payer",
        "inactive_customer",
    }

    assert expected_features.issubset(set(features))
