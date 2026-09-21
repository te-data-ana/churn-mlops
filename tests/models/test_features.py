import numpy as np
import pandas as pd
import pytest

from churn_mlops.models import FeatureBuilder


@pytest.mark.unit
def test_feature_builder_returns_finite_engineered_features(
    sample_features_df: pd.DataFrame,
) -> None:
    transformed_df = FeatureBuilder().transform(sample_features_df)

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


@pytest.mark.unit
def test_feature_builder_does_not_modify_input_dataframe(
    sample_features_df: pd.DataFrame,
) -> None:
    original_df = sample_features_df.copy(deep=True)

    transformed_df = FeatureBuilder().transform(sample_features_df)

    pd.testing.assert_frame_equal(
        sample_features_df,
        original_df,
    )

    pd.testing.assert_frame_equal(
        transformed_df[original_df.columns],
        original_df,
    )


@pytest.mark.unit
def test_feature_builder_handles_zero_tenure_without_division_error(
    sample_features_df: pd.DataFrame,
) -> None:
    # The first row of the sample_features_df fixture has zero tenure and exercises the division guard.
    transformed_df = FeatureBuilder().transform(sample_features_df)

    assert transformed_df["avg_spend_per_year"].iloc[0] == 0
    assert transformed_df["avg_s_calls_per_year"].iloc[0] == 0


@pytest.mark.unit
def test_feature_builder_maps_contract_length_to_months(
    sample_features_df: pd.DataFrame,
) -> None:
    # The first three rows of the sample_features_df fixture represent Monthly, Quarterly, and Annual plans.
    transformed_df = FeatureBuilder().transform(sample_features_df)

    assert transformed_df["contract_commitment"].iloc[0] == 1
    assert transformed_df["contract_commitment"].iloc[1] == 4
    assert transformed_df["contract_commitment"].iloc[2] == 12


@pytest.mark.unit
def test_feature_builder_calculates_relative_tenure(
    sample_features_df: pd.DataFrame,
) -> None:
    transformed_df = FeatureBuilder().transform(sample_features_df)

    assert transformed_df["tenure_rel2_commitment"].iloc[1] == 3
    assert transformed_df["tenure_rel2_commitment"].iloc[2] == 2


@pytest.mark.unit
def test_feature_builder_calculates_engagement(
    sample_features_df: pd.DataFrame,
) -> None:
    transformed_df = FeatureBuilder().transform(sample_features_df)

    assert transformed_df["engagement"].iloc[1] == 10
    assert transformed_df["engagement"].iloc[2] == 10


@pytest.mark.unit
def test_feature_builder_calculates_average_spend_per_year(
    sample_features_df: pd.DataFrame,
) -> None:
    transformed_df = FeatureBuilder().transform(sample_features_df)

    assert transformed_df["avg_spend_per_year"].iloc[1] == 400
    assert transformed_df["avg_spend_per_year"].iloc[2] == 500


@pytest.mark.unit
def test_feature_builder_calculates_average_support_calls_per_year(
    sample_features_df: pd.DataFrame,
) -> None:
    transformed_df = FeatureBuilder().transform(sample_features_df)

    assert transformed_df["avg_s_calls_per_year"].iloc[1] == 6
    assert transformed_df["avg_s_calls_per_year"].iloc[2] == 1 / 2


@pytest.mark.unit
def test_feature_builder_sets_late_payer_flag(
    sample_features_df: pd.DataFrame,
) -> None:
    transformed_df = FeatureBuilder().transform(sample_features_df)

    assert transformed_df["late_payer"].iloc[0] == 1
    assert transformed_df["late_payer"].iloc[1] == 0
    assert transformed_df["late_payer"].iloc[2] == 0


@pytest.mark.unit
def test_feature_builder_sets_inactive_customer_flag(
    sample_features_df: pd.DataFrame,
) -> None:
    transformed_df = FeatureBuilder().transform(sample_features_df)

    assert transformed_df["inactive_customer"].iloc[0] == 0
    assert transformed_df["inactive_customer"].iloc[1] == 0
    assert transformed_df["inactive_customer"].iloc[2] == 1


@pytest.mark.unit
def test_feature_builder_returns_original_and_engineered_feature_names(
    sample_features_df: pd.DataFrame,
) -> None:
    transformer = FeatureBuilder()

    features = transformer.get_feature_names_out(sample_features_df.columns)

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
