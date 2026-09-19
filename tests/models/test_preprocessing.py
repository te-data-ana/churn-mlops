import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from churn_mlops.models import create_preprocessor


@pytest.mark.unit
def test_create_preprocessor_returns_column_transformer_instance() -> None:
    preprocessor = create_preprocessor(scale=True)
    assert isinstance(preprocessor, ColumnTransformer)


@pytest.mark.unit
def test_preprocessor_transforms_dataframe_to_nonempty_output(
    sample_features_df: pd.DataFrame,
) -> None:
    preprocessor = create_preprocessor(scale=True)
    transformed = preprocessor.fit_transform(sample_features_df)

    assert transformed.shape[0] == len(sample_features_df)
    assert transformed.shape[1] > 0


@pytest.mark.unit
def test_preprocessor_one_hot_encodes_contract_length(
    sample_features_df: pd.DataFrame,
) -> None:
    # Encoded category columns should contain only binary indicators.
    preprocessor = create_preprocessor(scale=True)
    transformed = preprocessor.fit_transform(sample_features_df)
    feature_names = preprocessor.get_feature_names_out()
    transformed_df = pd.DataFrame(
        transformed,
        columns=feature_names,
    )

    expected_columns = [
        "contract_length_Annual",
        "contract_length_Monthly",
        "contract_length_Quarterly",
    ]

    assert set(expected_columns).issubset(set(feature_names))
    assert set(transformed_df[expected_columns].stack().unique()).issubset({0, 1})


@pytest.mark.unit
def test_preprocessor_imputes_missing_values_without_nonfinite_output(
    sample_features_df: pd.DataFrame,
) -> None:
    preprocessor = create_preprocessor(scale=True)
    transformed = preprocessor.fit_transform(sample_features_df)

    assert transformed.shape[0] == len(sample_features_df)
    assert np.isfinite(transformed).all()


@pytest.mark.unit
def test_scaled_preprocessor_includes_scaler_and_transforms_data(
    sample_features_df: pd.DataFrame,
) -> None:
    preprocessor = create_preprocessor(scale=True)
    transformed = preprocessor.fit_transform(sample_features_df)
    num_pipeline = preprocessor.transformers[0][1]

    assert "scaler" in num_pipeline.named_steps
    assert transformed.shape[0] == len(sample_features_df)
    assert transformed.shape[1] > 0
    assert np.isfinite(transformed).all()


@pytest.mark.unit
def test_unscaled_preprocessor_omits_scaler_and_transforms_data(
    sample_features_df: pd.DataFrame,
) -> None:
    preprocessor = create_preprocessor(scale=False)
    transformed = preprocessor.fit_transform(sample_features_df)
    num_pipeline = preprocessor.transformers[0][1]

    assert "scaler" not in num_pipeline.named_steps
    assert transformed.shape[0] == len(sample_features_df)
    assert transformed.shape[1] > 0
    assert np.isfinite(transformed).all()
