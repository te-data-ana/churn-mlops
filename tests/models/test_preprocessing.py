import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from churn_mlops.models import create_preprocessor


def test_create_preprocessor_returns_column_transformer():
    preprocessor = create_preprocessor(scale=True)
    assert isinstance(preprocessor, ColumnTransformer)


def test_preprocessor_can_transform(sample_df):
    preprocessor = create_preprocessor(scale=True)
    transformed = preprocessor.fit_transform(sample_df)

    assert transformed.shape[0] == len(sample_df)
    assert transformed.shape[1] > 0


def test_preprocessor_ohe_columns(sample_df):
    preprocessor = create_preprocessor(scale=True)
    transformed = preprocessor.fit_transform(sample_df)
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


def test_preprocessor_imputation(sample_df):
    preprocessor = create_preprocessor(scale=True)
    transformed = preprocessor.fit_transform(sample_df)

    assert transformed.shape[0] == len(sample_df)
    assert np.isfinite(transformed).all()


def test_scaled_preprocessor_can_transform(sample_df):
    preprocessor = create_preprocessor(scale=True)
    transformed = preprocessor.fit_transform(sample_df)
    num_pipeline = preprocessor.transformers[0][1]

    assert "scaler" in num_pipeline.named_steps
    assert transformed.shape[0] == len(sample_df)
    assert transformed.shape[1] > 0
    assert np.isfinite(transformed).all()


def test_unscaled_preprocessor_can_transform(sample_df):
    preprocessor = create_preprocessor(scale=False)
    transformed = preprocessor.fit_transform(sample_df)
    num_pipeline = preprocessor.transformers[0][1]

    assert "scaler" not in num_pipeline.named_steps
    assert transformed.shape[0] == len(sample_df)
    assert transformed.shape[1] > 0
    assert np.isfinite(transformed).all()
