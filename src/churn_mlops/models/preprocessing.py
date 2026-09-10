"""Preprocessing pipelines for tabular classification models."""

import numpy as np
from sklearn.compose import (
    ColumnTransformer,
    make_column_selector,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from churn_mlops.config import CATEGORICAL_IMPUTE_STRATEGY, NUMERIC_IMPUTE_STRATEGY


def _create_numerical_pipeline(scale: bool = False) -> Pipeline:
    """Create preprocessing pipeline for numerical features with optional scaling."""

    steps = [
        ("imputer", SimpleImputer(strategy=NUMERIC_IMPUTE_STRATEGY)),
    ]

    if scale:
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps)


def _create_categorical_pipeline() -> Pipeline:
    """Create preprocessing pipeline for categorical features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=CATEGORICAL_IMPUTE_STRATEGY)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def create_preprocessor(scale: bool = False) -> ColumnTransformer:
    """Create preprocessing for numerical and categorical features."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                _create_numerical_pipeline(scale=scale),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "cat",
                _create_categorical_pipeline(),
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
