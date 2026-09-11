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


def _create_numerical_pipeline(
    scale: bool = False, impute_strategy: str = "median"
) -> Pipeline:
    """Create preprocessing pipeline for numerical features with optional scaling."""

    steps = [
        ("imputer", SimpleImputer(strategy=impute_strategy)),
    ]

    if scale:
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps)


def _create_categorical_pipeline(impute_strategy: str = "most_frequent") -> Pipeline:
    """Create preprocessing pipeline for categorical features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=impute_strategy)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def create_preprocessor(
    scale: bool = False,
    num_impute_strategy: str = "median",
    cat_impute_strategy: str = "most_frequent",
) -> ColumnTransformer:
    """Create preprocessing for numerical and categorical features."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                _create_numerical_pipeline(
                    scale=scale, impute_strategy=num_impute_strategy
                ),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "cat",
                _create_categorical_pipeline(impute_strategy=cat_impute_strategy),
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
