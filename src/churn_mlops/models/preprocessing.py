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
    """Create an imputation pipeline for numerical features.

    Args:
        scale: Whether to standardize imputed values.
        impute_strategy: Strategy passed to ``SimpleImputer``.

    Returns:
        Pipeline that imputes numerical values and optionally scales them.
    """

    steps = [
        ("imputer", SimpleImputer(strategy=impute_strategy)),
    ]

    if scale:
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps)


def _create_categorical_pipeline(impute_strategy: str = "most_frequent") -> Pipeline:
    """Create an imputation and one-hot encoding pipeline.

    Args:
        impute_strategy: Strategy passed to ``SimpleImputer``.

    Returns:
        Pipeline that imputes categorical values and ignores unknown encoded
        categories during transformation.
    """
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
    """Create preprocessing for numerical and categorical features.

    Args:
        scale: Whether to standardize numerical features after imputation.
        num_impute_strategy: Imputation strategy for numerical features.
        cat_impute_strategy: Imputation strategy for categorical features.

    Returns:
        ColumnTransformer that processes numeric and categorical columns and
        drops all other columns.
    """
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
