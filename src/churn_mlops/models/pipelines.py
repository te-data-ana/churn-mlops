"""Classifier pipelines for tabular classification models, combining feature engineering, preprocessing and classifier."""

from typing import Any

from sklearn.pipeline import Pipeline

from .features import FeatureBuilder
from .preprocessing import create_preprocessor


def _requires_scaling(estimator) -> bool:
    module = estimator.__class__.__module__

    return module.startswith(
        (
            "sklearn.linear_model",
            "sklearn.svm",
            "sklearn.neighbors",
        )
    )


def build_classifier_pipeline(
    classifier,
    feature_params: dict[str, Any] | None = None,
    num_impute_strategy: str = "median",
    cat_impute_strategy: str = "most_frequent",
) -> Pipeline:
    """Build tabular classification pipeline, combining feature engineering with different preprocessing steps, depending on provided classifier."""

    scale = _requires_scaling(classifier)

    return Pipeline(
        steps=[
            ("features", FeatureBuilder(feature_params)),
            (
                "preprocessor",
                create_preprocessor(scale, num_impute_strategy, cat_impute_strategy),
            ),
            ("classifier", classifier),
        ]
    )
