"""Classifier pipelines for tabular classification models, combining feature engineering, preprocessing and classifier."""

from typing import Any

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from .features import FeatureBuilder
from .preprocessing import create_preprocessor


def _requires_scaling(estimator: BaseEstimator) -> bool:
    """Determine whether an estimator belongs to a scaling-sensitive family.

    Args:
        estimator: Scikit-learn estimator whose module is inspected.

    Returns:
        ``True`` for linear-model, SVM, and neighbor estimators; otherwise
        ``False``.
    """
    module = estimator.__class__.__module__

    return module.startswith(
        (
            "sklearn.linear_model",
            "sklearn.svm",
            "sklearn.neighbors",
        )
    )


def build_classifier_pipeline(
    classifier: ClassifierMixin,
    feature_params: dict[str, Any] | None = None,
    num_impute_strategy: str = "median",
    cat_impute_strategy: str = "most_frequent",
) -> Pipeline:
    """Build a feature-engineering, preprocessing, and classifier pipeline.

    Args:
        classifier: Scikit-learn classifier to use as the final pipeline step.
        feature_params: Optional thresholds for ``FeatureBuilder``.
        num_impute_strategy: Imputation strategy for numeric columns.
        cat_impute_strategy: Imputation strategy for categorical columns.

    Returns:
        Pipeline containing feature engineering, preprocessing, and the
        supplied classifier. Scaling is enabled for scaling-sensitive models.
    """

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
