"""Classifier pipelines for tabular classification models, combining feature engineering, preprocessing and classifier."""

from typing import Literal

from sklearn.pipeline import Pipeline

from churn_mlops.config import BASELINE_CLASSIFIERS

from .features import FeatureBuilder
from .preprocessing import create_preprocessor


def build_classifier_pipeline(
    classifier: Literal["lr", "rf", "hgb"] = "lr",
    classifier_params: dict[str, any] | None = None,
) -> Pipeline:
    """Build tabular classification pipeline, combining feature engineering with different preprocessing steps, depending on specified classifier."""

    if classifier not in BASELINE_CLASSIFIERS:
        raise ValueError(
            f"Unknown classifier '{classifier}'. "
            f"Must be one of {list(BASELINE_CLASSIFIERS.keys())}."
        )

    config = BASELINE_CLASSIFIERS[classifier]

    merged_params = config["default_params"] | (classifier_params or {})

    scale = classifier == "lr"

    return Pipeline(
        steps=[
            ("features", FeatureBuilder()),
            ("preprocessor", create_preprocessor(scale=scale)),
            ("classifier", config["clf"](**merged_params)),
        ]
    )
