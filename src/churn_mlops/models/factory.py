from collections.abc import Mapping
from typing import Any

from sklearn.base import BaseEstimator

from churn_mlops.models import MODEL_CATALOG


def create_model(
    model_alias: str, model_params: Mapping[str, Any] | None
) -> tuple[BaseEstimator, dict[str, Any]]:
    """Create a catalogued classifier with merged default/model parameters.

    Args:
        model_alias: Key identifying the classifier in ``MODEL_CATALOG``.
        model_params: Parameter overrides applied on top of catalog defaults.

    Returns:
        Tuple containing the configured classifier and its effective model
        configuration dictionary.

    Raises:
        ValueError: If ``model_alias`` is not present in ``MODEL_CATALOG``.
    """

    if model_alias not in MODEL_CATALOG:
        raise ValueError(
            f"Unknown classifier '{model_alias}'. "
            f"Must be one of {list(MODEL_CATALOG.keys())}."
        )

    catalog_entry = MODEL_CATALOG[model_alias]
    merged_params = catalog_entry["default_params"] | (model_params or {})
    model = catalog_entry["clf"](**merged_params)
    model_name = catalog_entry["clf"].__name__

    model_config = {
        "model_alias": model_alias,
        "model_name": model_name,
        **merged_params,
    }

    return model, model_config
