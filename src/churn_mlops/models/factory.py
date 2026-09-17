from churn_mlops.models import MODEL_CATALOG


def create_model(model_alias: str, model_params: dict):

    # lookup model in model catalog
    if model_alias not in MODEL_CATALOG:
        raise ValueError(
            f"Unknown classifier '{model_alias}'. "
            f"Must be one of {list(MODEL_CATALOG.keys())}."
        )

    # retrieve catalog entry
    catalog_entry = MODEL_CATALOG[model_alias]
    # mergre default parameters with provided model parameters
    merged_params = catalog_entry["default_params"] | (model_params or {})
    # configure model with parameters
    model = catalog_entry["clf"](**merged_params)
    # extract model class name
    model_name = catalog_entry["clf"].__name__

    # combine model components into effective model config
    model_config = {
        "model_alias": model_alias,
        "model_name": model_name,
        # "model_type": type(model),
        **merged_params,
    }

    return model, model_config
