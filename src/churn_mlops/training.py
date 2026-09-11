from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn_mlops.config import load_config
from churn_mlops.config.schemas import TrainingConfig
from churn_mlops.data import load_raw_data, validate_data
from churn_mlops.evaluation import Timer, evaluate_model
from churn_mlops.models import MODEL_REGISTRY, build_classifier_pipeline


@dataclass
class TrainingResult:
    trained_pipeline: Pipeline
    metrics: dict[str, any]
    metadata: dict[str, any]


def train_from_files(
    config_file: str = "training.yaml",
    training_file: str = "customer_churn_dataset-training.csv",
    index_col: str = "customerid",
) -> TrainingResult:

    # load training configuration from yaml file
    config = load_config(config_file)

    # load raw data
    df = load_raw_data(training_file, index_col)

    # validate data contract/schema
    df = validate_data(df)

    return train(config, df)


def train(config: TrainingConfig, df: pd.DataFrame) -> TrainingResult:

    # separate input features from target variable
    X, y = df.drop(columns=[config.data.target_column]), df[config.data.target_column]

    # keep 30% of the data for model evaluation, 70% for training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=y,
    )

    # retrieve classifier from model registry
    clf_name = config.model.classifier
    if clf_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown classifier '{clf_name}'. "
            f"Must be one of {list(MODEL_REGISTRY.keys())}."
        )
    registry = MODEL_REGISTRY[clf_name]
    merged_params = registry["default_params"] | (config.model.classifier_params or {})
    classifier = registry["clf"](**merged_params)

    # build model pipeline artifact: df_X -> features -> preprocessor -> classifier
    model_pipeline = build_classifier_pipeline(
        classifier=classifier,
        feature_params=config.feature_builder.feature_params,
        num_impute_strategy=config.preprocessing.numeric_impute_strategy,
        cat_impute_strategy=config.preprocessing.categorical_impute_strategy,
    )

    # fit model artifact on training data
    with Timer() as fit_timer:
        model_pipeline.fit(X_train, y_train)

    # predict and extract churn propensities based on unseen test data
    with Timer() as pred_timer:
        y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    # generate metrics artifact:
    # classifier name | quality metrics (partially dependent on threshold) | fit/pred timings
    metrics = evaluate_model(
        model_name=registry["clf"].__name__,
        y_true=y_test,
        y_prob=y_proba,
        fit_time_sec=fit_timer.duration,
        pred_time_sec=pred_timer.duration,
        threshold=config.evaluation.threshold,
    )

    # compile model metadata
    feature_names_out = (
        model_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    )
    metadata = {
        "training_config": config,
        "feature_names_in": list(X.columns),
        "feature_names_out": feature_names_out,
        "model_type": type(model_pipeline["classifier"]),
        "training_rows": len(X_train),
        "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    }

    return TrainingResult(
        trained_pipeline=model_pipeline, metrics=vars(metrics), metadata=metadata
    )
