from dataclasses import dataclass
from datetime import datetime, timezone

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn_mlops.config import load_config
from churn_mlops.config.schemas import TrainingConfig
from churn_mlops.data import load_raw_data, validate_data
from churn_mlops.evaluation import Timer, evaluate_model
from churn_mlops.models import MODEL_REGISTRY, build_classifier_pipeline
from churn_mlops.tracking import log_experiment_result, setup_local_experiment


@dataclass
class TrainingResult:
    trained_pipeline: Pipeline
    metrics: dict[str, any]
    classifier_config: dict[str, any]
    metadata: dict[str, any]


def run_training_job(
    config_file: str = "training.yaml",
    training_file: str = "customer_churn_dataset-training.csv",
    index_col: str = "customerid",
    experiment_name: str = "churn-baseline",
) -> TrainingResult:
    """
    Wrapper around actual model training:
    * loading `config` and raw data from files,
    * validating the loaded `pandas` DataFrame `df` fulfills the data contract defined with `pandera`,
    * executing training pipeline based on loaded `config` and `df`.
    """

    # load training configuration from yaml file
    config, config_file_path = load_config(config_file)
    # load raw data
    df = load_raw_data(training_file, index_col)
    # validate data contract/schema
    df = validate_data(df)

    # extract basic run details from config
    classifier_alias = config.model.classifier
    eval_threshold = config.evaluation.threshold
    # define experiment run name
    run_name = f"{classifier_alias}_{eval_threshold!s}"

    # execute training, using MLflow for experiment tracking
    experiment_id = setup_local_experiment(experiment_name)
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    experiment = mlflow.get_experiment(experiment_id)
    print(f"Experiment: {experiment}")

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        result = train(config, df)
        log_experiment_result(result, config, config_file_path)

    return result


def train(config: TrainingConfig, df: pd.DataFrame) -> TrainingResult:
    """
    Model training pipeline:
    * input: `config` and DataFrame `df` used for model training.
    * steps (depending on profided `config`):
        * split data into target/features and train/test datasets
        * prepare classifier and model parameters
        * compile and fit `model_pipeline` on train-split
        * predict target propensities for test-split
        * compute pre-defined set of `metrics`
        * store relevant `metadata`
        * and return training results, composed of `model_pipeline`, `metrics` and `metadata`
    """

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
    classifier_name = registry["clf"].__name__

    # combine classifier components into effective classifier config
    classifier_config = {
        "classifier_alias": clf_name,
        "classifier_name": classifier_name,
        # "classifier_type": type(classifier),
        **merged_params,
    }

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
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_count": len(feature_names_out),
        "feature_names_in": list(X.columns),
        "feature_names_out": feature_names_out,
        "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    }

    return TrainingResult(
        trained_pipeline=model_pipeline,
        metrics=vars(metrics),
        classifier_config=classifier_config,
        metadata=metadata,
    )
