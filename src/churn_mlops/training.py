import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn_mlops.config import configure_logging, load_config
from churn_mlops.config.schemas import TrainingConfig
from churn_mlops.data import load_raw_data, validate_training_data
from churn_mlops.evaluation import Timer, evaluate_model
from churn_mlops.models import build_classifier_pipeline, create_model
from churn_mlops.tracking import (
    ModelRegistry,
    PromotionService,
    log_experiment_result,
    setup_local_experiment,
)


@dataclass
class TrainingResult:
    trained_pipeline: Pipeline
    metrics: dict[str, any]
    classifier_config: dict[str, any]
    metadata: dict[str, any]


def run_training_job(
    config_file: str = "sample_training_config.yaml",
    training_file: str = "customer_churn_dataset-training.csv",
    index_col: str = "customerid",
    experiment_name: str = "test",
) -> TrainingResult:
    """
    Wrapper around actual model training:
    * loading `config` and raw data from files,
    * validating the loaded `pandas` DataFrame `df` fulfills the data contract defined with `pandera`,
    * executing training pipeline based on loaded `config` and `df`.
    """
    configure_logging()
    logger = logging.getLogger(__name__)

    # load training configuration from yaml file
    config, config_file_path = load_config(config_file)
    # load raw data
    df = load_raw_data(training_file, index_col)
    # validate data contract/schema
    df = validate_training_data(df)

    # extract basic run details from config
    classifier_alias = config.model.classifier
    eval_threshold = config.evaluation.threshold
    # define experiment run name
    run_name = f"{classifier_alias}_{eval_threshold!s}"

    # execute training, using MLflow for experiment tracking
    experiment_id = setup_local_experiment(experiment_name)
    logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
    experiment = mlflow.get_experiment(experiment_id)
    logger.info(str(experiment))

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        result = train(config, df)
        model_info = log_experiment_result(result, config, config_file_path)

    # if specified in config: register as candidate model
    if config.registry.register_model:
        # load config details
        model_name = config.registry.registry_params["model_name"]
        model_alias = config.registry.registry_params["alias"]

        logger.info(
            "Starting registration of '%s' model with alias '%s':",
            model_name,
            model_alias,
        )
        # initialize model registry
        registry = ModelRegistry()

        # register candidate and set alias
        candidate = registry.register_model(
            model_uri=model_info.model_uri,
            model_name=model_name,
        )
        registry.set_alias(
            model_name=model_name,
            alias=model_alias,
            version=candidate.version,
        )

        # retrieve current champion model
        champion = registry.get_champion_version(model_name=model_name)

        # retrieve model metric of candidate and champion (used for comparison)
        candidate_metric = result.metrics["roc_auc"]
        if champion:
            logger.info("Retrieving metric for current champion model.")
            champion_metric = registry.get_metric_by_alias(
                model_name=model_name,
                alias="champion",
                metric_name="roc_auc",
            )
        else:
            logger.info(
                "No champion model exists in registry for '%s'.",
                model_name,
            )
            champion_metric = None

        # decide if candidate model should be promoted to champion
        promotion_delta = config.registry.registry_params["promotion_delta"]
        promotion_service = PromotionService()
        promotion_decision = promotion_service.evaluate_candidate(
            candidate_metric=candidate_metric,
            champion_metric=champion_metric,
            promotion_delta=promotion_delta,
        )

        # promote depending on decision of promotion service
        if promotion_decision.promote:
            logger.info("Decision to promote candidate model to champion:")
            logger.info(promotion_decision.reason)
            if promotion_decision.metric_delta:
                logger.info(
                    "The AUC delta was '%.4f'.", promotion_decision.metric_delta
                )
            promotion_service.promote_candidate(
                decision=promotion_decision,
                registry=registry,
                model_name=model_name,
                candidate_version=candidate.version,
            )
        else:
            logger.info("No promotion of candidate model:")
            logger.info(promotion_decision.reason)

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
    classifier, classifier_config = create_model(
        model_alias=config.model.classifier, model_params=config.model.classifier_params
    )

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
