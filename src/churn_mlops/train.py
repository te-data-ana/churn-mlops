import argparse
import logging
from pathlib import Path

import mlflow

from churn_mlops.config import (
    configure_logging,
    load_config,
)
from churn_mlops.config.settings import RuntimeSettings
from churn_mlops.data import load_raw_data, validate_data
from churn_mlops.tracking import (
    ModelCardBuilder,
    ModelCardContext,
    ModelRegistry,
    PromotionService,
    log_experiment_result,
    log_model_card,
    setup_local_experiment,
)
from churn_mlops.training import TrainingResult, train_model


def run_training_job(
    config_file: str = "sample_training_config.yaml",
    config_dir: Path | None = None,
    training_file: str = "training.csv",
    index_col: str | None = "customerid",
    data_dir: Path | None = None,
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
    artifact_dir: Path | None = None,
) -> TrainingResult:
    """Run the configured training, tracking, registration, and promotion flow.

    Args:
        config_file: Training configuration YAML filename.
        config_dir: Directory containing the training configuration.
        training_file: Raw training data CSV filename.
        index_col: Column to use as the training DataFrame index.
        data_dir: Directory containing the training CSV.
        experiment_name: Optional MLflow experiment name override. When omitted,
            the ``MLFLOW_EXPERIMENT_NAME`` runtime setting is used.
        tracking_uri: Optional MLflow tracking URI.
        artifact_dir: Directory used for training artifacts.

    Returns:
        TrainingResult containing the fitted pipeline, metrics, classifier
        configuration, and training metadata.

    Raises:
        FileNotFoundError: If the configuration or training data file is not
            found.
        pandera.errors.SchemaError: If the training data violates its schema.
        ValueError: If the configured classifier or promotion operation is
            invalid.

    Side Effects:
        Logs an MLflow run and, when enabled by configuration, registers the
        candidate model and updates registry aliases.
    """

    # Load and establish all required configurations
    configure_logging()
    logger = logging.getLogger(__name__)

    settings = RuntimeSettings()
    resolved_config_dir = config_dir or settings.config_dir
    resolved_data_dir = data_dir or settings.raw_data_dir
    resolved_artifact_dir = artifact_dir or settings.artifact_dir
    resolved_tracking_uri = tracking_uri or settings.mlflow_tracking_uri
    resolved_experiment_name = experiment_name or settings.mlflow_experiment_name

    try:
        config, config_file_path = load_config(
            file=config_file, path=resolved_config_dir
        )
        classifier_alias = config.model.classifier
        eval_threshold = config.evaluation.threshold

        logger.info("Loaded training configuration from '%s'.", config_file_path)
        logger.info(
            "Configured model alias '%s' with evaluation threshold %.4f.",
            classifier_alias,
            eval_threshold,
        )

        run_name = f"{classifier_alias}_{eval_threshold!s}"
        experiment_id = setup_local_experiment(
            experiment_name=resolved_experiment_name,
            tracking_uri=resolved_tracking_uri,
            artifact_dir=resolved_artifact_dir,
        )
        experiment = mlflow.get_experiment(experiment_id)
        logger.info(
            "MLflow experiment '%s' initialized with id '%s'.",
            resolved_experiment_name,
            experiment_id,
        )
        logger.info("Tracking URI: %s", mlflow.get_tracking_uri())
        logger.info("Experiment metadata: %s", experiment)

        # Load and validate the training data
        df = load_raw_data(
            file_name=training_file,
            index_col=index_col,
            data_dir=resolved_data_dir,
        )
        df = validate_data(df)

        # Start an MLflow run for the training job and log the results
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            result = train_model(config, df)
            model_info = log_experiment_result(
                result=result,
                config=config,
                config_file_path=config_file_path,
                artifact_dir=resolved_artifact_dir,
            )

        # Register the candidate model and evaluate it for promotion to champion if enabled
        if config.registry.register_model:
            model_name = config.registry.registry_params["model_name"]
            model_alias = config.registry.registry_params["alias"]

            logger.info(
                "Starting registration of '%s' model with alias '%s':",
                model_name,
                model_alias,
            )
            registry = ModelRegistry()

            candidate = registry.register_model(
                model_uri=model_info.model_uri,
                model_name=model_name,
            )
            registry.set_alias(
                model_name=model_name,
                alias=model_alias,
                version=candidate.version,
            )

            champion = registry.get_champion_version(model_name=model_name)

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

            # Promotion requires a strict improvement over the configured delta.
            promotion_delta = config.registry.registry_params["promotion_delta"]
            promotion_service = PromotionService()
            promotion_decision = promotion_service.evaluate_candidate(
                candidate_metric=candidate_metric,
                champion_metric=champion_metric,
                promotion_delta=promotion_delta,
            )

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
                logger.info(
                    "Candidate model '%s' version %s was promoted to champion.",
                    model_name,
                    candidate.version,
                )
            else:
                logger.warning(
                    "No promotion of candidate model: %s", promotion_decision.reason
                )

            # Registered models are documented using model cards
            context = ModelCardContext(
                model_name=model_name,
                model_version=candidate.version,
                promotion_decision=promotion_decision,
            )

            model_card = ModelCardBuilder().build(
                result=result,
                config=config,
                context=context,
            )

            registry.update_model_description(
                model_name=model_name,
                version=candidate.version,
                description=model_card,
            )

            with mlflow.start_run(run_id=candidate.run_id):
                log_model_card(model_card)

        return result
    except Exception:
        logger.exception(
            "Training job failed while processing config '%s' and file '%s'.",
            config_file,
            training_file,
        )
        raise


def main() -> None:
    """Parse CLI arguments, run training, and log the resulting ROC AUC."""

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment_name", required=False)

    args = parser.parse_args()

    training_kwargs = {"config_file": args.config}
    if args.experiment_name is not None:
        training_kwargs["experiment_name"] = args.experiment_name

    result = run_training_job(**training_kwargs)

    logger.info(
        "Successfully trained %s model: AUC=%.4f",
        result.classifier_config["model_name"],
        result.metrics["roc_auc"],
    )


if __name__ == "__main__":
    configure_logging()
    main()
