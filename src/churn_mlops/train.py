import json
import logging
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import mlflow

from churn_mlops.config import (
    configure_logging,
    load_config,
)
from churn_mlops.config.schemas import TuningConfig
from churn_mlops.config.settings import RuntimeSettings
from churn_mlops.data.splitting import load_and_validate_training_splits
from churn_mlops.tracking import (
    ModelCardBuilder,
    ModelRegistry,
    PromotionService,
    TrainingManifest,
    log_experiment_result,
    log_model_card,
    setup_local_experiment,
)
from churn_mlops.training import (
    TrainingResult,
    optimize_hyperparameters,
    train_model,
)


def run_training_job(
    config_file: str,
    split_name: str,
    config_dir: Path | None = None,
    split_dir: Path | None = None,
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
    artifact_dir: Path | None = None,
    register_model: bool | None = None,
    promote_model: bool | None = None,
    write_manifest: bool = True,
) -> TrainingResult:
    """Run the configured training, tracking, registration, and promotion flow.

    Args:
        config_file: Training configuration YAML filename.
        split_name: Prefix of the prepared train/test Parquet split files.
        config_dir: Directory containing the training configuration.
        split_dir: Directory containing the prepared split files.
        experiment_name: Optional MLflow experiment name override. When omitted,
            the ``MLFLOW_EXPERIMENT_NAME`` runtime setting is used.
        tracking_uri: Optional MLflow tracking URI.
        artifact_dir: Directory used for training artifacts.
        write_manifest: Boolean flag for writing manifest JSON file, which
            contains experiment details like run and registered model version
            (default: True).
        register_model: Optional override for model registration.
        promote_model: Optional override for alias updates and promotion.

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
        Logs an MLflow run and, when enabled, registers an immutable model
        version. Alias changes and promotion can be configured separately.
    """

    # Load and establish all required configurations
    configure_logging()
    logger = logging.getLogger(__name__)

    settings = RuntimeSettings()
    resolved_config_dir = config_dir or settings.config_dir
    resolved_split_dir = split_dir or settings.data_dir / "splits"
    resolved_artifact_dir = artifact_dir or settings.artifact_dir
    resolved_tracking_uri = tracking_uri or settings.mlflow_tracking_uri
    resolved_experiment_name = experiment_name or settings.mlflow_experiment_name

    try:
        config, config_file_path = load_config(
            file=config_file, path=resolved_config_dir
        )
        classifier_alias = config.model.classifier
        eval_threshold = config.evaluation.threshold

        run_name = f"{classifier_alias}_{split_name}"
        manifest_file = settings.data_dir / f"manifests/{run_name}.json"
        resolved_register_model = (
            config.registry.register_model if register_model is None else register_model
        )
        resolved_promote_model = (
            getattr(config.registry, "promote_model", True)
            if promote_model is None
            else promote_model
        )

        split_metadata: dict[str, str | int] = {}
        split_metadata_path = resolved_split_dir / f"{split_name}.json"
        if split_metadata_path.exists():
            payload = json.loads(split_metadata_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not all(
                key in payload
                for key in (
                    "train_rows",
                    "test_rows",
                    "train_start",
                    "train_end",
                    "test_start",
                    "test_end",
                )
            ):
                raise ValueError(
                    f"Split metadata '{split_metadata_path}' is missing required fields."
                )
            split_metadata = {
                "train_rows": int(payload["train_rows"]),
                "train_start": str(payload["train_start"]),
                "train_end": str(payload["train_end"]),
                "test_rows": int(payload["test_rows"]),
                "test_start": str(payload["test_start"]),
                "test_end": str(payload["test_end"]),
            }

        if write_manifest and not resolved_register_model:
            raise ValueError(
                "A training manifest requires model registration. "
                "Enable registration for this training run."
            )

        logger.info("Loaded training configuration from '%s'.", config_file_path)
        logger.info(
            "Configured model alias '%s' with evaluation threshold %.4f.",
            classifier_alias,
            eval_threshold,
        )

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

        tuning_config = getattr(config, "tuning", TuningConfig())
        if tuning_config.enabled:
            train_df, test_df = load_and_validate_training_splits(
                split_name=split_name,
                split_dir=resolved_split_dir,
                preserve_time_column=tuning_config.time_column,
            )
        else:
            train_df, test_df = load_and_validate_training_splits(
                split_name=split_name,
                split_dir=resolved_split_dir,
            )

        # Start an MLflow run for the training job and log the results
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
        ):
            active_run = mlflow.active_run()
            if active_run is None:
                raise RuntimeError("MLflow did not create an active training run.")
            run_id = active_run.info.run_id
            mlflow.set_tags(
                {
                    "config_file": config_file,
                    "run_name": run_name,
                    "classifier": classifier_alias,
                    "split_name": split_name,
                }
            )
            try:
                git_revision = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).resolve().parents[2],
                ).stdout.strip()
            except (OSError, subprocess.CalledProcessError) as exc:
                logger.warning("Unable to resolve Git revision for the run: %s", exc)
            else:
                mlflow.set_tag("git_commit", git_revision)
            for env_name, tag_name in (
                ("DVC_EXP_NAME", "dvc_experiment_name"),
                ("DVC_EXP_REV", "dvc_experiment_revision"),
            ):
                if env_value := os.environ.get(env_name):
                    mlflow.set_tag(tag_name, env_value)
            if split_metadata:
                mlflow.log_params(
                    {
                        "train_rows": split_metadata["train_rows"],
                        "train_start": split_metadata["train_start"],
                        "train_end": split_metadata["train_end"],
                        "test_rows": split_metadata["test_rows"],
                        "test_start": split_metadata["test_start"],
                        "test_end": split_metadata["test_end"],
                    }
                )
            if tuning_config.enabled:
                tuning_result = optimize_hyperparameters(
                    config=config,
                    train_df=train_df,
                )
                mlflow.log_params(
                    {
                        "tuning_enabled": "true",
                        "tuning_n_trials": tuning_result.n_trials,
                        "tuning_n_splits": tuning_result.n_splits,
                        "tuning_time_column": tuning_result.time_column,
                        "tuning_metric": tuning_result.metric,
                        "tuning_random_state": tuning_result.random_state,
                        "tuning_best_params": json.dumps(
                            tuning_result.best_params,
                            sort_keys=True,
                        ),
                    }
                )
                mlflow.log_metric(
                    f"tuning_cv_{tuning_result.metric}",
                    tuning_result.best_score,
                )
                result = train_model(
                    config=config,
                    train_df=train_df,
                    test_df=test_df,
                    model_params_override=tuning_result.best_params,
                    exclude_columns=[tuning_result.time_column],
                )
            else:
                result = train_model(
                    config=config,
                    train_df=train_df,
                    test_df=test_df,
                )
            model_info = log_experiment_result(
                result=result,
                config=config,
                config_file_path=config_file_path,
                artifact_dir=resolved_artifact_dir,
            )
            result.metadata.update(
                {
                    "mlflow_run_id": run_id,
                    "run_name": run_name,
                    "split_name": split_name,
                    "split_metadata": split_metadata,
                }
            )

        # Register the candidate model and evaluate it for promotion to champion if enabled
        if resolved_register_model:
            model_name = config.registry.registry_params["model_name"]
            registry = ModelRegistry()
            registered_version = registry.register_model(
                model_uri=model_info.model_uri,
                model_name=model_name,
            )
            registered_model_version = registered_version.version
            logger.info(
                "Registration of '%s' model, version '%s'.",
                model_name,
                registered_model_version,
            )
            result.metadata.update(
                {
                    "model_name": model_name,
                    "model_version": registered_model_version,
                }
            )

            promotion_decision_dict = {}
            if resolved_promote_model:
                model_alias = config.registry.registry_params["alias"]
                registry.set_alias(
                    model_name=model_name,
                    alias=model_alias,
                    version=registered_model_version,
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
                promotion_decision_dict = {
                    **asdict(promotion_decision),
                    "promote": promotion_decision.promote,
                }

                if promotion_decision.promote:
                    logger.info("Decision to promote candidate model to champion:")
                    logger.info(promotion_decision.reason)
                    if promotion_decision.metric_delta:
                        logger.info(
                            "The AUC delta was '%.4f'.",
                            promotion_decision.metric_delta,
                        )
                    promotion_service.promote_candidate(
                        decision=promotion_decision,
                        registry=registry,
                        model_name=model_name,
                        candidate_version=registered_model_version,
                    )
                    logger.info(
                        "Candidate model '%s' version %s was promoted to champion.",
                        model_name,
                        registered_model_version,
                    )
                else:
                    logger.warning(
                        "No promotion of candidate model: %s",
                        promotion_decision.reason,
                    )

            # Registered models are documented using model cards
            manifest = TrainingManifest(
                run_id=run_id,
                run_name=run_name,
                model_name=model_name,
                model_version=registered_model_version,
                target_column=config.data.target_column,
                classifier_name=result.classifier_config["model_name"],
                classifier_alias=classifier_alias,
                split_name=split_name,
                split_metadata=split_metadata,
                feature_count=result.metadata["feature_count"],
                feature_names_in=result.metadata["feature_names_in"],
                feature_names_out=result.metadata["feature_names_out"],
                threshold=eval_threshold,
                metrics=result.metrics,
                promotion_decision=promotion_decision_dict,
            )

            model_card = ModelCardBuilder().build(
                config=config,
                context=manifest,
            )

            registry.update_model_description(
                model_name=model_name,
                version=registered_model_version,
                description=model_card,
            )

            with mlflow.start_run(run_id=run_id):
                log_model_card(model_card)
                if write_manifest:
                    manifest.write(manifest_file)
                    mlflow.log_artifact(
                        local_path=str(manifest_file),
                        artifact_path="manifest",
                    )

        return result
    except Exception:
        logger.exception(
            "Training job failed while processing config '%s' and split '%s'.",
            config_file,
            split_name,
        )
        raise
