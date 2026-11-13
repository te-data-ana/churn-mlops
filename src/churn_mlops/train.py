import json
import logging
import os
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, replace
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
    LoggedModelReference,
    ModelCardBuilder,
    ModelRegistry,
    PromotionService,
    TrainingManifest,
    log_experiment_result,
    log_model_card,
    setup_local_experiment,
)
from churn_mlops.tracking.manifest import (
    ManifestData,
    ManifestEvaluation,
    ManifestLifecycle,
    ManifestModel,
    ManifestRun,
    ManifestTraining,
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
    exclude_columns: Sequence[str] = (),
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
    artifact_dir: Path | None = None,
    register_model: bool | None = None,
    promote_model: bool | None = None,
    calibration_method: str | None = None,
    n_jobs: int | None = None,
    write_manifest: bool = True,
) -> TrainingResult:
    """Run the configured training, tracking, registration, and promotion flow.

    Args:
        config_file: Training configuration YAML filename.
        split_name: Prefix of the prepared train/test Parquet split files.
        config_dir: Directory containing the training configuration.
        split_dir: Directory containing the prepared split files.
        exclude_columns: Optional non-feature columns to exclude from
            hyperparameter tuning and model fitting.
        experiment_name: Optional MLflow experiment name override. When omitted,
            the ``MLFLOW_EXPERIMENT_NAME`` runtime setting is used.
        tracking_uri: Optional MLflow tracking URI.
        artifact_dir: Directory used for training artifacts.
        register_model: Optional override for model registration.
        promote_model: Optional override for alias updates and promotion.
        calibration_method: Optional override for the configured calibration
            method.
        n_jobs: Optional override for the configured total CPU job budget.
        write_manifest: Boolean flag for writing manifest JSON file, which
            contains experiment details like run and registered model version
            (default: True).

    Returns:
        TrainingResult containing the fitted pipeline, metrics, classifier
        configuration, feature names, and selected threshold.

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
        if calibration_method is not None:
            config = replace(
                config,
                calibration=replace(config.calibration, method=calibration_method),
            )
        if n_jobs is not None:
            config = replace(config, n_jobs=n_jobs)
        classifier_alias = config.model.classifier
        threshold_strategy = config.threshold.strategy

        run_name = f"{classifier_alias}_{split_name}"
        manifest_file = settings.data_dir / f"manifests/{run_name}.json"
        resolved_register_model = (
            config.registry.register_model if register_model is None else register_model
        )
        resolved_promote_model = (
            config.promotion.promote_model if promote_model is None else promote_model
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

        logger.info("Loaded training configuration from '%s'.", config_file_path)
        logger.info(
            "Configured model alias '%s' with threshold strategy '%s'.",
            classifier_alias,
            threshold_strategy,
        )
        logger.info(
            "Model registration is %s; model promotion is %s.",
            "enabled" if resolved_register_model else "disabled",
            "enabled"
            if resolved_register_model and resolved_promote_model
            else "disabled",
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
        logger.info("MLflow artifact location: %s", experiment.artifact_location)

        tuning_config = getattr(config, "tuning", TuningConfig())
        train_df, test_df = load_and_validate_training_splits(
            split_name=split_name,
            split_dir=resolved_split_dir,
            preserve_time_column=config.data.time_column,
        )
        logger.info(
            "Loaded validated split '%s': %d train rows and %d test rows.",
            split_name,
            len(train_df),
            len(test_df),
        )

        # Start an MLflow run for the training job and log the results
        logger.info(
            "Starting MLflow run '%s' in experiment '%s'.",
            run_name,
            resolved_experiment_name,
        )
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
        ):
            active_run = mlflow.active_run()
            if active_run is None:
                raise RuntimeError("MLflow did not create an active training run.")
            run_id = active_run.info.run_id
            logger.info("Started MLflow run '%s'.", run_id)
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
                logger.info("Logged split metadata to MLflow: %s.", split_metadata)
            if tuning_config.enabled:
                tuning_result = optimize_hyperparameters(
                    config=config,
                    train_df=train_df,
                    exclude_columns=[
                        *exclude_columns,
                        config.data.time_column,
                    ],
                )
                mlflow.log_params(
                    {
                        "tuning_enabled": "true",
                        "tuning_n_trials": tuning_result.n_trials,
                        "tuning_n_splits": tuning_result.n_splits,
                        "tuning_n_jobs": config.n_jobs,
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
                logger.info(
                    "Logged tuning results to MLflow: %s=%.6f; best parameters=%s.",
                    tuning_result.metric,
                    tuning_result.best_score,
                    tuning_result.best_params,
                )
                result = train_model(
                    config=config,
                    train_df=train_df,
                    test_df=test_df,
                    model_params_override=tuning_result.best_params,
                    exclude_columns=[
                        *exclude_columns,
                        tuning_result.time_column,
                    ],
                )
            else:
                logger.info(
                    "Hyperparameter tuning is disabled; using configured classifier "
                    "parameters."
                )
                result = train_model(
                    config=config,
                    train_df=train_df,
                    test_df=test_df,
                    exclude_columns=exclude_columns,
                )
            result.metrics["promotion_score"] = (
                PromotionService.calculate_weighted_score(
                    result.metrics,
                    config.promotion.metric_weights,
                )
            )
            model_info = log_experiment_result(
                result=result,
                config=config,
                config_file_path=config_file_path,
            )
            logger.info(
                "Training results logged to MLflow run '%s' (model URI: '%s').",
                run_id,
                model_info.model_uri,
            )
        registered_model_version = None
        model_name = config.registry.model_name
        promotion_decision_dict = {}
        if resolved_register_model:
            registry = ModelRegistry()
            registered_version = registry.register_model(
                model_info=model_info,
                model_name=model_name,
            )
            registered_model_version = int(registered_version.version)
            if resolved_promote_model:
                model_alias = config.registry.alias
                registry.set_alias(
                    model_name=model_name,
                    model_alias=model_alias,
                    model_version=registered_model_version,
                )
                champion = registry.get_champion_version(model_name=model_name)

                if champion:
                    logger.info("Retrieving metric for current champion model.")
                    champion_metrics = registry.get_metrics_by_alias(
                        model_name=model_name,
                        model_alias="champion",
                    )
                else:
                    logger.info(
                        "No champion model exists in registry for '%s'.",
                        model_name,
                    )
                    champion_metrics = None

                # Promotion requires a strict weighted-score improvement.
                promotion_service = PromotionService()
                promotion_decision = promotion_service.evaluate_weighted_candidate(
                    candidate_metrics=result.metrics,
                    champion_metrics=champion_metrics,
                    metric_weights=config.promotion.metric_weights,
                    min_delta=config.promotion.min_delta,
                )
                promotion_decision_dict = asdict(promotion_decision)

                if promotion_decision.promote:
                    logger.info(
                        "Promoting candidate model to champion: %s",
                        promotion_decision.reason,
                    )
                    if promotion_decision.score_delta is not None:
                        logger.info(
                            "The weighted quality score delta was '%.4f'.",
                            promotion_decision.score_delta,
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
            else:
                logger.info(
                    "Model promotion is disabled; registered model remains a candidate."
                )
        else:
            logger.info(
                "Model registration is disabled; skipping registry and promotion steps."
            )

        if resolved_register_model or write_manifest:
            manifest = TrainingManifest(
                run=ManifestRun(run_id=run_id, run_name=run_name),
                model=ManifestModel(
                    model_name=model_name,
                    model_version=registered_model_version,
                    model_reference=LoggedModelReference.from_model_info(model_info),
                ),
                data=ManifestData(
                    target_column=config.data.target_column,
                    time_column=config.data.time_column,
                    split_name=split_name,
                    split_metadata=split_metadata,
                ),
                training=ManifestTraining(
                    classifier_name=result.classifier_config["model_name"],
                    classifier_alias=classifier_alias,
                    feature_count=len(result.feature_names_out),
                    feature_names_in=result.feature_names_in,
                    feature_names_out=result.feature_names_out,
                    tuning_enabled=config.tuning.enabled,
                    calibration=config.calibration.method,
                    threshold_strategy=config.threshold.strategy,
                    threshold_data=result.threshold_data,
                    threshold=result.threshold,
                ),
                evaluation=ManifestEvaluation(metrics=result.metrics),
                lifecycle=ManifestLifecycle(
                    registration_enabled=resolved_register_model,
                    promotion_enabled=(
                        resolved_register_model and resolved_promote_model
                    ),
                    promotion_decision=promotion_decision_dict,
                ),
            )

            model_card = ModelCardBuilder().build(
                config=config,
                context=manifest,
            )

            if resolved_register_model:
                registry.update_model_description(
                    model_name=model_name,
                    model_version=registered_model_version,
                    description=model_card,
                )

            with mlflow.start_run(run_id=run_id):
                logger.info("Logging model card to MLflow.")
                log_model_card(model_card)
                if write_manifest:
                    manifest.write(manifest_file)
                    mlflow.log_artifact(
                        local_path=str(manifest_file),
                        artifact_path="manifest",
                    )
                    logger.info(
                        "Wrote training manifest to '%s' and logged it to MLflow.",
                        manifest_file,
                    )

        logger.info(
            "Training job completed for classifier '%s' on split '%s' "
            "(MLflow run '%s').",
            classifier_alias,
            split_name,
            run_id,
        )
        return result
    except Exception:
        logger.exception(
            "Training job failed while processing config '%s' and split '%s'.",
            config_file,
            split_name,
        )
        raise
