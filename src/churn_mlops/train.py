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
from churn_mlops.use_cases import get_use_case


def run_training_job(
    use_case: str,
    config_file: str,
    split_name: str,
    config_dir: Path | None = None,
    split_dir: Path | None = None,
    tracking_uri: str | None = None,
    artifact_dir: Path | None = None,
    calibration_method: str = "none",
    n_jobs: int = 1,
    tune_model: bool = False,
    register_model: bool = False,
    promote_model: bool = False,
    write_manifest: bool = True,
) -> TrainingResult:
    """Run the configured training, tracking, registration, and promotion flow.

    Args:
        use_case: Required name of the registered use-case contract.
        config_file: Training configuration YAML filename.
        split_name: Prefix of the prepared train/test Parquet split files.
        config_dir: Directory containing the training configuration.
        split_dir: Directory containing the prepared split files.
        tracking_uri: Optional MLflow tracking URI.
        artifact_dir: Directory used for training artifacts.
        calibration_method: Calibration method for this run (default: "none").
            Options are "none", "sigmoid", or "isotonic".
        n_jobs: Total CPU job budget for this run (default: 1).
        tune_model: Whether to run hyperparameter tuning (default: False).
        register_model: Whether to register the trained model (default: False).
        promote_model: Whether to promote a registered model (default: False).
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
        TypeError: If run-time boolean options are not booleans.
        ValueError: If a run-time training option, classifier, or promotion
            operation is invalid.

    Side Effects:
        Logs an MLflow run and, when enabled, registers an immutable model
        version. The ``champion`` alias changes only after successful promotion.
    """

    # Load and establish all required configurations
    configure_logging()
    logger = logging.getLogger(__name__)

    if not isinstance(calibration_method, str) or calibration_method not in {
        "none",
        "sigmoid",
        "isotonic",
    }:
        raise ValueError(
            "Calibration method must be one of 'none', 'sigmoid', or 'isotonic'."
        )
    if type(n_jobs) is not int or n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be a positive integer or -1.")
    if (
        type(register_model) is not bool
        or type(promote_model) is not bool
        or type(tune_model) is not bool
    ):
        raise TypeError(
            "register_model, promote_model, and tune_model must be booleans."
        )

    settings = RuntimeSettings()
    resolved_config_dir = config_dir or settings.config_dir
    resolved_split_dir = split_dir or settings.data_dir / "splits"
    resolved_artifact_dir = artifact_dir or settings.artifact_dir
    resolved_tracking_uri = tracking_uri or settings.mlflow_tracking_uri

    try:
        config, config_file_path = load_config(
            file=config_file, path=resolved_config_dir
        )
        classifier_alias = config.model.classifier
        threshold_strategy = config.threshold.strategy
        selected_use_case = get_use_case(use_case)
        experiment_name = f"{selected_use_case.name}-{split_name}"
        model_name = f"{selected_use_case.name}-model"
        run_name = f"{selected_use_case.name}-{classifier_alias}-{split_name}"
        manifest_file = settings.data_dir / f"manifests/{run_name}.json"

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
            "Configured classifier alias '%s' with threshold strategy '%s'.",
            classifier_alias,
            threshold_strategy,
        )
        logger.info(
            "Model registration is %s; model promotion is %s.",
            "enabled" if register_model else "disabled",
            "enabled" if register_model and promote_model else "disabled",
        )

        experiment_id = setup_local_experiment(
            experiment_name=experiment_name,
            tracking_uri=resolved_tracking_uri,
            artifact_dir=resolved_artifact_dir,
        )
        experiment = mlflow.get_experiment(experiment_id)
        logger.info(
            "MLflow experiment '%s' initialized with id '%s'.",
            experiment_name,
            experiment_id,
        )
        logger.info("Tracking URI: %s", mlflow.get_tracking_uri())
        logger.info("MLflow artifact location: %s", experiment.artifact_location)

        train_df, test_df = load_and_validate_training_splits(
            use_case=selected_use_case,
            split_name=split_name,
            split_dir=resolved_split_dir,
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
            experiment_name,
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
            if tune_model:
                tuning_result = optimize_hyperparameters(
                    use_case=selected_use_case,
                    config=config,
                    train_df=train_df,
                    n_jobs=n_jobs,
                )
                mlflow.log_params(
                    {
                        "tuning_n_trials": tuning_result.n_trials,
                        "tuning_n_splits": tuning_result.n_splits,
                        "tuning_n_jobs": n_jobs,
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
                    use_case=selected_use_case,
                    config=config,
                    train_df=train_df,
                    test_df=test_df,
                    model_params_override=tuning_result.best_params,
                    calibration_method=calibration_method,
                    n_jobs=n_jobs,
                )
            else:
                logger.info(
                    "Hyperparameter tuning is disabled for this run; using configured "
                    "classifier parameters."
                )
                result = train_model(
                    use_case=selected_use_case,
                    config=config,
                    train_df=train_df,
                    test_df=test_df,
                    calibration_method=calibration_method,
                    n_jobs=n_jobs,
                )
            result.metrics["promotion_score"] = (
                PromotionService.calculate_weighted_score(
                    result.metrics,
                    config.promotion.metric_weights,
                )
            )
            model_info = log_experiment_result(
                use_case=selected_use_case,
                result=result,
                config=config,
                config_file_path=config_file_path,
                calibration_method=calibration_method,
                tune_model=tune_model,
                register_model=register_model,
                promote_model=promote_model,
            )
            logger.info(
                "Training results logged to MLflow run '%s' (model URI: '%s').",
                run_id,
                model_info.model_uri,
            )
        registered_model_version = None
        promotion_decision_dict = {}
        if register_model:
            registry = ModelRegistry()
            registered_version = registry.register_model(
                model_info=model_info,
                model_name=model_name,
            )
            registered_model_version = int(registered_version.version)
            if promote_model:
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
                    "Model promotion is disabled; registered model has no alias."
                )
        else:
            logger.info(
                "Model registration is disabled; skipping registry and promotion steps."
            )

        if register_model or write_manifest:
            manifest = TrainingManifest(
                run=ManifestRun(run_id=run_id, run_name=run_name),
                model=ManifestModel(
                    use_case=selected_use_case.name,
                    model_name=model_name,
                    model_version=registered_model_version,
                    model_reference=LoggedModelReference.from_model_info(model_info),
                ),
                data=ManifestData(
                    target_column=selected_use_case.target_column,
                    time_column=selected_use_case.time_column,
                    split_name=split_name,
                    split_metadata=split_metadata,
                ),
                training=ManifestTraining(
                    classifier_name=result.classifier_config["model_name"],
                    classifier_alias=classifier_alias,
                    feature_count=len(result.feature_names_out),
                    feature_names_in=result.feature_names_in,
                    feature_names_out=result.feature_names_out,
                    calibration=calibration_method,
                    n_jobs=n_jobs,
                    tuning_enabled=tune_model,
                    threshold_strategy=config.threshold.strategy,
                    threshold_data=result.threshold_data,
                    threshold=result.threshold,
                ),
                evaluation=ManifestEvaluation(metrics=result.metrics),
                lifecycle=ManifestLifecycle(
                    registration_enabled=register_model,
                    promotion_enabled=(register_model and promote_model),
                    promotion_decision=promotion_decision_dict,
                ),
            )

            model_card = ModelCardBuilder().build(
                config=config,
                context=manifest,
            )

            if register_model:
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
