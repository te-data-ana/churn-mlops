import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from .config import ServingSettings, configure_logging

logger = logging.getLogger(__name__)

__all__ = ["main"]

__version__ = "0.4.0"


def main(argv: Sequence[str] | None = None) -> None:
    """Parse and dispatch commands for the package CLI.

    Args:
        argv: Optional argument list to parse. When omitted, the process argv is
            used.
    """

    configure_logging()

    settings = ServingSettings()

    parser = argparse.ArgumentParser(
        prog="churn-mlops",
        description=(
            "Train, score, and serve churn prediction models with the local "
            "MLflow workflow."
        ),
        epilog=(
            "Examples:\n"
            "  churn-mlops train --config sample_training_config.yaml\n"
            "  churn-mlops batch-predict --input_csv customer.csv --output_csv predictions.csv\n"
            "  churn-mlops serve --host 127.0.0.1 --port 8000 --reload"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    prepare_parser = subparsers.add_parser(
        "prepare-data",
        help="Convert raw CSV datasets into partitioned Parquet datasets.",
    )
    prepare_parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Input CSV file path.",
    )
    prepare_parser.add_argument(
        "--timestamp_column",
        type=str,
        default="reference_date",
        help="Timestamp column used for partitioning.",
    )

    split_parser = subparsers.add_parser(
        "create-split",
        help="Create train and test datasets using an out-of-time split.",
    )
    split_parser.add_argument(
        "--dataset_name",
        required=True,
        help="Partitioned dataset name.",
    )
    split_parser.add_argument(
        "--timestamp_column",
        default="reference_date",
        help="Timestamp column used for splitting.",
    )
    split_parser.add_argument(
        "--start_date",
        required=True,
        help="Inclusive extraction start date.",
    )
    split_parser.add_argument(
        "--split_date",
        required=True,
        help="First date belonging to the test split.",
    )
    split_parser.add_argument(
        "--end_date",
        required=True,
        help="Inclusive extraction end date.",
    )
    split_parser.add_argument(
        "--split_name",
        default="default",
        help="Output filename prefix.",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train and evaluate a churn model from a YAML config.",
    )
    train_parser.add_argument(
        "--config", required=True, help="Training config file path."
    )
    train_parser.add_argument(
        "--split_name",
        default="default",
        help="Name prefix of prepared train/test Parquet files in data/splits.",
    )
    train_parser.add_argument(
        "--exclude-columns",
        nargs="+",
        default=(),
        help="Column names to exclude from model features/training/hyperparameter tuning.",
    )
    train_parser.add_argument(
        "--experiment_name",
        required=False,
        help="Experiment name used for MLflow tracking.",
    )
    train_parser.add_argument(
        "--register-model",
        action="store_true",
        default=None,
        help="Register this run's model even when the config disables registration.",
    )
    train_parser.add_argument(
        "--promote-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable candidate/champion alias updates for this run.",
    )

    batch_parser = subparsers.add_parser(
        "batch-predict",
        help="Generate batch predictions from CSV or partitioned data.",
    )
    batch_input = batch_parser.add_mutually_exclusive_group(required=True)
    batch_input.add_argument(
        "--input_csv",
        help="Input CSV filename under configured directory for raw data input.",
    )
    batch_input.add_argument(
        "--input_dataset",
        help="Name of partitioned input dataset under configured data directory.",
    )
    batch_parser.add_argument(
        "--output_csv",
        help="Output CSV filename under configured output directory.",
    )
    batch_parser.add_argument(
        "--output_dataset",
        type=str,
        default="batch_predictions",
        help="Name of ouput dataset under configured data directory.",
    )
    batch_parser.add_argument(
        "--index_col",
        required=False,
        default=None,
        help="Optional index column name for the input CSV.",
    )
    batch_parser.add_argument(
        "--start_date",
        type=str,
        help="Inclusive start date (partitioned input).",
    )
    batch_parser.add_argument(
        "--end_date",
        type=str,
        help="Inclusive end date (partitioned input).",
    )
    batch_parser.add_argument(
        "--timestamp_column",
        type=str,
        default="reference_date",
        help="Timestamp column for partitioned in-/output.",
    )
    batch_parser.add_argument(
        "--model_name",
        help="Registered model name (defaults to the configured serving model).",
    )
    batch_parser.add_argument(
        "--model_version",
        type=int,
        help="Exact registered model version; mutually exclusive with a manifest.",
    )
    batch_parser.add_argument(
        "--model_manifest",
        type=Path,
        help="Path of model manifest in configured data directory.",
    )

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the FastAPI prediction service.",
    )
    serve_parser.add_argument(
        "--host", default=settings.api_host, help="Hostname for the API server."
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=settings.api_port,
        help="Port for the API server.",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for local development.",
    )
    serve_parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Registered model name (defaults to the configured serving model).",
    )
    serve_parser.add_argument(
        "--model-version",
        type=int,
        default=None,
        help="Exact registered model version; mutually exclusive with a manifest.",
    )
    serve_parser.add_argument(
        "--model_manifest",
        type=Path,
        help="Path of model manifest in configured data directory.",
    )

    ingest_prediction_log_parser = subparsers.add_parser(
        "ingest-prediction-logs",
        help="Convert API prediction logs into a partitioned Parquet dataset.",
    )
    ingest_prediction_log_parser.add_argument(
        "--dataset_name",
        type=str,
        default="api_predictions",
        help="Target dataset name.",
    )
    ingest_prediction_log_parser.add_argument(
        "--timestamp_column",
        type=str,
        default="reference_date",
        help="Timestamp column used for partitioning.",
    )

    ingest_error_log_parser = subparsers.add_parser(
        "ingest-error-logs",
        help="Convert API prediction error logs into a partitioned Parquet dataset.",
    )
    ingest_error_log_parser.add_argument(
        "--dataset_name",
        type=str,
        default="api_errors",
        help="Target dataset name.",
    )
    ingest_error_log_parser.add_argument(
        "--timestamp_column",
        type=str,
        default="reference_date",
        help="Timestamp column used for partitioning.",
    )

    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Monitor prediction quality based on files or partitioned datasets.",
    )
    monitor_reference = monitor_parser.add_mutually_exclusive_group(required=True)
    monitor_reference.add_argument(
        "--reference_csv",
        type=Path,
        help="Path to the labeled reference CSV file.",
    )
    monitor_reference.add_argument(
        "--reference_dataset",
        type=str,
        help="Name of partitioned reference dataset in configured data directory.",
    )
    monitor_source = monitor_parser.add_mutually_exclusive_group(required=True)
    monitor_source.add_argument(
        "--analysis_csv",
        type=Path,
        help="Path to the scored analysis CSV file.",
    )
    monitor_source.add_argument(
        "--prediction_log",
        type=Path,
        help="API prediction JSONL path; defaults to the configured serving log.",
    )
    monitor_source.add_argument(
        "--api",
        action="store_true",
        help="Monitor the configured API prediction and error logs.",
    )
    monitor_source.add_argument(
        "--analysis_dataset",
        type=str,
        help="Name of partitioned analysis dataset in configured data directory.",
    )
    monitor_errors = monitor_parser.add_mutually_exclusive_group()
    monitor_errors.add_argument(
        "--error_log",
        type=Path,
        help="API prediction error JSONL path (API file mode only).",
    )
    monitor_errors.add_argument(
        "--error_dataset",
        type=str,
        help="Name of partitioned error dataset in the configured data directory.",
    )
    monitor_parser.add_argument(
        "--timestamp_column",
        type=str,
        default="reference_date",
        help="Timestamp/partition column for partitioned datasets (default: reference_date).",
    )
    monitor_parser.add_argument(
        "--model_name",
        type=str,
        help="Registered model name to include in MLflow monitoring tags.",
    )
    monitor_parser.add_argument(
        "--model_version",
        type=int,
        help="Exact model version to include in the monitoring report.",
    )
    monitor_parser.add_argument(
        "--model_manifest",
        type=Path,
        help="Path of model manifest in configured data directory.",
    )
    monitor_parser.add_argument(
        "--mlflow_experiment_name",
        type=str,
        help="MLflow experiment name for logging the monitoring run.",
    )
    monitor_parser.add_argument(
        "--run_name",
        type=str,
        help="MLflow run name (default: monitoring).",
    )
    monitor_parser.add_argument(
        "--reference_name",
        type=str,
        help="Reference input label for MLflow monitoring tags.",
    )
    monitor_parser.add_argument(
        "--analysis_name",
        type=str,
        help="Analysis input label for MLflow monitoring tags.",
    )
    monitor_parser.add_argument(
        "--source_reference",
        choices=("batch", "api"),
        help="Source label for partitioned reference data (default: batch).",
    )
    monitor_parser.add_argument(
        "--source_analysis",
        choices=("batch", "api"),
        help="Source label for partitioned analysis data (default: api).",
    )
    monitor_parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for monitoring reports (default: configured output directory).",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        raise SystemExit(0)

    if args.command == "prepare-data":
        from .data import load_raw_data, validate_data
        from .data.storage import write_partitioned_dataset

        df = load_raw_data(file_name=args.input_csv, data_dir=settings.data_dir / "raw")
        df = validate_data(df)

        write_partitioned_dataset(
            df=df,
            dataset_name="partitioned",
            timestamp_column=args.timestamp_column,
            overwrite_partitions=True,
        )

        return

    if args.command == "create-split":
        from .data.splitting import create_time_based_split

        split_metadata = create_time_based_split(
            dataset_name=args.dataset_name,
            timestamp_column=args.timestamp_column,
            start_date=args.start_date,
            split_date=args.split_date,
            end_date=args.end_date,
            split_name=args.split_name,
        )

        logger.info("OOT train/test split created with: %s", split_metadata)

        return

    if args.command == "train":
        from .train import run_training_job

        training_kwargs: dict[str, object] = {
            "config_file": args.config,
            "split_name": args.split_name,
            "exclude_columns": args.exclude_columns,
        }
        optional_kwargs = {
            "experiment_name": args.experiment_name,
            "register_model": args.register_model,
            "promote_model": args.promote_model,
        }
        training_kwargs.update(
            {key: value for key, value in optional_kwargs.items() if value is not None}
        )
        result = run_training_job(**training_kwargs)
        logger.info(
            "Successfully trained %s model: AUC=%.4f",
            result.classifier_config["model_name"],
            result.metrics["roc_auc"],
        )
        return

    if args.command == "batch-predict":
        is_csv_input = args.input_csv is not None
        if is_csv_input and args.output_csv is None:
            parser.error("CSV input requires --output_csv.")
        if not is_csv_input and args.output_csv is not None:
            parser.error("Partitioned input cannot use --output_csv.")
        if (args.start_date is None) != (args.end_date is None):
            parser.error("--start_date and --end_date must be provided together.")
        if is_csv_input and (args.start_date is not None or args.end_date is not None):
            parser.error("Date bounds can only be used with partitioned input.")
        if not is_csv_input and args.index_col is not None:
            parser.error("--index_col can only be used with CSV input.")
        if args.model_manifest is not None and (
            args.model_name is not None or args.model_version is not None
        ):
            parser.error(
                "--model_manifest cannot be combined with --model_name or --model_version."
            )
        if (args.model_name is None) != (args.model_version is None):
            parser.error("--model_name and --model_version must be provided together.")

        model_name = args.model_name
        model_version = args.model_version
        if args.model_manifest is not None:
            from .tracking import TrainingManifest

            model_manifest_file = settings.data_dir / args.model_manifest
            manifest = TrainingManifest.read(model_manifest_file)
            model_name = manifest.model_name
            model_version = manifest.model_version

        from .batch_predict import run_batch_prediction

        if is_csv_input:
            from .data import load_raw_data

            input_dir = settings.data_dir / "raw"
            df = load_raw_data(
                file_name=args.input_csv,
                index_col=args.index_col,
                data_dir=input_dir,
            )
            prediction_kwargs: dict[str, object] = {"df": df}
            if model_name is not None:
                prediction_kwargs["model_name"] = model_name
            if model_version is not None:
                prediction_kwargs["model_version"] = model_version
            df_pred = run_batch_prediction(**prediction_kwargs)

            output_path = settings.output_dir / args.output_csv
            settings.output_dir.mkdir(parents=True, exist_ok=True)
            df_pred.to_csv(output_path)
            logger.info("Batch prediction output written to '%s'.", output_path)
        else:
            from .data.storage import (
                read_partitioned_dataset,
                write_partitioned_dataset,
            )

            df = read_partitioned_dataset(
                dataset_name=args.input_dataset,
                timestamp_column=args.timestamp_column,
                start=args.start_date,
                end=args.end_date,
            )
            prediction_kwargs: dict[str, object] = {"df": df}
            if model_name is not None:
                prediction_kwargs["model_name"] = model_name
            if model_version is not None:
                prediction_kwargs["model_version"] = model_version
            df_pred = run_batch_prediction(**prediction_kwargs)
            write_partitioned_dataset(
                df=df_pred,
                dataset_name=args.output_dataset,
                timestamp_column=args.timestamp_column,
                overwrite_partitions=True,
            )
            logger.info(
                "Partitioned batch prediction written to '%s'.",
                args.output_dataset,
            )
        return

    if args.command == "serve":
        if args.model_manifest is not None and (
            args.model_name is not None or args.model_version is not None
        ):
            parser.error(
                "--model_manifest cannot be combined with --model_name or --model_version."
            )
        if (args.model_name is None) != (args.model_version is None):
            parser.error("--model_name and --model_version must be provided together.")

        if args.model_manifest is not None:
            from .tracking import TrainingManifest

            model_manifest_file = settings.data_dir / args.model_manifest
            manifest = TrainingManifest.read(model_manifest_file)
            os.environ["MODEL_NAME"] = manifest.model_name
            os.environ["MODEL_VERSION"] = str(manifest.model_version)

        if args.model_name is not None and args.model_version is not None:
            os.environ["MODEL_NAME"] = args.model_name
            os.environ["MODEL_VERSION"] = str(args.model_version)

        import uvicorn

        uvicorn.run(
            "churn_mlops.serving.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return

    if args.command == "ingest-prediction-logs":
        from .data.ingestion import ingest_prediction_logs

        ingest_prediction_logs(
            dataset_name=args.dataset_name,
            timestamp_column=args.timestamp_column,
        )
        return

    if args.command == "ingest-error-logs":
        from .data.ingestion import ingest_error_logs

        ingest_error_logs(
            dataset_name=args.dataset_name,
            timestamp_column=args.timestamp_column,
        )
        return

    if args.command == "monitor":
        dataset_mode = args.reference_dataset is not None
        if dataset_mode != (args.analysis_dataset is not None):
            parser.error(
                "--reference_dataset and --analysis_dataset must be used together."
            )
        if args.error_dataset is not None and not dataset_mode:
            parser.error("--error_dataset can only be used with dataset inputs.")
        if args.error_log is not None and (
            dataset_mode or (args.prediction_log is None and not args.api)
        ):
            parser.error("--error_log can only be used with API file inputs.")
        if not dataset_mode and (
            args.source_reference is not None or args.source_analysis is not None
        ):
            parser.error(
                "--source_reference and --source_analysis can only be used with "
                "dataset inputs."
            )
        if args.model_manifest is not None and (
            args.model_name is not None or args.model_version is not None
        ):
            parser.error(
                "--model_manifest cannot be combined with --model_name or "
                "--model_version."
            )

        model_name = args.model_name
        model_version = args.model_version
        if args.model_manifest is not None:
            from .tracking import TrainingManifest

            manifest = TrainingManifest.read(settings.data_dir / args.model_manifest)
            model_name = manifest.model_name
            model_version = manifest.model_version

        output_dir = args.output_dir or settings.output_dir
        if dataset_mode:
            from .data import read_partitioned_dataset
            from .monitoring.core import build_monitoring_report

            reference = read_partitioned_dataset(
                dataset_name=args.reference_dataset,
                timestamp_column=args.timestamp_column,
            )
            analysis = read_partitioned_dataset(
                dataset_name=args.analysis_dataset,
                timestamp_column=args.timestamp_column,
            )
            errors = (
                read_partitioned_dataset(
                    dataset_name=args.error_dataset,
                    timestamp_column=args.timestamp_column,
                )
                if args.error_dataset is not None
                else None
            )
            report_kwargs: dict[str, object] = {
                "reference": reference,
                "analysis": analysis,
                "source_reference": args.source_reference or "batch",
                "source_analysis": args.source_analysis or "api",
                "errors": errors,
                "output_dir": output_dir,
            }
            if model_version is not None:
                report_kwargs["model_version"] = model_version
            report = build_monitoring_report(**report_kwargs)
        else:
            from .monitoring.core import run_monitoring

            prediction_log = args.prediction_log
            if args.api and prediction_log is None:
                prediction_log = settings.prediction_log_path
            error_log = args.error_log
            if (args.api or prediction_log is not None) and error_log is None:
                error_log = settings.error_log_path
            report_kwargs = {
                "reference_csv": args.reference_csv,
                "analysis_csv": args.analysis_csv,
                "prediction_log": prediction_log,
                "error_log": error_log,
                "output_dir": output_dir,
            }
            if model_version is not None:
                report_kwargs["model_version"] = model_version
            report = run_monitoring(**report_kwargs)

        if args.mlflow_experiment_name is not None or args.run_name is not None:
            from .monitoring.core import log_monitoring_run

            reference_input = args.reference_dataset or str(args.reference_csv)
            if args.analysis_dataset is not None:
                analysis_input = args.analysis_dataset
            elif args.analysis_csv is not None:
                analysis_input = str(args.analysis_csv)
            else:
                analysis_input = str(args.prediction_log or "api")
            tags = {
                "reference_input": reference_input,
                "analysis_input": analysis_input,
                "reference_name": args.reference_name or reference_input,
                "analysis_name": args.analysis_name or analysis_input,
            }
            if model_name is not None:
                tags["model_name"] = model_name
            if model_version is not None:
                tags["model_version"] = str(model_version)
            log_monitoring_run(
                report,
                experiment_name=(
                    args.mlflow_experiment_name or settings.mlflow_experiment_name
                ),
                run_name=args.run_name or "monitoring",
                tags=tags,
            )
        return

    parser.print_help()
    raise SystemExit(1)


if __name__ == "__main__":
    main()
