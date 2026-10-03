import argparse
import logging
import sys
from collections.abc import Sequence

from .config import ServingSettings, configure_logging

logger = logging.getLogger(__name__)

__all__ = ["main"]

__version__ = "0.3.0"


def main(argv: Sequence[str] | None = None) -> None:
    """Parse and dispatch commands for the package CLI.

    Args:
        argv: Optional argument list to parse. When omitted, the process argv is
            used.
    """
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
        "--experiment_name",
        required=False,
        help="Experiment name used for MLflow tracking.",
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
        help="Monitor scored batch predictions or live API events.",
    )
    monitor_parser.add_argument("--reference_csv", required=True, type=str)
    monitor_source = monitor_parser.add_mutually_exclusive_group(required=True)
    monitor_source.add_argument("--analysis_csv", type=str)
    monitor_source.add_argument(
        "--prediction_log",
        type=str,
        help="API prediction JSONL path; defaults to the configured serving log.",
    )
    monitor_source.add_argument(
        "--api",
        action="store_true",
        help="Monitor the configured API prediction and error logs.",
    )
    monitor_parser.add_argument("--error_log", type=str)
    monitor_parser.add_argument("--index_col", default="customerid")
    monitor_parser.add_argument("--output_dir", type=str, default=None)

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

        print(split_metadata)

        return

    if args.command == "train":
        from .train import run_training_job

        training_kwargs = {"config_file": args.config, "split_name": args.split_name}
        if args.experiment_name is not None:
            training_kwargs["experiment_name"] = args.experiment_name
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

        from .batch_predict import run_batch_prediction

        if is_csv_input:
            from .data import load_raw_data

            input_dir = settings.data_dir / "raw"
            df = load_raw_data(
                file_name=args.input_csv,
                index_col=args.index_col,
                data_dir=input_dir,
            )
            df_pred = run_batch_prediction(df=df)

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
            df_pred = run_batch_prediction(df=df)
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
        from .monitoring.core import main as monitoring_main

        command = [
            "churn-mlops.monitoring",
            "--reference_csv",
            args.reference_csv,
        ]
        if args.analysis_csv is not None:
            command.extend(["--analysis_csv", args.analysis_csv])
        elif args.prediction_log is not None:
            command.extend(["--prediction_log", args.prediction_log])
        else:
            command.append("--api")
        if args.error_log is not None:
            command.extend(["--error_log", args.error_log])
        if args.output_dir is not None:
            command.extend(["--output_dir", args.output_dir])
        sys.argv = command
        monitoring_main()
        return

    parser.print_help()
    raise SystemExit(1)


if __name__ == "__main__":
    configure_logging()
    main()
