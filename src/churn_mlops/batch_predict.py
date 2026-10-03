import argparse
import logging
from pathlib import Path

import pandas as pd

from churn_mlops.config import ServingSettings, configure_logging
from churn_mlops.data import load_raw_data, validate_data
from churn_mlops.data.storage import (
    read_partitioned_dataset,
    write_partitioned_dataset,
)
from churn_mlops.serving import Predictor, load_model

logger = logging.getLogger(__name__)


def run_batch_prediction(
    df: pd.DataFrame,
    tracking_uri: str | None = None,
    model_name: str | None = None,
    model_alias: str | None = None,
) -> pd.DataFrame:
    """Validate input data and return a DataFrame containing predictions.

    Args:
        df: Customer feature DataFrame to score.
        tracking_uri: Optional MLflow tracking URI.
        model_name: Optional registered model name.
        model_alias: Optional registered model alias.

    Returns:
        Validated input data with model predictions.
    """
    settings = ServingSettings()
    resolved_tracking_uri = tracking_uri or settings.mlflow_tracking_uri
    resolved_model_name = model_name or settings.model_name
    resolved_model_alias = model_alias or settings.model_alias

    try:
        logger.info("Starting batch prediction for %d rows.", len(df))
        df = validate_data(df)

        loaded_model = load_model(
            tracking_uri=resolved_tracking_uri,
            model_name=resolved_model_name,
            model_alias=resolved_model_alias,
        )

        predictor = Predictor(loaded_model)
        df_pred = predictor.predict_batch(df=df)

        logger.info("Generated %d predictions.", len(df_pred))
        logger.info(
            "Average churn probability: %.4f",
            df_pred["predicted_probability"].mean(),
        )
        return df_pred
    except Exception:
        logger.exception("Batch prediction failed.")
        raise


def main() -> None:
    """Read batch input, generate predictions, and persist the results.

    CSV input is written to CSV output. Partitioned input is read with
    optional date bounds and written as a partitioned Parquet dataset.
    """
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_csv")
    input_group.add_argument("--input_dataset")
    parser.add_argument("--output_csv")
    parser.add_argument("--output_dataset")
    parser.add_argument("--index_col")
    parser.add_argument("--input_dir")
    parser.add_argument("--start_date")
    parser.add_argument("--end_date")
    parser.add_argument("--timestamp_column", default="reference_date")

    args = parser.parse_args()
    is_csv_input = args.input_csv is not None

    if is_csv_input and (args.output_csv is None or args.output_dataset is not None):
        parser.error("CSV input requires --output_csv and cannot use --output_dataset.")
    if not is_csv_input and args.output_csv is not None:
        parser.error("Partitioned input cannot use --output_csv.")
    if (args.start_date is None) != (args.end_date is None):
        parser.error("--start_date and --end_date must be provided together.")
    if is_csv_input and (args.start_date is not None or args.end_date is not None):
        parser.error("Date bounds can only be used with partitioned input.")
    if not is_csv_input and args.index_col is not None:
        parser.error("--index_col can only be used with CSV input.")

    if is_csv_input:
        settings = ServingSettings()
        input_dir = (
            Path(args.input_dir) if args.input_dir else settings.data_dir / "raw"
        )
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
        df = read_partitioned_dataset(
            dataset_name=args.input_dataset,
            timestamp_column=args.timestamp_column,
            start=args.start_date,
            end=args.end_date,
        )
        df_pred = run_batch_prediction(df=df)
        output_dataset = args.output_dataset or "batch_predictions"
        write_partitioned_dataset(
            df=df_pred,
            dataset_name=output_dataset,
            timestamp_column=args.timestamp_column,
            overwrite_partitions=True,
        )
        logger.info("Partitioned batch prediction written to '%s'.", output_dataset)


if __name__ == "__main__":
    configure_logging()
    main()
