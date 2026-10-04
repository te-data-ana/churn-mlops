import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from churn_mlops.config import RuntimeSettings, ServingSettings
from churn_mlops.data.jsonl_logs import read_jsonl_log
from churn_mlops.data.storage import write_partitioned_dataset
from churn_mlops.data.validation import validate_data

logger = logging.getLogger(__name__)


def normalize_strings(seq: Sequence[str]) -> Sequence[str]:
    return [s.strip().lower().replace(" ", "_") for s in seq]


def load_raw_data(
    file_name: str,
    index_col: str | None = None,
    data_dir: Path | None = None,
    drop_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load and normalize a raw CSV file from the configured data directory.

    Args:
        file_name: Name of the CSV file in ``data_dir``.
        index_col: Optional column name to use as the DataFrame index.
        data_dir: Directory containing the CSV file.
        drop_columns: Optional list of column names to drop from loaded data.

    Returns:
        DataFrame with normalized column names, nullable dtypes, the optional
        index applied, and rows containing only missing values removed.

    Raises:
        FileNotFoundError: If the requested CSV file does not exist.
    """
    settings = RuntimeSettings()
    resolved_data_dir = data_dir or settings.data_dir / "raw"

    try:
        logger.info("Loading raw dataset '%s' from '%s'.", file_name, resolved_data_dir)
        df = pd.read_csv(resolved_data_dir / file_name)
        df.columns = normalize_strings(df.columns)
        df = df.convert_dtypes()
        if drop_columns:
            df.drop(
                columns=normalize_strings(drop_columns), errors="ignore", inplace=True
            )
        if index_col:
            df.set_index(index_col, inplace=True)
            logger.info("Set index column '%s' on dataset '%s'.", index_col, file_name)
        prior_rows = len(df)
        df.dropna(how="all", inplace=True)
        dropped_rows = prior_rows - len(df)
        if dropped_rows:
            logger.warning(
                "Dropped %d all-missing rows from dataset '%s'.",
                dropped_rows,
                file_name,
            )
        logger.info(
            "Loaded dataset '%s' with %d rows and %d columns.",
            file_name,
            len(df),
            len(df.columns),
        )
        return df
    except Exception:
        logger.exception("Failed to load raw dataset '%s'.", file_name)
        raise


def combine_sample_predictions(data_dir: Path | None = None) -> pd.DataFrame:
    """Combine sample prediction files into a single DataFrame.

    Reads all CSV files matching the pattern
    ``dddd_sample_predictions.csv`` from the specified directory,
    adds a ``run_id`` column derived from the four-digit filename
    prefix, and concatenates the files into a single DataFrame.

    Args:
        data_dir: Directory containing sample prediction CSV files. If
            not provided, the default output data directory from the
            runtime settings is used.

    Returns:
        A DataFrame containing the combined prediction records from all
        matching files, including a ``run_id`` column identifying the
        source file of each record.
    """
    settings = RuntimeSettings()
    resolved_data_dir = data_dir or settings.output_dir

    dfs = []

    for path in resolved_data_dir.glob("[0-9][0-9][0-9][0-9]_sample_predictions.csv"):
        df = pd.read_csv(path)
        df["run_id"] = path.stem[:4]
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def ingest_sample_data(
    dataset_name: str,
    timestamp_column: str,
) -> None:
    """
    Combine sample prediction files, validate the resulting data,
    and append it as a partitioned Parquet dataset.

    Args:
        dataset_name:
            Name of the target dataset directory.
        timestamp_column:
            Name of the timestamp column used for partitioning.
    """
    df = combine_sample_predictions()
    df = validate_data(df)

    write_partitioned_dataset(
        df=df,
        dataset_name=dataset_name,
        timestamp_column=timestamp_column,
        overwrite_partitions=True,
    )


def ingest_jsonl_log(
    dataset_name: str,
    timestamp_column: str,
    jsonl_path: Path,
) -> None:
    """Ingest JSONL logs into a partitioned dataset.

    Reads records from a JSONL log and writes them to a partitioned
    Parquet dataset. Existing partitions are overwritten when records
    for the same partition are ingested again.

    If no valid records are found, the function logs an informational
    message and returns without writing any data.

    Args:
        dataset_name: Name of the target dataset directory.
        timestamp_column: Name of the timestamp column used for
            partitioning the dataset.
        jsonl_path: Path to the source JSONL log file.

    Raises:
        KeyError: If ``timestamp_column`` is not present in the
            prediction data.
    """

    df = read_jsonl_log(jsonl_path)

    if df.empty:
        logger.info(
            "No valid records found in %s. Skipping ingestion.",
            jsonl_path,
        )
        return

    write_partitioned_dataset(
        df=df,
        dataset_name=dataset_name,
        timestamp_column=timestamp_column,
        overwrite_partitions=True,
    )


def ingest_prediction_logs(
    dataset_name: str,
    timestamp_column: str,
    jsonl_path: Path | None = None,
) -> None:
    """Ingest API prediction logs into a partitioned dataset.

    Uses the configured prediction log path when ``jsonl_path`` is not provided.
    """
    settings = ServingSettings()

    ingest_jsonl_log(
        dataset_name=dataset_name,
        timestamp_column=timestamp_column,
        jsonl_path=jsonl_path or settings.prediction_log_path,
    )


def ingest_error_logs(
    dataset_name: str,
    timestamp_column: str,
    jsonl_path: Path | None = None,
) -> None:
    """Ingest API error logs into a partitioned dataset.

    Uses the configured error log path when ``jsonl_path`` is not provided.
    """
    settings = ServingSettings()

    ingest_jsonl_log(
        dataset_name=dataset_name,
        timestamp_column=timestamp_column,
        jsonl_path=jsonl_path or settings.error_log_path,
    )
