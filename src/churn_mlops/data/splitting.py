import logging
from pathlib import Path
from typing import TypedDict

import pandas as pd

from churn_mlops.config import RuntimeSettings
from churn_mlops.data.ingestion import normalize_strings
from churn_mlops.data.storage import read_partitioned_dataset
from churn_mlops.data.validation import validate_data

logger = logging.getLogger(__name__)


class SplitMetadata(TypedDict):
    train_rows: int
    test_rows: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def load_and_validate_training_splits(
    split_name: str = "default",
    split_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the prepared train and test Parquet files.

    Args:
        split_name: Prefix shared by the train and test split filenames.
        split_dir: Directory containing the split files. Defaults to the
            configured ``data/splits`` directory.

    Returns:
        Validated training and test DataFrames, in that order.

    Raises:
        FileNotFoundError: If either split file does not exist.
        pandera.errors.SchemaError: If either split violates the data schema.
    """
    settings = RuntimeSettings()
    resolved_split_dir = split_dir or settings.data_dir / "splits"
    split_data: dict[str, pd.DataFrame] = {}

    for split in ("train", "test"):
        split_file = resolved_split_dir / f"{split_name}_{split}.parquet"
        logger.info("Loading %s split from '%s'.", split, split_file)
        df = pd.read_parquet(split_file)
        df.columns = normalize_strings(df.columns)
        df.drop(columns=["reference_date"], errors="ignore", inplace=True)
        split_data[split] = validate_data(df.convert_dtypes())

    return split_data["train"], split_data["test"]


def create_time_based_split(
    dataset_name: str,
    timestamp_column: str,
    start_date: str | pd.Timestamp,
    split_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    split_name: str = "default",
) -> SplitMetadata:
    """
    Create train and test datasets using an out-of-time split and
    persist them as Parquet files.

    Records with timestamps prior to ``split_date`` are assigned to the
    training dataset. Records with timestamps on or after ``split_date``
    are assigned to the test dataset.

    The resulting datasets are written to:

        data/splits/{split_name}_train.parquet
        data/splits/{split_name}_test.parquet

    Args:
        dataset_name:
            Name of the partitioned dataset to read from.

        timestamp_column:
            Name of the timestamp column used for filtering and
            splitting.

        start_date:
            Inclusive start of the extraction window.

        split_date:
            First timestamp that belongs to the test dataset.

        end_date:
            Inclusive end of the extraction window.

        split_name:
            Prefix used for the generated split filenames.

    Returns:
        Metadata describing the generated split, including row counts
        and date boundaries for the train and test datasets.

    Raises:
        ValueError:
            If ``split_date`` is not strictly later than
            ``start_date``.

        ValueError:
            If ``split_date`` is later than ``end_date``.

        ValueError:
            If either the training or test dataset is empty.
    """

    start_date = pd.Timestamp(start_date)
    split_date = pd.Timestamp(split_date)
    end_date = pd.Timestamp(end_date)

    if start_date >= split_date:
        msg = "'split_date' must be later than 'start_date'."
        raise ValueError(msg)

    if split_date > end_date:
        msg = "'split_date' must be before or equal to 'end_date'."
        raise ValueError(msg)

    df = read_partitioned_dataset(
        dataset_name=dataset_name,
        timestamp_column=timestamp_column,
        start=start_date,
        end=end_date,
    )

    train = df.loc[df[timestamp_column] < split_date].reset_index(drop=True)

    test = df.loc[df[timestamp_column] >= split_date].reset_index(drop=True)

    if train.empty:
        msg = "Training split is empty."
        raise ValueError(msg)

    if test.empty:
        msg = "Test split is empty."
        raise ValueError(msg)

    settings = RuntimeSettings()

    output_dir = settings.data_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(
        output_dir / f"{split_name}_train.parquet",
        index=False,
    )

    test.to_parquet(
        output_dir / f"{split_name}_test.parquet",
        index=False,
    )

    return SplitMetadata(
        train_rows=len(train),
        test_rows=len(test),
        train_start=train[timestamp_column].min(),
        train_end=train[timestamp_column].max(),
        test_start=test[timestamp_column].min(),
        test_end=test[timestamp_column].max(),
    )
