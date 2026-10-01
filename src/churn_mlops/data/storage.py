"""
Utilities for storing and retrieving partitioned Parquet datasets.

Datasets are partitioned by year and month to enable efficient
time-range queries for model training, evaluation, monitoring,
and retraining workflows.
"""

import pandas as pd

from churn_mlops.config import RuntimeSettings

type PartitionFilter = list[list[tuple[str, str, int]]]


def write_partitioned_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    timestamp_column: str,
) -> None:
    """
    Write a DataFrame as a year/month partitioned Parquet dataset.

    The timestamp column is converted to datetime if necessary and used
    to derive ``year`` and ``month`` partition columns. The resulting
    dataset is written to ``data/{dataset_name}`` using Hive-style
    partitioning.

    Args:
        df:
            DataFrame to persist.

        dataset_name:
            Name of the output dataset directory.

        timestamp_column:
            Name of the timestamp column used to derive partition keys.

    Returns:
        None.
    """

    if df.empty:
        return

    df = df.copy()

    df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    df["year"] = df[timestamp_column].dt.year
    df["month"] = df[timestamp_column].dt.month

    settings = RuntimeSettings()

    output_dir = settings.data_dir / dataset_name

    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        path=output_dir,
        partition_cols=["year", "month"],
        index=False,
    )


def _partition_filters(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> PartitionFilter:
    """
    Create Parquet partition filters for a date range.

    Builds Hive-style year/month partition filters that allow
    ``pd.read_parquet`` to read only the partitions intersecting the
    requested date range.

    Args:
        start:
            Inclusive start date.

        end:
            Inclusive end date.

    Returns:
        A nested filter structure suitable for the ``filters`` argument
        of ``pd.read_parquet``.

    Raises:
        ValueError:
            If ``start`` is later than ``end``.
    """

    if start > end:
        msg = "'start' must be before or equal to 'end'."
        raise ValueError(msg)

    months = pd.period_range(start=start, end=end, freq="M")

    filters: PartitionFilter = []

    for month in months:
        filters.append(
            [
                ("year", "==", month.year),
                ("month", "==", month.month),
            ]
        )

    return filters


def read_partitioned_dataset(
    dataset_name: str,
    timestamp_column: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """
    Read rows from a partitioned Parquet dataset within a date range.

    The function first applies year/month partition pruning to limit the
    number of Parquet files read and then applies an exact timestamp
    filter to ensure that only records within the requested date range
    are returned.

    Args:
        dataset_name:
            Name of the partitioned dataset directory.

        timestamp_column:
            Name of the timestamp column used for filtering.

        start:
            Inclusive start date of the requested time window.

        end:
            Inclusive end date of the requested time window.

    Returns:
        A DataFrame containing all rows whose timestamp falls within the
        specified date range.

    Raises:
        FileNotFoundError:
            If the dataset directory does not exist.

        ValueError:
            If ``start`` is later than ``end``.
    """

    settings = RuntimeSettings()

    dataset_dir = settings.data_dir / dataset_name
    if not dataset_dir.exists():
        msg = f"Dataset does not exist: {dataset_dir}"
        raise FileNotFoundError(msg)

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    partition_filters = _partition_filters(start, end)

    df = pd.read_parquet(dataset_dir, filters=partition_filters)

    df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    date_filter = df[timestamp_column].between(start, end, inclusive="both")
    df = df[date_filter]

    return df.reset_index(drop=True)
