"""
Utilities for storing and retrieving partitioned Parquet datasets.

Datasets are partitioned by year and month to enable efficient
time-range queries for model training, evaluation, monitoring,
and retraining workflows.
"""

import shutil

import pandas as pd

from churn_mlops.config import RuntimeSettings

type PartitionFilter = list[list[tuple[str, str, int]]]


def write_partitioned_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    timestamp_column: str,
    overwrite_partitions: bool = False,
) -> None:
    """
    Write a DataFrame as a year/month partitioned Parquet dataset.

    The timestamp column is converted to datetime if necessary and used
    to derive ``year`` and ``month`` partition columns. The dataset is
    written to ``data/{dataset_name}`` using Hive-style partitioning.

    If ``overwrite_partitions`` is ``True``, existing partitions whose
    year/month combinations are present in the input DataFrame are
    removed before writing new data. This prevents duplicate records
    when re-ingesting data for an existing period.

    Args:
        df:
            DataFrame to persist. If the input DataFrame is empty, no
            output is written.

        dataset_name:
            Name of the output dataset directory.

        timestamp_column:
            Name of the timestamp column used to derive partition keys.
            Note that the column must be convertible to datetime via
            ``pd.to_datetime``.

        overwrite_partitions:
            Whether to replace existing year/month partitions before
            writing new data. Defaults to ``True``.

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
    dataset_dir = settings.data_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if overwrite_partitions:
        year_month_comb = df[["year", "month"]].drop_duplicates().values
        for year, month in year_month_comb:
            partition_dir = dataset_dir / f"year={year}" / f"month={month}"

            if partition_dir.exists():
                shutil.rmtree(partition_dir)

    df.to_parquet(
        path=dataset_dir,
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
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Read rows from a partitioned Parquet dataset.

    When ``start`` and ``end`` are provided, the function first applies
    year/month partition pruning to limit the number of Parquet files read
    and then applies an exact timestamp filter. A date-only string supplied
    as ``end`` includes the entire end calendar day; explicit timestamps are
    treated as exact inclusive bounds.

    When both ``start`` and ``end`` are ``None``, the entire dataset is
    loaded without partition pruning or timestamp filtering.

    Args:
        dataset_name:
            Name of the partitioned dataset directory.

        timestamp_column:
            Name of the timestamp column used for filtering when a date
            range is specified.

        start:
            Inclusive start date of the requested time window. Must be
            provided together with ``end``. If both ``start`` and ``end``
            are ``None``, the entire dataset is loaded. Defaults to
            ``None``.

        end:
            Inclusive end bound of the requested time window. A date-only
            string includes that entire calendar day. Must be provided
            together with ``start``. If both ``start`` and ``end`` are
            ``None``, the entire dataset is loaded. Defaults to ``None``.

    Returns:
        A DataFrame containing either:

        * all rows in the dataset if ``start`` and ``end`` are ``None``; or
        * all rows whose timestamp falls within the specified date range.

    Raises:
        FileNotFoundError:
            If the dataset directory does not exist.

        ValueError:
            If only one of ``start`` or ``end`` is provided, or if
            ``start`` is later than ``end``.
    """

    settings = RuntimeSettings()
    dataset_dir = settings.data_dir / dataset_name

    if not dataset_dir.exists():
        msg = f"Dataset does not exist: {dataset_dir}"
        raise FileNotFoundError(msg)

    if start is None and end is None:
        df = pd.read_parquet(dataset_dir)
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    else:
        if start is None or end is None:
            msg = "'start' and 'end' must either both be provided or both be None."
            raise ValueError(msg)

        start = pd.Timestamp(start)
        end_is_date_only = (
            isinstance(end, str) and pd.Timestamp(end).strftime("%Y-%m-%d") == end
        )
        end = pd.Timestamp(end)
        if end_is_date_only:
            end += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

        partition_filters = _partition_filters(start, end)

        df = pd.read_parquet(dataset_dir, filters=partition_filters)

        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

        date_filter = df[timestamp_column].between(start, end, inclusive="both")

        df = df[date_filter]

    return df.reset_index(drop=True)
