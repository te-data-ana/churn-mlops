import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from churn_mlops.config import RuntimeSettings, configure_logging
from churn_mlops.data.ingestion import load_raw_data
from churn_mlops.data.validation import validate_data


def jsonl_prediction_log_to_csv(jsonl_path: Path, csv_path: Path) -> None:
    """Convert a prediction log from JSONL to CSV format.

    Flattens the nested ``features`` object into separate columns and
    writes the transformed records to a CSV file.

    Args:
        jsonl_path: Path to the source JSONL prediction log.
        csv_path: Path where the CSV file will be written.
    """
    df = pd.read_json(jsonl_path, lines=True)
    features = pd.json_normalize(df["features"])
    df = df.drop(columns=["features"])
    df = pd.concat([df, features], axis=1)
    df.to_csv(csv_path, index=False)


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

    return pd.concat(dfs, ignore_index=True)


def assign_split_values(
    df: pd.DataFrame, values: list, column_name: str, random_state: int = 42
) -> pd.DataFrame:
    """
    Randomly split a DataFrame into n approximately equal parts and assign
    a split-specific value to each row.

    Args:
        df : pd.DataFrame
            Input DataFrame.
        values : list
            List of values to assign, one per split. Length determines n.
        column_name : str
            Name of the new column.
        random_state: int
            Random seed value can be set for reproducibility.

    Returns:
        Copy of the input DataFrame with the new column added.
    """
    n = len(values)

    # Shuffle rows
    shuffled_idx = df.sample(frac=1, random_state=random_state).index

    # Split indices into n roughly equal groups
    splits = np.array_split(shuffled_idx, n)

    result = df.copy()

    # Assign values
    for split_idx, value in zip(splits, values):
        result.loc[split_idx, column_name] = value

    return result


def preprocess_raw_data(
    input_csv: str,
    output_csv: str,
    start_date: str,
    end_date: str,
    freq: str = "MS",
    date_column: str = "reference_date",
    data_dir: Path | None = None,
) -> Path:
    """Pre-process raw data and store the result as a CSV file.

    Loads a raw CSV dataset, validates its contents, and augments it with a
    date column containing values generated from the specified date range.
    The resulting dataset is written to ``output_csv`` and the path to the
    created file is returned.

    Args:
        input_csv: Name of the input CSV file containing the raw data.
        output_csv: Name of the output CSV file to create.
        start_date: Start date for generating date values. Passed directly to
            :func:`pandas.date_range`.
        end_date: End date for generating date values. Passed directly to
            :func:`pandas.date_range`.
        freq: Frequency string used to generate dates via
            :func:`pandas.date_range`. Defaults to ``"MS"`` (month start).
        date_column: Name of the column to add containing the generated date
            values. Defaults to ``"reference_date"``.
        data_dir: Directory containing the input file and where the output
            file will be written. If ``None``, the raw data directory from
            :class:`RuntimeSettings` is used.

    Returns:
        Path: Path to the generated output CSV file.

    Raises:
        pandera.errors.SchemaError: If the input data fails validation.
    """

    settings = RuntimeSettings()
    resolved_data_dir = data_dir or settings.raw_data_dir

    df = load_raw_data(
        file_name=input_csv,
        index_col=None,
        data_dir=resolved_data_dir,
    )
    df = validate_data(df)

    dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    df_dates = assign_split_values(df, values=dates, column_name=date_column)

    output_path = resolved_data_dir / output_csv
    resolved_data_dir.mkdir(parents=True, exist_ok=True)

    df_dates.to_csv(output_path, index=False)

    return output_path


def main() -> None:
    """Pre-process raw data files and store result as CSV.

    The in-/output paths and start/end dates are supplied as command-line arguments.
    The original data is enhanced by a ``reference_date`` column.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)

    args = parser.parse_args()

    preprocess_raw_data(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        start_date=args.start_date,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    configure_logging()
    main()
