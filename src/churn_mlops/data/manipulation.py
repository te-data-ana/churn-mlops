import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from churn_mlops.config import RuntimeSettings, configure_logging
from churn_mlops.data.ingestion import load_raw_data
from churn_mlops.data.validation import validate_data


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
    date_column: str = "reference_data",
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
            values. Defaults to ``"reference_data"``.
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
