from pathlib import Path

import pandas as pd

from churn_mlops.config import RuntimeSettings


def read_jsonl_prediction_log(jsonl_path: Path) -> pd.DataFrame:
    """Convert a prediction log from JSONL to a pandas DataFrame.

    Flattens the nested ``features`` object into separate columns when
    present. Returns an empty DataFrame if the file does not exist,
    is empty, contains invalid JSONL, or does not contain any valid
    records.

    Args:
        jsonl_path: Path to the source JSONL prediction log.

    Returns:
        Flattened prediction log as a pandas DataFrame.
    """
    try:
        df = pd.read_json(jsonl_path, lines=True)
    except (FileNotFoundError, ValueError):
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if "features" not in df.columns:
        return df

    features = pd.json_normalize(df["features"])

    return pd.concat(
        [df.drop(columns=["features"]), features],
        axis=1,
    )


def jsonl_prediction_log_to_csv(jsonl_path: Path, csv_path: Path) -> None:
    """Convert a prediction log from JSONL to CSV format.

    Flattens the nested ``features`` object into separate columns and
    writes the transformed records to a CSV file.

    Args:
        jsonl_path: Path to the source JSONL prediction log.
        csv_path: Path where the CSV file will be written.
    """
    df = read_jsonl_prediction_log(jsonl_path)
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
