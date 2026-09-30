import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from churn_mlops.config import RuntimeSettings

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
