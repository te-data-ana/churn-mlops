import logging
from pathlib import Path

import pandas as pd

from churn_mlops.config.settings import RAW_DATA_DIR

logger = logging.getLogger(__name__)


def load_raw_data(
    file_name: str,
    index_col: str | None = None,
    data_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Load and normalize a raw CSV file from the configured data directory.

    Args:
        file_name: Name of the CSV file in ``data_dir``.
        index_col: Optional column name to use as the DataFrame index.
        data_dir: Directory containing the CSV file. Defaults to ``RAW_DATA_DIR``.

    Returns:
        DataFrame with normalized column names, nullable dtypes, the optional
        index applied, and rows containing only missing values removed.

    Raises:
        FileNotFoundError: If the requested CSV file does not exist.
    """
    try:
        logger.info("Loading raw dataset '%s' from '%s'.", file_name, data_dir)
        df = pd.read_csv(data_dir / file_name)
        df.columns = (
            df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
        )
        df = df.convert_dtypes()
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
