import pandas as pd

from churn_mlops.config.settings import RAW_DATA_DIR


def load_raw_data(file_name: str, index_col: str | None = None) -> pd.DataFrame:
    """Load and normalize a raw CSV file from the configured data directory.

    Args:
        file_name: Name of the CSV file in ``RAW_DATA_DIR``.
        index_col: Optional column name to use as the DataFrame index.

    Returns:
        DataFrame with normalized column names, nullable dtypes, the optional
        index applied, and rows containing only missing values removed.

    Raises:
        FileNotFoundError: If the requested CSV file does not exist.
    """
    df = pd.read_csv(RAW_DATA_DIR / file_name)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    df = df.convert_dtypes()
    if index_col:
        df.set_index(index_col, inplace=True)
    df.dropna(how="all", inplace=True)
    return df
