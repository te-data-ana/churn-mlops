import pandas as pd

from churn_mlops.config.settings import RAW_DATA_DIR


def load_raw_data(file_name: str, index_col=None) -> pd.DataFrame:
    """
    Using pandas, load raw data from standard csv-file stored in raw data directory.
    Column names are converted to snake_case.
    An index column is set based on user input.
    Records with missing values only are dropped.
    """
    df = pd.read_csv(
        # retrieve raw data from specified file
        RAW_DATA_DIR / file_name,
    )
    # convert column names to snake_case
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    # convert columns to nullable data types
    df = df.convert_dtypes()
    # set index column
    if index_col:
        df.set_index(index_col, inplace=True)
    # drop rows containing only missing values
    df.dropna(how="all", inplace=True)
    return df
