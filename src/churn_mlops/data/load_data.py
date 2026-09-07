import pandas as pd

from churn_mlops.config import DTYPES, RAW_DATA_DIR


def load_raw_data(file_name):
    """Using pandas, load raw data from standard csv-file stored in raw data directory."""
    df = pd.read_csv(
        # retrieve raw data from specified file
        RAW_DATA_DIR / file_name,
        # cast to pre-defined data types
        dtype=DTYPES,
        # set index to ID column(s)
        index_col=[col for col in DTYPES if col.find("ID") > 0],
    )
    # drop rows containing only missing values
    df = df.dropna(how="all")
    return df
