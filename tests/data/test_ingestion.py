import pandas as pd

from churn_mlops.data import load_raw_data


def test_load_raw_data():
    df = load_raw_data("customer_churn_dataset-training.csv", index_col="customerid")

    # pandas dataframe containing data
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # index column as specified
    assert df.index.name == "customerid"
    # no completely empty rows
    assert not df.isna().all(axis=1).any()
    # snake_case column names
    assert ~df.columns.str.contains(" ", regex=False).any()
    assert all(col == col.lower() for col in df.columns)
