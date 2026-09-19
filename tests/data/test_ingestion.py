import pandas as pd
import pytest

from churn_mlops.data import load_raw_data


@pytest.mark.unit
def test_load_raw_data_returns_clean_indexed_dataframe() -> None:
    df = load_raw_data("customer_churn_dataset-training.csv", index_col="customerid")

    # Verify ingestion returns indexed data, without empty rows and snake-case column names.
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df.index.name == "customerid"
    assert not df.isna().all(axis=1).any()
    assert ~df.columns.str.contains(" ", regex=False).any()
    assert all(col == col.lower() for col in df.columns)
