from pathlib import Path

import pandas as pd
import pytest

from churn_mlops.data.ingestion import combine_sample_predictions, load_raw_data


@pytest.mark.unit
def test_load_raw_data_returns_clean_indexed_dataframe(generated_training_csv) -> None:
    df = load_raw_data(generated_training_csv, index_col="customerid")

    # Verify ingestion returns indexed data, without empty rows and snake-case column names.
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df.index.name == "customerid"
    assert not df.isna().all(axis=1).any()
    assert ~df.columns.str.contains(" ", regex=False).any()
    assert all(col == col.lower() for col in df.columns)


@pytest.mark.unit
def test_combine_sample_predictions_adds_run_id(tmp_path: Path) -> None:
    df1 = pd.DataFrame({"prediction": [0, 1]})
    df2 = pd.DataFrame({"prediction": [1]})

    df1.to_csv(
        tmp_path / "0001_sample_predictions.csv",
        index=False,
    )
    df2.to_csv(
        tmp_path / "0042_sample_predictions.csv",
        index=False,
    )

    result = combine_sample_predictions(tmp_path)

    assert len(result) == 3
    assert "run_id" in result.columns
    assert set(result["run_id"]) == {"0001", "0042"}
