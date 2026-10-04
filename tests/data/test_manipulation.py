from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from churn_mlops.data.manipulation import (
    assign_split_values,
    main,
    preprocess_raw_data,
)


@pytest.mark.unit
def test_main_passes_cli_config_to_preprocess_raw_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    mock_preprocessing_job = MagicMock()

    monkeypatch.setattr(
        "churn_mlops.data.manipulation.preprocess_raw_data",
        mock_preprocessing_job,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "churn_mlops.data.manipulation.py",
            "--input_csv",
            "in.csv",
            "--output_csv",
            "out.csv",
            "--start_date",
            "2026-01-01",
            "--end_date",
            "2026-06-01",
        ],
    )

    main()

    mock_preprocessing_job.assert_called_once_with(
        input_csv="in.csv",
        output_csv="out.csv",
        start_date="2026-01-01",
        end_date="2026-06-01",
    )


@pytest.mark.unit
def test_assign_split_values_creates_balanced_splits() -> None:
    df = pd.DataFrame({"x": range(100)})

    result = assign_split_values(
        df=df,
        values=["A", "B", "C"],
        column_name="group",
    )

    counts = result["group"].value_counts()

    assert len(result) == len(df)
    assert set(result["group"]) == {"A", "B", "C"}
    assert counts.max() - counts.min() <= 1


@pytest.mark.unit
def test_assign_split_values_is_reproducible() -> None:
    df = pd.DataFrame({"x": range(42)})
    vals = ["A", "B", "C"]
    col_name = "group"
    seed = 42

    result1 = assign_split_values(df, vals, col_name, seed)

    result2 = assign_split_values(df, vals, col_name, seed)

    pd.testing.assert_series_equal(
        result1["group"],
        result2["group"],
    )


@pytest.mark.unit
def test_preprocess_raw_data_creates_output_file(
    generated_training_csv: Path,
    tmp_path: Path,
) -> None:
    output_path = preprocess_raw_data(
        input_csv=generated_training_csv.name,
        output_csv="processed.csv",
        start_date="2024-01-01",
        end_date="2024-12-01",
        data_dir=tmp_path,
    )

    assert output_path.exists()

    result = pd.read_csv(output_path)

    assert "reference_date" in result.columns
