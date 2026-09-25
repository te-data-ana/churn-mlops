import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from churn_mlops.data.manipulation import (
    assign_split_values,
    combine_sample_predictions,
    jsonl_prediction_log_to_csv,
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
def test_jsonl_prediction_log_to_csv(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "predictions.jsonl"
    csv_path = tmp_path / "predictions.csv"

    records = [
        {
            "request_id": "123",
            "predicted_class": 0,
            "features": {
                "age": 42,
                "gender": "Female",
            },
        },
        {
            "request_id": "456",
            "predicted_class": 1,
            "features": {
                "age": 52,
                "gender": "Male",
            },
        },
    ]

    jsonl_path.write_text("\n".join(json.dumps(record) for record in records))

    jsonl_prediction_log_to_csv(jsonl_path, csv_path)

    result = pd.read_csv(csv_path)

    assert len(result) == 2
    assert "features" not in result.columns
    assert result.loc[0, "age"] == 42
    assert result.loc[0, "gender"] == "Female"
    assert result.loc[0, "predicted_class"] == 0


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
