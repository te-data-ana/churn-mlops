import json
from pathlib import Path

import pandas as pd
import pytest

from churn_mlops.data.prediction_logs import (
    combine_sample_predictions,
    jsonl_prediction_log_to_csv,
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
