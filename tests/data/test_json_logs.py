import json
from pathlib import Path

import pandas as pd
import pytest

from churn_mlops.data.jsonl_logs import jsonl_log_to_csv


@pytest.mark.unit
def test_jsonl_log_to_csv(tmp_path: Path) -> None:
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

    jsonl_log_to_csv(jsonl_path, csv_path)

    result = pd.read_csv(csv_path)

    assert len(result) == 2
    assert "features" not in result.columns
    assert result.loc[0, "age"] == 42
    assert result.loc[0, "gender"] == "Female"
    assert result.loc[0, "predicted_class"] == 0
