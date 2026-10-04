from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from churn_mlops.data import ingestion
from churn_mlops.data.ingestion import (
    combine_sample_predictions,
    load_raw_data,
    normalize_strings,
)


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


@pytest.mark.unit
def test_normalize_strings_strips_and_normalizes_column_names() -> None:
    assert normalize_strings([" Customer ID ", "Subscription Type"]) == [
        "customer_id",
        "subscription_type",
    ]


@pytest.mark.unit
def test_ingest_jsonl_log_writes_nonempty_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    df = pd.DataFrame({"reference_date": ["2026-01-01"], "predicted_class": [1]})
    observed: dict[str, object] = {}

    def fake_read_jsonl_log(path: Path) -> pd.DataFrame:
        observed["source"] = path
        return df

    def fake_write_partitioned_dataset(**kwargs: object) -> None:
        observed["write"] = kwargs

    monkeypatch.setattr(ingestion, "read_jsonl_log", fake_read_jsonl_log)
    monkeypatch.setattr(
        ingestion,
        "write_partitioned_dataset",
        fake_write_partitioned_dataset,
    )
    log_path = tmp_path / "predictions.jsonl"

    ingestion.ingest_jsonl_log(
        dataset_name="predictions",
        timestamp_column="reference_date",
        jsonl_path=log_path,
    )

    assert observed["source"] == log_path
    write_args = observed["write"]
    assert isinstance(write_args, dict)
    assert write_args["df"] is df
    assert write_args["dataset_name"] == "predictions"
    assert write_args["timestamp_column"] == "reference_date"
    assert write_args["overwrite_partitions"] is True


@pytest.mark.unit
def test_ingest_jsonl_log_skips_empty_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    writes: list[dict[str, object]] = []
    monkeypatch.setattr(
        ingestion,
        "read_jsonl_log",
        lambda _: pd.DataFrame(),
    )
    monkeypatch.setattr(
        ingestion,
        "write_partitioned_dataset",
        lambda **kwargs: writes.append(kwargs),
    )

    ingestion.ingest_jsonl_log(
        dataset_name="predictions",
        timestamp_column="reference_date",
        jsonl_path=tmp_path / "empty.jsonl",
    )

    assert writes == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("function_name", "log_path_attribute", "dataset_name"),
    [
        ("ingest_prediction_logs", "prediction_log_path", "predictions"),
        ("ingest_error_logs", "error_log_path", "prediction_errors"),
    ],
)
def test_log_ingestion_wrappers_use_configured_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    function_name: str,
    log_path_attribute: str,
    dataset_name: str,
) -> None:
    prediction_path = tmp_path / "predictions.jsonl"
    error_path = tmp_path / "prediction_errors.jsonl"
    settings = SimpleNamespace(
        prediction_log_path=prediction_path,
        error_log_path=error_path,
    )
    observed: dict[str, object] = {}

    def fake_ingest_jsonl_log(
        dataset_name: str,
        timestamp_column: str,
        jsonl_path: Path,
    ) -> None:
        observed.update(
            dataset_name=dataset_name,
            timestamp_column=timestamp_column,
            jsonl_path=jsonl_path,
        )

    monkeypatch.setattr(ingestion, "ServingSettings", lambda: settings)
    monkeypatch.setattr(ingestion, "ingest_jsonl_log", fake_ingest_jsonl_log)

    ingest_function = getattr(ingestion, function_name)
    ingest_function(dataset_name, "reference_date")

    assert observed == {
        "dataset_name": dataset_name,
        "timestamp_column": "reference_date",
        "jsonl_path": getattr(settings, log_path_attribute),
    }


@pytest.mark.unit
def test_ingest_sample_data_validates_and_writes_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_data = pd.DataFrame({"reference_date": ["2026-01-01"]})
    validated_data = pd.DataFrame({"reference_date": [pd.Timestamp("2026-01-01")]})
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        ingestion,
        "combine_sample_predictions",
        lambda: sample_data,
    )

    def fake_validate_data(df: pd.DataFrame) -> pd.DataFrame:
        observed["validated_input"] = df
        return validated_data

    def fake_write_partitioned_dataset(**kwargs: object) -> None:
        observed["write"] = kwargs

    monkeypatch.setattr(ingestion, "validate_data", fake_validate_data)
    monkeypatch.setattr(
        ingestion,
        "write_partitioned_dataset",
        fake_write_partitioned_dataset,
    )

    ingestion.ingest_sample_data("samples", "reference_date")

    assert observed["validated_input"] is sample_data
    write_args = observed["write"]
    assert isinstance(write_args, dict)
    assert write_args["df"] is validated_data
    assert write_args["dataset_name"] == "samples"
    assert write_args["timestamp_column"] == "reference_date"
    assert write_args["overwrite_partitions"] is True
