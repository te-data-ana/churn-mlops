import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from churn_mlops import batch_predict


@pytest.mark.unit
def test_batch_predict_main_reads_csv_and_writes_predictions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_df = pd.DataFrame({"age": [45], "tenure": [24]})
    validated_df = input_df.copy()
    prediction_df = pd.DataFrame(
        {
            "predicted_probability": [0.8],
            "predicted_class": [1],
        },
        index=input_df.index,
    )
    settings = SimpleNamespace(
        data_dir=tmp_path,
        output_dir=tmp_path,
        mlflow_tracking_uri=None,
        model_name=None,
        model_alias=None,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_predict",
            "--input_csv",
            "customers.csv",
            "--output_csv",
            "predictions.csv",
            "--index_col",
            "customerid",
        ],
    )
    monkeypatch.setattr(batch_predict, "ServingSettings", lambda: settings)

    load_raw_data = Mock(return_value=input_df)
    validate_data = Mock(return_value=validated_df)
    predictor = Mock()
    predictor.predict_batch.return_value = prediction_df
    loaded_model = Mock()
    load_model = Mock(return_value=loaded_model)
    predictor_factory = Mock(return_value=predictor)
    monkeypatch.setattr(batch_predict, "load_raw_data", load_raw_data)
    monkeypatch.setattr(batch_predict, "validate_data", validate_data)
    monkeypatch.setattr(batch_predict, "load_model", load_model)
    monkeypatch.setattr(batch_predict, "Predictor", predictor_factory)

    batch_predict.main()

    load_raw_data.assert_called_once_with(
        file_name="customers.csv",
        index_col="customerid",
        data_dir=tmp_path / "raw",
    )
    validate_data.assert_called_once_with(input_df)
    load_model.assert_called_once_with(
        tracking_uri=None,
        model_name=None,
        model_alias=None,
    )
    predictor_factory.assert_called_once_with(loaded_model)
    predictor.predict_batch.assert_called_once_with(df=validated_df)

    output_path = tmp_path / "predictions.csv"
    assert output_path.exists()
    pd.testing.assert_frame_equal(
        pd.read_csv(output_path, index_col=0),
        prediction_df,
    )


@pytest.mark.unit
def test_run_batch_prediction_validates_and_returns_prediction_dataframe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = SimpleNamespace(
        mlflow_tracking_uri="uri",
        model_name="my_model",
        model_alias="prod",
    )
    input_df = pd.DataFrame({"age": [45]})
    validated_df = pd.DataFrame({"age": [45]})
    prediction_df = pd.DataFrame({"predicted_probability": [0.5]})

    monkeypatch.setattr(batch_predict, "ServingSettings", lambda: settings)
    validate_data = Mock(return_value=validated_df)
    load_model = Mock(return_value=Mock())
    predictor = Mock()
    predictor.predict_batch.return_value = prediction_df
    monkeypatch.setattr(batch_predict, "validate_data", validate_data)
    monkeypatch.setattr(batch_predict, "load_model", load_model)
    monkeypatch.setattr(batch_predict, "Predictor", Mock(return_value=predictor))

    result = batch_predict.run_batch_prediction(df=input_df)

    validate_data.assert_called_once_with(input_df)
    load_model.assert_called_once_with(
        tracking_uri="uri",
        model_name="my_model",
        model_alias="prod",
    )
    predictor.predict_batch.assert_called_once_with(df=validated_df)
    pd.testing.assert_frame_equal(result, prediction_df)


@pytest.mark.unit
def test_batch_prediction_logs_and_reraises_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        batch_predict,
        "ServingSettings",
        lambda: SimpleNamespace(
            mlflow_tracking_uri=None,
            model_name=None,
            model_alias=None,
        ),
    )
    monkeypatch.setattr(
        batch_predict,
        "validate_data",
        Mock(side_effect=RuntimeError("boom")),
    )
    log_exception = Mock()
    monkeypatch.setattr(batch_predict.logger, "exception", log_exception)

    with pytest.raises(RuntimeError, match="boom"):
        batch_predict.run_batch_prediction(df=pd.DataFrame({"age": [45]}))

    log_exception.assert_called_once()


@pytest.mark.unit
def test_batch_predict_main_reads_and_writes_partitioned_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_df = pd.DataFrame({"reference_date": pd.to_datetime(["2026-02-15"])})
    prediction_df = input_df.assign(predicted_probability=[0.7])
    read_dataset = Mock(return_value=input_df)
    predict = Mock(return_value=prediction_df)
    write_dataset = Mock()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_predict",
            "--input_dataset",
            "partitioned",
            "--start_date",
            "2026-02-01",
            "--end_date",
            "2026-02-28",
        ],
    )
    monkeypatch.setattr(batch_predict, "read_partitioned_dataset", read_dataset)
    monkeypatch.setattr(batch_predict, "run_batch_prediction", predict)
    monkeypatch.setattr(batch_predict, "write_partitioned_dataset", write_dataset)

    batch_predict.main()

    read_dataset.assert_called_once_with(
        dataset_name="partitioned",
        timestamp_column="reference_date",
        start="2026-02-01",
        end="2026-02-28",
    )
    predict.assert_called_once_with(df=input_df)
    write_dataset.assert_called_once_with(
        df=prediction_df,
        dataset_name="batch_predictions",
        timestamp_column="reference_date",
        overwrite_partitions=True,
    )


@pytest.mark.integration
def test_batch_prediction_uses_registered_model(
    generated_inference_csv: Path,
    registered_model: dict[str, str | Path],
) -> None:
    input_df = batch_predict.load_raw_data(
        file_name=generated_inference_csv.name,
        index_col="customerid",
        data_dir=generated_inference_csv.parent,
    )

    predictions = batch_predict.run_batch_prediction(
        df=input_df,
        tracking_uri=str(registered_model["tracking_uri"]),
        model_name=str(registered_model["model_name"]),
        model_alias=str(registered_model["model_alias"]),
    )

    assert len(predictions) == len(pd.read_csv(generated_inference_csv))
    assert predictions["predicted_probability"].between(0, 1).all()
    assert predictions["predicted_class"].isin([0, 1]).all()
    assert (predictions["threshold"] == 0.5).all()
    assert predictions["model_version"].notna().all()
