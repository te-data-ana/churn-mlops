from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import churn_mlops
from churn_mlops import batch_predict, data
from churn_mlops.data import storage
from churn_mlops.tracking.manifest import TrainingManifest


@pytest.mark.unit
def test_batch_predict_cli_reads_csv_and_writes_predictions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_df = pd.DataFrame({"age": [45], "tenure": [24]})
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
        api_host="127.0.0.1",
        api_port=8000,
    )
    load_raw_data = Mock(return_value=input_df)
    predict = Mock(return_value=prediction_df)
    monkeypatch.setattr(churn_mlops, "ServingSettings", lambda: settings)
    monkeypatch.setattr(data, "load_raw_data", load_raw_data)
    monkeypatch.setattr(batch_predict, "run_batch_prediction", predict)

    churn_mlops.main(
        [
            "batch-predict",
            "--input_csv",
            "customers.csv",
            "--output_csv",
            "predictions.csv",
            "--index_col",
            "customerid",
        ]
    )

    load_raw_data.assert_called_once_with(
        file_name="customers.csv",
        index_col="customerid",
        data_dir=tmp_path / "raw",
    )
    predict.assert_called_once_with(df=input_df)

    output_path = tmp_path / "predictions.csv"
    assert output_path.exists()
    pd.testing.assert_frame_equal(
        pd.read_csv(output_path, index_col=0),
        prediction_df,
    )


@pytest.mark.unit
def test_run_batch_prediction_validates_and_returns_prediction_dataframe(
    monkeypatch: pytest.MonkeyPatch,
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
def test_run_batch_prediction_uses_exact_model_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    load_model = Mock(return_value=Mock())
    predictor = Mock()
    predictor.predict_batch.return_value = pd.DataFrame(
        {"predicted_probability": [0.7]}
    )

    monkeypatch.setattr(
        batch_predict,
        "ServingSettings",
        lambda: SimpleNamespace(
            mlflow_tracking_uri="uri",
            model_name="configured-model",
            model_alias="champion",
        ),
    )
    monkeypatch.setattr(batch_predict, "validate_data", lambda df: df)
    monkeypatch.setattr(batch_predict, "load_model", load_model)
    monkeypatch.setattr(batch_predict, "Predictor", Mock(return_value=predictor))

    batch_predict.run_batch_prediction(
        df=pd.DataFrame({"age": [30]}),
        model_name="churn-propensity",
        model_version=8,
    )

    load_model.assert_called_once_with(
        tracking_uri="uri",
        model_name="churn-propensity",
        model_alias="champion",
        model_version=8,
    )


@pytest.mark.unit
def test_batch_predict_cli_reads_model_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    training_manifest_factory: TrainingManifest,
) -> None:
    from churn_mlops import tracking

    settings = SimpleNamespace(
        data_dir=tmp_path,
        output_dir=tmp_path,
        api_host="127.0.0.1",
        api_port=8000,
    )
    input_df = pd.DataFrame({"age": [35]})
    prediction_df = pd.DataFrame({"predicted_probability": [0.7]})
    manifest = training_manifest_factory()
    predict = Mock(return_value=prediction_df)

    monkeypatch.setattr(churn_mlops, "ServingSettings", lambda: settings)
    monkeypatch.setattr(data, "load_raw_data", Mock(return_value=input_df))
    monkeypatch.setattr(tracking.TrainingManifest, "read", Mock(return_value=manifest))
    monkeypatch.setattr(batch_predict, "run_batch_prediction", predict)

    churn_mlops.main(
        [
            "batch-predict",
            "--input_csv",
            "inference.csv",
            "--output_csv",
            "scored.csv",
            "--model_manifest",
            str(tmp_path / "manifest.json"),
        ]
    )

    predict.assert_called_once_with(
        df=input_df,
        model_name="churn-propensity",
        model_version=4,
    )


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
def test_batch_predict_cli_reads_and_writes_partitioned_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_df = pd.DataFrame({"reference_date": pd.to_datetime(["2026-02-15"])})
    prediction_df = input_df.assign(predicted_probability=[0.7])
    read_dataset = Mock(return_value=input_df)
    predict = Mock(return_value=prediction_df)
    write_dataset = Mock()

    monkeypatch.setattr(
        churn_mlops,
        "ServingSettings",
        lambda: SimpleNamespace(
            data_dir=Path("data"),
            output_dir=Path("output"),
            api_host="127.0.0.1",
            api_port=8000,
        ),
    )
    monkeypatch.setattr(storage, "read_partitioned_dataset", read_dataset)
    monkeypatch.setattr(batch_predict, "run_batch_prediction", predict)
    monkeypatch.setattr(storage, "write_partitioned_dataset", write_dataset)

    churn_mlops.main(
        [
            "batch-predict",
            "--input_dataset",
            "partitioned",
            "--start_date",
            "2026-02-01",
            "--end_date",
            "2026-02-28",
        ]
    )

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


@pytest.mark.unit
@pytest.mark.parametrize(
    ("arguments", "error_message"),
    [
        (
            ["--input_csv", "input.csv"],
            "CSV input requires --output_csv",
        ),
        (
            [
                "--input_dataset",
                "partitioned",
                "--output_csv",
                "output.csv",
            ],
            "Partitioned input cannot use --output_csv",
        ),
        (
            ["--input_dataset", "partitioned", "--start_date", "2026-01-01"],
            "--start_date and --end_date must be provided together",
        ),
        (
            [
                "--input_csv",
                "input.csv",
                "--output_csv",
                "output.csv",
                "--start_date",
                "2026-01-01",
                "--end_date",
                "2026-01-31",
            ],
            "Date bounds can only be used with partitioned input",
        ),
        (
            ["--input_dataset", "partitioned", "--index_col", "customerid"],
            "--index_col can only be used with CSV input",
        ),
    ],
)
def test_batch_predict_cli_rejects_invalid_arguments(
    arguments: list[str],
    error_message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        churn_mlops.main(["batch-predict", *arguments])

    assert exc_info.value.code == 2
    assert error_message in capsys.readouterr().err


@pytest.mark.integration
def test_batch_prediction_uses_registered_model(
    generated_inference_csv: Path,
    registered_model: dict[str, str | Path],
) -> None:
    input_df = data.load_raw_data(
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
