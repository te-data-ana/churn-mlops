import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from churn_mlops import batch_predict


@pytest.mark.unit
def test_batch_predict_main_orchestrates_prediction_and_writes_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Prepare separate frames for raw input, validated input, and model output.
    input_df = pd.DataFrame({"age": [45], "tenure": [24]})
    validated_df = input_df.copy()
    prediction_df = pd.DataFrame(
        {
            "predicted_probability": [0.8],
            "predicted_class": [1],
        },
        index=input_df.index,
    )

    # Supply CLI arguments and redirect in-/output from/to a temporary directory.
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

    # Mock settings
    settings = SimpleNamespace(
        raw_data_dir=tmp_path,
        tmp_dir=tmp_path,
        mlflow_tracking_uri=None,
        model_name=None,
        model_alias=None,
    )

    monkeypatch.setattr(batch_predict, "ServingSettings", lambda: settings)

    # Mock the model and predictor so this test covers orchestration only.
    loaded_model = Mock(tracking_uri=None, model_name=None, model_alias=None)
    predictor = Mock()
    predictor.predict_batch.return_value = prediction_df

    # Replace filesystem, validation, model-loading, and prediction boundaries.
    load_raw_data = Mock(return_value=input_df)
    validate_data = Mock(return_value=validated_df)
    predictor_factory = Mock(return_value=predictor)
    load_model = Mock(return_value=loaded_model)
    monkeypatch.setattr(batch_predict, "load_raw_data", load_raw_data)
    monkeypatch.setattr(batch_predict, "validate_data", validate_data)
    monkeypatch.setattr(batch_predict, "Predictor", predictor_factory)
    monkeypatch.setattr(batch_predict, "load_model", load_model)

    # Run the real entry-point orchestration with the dependencies controlled.
    batch_predict.main()

    # Confirm arguments flow through each stage in the expected order and shape.
    load_raw_data.assert_called_once_with(
        file_name="customers.csv",
        index_col="customerid",
        data_dir=tmp_path,
    )
    validate_data.assert_called_once_with(input_df)
    load_model.assert_called_once_with(
        tracking_uri=None,
        model_name=None,
        model_alias=None,
    )
    predictor_factory.assert_called_once_with(loaded_model)
    predictor.predict_batch.assert_called_once_with(df=validated_df)

    # Confirm the mocked predictions are persisted to the requested output file.
    output_path = tmp_path / "predictions.csv"
    assert output_path.exists()
    pd.testing.assert_frame_equal(
        pd.read_csv(output_path, index_col=0),
        prediction_df,
    )


@pytest.mark.unit
def test_batch_prediction_uses_default_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = SimpleNamespace(
        raw_data_dir=tmp_path / "raw",
        tmp_dir=tmp_path / "tmp",
        mlflow_tracking_uri="uri",
        model_name="my_model",
        model_alias="prod",
    )

    monkeypatch.setattr(batch_predict, "ServingSettings", lambda: settings)

    load_raw_data = Mock(return_value=pd.DataFrame({"x": [1]}))
    validate = Mock(return_value=pd.DataFrame({"x": [1]}))
    load_model = Mock()
    predictor = Mock()
    predictor.predict_batch.return_value = pd.DataFrame(
        {"predicted_probability": [0.5]}
    )

    monkeypatch.setattr(batch_predict, "load_raw_data", load_raw_data)
    monkeypatch.setattr(batch_predict, "validate_data", validate)
    monkeypatch.setattr(batch_predict, "load_model", load_model)
    monkeypatch.setattr(batch_predict, "Predictor", Mock(return_value=predictor))

    batch_predict.run_batch_prediction(
        input_csv="input.csv",
        output_csv="output.csv",
    )

    load_raw_data.assert_called_once_with(
        file_name="input.csv",
        index_col=None,
        data_dir=settings.raw_data_dir,
    )

    load_model.assert_called_once_with(
        tracking_uri="uri",
        model_name="my_model",
        model_alias="prod",
    )


@pytest.mark.unit
def test_batch_prediction_logs_and_reraises_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        batch_predict,
        "ServingSettings",
        lambda: SimpleNamespace(
            raw_data_dir=tmp_path,
            tmp_dir=tmp_path,
            mlflow_tracking_uri=None,
            model_name=None,
            model_alias=None,
        ),
    )

    monkeypatch.setattr(
        batch_predict,
        "load_raw_data",
        Mock(side_effect=RuntimeError("boom")),
    )

    log_exception = Mock()
    monkeypatch.setattr(batch_predict.logger, "exception", log_exception)

    with pytest.raises(RuntimeError, match="boom"):
        batch_predict.run_batch_prediction(
            input_csv="input.csv",
            output_csv="output.csv",
        )

    log_exception.assert_called_once()


@pytest.mark.integration
def test_batch_prediction_uses_registered_model(
    generated_inference_csv: Path,
    registered_model: dict[str, str | Path],
    tmp_path: Path,
) -> None:
    output_path = batch_predict.run_batch_prediction(
        input_csv=generated_inference_csv.name,
        output_csv="predictions.csv",
        index_col="customerid",
        input_dir=generated_inference_csv.parent,
        output_dir=tmp_path / "output",
        tracking_uri=str(registered_model["tracking_uri"]),
        model_name=str(registered_model["model_name"]),
        model_alias=str(registered_model["model_alias"]),
    )

    predictions = pd.read_csv(output_path, index_col=0)

    assert output_path == tmp_path / "output" / "predictions.csv"
    assert len(predictions) == len(pd.read_csv(generated_inference_csv))
    assert predictions["predicted_probability"].between(0, 1).all()
    assert predictions["predicted_class"].isin([0, 1]).all()
    assert (predictions["threshold"] == 0.5).all()
    assert predictions["model_version"].notna().all()
