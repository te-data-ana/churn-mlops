from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from churn_mlops.config import RuntimeSettings
from churn_mlops.serving import serve_samples
from churn_mlops.serving.serve_samples import main


class MockResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def json(self) -> dict:
        return self.payload

    def raise_for_status(self) -> None:
        pass


@pytest.mark.unit
def test_load_sample_data_is_reproducible(
    monkeypatch: pytest.MonkeyPatch,
    generated_training_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(
        serve_samples,
        "load_raw_data",
        lambda **kwargs: generated_training_df.drop(columns=["Churn"]),
    )

    monkeypatch.setattr(
        serve_samples,
        "validate_data",
        lambda df: df,
    )

    settings = RuntimeSettings()

    result_1 = serve_samples.load_sample_data(
        sample_size=5,
        random_state=42,
        settings=settings,
    )

    result_2 = serve_samples.load_sample_data(
        sample_size=5,
        random_state=42,
        settings=settings,
    )

    assert len(result_1) == 5
    pd.testing.assert_frame_equal(result_1, result_2)


@pytest.mark.unit
def test_predict_samples_combines_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame(
        {
            "age": [42, 52],
            "gender": ["Female", "Male"],
        }
    )

    monkeypatch.setattr(
        serve_samples.requests,
        "post",
        lambda *args, **kwargs: MockResponse(
            {
                "predicted_class": 1,
                "predicted_probability": 0.9,
            }
        ),
    )

    settings = RuntimeSettings()

    result = serve_samples.predict_samples(
        df=df,
        settings=settings,
    )

    assert "predicted_class" in result.columns
    assert "predicted_probability" in result.columns

    assert len(result) == 2


@pytest.mark.unit
def test_predict_samples_raises_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ErrorResponse:
        def raise_for_status(self) -> None:
            raise requests.HTTPError()

    monkeypatch.setattr(
        serve_samples.requests,
        "post",
        lambda *args, **kwargs: ErrorResponse(),
    )

    df = pd.DataFrame({"age": [42]})

    with pytest.raises(requests.HTTPError):
        serve_samples.predict_samples(
            df=df,
            settings=RuntimeSettings(),
        )


@pytest.mark.unit
def test_save_predictions_writes_csv(
    tmp_path: Path,
) -> None:
    settings = RuntimeSettings()
    settings.output_dir = tmp_path

    df = pd.DataFrame({"prediction": [0, 1]})

    output_path = serve_samples.save_predictions(
        df=df,
        run_id="0042",
        settings=settings,
    )

    assert output_path.exists()
    assert output_path.name == "0042_sample_predictions.csv"

    result = pd.read_csv(output_path)

    assert len(result) == 2


@pytest.mark.unit
def test_serve_samples_calls_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = RuntimeSettings()
    settings.output_dir = tmp_path

    sample_df = pd.DataFrame({"age": [42]})
    prediction_df = pd.DataFrame(
        {
            "age": [42],
            "predicted_class": [1],
        }
    )

    expected = tmp_path / "0042_sample_predictions.csv"

    load_sample_data_mock = Mock(return_value=sample_df)
    predict_samples_mock = Mock(return_value=prediction_df)
    save_predictions_mock = Mock(return_value=expected)

    monkeypatch.setattr(serve_samples, "load_sample_data", load_sample_data_mock)
    monkeypatch.setattr(serve_samples, "predict_samples", predict_samples_mock)
    monkeypatch.setattr(serve_samples, "save_predictions", save_predictions_mock)

    result = serve_samples.serve_samples(
        sample_size=1,
        random_state=42,
        settings=settings,
        drop_columns=None,
    )

    assert result == expected
    load_sample_data_mock.assert_called_once_with(
        sample_size=1,
        random_state=42,
        settings=settings,
        drop_columns=None,
    )
    predict_samples_mock.assert_called_once_with(
        df=sample_df,
        settings=settings,
        reference_date=None,
    )
    save_predictions_mock.assert_called_once_with(
        df=prediction_df,
        run_id="0042",
        settings=settings,
    )


@pytest.mark.unit
def test_main_starts_api_serves_samples_and_stops_server(
    monkeypatch,
) -> None:
    """Test that main orchestrates API startup, serving, and shutdown."""

    mock_settings = Mock()
    mock_process = Mock()

    args = Namespace(
        sample_size=100,
        random_state=42,
        reference_date=datetime(2026, 9, 30, tzinfo=UTC),
        drop_columns=["reference_date"],
        host="localhost",
        port=8000,
        reload=False,
        model_name=None,
        model_version=None,
        model_manifest=None,
    )

    mock_parser = Mock()
    mock_parser.parse_args.return_value = args

    configure_logging_mock = Mock()
    serving_settings_mock = Mock(return_value=mock_settings)
    custom_parser_mock = Mock(return_value=mock_parser)
    start_api_server_mock = Mock(return_value=mock_process)
    wait_for_api_mock = Mock()
    serve_samples_mock = Mock()
    stop_api_server_mock = Mock()

    monkeypatch.setattr(
        "churn_mlops.serving.serve_samples.configure_logging",
        configure_logging_mock,
    )
    monkeypatch.setattr(
        "churn_mlops.serving.serve_samples.ServingSettings",
        serving_settings_mock,
    )
    monkeypatch.setattr(
        "churn_mlops.serving.serve_samples.custom_parser",
        custom_parser_mock,
    )
    monkeypatch.setattr(
        "churn_mlops.serving.serve_samples.start_api_server",
        start_api_server_mock,
    )
    monkeypatch.setattr(
        "churn_mlops.serving.serve_samples.wait_for_api",
        wait_for_api_mock,
    )
    monkeypatch.setattr(
        "churn_mlops.serving.serve_samples.serve_samples",
        serve_samples_mock,
    )
    monkeypatch.setattr(
        "churn_mlops.serving.serve_samples.stop_api_server",
        stop_api_server_mock,
    )

    main()

    configure_logging_mock.assert_called_once()
    custom_parser_mock.assert_called_once_with(settings=mock_settings)

    start_api_server_mock.assert_called_once_with(
        host="localhost",
        port=8000,
    )

    wait_for_api_mock.assert_called_once_with(
        host="localhost",
        port=8000,
    )

    serve_samples_mock.assert_called_once_with(
        sample_size=100,
        random_state=42,
        settings=mock_settings,
        reference_date=args.reference_date,
        drop_columns=["reference_date"],
    )

    stop_api_server_mock.assert_called_once_with(mock_process)


@pytest.mark.unit
def test_main_stops_server_when_serving_fails(monkeypatch) -> None:
    """Test that API process is stopped even if serving fails."""

    mock_process = Mock()

    args = Namespace(
        sample_size=100,
        random_state=42,
        reference_date=None,
        drop_columns=[],
        host="localhost",
        port=8000,
        reload=False,
        model_name=None,
        model_version=None,
        model_manifest=None,
    )

    parser = Mock()
    parser.parse_args.return_value = args

    monkeypatch.setattr(
        "churn_mlops.serving.serve_samples.configure_logging",
        Mock(),
    )
    monkeypatch.setattr(
        "churn_mlops.serving.serve_samples.ServingSettings",
        Mock(),
    )
    monkeypatch.setattr(
        "churn_mlops.serving.serve_samples.custom_parser",
        Mock(return_value=parser),
    )
    monkeypatch.setattr(
        "churn_mlops.serving.serve_samples.start_api_server",
        Mock(return_value=mock_process),
    )
    monkeypatch.setattr(
        "churn_mlops.serving.serve_samples.wait_for_api",
        Mock(),
    )
    monkeypatch.setattr(
        "churn_mlops.serving.serve_samples.serve_samples",
        Mock(side_effect=RuntimeError("boom")),
    )

    stop_api_server_mock = Mock()
    monkeypatch.setattr(
        "churn_mlops.serving.serve_samples.stop_api_server",
        stop_api_server_mock,
    )

    import pytest

    with pytest.raises(RuntimeError, match="boom"):
        main()

    stop_api_server_mock.assert_called_once_with(mock_process)
