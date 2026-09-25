import argparse
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from churn_mlops.config import RuntimeSettings
from churn_mlops.serving import serve_samples


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
    )

    assert result == expected
    load_sample_data_mock.assert_called_once_with(
        sample_size=1,
        random_state=42,
        settings=settings,
    )
    predict_samples_mock.assert_called_once_with(
        df=sample_df,
        settings=settings,
    )
    save_predictions_mock.assert_called_once_with(
        df=prediction_df,
        run_id="0042",
        settings=settings,
    )


@pytest.mark.unit
def test_main_uses_supplied_random_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            sample_size="10",
            random_state="42",
        ),
    )

    called = {}

    monkeypatch.setattr(
        serve_samples,
        "serve_samples",
        lambda **kwargs: called.update(kwargs),
    )

    serve_samples.main()

    assert called["sample_size"] == 10
    assert called["random_state"] == 42
