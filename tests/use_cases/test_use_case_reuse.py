import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from churn_mlops import batch_predict
from churn_mlops.models import IdentityFeatureBuilder
from churn_mlops.monitoring import core
from churn_mlops.serving import api
from churn_mlops.serving.model_loader import LoadedModel, ModelMetadata
from churn_mlops.training import train_model
from churn_mlops.use_cases import UseCase


class AlternateInput(BaseModel):
    amount: float
    category: str


def _validate_alternate_data(df: pd.DataFrame) -> pd.DataFrame:
    required = {"amount", "category"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing alternate-use-case columns: {sorted(missing)}")
    if "label" in df and not df["label"].dropna().isin([0, 1]).all():
        raise ValueError("The alternate target must contain binary labels.")
    return df.copy()


ALTERNATE_USE_CASE = UseCase(
    name="alternate",
    target_column="label",
    time_column="event_date",
    feature_columns=("amount", "category"),
    data_validator=_validate_alternate_data,
    feature_builder_factory=IdentityFeatureBuilder,
    request_schema=AlternateInput,
)


def _alternate_df(rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": np.arange(rows, dtype=float),
            "category": ["even" if row % 2 == 0 else "odd" for row in range(rows)],
            "customer_id": range(rows),
            "label": [row % 2 for row in range(rows)],
            "event_date": pd.date_range("2026-01-01", periods=rows),
        }
    )


@pytest.mark.unit
def test_alternate_use_case_runs_training_batch_api_and_monitoring(
    config_factory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = replace(config_factory(classifier="dt"), use_case="alternate")
    df = _alternate_df()
    result = train_model(
        config,
        df.iloc[:8],
        df.iloc[8:],
        use_case=ALTERNATE_USE_CASE,
    )
    assert result.feature_names_in == list(ALTERNATE_USE_CASE.feature_columns)
    loaded_model = LoadedModel(
        model=result.trained_pipeline,
        metadata=ModelMetadata(
            model_name="alternate-model",
            model_alias=None,
            model_version=1,
            model_threshold=result.threshold,
        ),
        use_case=ALTERNATE_USE_CASE,
    )
    monkeypatch.setattr(batch_predict, "load_model", lambda **_: loaded_model)

    scored = batch_predict.run_batch_prediction(
        df=df.drop(columns="label"),
        use_case=ALTERNATE_USE_CASE,
    )
    assert scored["predicted_probability"].between(0, 1).all()
    assert scored["customer_id"].tolist() == df["customer_id"].tolist()

    api_settings = SimpleNamespace(
        model_manifest=None,
        model_version=None,
        model_name="alternate-model",
        prediction_log_path=tmp_path / "predictions.jsonl",
        error_log_path=tmp_path / "errors.jsonl",
        log_features=True,
        request_id_header="X-Request-ID",
    )
    monkeypatch.setattr(api, "ServingSettings", lambda: api_settings)
    monkeypatch.setattr(api, "load_model", lambda **_: loaded_model)
    with TestClient(api.create_app(ALTERNATE_USE_CASE)) as client:
        valid = client.post(
            "/predict",
            json={"amount": 12, "category": "even"},
            params={"timestamp": "2026-01-11T00:00:00"},
        )
        invalid = client.post("/predict", json={"amount": "invalid"})

    assert valid.status_code == 200
    assert valid.json()["predicted_class"] in {0, 1}
    assert valid.json()["event_date"] == "2026-01-11T00:00:00"
    assert "reference_date" not in valid.json()
    openapi = client.get("/openapi.json").json()
    response_schema = openapi["components"]["schemas"]["PredictionResponse_event_date"]
    assert "event_date" in response_schema["properties"]
    assert "reference_date" not in response_schema["properties"]
    assert invalid.status_code == 422
    prediction_event = json.loads((tmp_path / "predictions.jsonl").read_text())
    assert prediction_event["event_date"] == "2026-01-11T00:00:00"
    assert "reference_date" not in prediction_event

    reference = _alternate_df(4).assign(
        predicted_probability=[0.1, 0.9, 0.2, 0.8],
        predicted_class=[0, 1, 0, 1],
        model_version=1,
    )
    analysis = _alternate_df(4).assign(
        event_date=pd.date_range("2026-02-01", periods=4),
        predicted_probability=[0.1, 0.9, 0.2, 0.8],
        predicted_class=[0, 1, 0, 1],
        model_version=1,
    )
    prepared = core._prepare_validated_df(
        df=reference.copy(),
        name="reference",
        source="batch",
        use_case=ALTERNATE_USE_CASE,
        require_target=True,
        require_features=True,
    )
    assert ALTERNATE_USE_CASE.time_column in prepared
    assert prepared[ALTERNATE_USE_CASE.target_column].nunique() == 2
    report = core.build_monitoring_report(
        reference=reference,
        analysis=analysis,
        use_case=ALTERNATE_USE_CASE,
        source_reference="batch",
        source_analysis="batch",
        drift=False,
    )
    assert not report.summary.empty


@pytest.mark.unit
@pytest.mark.parametrize("feature_column", ["label", "event_date"])
def test_use_case_rejects_target_and_time_as_features(
    feature_column: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="feature_columns must not include target or time columns",
    ):
        UseCase(
            name="invalid",
            target_column="label",
            time_column="event_date",
            feature_columns=("amount", feature_column),
            data_validator=_validate_alternate_data,
            feature_builder_factory=lambda _: IdentityFeatureBuilder(),
        )
