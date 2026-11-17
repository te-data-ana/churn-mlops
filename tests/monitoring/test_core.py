import warnings
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from churn_mlops.monitoring import core
from churn_mlops.use_cases import get_use_case

CHURN_USE_CASE = get_use_case("churn")


class _FakeResult:
    def __init__(self, columns: list[tuple[str, str]]) -> None:
        self.df = pd.DataFrame(
            [
                [
                    "analysis",
                    pd.Timestamp("2026-02-01"),
                    False if columns[-1][-1] == "alert" else 0.75,
                ]
            ],
            columns=pd.MultiIndex.from_tuples(columns),
        )

    def to_df(self) -> pd.DataFrame:
        return self.df


class _FakeCBPE:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit(self, reference: pd.DataFrame) -> "_FakeCBPE":
        self.reference = reference
        return self

    def estimate(self, analysis: pd.DataFrame) -> _FakeResult:
        return _FakeResult(
            [
                ("chunk", "period"),
                ("chunk", "start_date"),
                ("roc_auc", "value"),
            ]
        )


class _FakeDriftCalculator:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit(self, reference: pd.DataFrame) -> "_FakeDriftCalculator":
        self.reference = reference
        return self

    def calculate(self, analysis: pd.DataFrame) -> _FakeResult:
        return _FakeResult(
            [
                ("chunk", "period"),
                ("chunk", "start_date"),
                ("drift", "alert"),
            ]
        )


@pytest.mark.unit
def test_read_scored_csv_normalizes_columns_and_raises_for_missing_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scores.csv"
    pd.DataFrame({"Age": [27], "Predicted Probability": [0.3]}).to_csv(
        path, index=False
    )

    df = core._read_scored_csv(path)

    assert list(df.columns) == ["age", "predicted_probability"]

    with pytest.raises(FileNotFoundError):
        core._read_scored_csv(tmp_path / "missing.csv")


@pytest.mark.unit
def test_event_log_to_df_handles_empty_invalid_and_valid_logs(
    tmp_path: Path,
    valid_prediction_log_path: Path,
) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    assert core._event_log_to_df(
        path=empty, event="prediction", use_case=CHURN_USE_CASE
    ).empty
    assert core._event_log_to_df(
        path=None, event="prediction_error", use_case=CHURN_USE_CASE
    ).empty

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"event":"prediction_error"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="other than prediction"):
        core._event_log_to_df(path=bad, event="prediction", use_case=CHURN_USE_CASE)

    df = core._event_log_to_df(
        path=valid_prediction_log_path, event="prediction", use_case=CHURN_USE_CASE
    )
    expected = {
        "event",
        CHURN_USE_CASE.time_column,
        "latency_ms",
        core.MODEL_VERSION_COLUMN,
    }

    assert expected.issubset(df.columns)
    assert df[CHURN_USE_CASE.time_column].dtype.kind == "M"
    assert df["latency_ms"].tolist() == pytest.approx([12.5, 18.0])


@pytest.mark.unit
def test_event_log_to_df_uses_alternate_use_case_time_column(
    tmp_path: Path,
) -> None:
    use_case = replace(CHURN_USE_CASE, time_column="event_date")
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"event":"prediction","event_date":"2026-02-01T00:00:00Z",'
        '"model_version":1,"latency_ms":12.5}\n',
        encoding="utf-8",
    )

    df = core._event_log_to_df(path=path, event="prediction", use_case=use_case)

    assert df["event_date"].dtype.kind == "M"
    assert "reference_date" not in df.columns


@pytest.mark.unit
def test_prepare_validated_df_requires_predictions_target_and_features(
    valid_reference_df: pd.DataFrame,
) -> None:
    df = valid_reference_df.copy()

    with pytest.raises(ValueError, match="probabilities outside"):
        bad_probability = df.copy()
        bad_probability[core.PRED_PROBA_COLUMN] = [1.5, 0.2, 0.8]
        core._prepare_validated_df(
            df=bad_probability,
            name="reference",
            source="batch",
            use_case=CHURN_USE_CASE,
            require_predictions=True,
            require_target=True,
        )

    missing_prediction_cols = df.drop(
        columns=[core.PRED_PROBA_COLUMN, core.PRED_CLASS_COLUMN]
    )
    with pytest.raises(ValueError, match="missing required columns"):
        core._prepare_validated_df(
            df=missing_prediction_cols,
            name="reference",
            source="batch",
            use_case=CHURN_USE_CASE,
            require_predictions=True,
            require_target=True,
        )

    with pytest.raises(ValueError, match="both target classes"):
        single_target = df.copy()
        single_target[CHURN_USE_CASE.target_column] = [0, 0, 0]
        core._prepare_validated_df(
            df=single_target,
            name="reference",
            source="batch",
            use_case=CHURN_USE_CASE,
            require_predictions=True,
            require_target=True,
        )

    prepared = core._prepare_validated_df(
        df=df.copy(),
        name="reference",
        source="batch",
        use_case=CHURN_USE_CASE,
        require_predictions=True,
        require_target=True,
        require_features=False,
    )
    assert prepared["source"].unique().tolist() == ["batch"]
    assert prepared[CHURN_USE_CASE.target_column].nunique() == 2
    assert prepared[core.MODEL_VERSION_COLUMN].dtype.kind in {"i", "u"}


@pytest.mark.unit
def test_flatten_and_extract_columns_handle_multiindex() -> None:
    df = pd.DataFrame(
        [
            [1, 2.5, True],
        ],
        columns=pd.MultiIndex.from_tuples(
            [("chunk", "period"), ("chunk", "start_date"), ("drift", "alert")]
        ),
    )

    flat = core._flatten_columns(df)
    assert list(flat.columns) == ["chunk__period", "chunk__start_date", "drift__alert"]
    assert core._extract_columns(flat, {"period", "alert"}) == [
        "chunk__period",
        "drift__alert",
    ]


@pytest.mark.unit
def test_drift_alert_summary_and_cbpe_summary_build_monitoring_tables() -> None:
    result = _FakeResult(
        [
            ("chunk", "period"),
            ("chunk", "start_date"),
            ("drift", "alert"),
        ]
    )
    summary = core._drift_alert_summary(
        result=result,
        model_version=2,
        calculator_name="univariate",
        time_column=CHURN_USE_CASE.time_column,
    )
    assert list(summary.columns) == [
        core.PERIOD_COLUMN,
        core.MODEL_VERSION_COLUMN,
        CHURN_USE_CASE.time_column,
        "univariate_alert_count",
    ]
    assert summary.iloc[0][core.MODEL_VERSION_COLUMN] == 2
    assert summary.iloc[0]["univariate_alert_count"] == 0

    cbpe_result = _FakeResult(
        [
            ("chunk", "period"),
            ("chunk", "start_date"),
            ("roc_auc", "value"),
        ]
    )
    cbpe_summary = core._cbpe_summary(
        result=cbpe_result,
        model_version=3,
        calculator_name="cbpe",
        time_column=CHURN_USE_CASE.time_column,
    )
    assert cbpe_summary.columns.tolist() == [
        core.PERIOD_COLUMN,
        core.MODEL_VERSION_COLUMN,
        CHURN_USE_CASE.time_column,
        "cbpe_roc_auc",
    ]
    assert cbpe_summary.iloc[0][core.MODEL_VERSION_COLUMN] == 3


@pytest.mark.unit
def test_base_summary_and_actual_metrics_compute_expected_values(
    valid_reference_df: pd.DataFrame,
    valid_analysis_df: pd.DataFrame,
) -> None:
    reference_prepared = core._prepare_validated_df(
        df=valid_reference_df.copy(),
        name="reference",
        source="batch",
        use_case=CHURN_USE_CASE,
        require_predictions=True,
        require_target=True,
        require_features=False,
    )
    analysis_prepared = core._prepare_validated_df(
        df=valid_analysis_df.copy(),
        name="analysis",
        source="batch",
        use_case=CHURN_USE_CASE,
        require_predictions=True,
        require_target=False,
        require_features=False,
    )

    base = core._base_summary(
        reference=reference_prepared,
        analysis=analysis_prepared,
        time_column=CHURN_USE_CASE.time_column,
    )
    assert base["prediction_count"].sum() == 5
    assert base["avg_predicted_probability"].notna().all()
    assert base["source"].unique().tolist() == ["batch"]

    actual = core._actual_metrics(
        reference=reference_prepared,
        analysis=analysis_prepared,
        target_column=CHURN_USE_CASE.target_column,
        time_column=CHURN_USE_CASE.time_column,
    )
    assert actual.empty is False
    assert set(actual.columns) >= {
        core.PERIOD_COLUMN,
        core.MODEL_VERSION_COLUMN,
        CHURN_USE_CASE.time_column,
        "observed_positive_rate",
        "realized_precision",
        "realized_recall",
        "realized_roc_auc",
    }


@pytest.mark.unit
def test_calculate_version_drift_uses_nannyml_and_merges_alerts(
    valid_reference_df: pd.DataFrame,
    valid_analysis_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core.nml, "CBPE", _FakeCBPE)
    monkeypatch.setattr(
        core.nml, "DataReconstructionDriftCalculator", _FakeDriftCalculator
    )
    monkeypatch.setattr(core.nml, "UnivariateDriftCalculator", _FakeDriftCalculator)

    reference_prepared = core._prepare_validated_df(
        df=valid_reference_df.copy(),
        name="reference",
        source="batch",
        use_case=CHURN_USE_CASE,
        require_predictions=True,
        require_target=True,
        require_features=True,
    )
    analysis_prepared = core._prepare_validated_df(
        df=valid_analysis_df.copy(),
        name="analysis",
        source="batch",
        use_case=CHURN_USE_CASE,
        require_predictions=True,
        require_target=False,
        require_features=True,
    )
    errors = pd.DataFrame(
        {
            core.PERIOD_COLUMN: ["analysis"],
            core.MODEL_VERSION_COLUMN: [1],
            CHURN_USE_CASE.time_column: pd.to_datetime(["2026-02-03"]),
            "latency_ms": [12.0],
            **{column: [0] for column in CHURN_USE_CASE.feature_columns},
        }
    )

    result = core._calculate_version_drift(
        reference=reference_prepared,
        analysis=analysis_prepared,
        errors=errors,
        model_version=1,
        use_case=CHURN_USE_CASE,
    )

    assert list(result.columns) == [
        core.PERIOD_COLUMN,
        core.MODEL_VERSION_COLUMN,
        CHURN_USE_CASE.time_column,
        "cbpe_roc_auc",
        "reconstruction_drift_alert_count",
        "univariate_drift_alert_count",
    ]


@pytest.mark.unit
def test_operational_summary_combines_success_and_error_events() -> None:
    analysis = pd.DataFrame(
        {
            core.PERIOD_COLUMN: ["api", "api"],
            core.MODEL_VERSION_COLUMN: [1, 1],
            CHURN_USE_CASE.time_column: pd.to_datetime(["2026-02-01", "2026-02-02"]),
            "latency_ms": [10.0, 30.0],
        }
    )
    errors = pd.DataFrame(
        {
            core.PERIOD_COLUMN: ["api"],
            core.MODEL_VERSION_COLUMN: [1],
            CHURN_USE_CASE.time_column: pd.to_datetime(["2026-02-01"]),
            "latency_ms": [95.0],
        }
    )

    summary = core._operational_summary(
        analysis=analysis,
        errors=errors,
        time_column=CHURN_USE_CASE.time_column,
    )

    assert summary["api_request_count"].tolist() == [2, 1]
    assert summary["api_error_rate"].iloc[0] == pytest.approx(0.5)
    assert summary["prediction_latency_p95_ms"].notna().all()


@pytest.mark.unit
def test_build_monitoring_report_generates_summary_and_csv(
    valid_reference_df: pd.DataFrame,
    valid_analysis_df: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core.nml, "CBPE", _FakeCBPE)
    monkeypatch.setattr(
        core.nml, "DataReconstructionDriftCalculator", _FakeDriftCalculator
    )
    monkeypatch.setattr(core.nml, "UnivariateDriftCalculator", _FakeDriftCalculator)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="obj.round has no effect with datetime",
            category=UserWarning,
        )
        report = core.build_monitoring_report(
            reference=valid_reference_df,
            analysis=valid_analysis_df,
            use_case=CHURN_USE_CASE,
            source_reference="batch",
            source_analysis="batch",
            drift=True,
            output_dir=tmp_path,
        )

    assert isinstance(report, core.MonitoringReport)
    assert not report.summary.empty
    assert report.summary[CHURN_USE_CASE.time_column].dtype.kind == "M"
    assert report.summary_path is not None
    assert report.summary_path.exists()
    assert list(report.summary.columns)[:4] == [
        "source",
        core.PERIOD_COLUMN,
        core.MODEL_VERSION_COLUMN,
        CHURN_USE_CASE.time_column,
    ]


@pytest.mark.unit
def test_build_monitoring_report_can_select_an_exact_model_version(
    valid_reference_df: pd.DataFrame,
    valid_analysis_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core.nml, "CBPE", _FakeCBPE)
    reference_v2 = valid_reference_df.assign(model_version=2)
    analysis_v2 = valid_analysis_df.assign(model_version=2)

    report = core.build_monitoring_report(
        reference=pd.concat([valid_reference_df, reference_v2], ignore_index=True),
        analysis=pd.concat([valid_analysis_df, analysis_v2], ignore_index=True),
        use_case=CHURN_USE_CASE,
        source_reference="batch",
        source_analysis="batch",
        drift=False,
        model_version=2,
    )

    assert report.summary[core.MODEL_VERSION_COLUMN].unique().tolist() == [2]


@pytest.mark.unit
@pytest.mark.filterwarnings(
    "ignore:'future.no_silent_downcasting' is deprecated",
    "ignore:The resulting number of chunks is too low.",
)
def test_run_monitoring_handles_batch_and_api_sources(
    valid_reference_df: pd.DataFrame,
    valid_prediction_log_path: Path,
    valid_error_log_path: Path,
    tmp_path: Path,
) -> None:
    reference_csv = tmp_path / "reference.csv"
    analysis_csv = tmp_path / "analysis.csv"
    valid_reference_df.to_csv(reference_csv, index=False)
    valid_reference_df.to_csv(analysis_csv, index=False)

    batch_report = core.run_monitoring(
        reference_csv=reference_csv,
        analysis_csv=analysis_csv,
        use_case=CHURN_USE_CASE,
        output_dir=tmp_path,
    )
    assert isinstance(batch_report, core.MonitoringReport)
    assert batch_report.summary.empty is False

    api_report = core.run_monitoring(
        reference_csv=reference_csv,
        use_case=CHURN_USE_CASE,
        prediction_log=valid_prediction_log_path,
        error_log=valid_error_log_path,
        output_dir=tmp_path,
    )
    assert isinstance(api_report, core.MonitoringReport)
    assert api_report.summary.empty is False

    with pytest.raises(ValueError, match="exactly one"):
        core.run_monitoring(reference_csv=reference_csv, use_case=CHURN_USE_CASE)


@pytest.mark.unit
def test_log_monitoring_run_logs_tags_and_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "monitoring_summary_batch.csv"
    report = core.MonitoringReport(
        summary=pd.DataFrame(
            {
                "model_version": [7, 7],
                CHURN_USE_CASE.time_column: pd.to_datetime(
                    ["2026-01-01", "2026-02-01"]
                ),
            }
        ),
        summary_path=report_path,
    )
    setup_experiment = MagicMock(return_value="experiment-id")
    monkeypatch.setattr(
        "churn_mlops.tracking.setup_local_experiment",
        setup_experiment,
    )
    monkeypatch.setattr(core.mlflow, "start_run", MagicMock())
    set_tags = MagicMock()
    log_artifact = MagicMock()
    log_metrics = MagicMock()
    monkeypatch.setattr(core.mlflow, "set_tags", set_tags)
    monkeypatch.setattr(core.mlflow, "log_artifact", log_artifact)
    monkeypatch.setattr(core.mlflow, "log_metrics", log_metrics)

    core.log_monitoring_run(
        report,
        experiment_name="churn-monitoring",
        run_name="rf__followup",
        tags={"model_name": "churn-risk", "model_version": 7},
        tracking_uri="sqlite:///tracking.db",
    )

    setup_experiment.assert_called_once_with(
        experiment_name="churn-monitoring",
        tracking_uri="sqlite:///tracking.db",
    )
    set_tags.assert_called_once_with({"model_name": "churn-risk", "model_version": 7})
    log_artifact.assert_called_once_with(
        local_path=str(report_path),
        artifact_path="monitoring",
    )
