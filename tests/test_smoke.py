import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.smoke
def test_project_package_imports_successfully() -> None:
    import churn_mlops

    assert churn_mlops is not None


@pytest.mark.smoke
def test_package_main_dispatches_train_command(monkeypatch) -> None:
    import churn_mlops

    observed: dict[str, object] = {}
    train_module = importlib.import_module("churn_mlops.train")

    def fake_run_training_job(**kwargs: object) -> SimpleNamespace:
        observed["kwargs"] = kwargs
        return SimpleNamespace(
            classifier_config={"model_name": "test-model"},
            metrics={"roc_auc": 0.9},
        )

    monkeypatch.setattr(train_module, "run_training_job", fake_run_training_job)

    churn_mlops.main(
        [
            "train",
            "--config",
            "config.yaml",
            "--exclude-columns",
            "customerid",
            "legacy_id",
            "--experiment_name",
            "my-exp",
        ]
    )

    assert observed["kwargs"] == {
        "config_file": "config.yaml",
        "split_name": "default",
        "exclude_columns": ["customerid", "legacy_id"],
        "experiment_name": "my-exp",
    }


@pytest.mark.smoke
def test_package_main_prints_help_for_empty_args(capsys) -> None:
    import churn_mlops

    try:
        churn_mlops.main([])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError(
            "Top-level CLI should exit with code 0 when no subcommand is provided."
        )

    captured = capsys.readouterr()
    assert "usage: churn-mlops" in captured.out
    assert "train" in captured.out
    assert "batch-predict" in captured.out
    assert "monitor" in captured.out
    assert "serve" in captured.out


@pytest.mark.smoke
def test_package_main_dispatches_monitor_command(monkeypatch) -> None:
    import churn_mlops

    observed: dict[str, object] = {}

    def fake_run_monitoring(**kwargs: object) -> None:
        observed.update(kwargs)

    monitoring_module = importlib.import_module("churn_mlops.monitoring.core")
    monkeypatch.setattr(monitoring_module, "run_monitoring", fake_run_monitoring)

    churn_mlops.main(
        [
            "monitor",
            "--reference_csv",
            "reference.csv",
            "--analysis_csv",
            "analysis.csv",
            "--output_dir",
            "reports",
        ]
    )

    assert observed == {
        "reference_csv": Path("reference.csv"),
        "analysis_csv": Path("analysis.csv"),
        "prediction_log": None,
        "error_log": None,
        "output_dir": Path("reports"),
    }


@pytest.mark.smoke
def test_package_main_builds_report_from_partitioned_datasets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import churn_mlops
    from churn_mlops import data
    from churn_mlops.monitoring import core

    datasets = {
        "partitioned": object(),
        "api_predictions": object(),
        "api_errors": object(),
    }
    loaded: list[tuple[str, str]] = []
    observed: dict[str, object] = {}

    def fake_read_partitioned_dataset(
        dataset_name: str, timestamp_column: str
    ) -> object:
        loaded.append((dataset_name, timestamp_column))
        return datasets[dataset_name]

    def fake_build_monitoring_report(**kwargs: object) -> None:
        observed.update(kwargs)

    monkeypatch.setattr(data, "read_partitioned_dataset", fake_read_partitioned_dataset)
    monkeypatch.setattr(core, "build_monitoring_report", fake_build_monitoring_report)

    churn_mlops.main(
        [
            "monitor",
            "--reference_dataset",
            "partitioned",
            "--analysis_dataset",
            "api_predictions",
            "--error_dataset",
            "api_errors",
            "--output_dir",
            "reports",
        ]
    )

    assert loaded == [
        ("partitioned", "reference_date"),
        ("api_predictions", "reference_date"),
        ("api_errors", "reference_date"),
    ]
    assert observed == {
        "reference": datasets["partitioned"],
        "analysis": datasets["api_predictions"],
        "source_reference": "batch",
        "source_analysis": "api",
        "errors": datasets["api_errors"],
        "output_dir": Path("reports"),
    }


@pytest.mark.smoke
def test_package_main_uses_monitoring_model_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import churn_mlops
    from churn_mlops import data
    from churn_mlops.monitoring import core
    from churn_mlops.tracking import TrainingManifest

    settings = SimpleNamespace(
        data_dir=tmp_path,
        output_dir=tmp_path / "reports",
        api_host="127.0.0.1",
        api_port=8000,
    )
    manifest = SimpleNamespace(model_name="churn-risk", model_version=11)
    datasets = {"reference": object(), "analysis": object()}
    observed_report: dict[str, object] = {}
    observed_log: dict[str, object] = {}

    monkeypatch.setattr(churn_mlops, "ServingSettings", lambda: settings)

    def fake_read_partitioned_dataset(
        dataset_name: str, timestamp_column: str
    ) -> object:
        return datasets[dataset_name]

    monkeypatch.setattr(data, "read_partitioned_dataset", fake_read_partitioned_dataset)
    manifest_read = Mock(return_value=manifest)
    monkeypatch.setattr(TrainingManifest, "read", manifest_read)

    def fake_build_monitoring_report(**kwargs: object) -> str:
        observed_report.update(kwargs)
        return "report"

    def fake_log_monitoring_run(report: object, **kwargs: object) -> None:
        observed_log["report"] = report
        observed_log.update(kwargs)

    monkeypatch.setattr(core, "build_monitoring_report", fake_build_monitoring_report)
    monkeypatch.setattr(core, "log_monitoring_run", fake_log_monitoring_run)

    churn_mlops.main(
        [
            "monitor",
            "--reference_dataset",
            "reference",
            "--analysis_dataset",
            "analysis",
            "--model_manifest",
            "manifests/hgb_OOT_26Q2.json",
            "--mlflow_experiment_name",
            "churn-monitoring",
        ]
    )

    manifest_read.assert_called_once_with(tmp_path / "manifests/hgb_OOT_26Q2.json")
    assert observed_report == {
        "reference": datasets["reference"],
        "analysis": datasets["analysis"],
        "source_reference": "batch",
        "source_analysis": "api",
        "errors": None,
        "output_dir": settings.output_dir,
        "model_version": 11,
    }
    assert observed_log == {
        "report": "report",
        "experiment_name": "churn-monitoring",
        "run_name": "monitoring",
        "tags": {
            "reference_input": "reference",
            "analysis_input": "analysis",
            "reference_name": "reference",
            "analysis_name": "analysis",
            "model_name": "churn-risk",
            "model_version": "11",
        },
    }


@pytest.mark.smoke
def test_package_main_rejects_monitor_manifest_with_explicit_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import churn_mlops

    with pytest.raises(SystemExit, match="2"):
        churn_mlops.main(
            [
                "monitor",
                "--reference_dataset",
                "reference",
                "--analysis_dataset",
                "analysis",
                "--model_manifest",
                "manifests/model.json",
                "--model_version",
                "11",
            ]
        )

    assert "--model_manifest cannot be combined" in capsys.readouterr().err


@pytest.mark.smoke
def test_package_main_dispatches_api_file_monitoring_with_configured_error_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import churn_mlops
    from churn_mlops.monitoring import core

    settings = SimpleNamespace(
        output_dir=tmp_path / "reports",
        prediction_log_path=tmp_path / "predictions.jsonl",
        error_log_path=tmp_path / "errors.jsonl",
        api_host="127.0.0.1",
        api_port=8000,
    )
    observed: dict[str, object] = {}

    def fake_run_monitoring(**kwargs: object) -> None:
        observed.update(kwargs)

    monkeypatch.setattr(churn_mlops, "ServingSettings", lambda: settings)
    monkeypatch.setattr(core, "run_monitoring", fake_run_monitoring)

    churn_mlops.main(
        [
            "monitor",
            "--reference_csv",
            "reference.csv",
            "--prediction_log",
            "custom_predictions.jsonl",
        ]
    )

    assert observed == {
        "reference_csv": Path("reference.csv"),
        "analysis_csv": None,
        "prediction_log": Path("custom_predictions.jsonl"),
        "error_log": settings.error_log_path,
        "output_dir": settings.output_dir,
    }


@pytest.mark.smoke
def test_package_main_dispatches_serve_command(monkeypatch) -> None:
    import churn_mlops

    called = {}

    class FakeUvicorn:
        @staticmethod
        def run(app: str, host: str, port: int, reload: bool) -> None:
            called["app"] = app
            called["host"] = host
            called["port"] = port
            called["reload"] = reload

    fake_uvicorn_module = type("module", (), {"run": staticmethod(FakeUvicorn.run)})
    monkeypatch.setitem(importlib.sys.modules, "uvicorn", fake_uvicorn_module)

    churn_mlops.main(["serve", "--host", "0.0.0.0", "--port", "9000", "--reload"])

    assert called == {
        "app": "churn_mlops.serving.api:app",
        "host": "0.0.0.0",
        "port": 9000,
        "reload": True,
    }


@pytest.mark.smoke
def test_package_main_dispatches_prepare_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import churn_mlops
    from churn_mlops import data
    from churn_mlops.data import storage

    raw_data = object()
    validated_data = object()
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        churn_mlops,
        "ServingSettings",
        lambda: SimpleNamespace(
            data_dir=tmp_path,
            api_host="127.0.0.1",
            api_port=8000,
        ),
    )

    def fake_load_raw_data(**kwargs: object) -> object:
        observed["load"] = kwargs
        return raw_data

    def fake_validate_data(df: object) -> object:
        observed["validated_input"] = df
        return validated_data

    def fake_write_partitioned_dataset(**kwargs: object) -> None:
        observed["write"] = kwargs

    monkeypatch.setattr(data, "load_raw_data", fake_load_raw_data)
    monkeypatch.setattr(data, "validate_data", fake_validate_data)
    monkeypatch.setattr(
        storage,
        "write_partitioned_dataset",
        fake_write_partitioned_dataset,
    )

    churn_mlops.main(["prepare-data", "--input_csv", "training.csv"])

    assert observed == {
        "load": {"file_name": "training.csv", "data_dir": tmp_path / "raw"},
        "validated_input": raw_data,
        "write": {
            "df": validated_data,
            "dataset_name": "partitioned",
            "timestamp_column": "reference_date",
            "overwrite_partitions": True,
        },
    }


@pytest.mark.smoke
def test_package_main_dispatches_create_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import churn_mlops
    from churn_mlops.data import splitting

    observed: dict[str, object] = {}

    def fake_create_time_based_split(**kwargs: object) -> dict[str, int]:
        observed.update(kwargs)
        return

    settings = SimpleNamespace(data_dir=tmp_path)
    monkeypatch.setattr(splitting, "RuntimeSettings", lambda: settings)
    monkeypatch.setattr(
        splitting,
        "create_time_based_split",
        fake_create_time_based_split,
    )

    churn_mlops.main(
        [
            "create-split",
            "--dataset_name",
            "partitioned",
            "--start_date",
            "2026-01-01",
            "--split_date",
            "2026-04-01",
            "--end_date",
            "2026-06-30",
            "--split_name",
            "pytest",
        ]
    )

    assert observed == {
        "dataset_name": "partitioned",
        "timestamp_column": "reference_date",
        "start_date": "2026-01-01",
        "split_date": "2026-04-01",
        "end_date": "2026-06-30",
        "split_name": "pytest",
    }


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("command", "function_name", "dataset_name"),
    [
        ("ingest-prediction-logs", "ingest_prediction_logs", "api_predictions"),
        ("ingest-error-logs", "ingest_error_logs", "api_errors"),
    ],
)
def test_package_main_dispatches_log_ingestion(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    function_name: str,
    dataset_name: str,
) -> None:
    import churn_mlops

    ingestion = importlib.import_module("churn_mlops.data.ingestion")
    observed: dict[str, str] = {}

    def fake_ingest(dataset_name: str, timestamp_column: str) -> None:
        observed["dataset_name"] = dataset_name
        observed["timestamp_column"] = timestamp_column

    monkeypatch.setattr(ingestion, function_name, fake_ingest)

    churn_mlops.main([command])

    assert observed == {
        "dataset_name": dataset_name,
        "timestamp_column": "reference_date",
    }
