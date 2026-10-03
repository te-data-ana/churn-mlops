import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.smoke
def test_project_package_imports_successfully() -> None:
    import churn_mlops

    assert churn_mlops is not None


@pytest.mark.smoke
def test_package_main_dispatches_train_command(monkeypatch) -> None:
    import churn_mlops

    observed = {}

    def fake_train_main() -> None:
        observed["argv"] = importlib.sys.argv.copy()

    train_module = importlib.import_module("churn_mlops.train")
    monkeypatch.setattr(train_module, "main", fake_train_main)

    churn_mlops.main(
        [
            "train",
            "--config",
            "config.yaml",
            "--experiment_name",
            "my-exp",
        ]
    )

    assert observed["argv"] == [
        "churn-mlops.train",
        "--config",
        "config.yaml",
        "--split_name",
        "default",
        "--experiment_name",
        "my-exp",
    ]


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
def test_package_main_dispatches_batch_predict(monkeypatch) -> None:
    import churn_mlops

    observed = {}

    def fake_batch_predict_main() -> None:
        observed["argv"] = importlib.sys.argv.copy()

    batch_module = importlib.import_module("churn_mlops.batch_predict")
    monkeypatch.setattr(batch_module, "main", fake_batch_predict_main)

    churn_mlops.main(
        [
            "batch-predict",
            "--input_csv",
            "input.csv",
            "--output_csv",
            "output.csv",
            "--index_col",
            "customer_id",
        ]
    )

    assert observed["argv"] == [
        "churn-mlops.batch_predict",
        "--input_csv",
        "input.csv",
        "--output_csv",
        "output.csv",
        "--index_col",
        "customer_id",
    ]


@pytest.mark.smoke
def test_package_main_dispatches_partitioned_batch_predict(monkeypatch) -> None:
    import churn_mlops

    observed = {}

    def fake_batch_predict_main() -> None:
        observed["argv"] = importlib.sys.argv.copy()

    batch_module = importlib.import_module("churn_mlops.batch_predict")
    monkeypatch.setattr(batch_module, "main", fake_batch_predict_main)

    churn_mlops.main(
        [
            "batch-predict",
            "--input_dataset",
            "partitioned",
            "--output_dataset",
            "scored",
            "--start_date",
            "2026-01-01",
            "--end_date",
            "2026-06-30",
        ]
    )

    assert observed["argv"] == [
        "churn-mlops.batch_predict",
        "--input_dataset",
        "partitioned",
        "--output_dataset",
        "scored",
        "--start_date",
        "2026-01-01",
        "--end_date",
        "2026-06-30",
    ]


@pytest.mark.smoke
def test_package_main_dispatches_monitor_command(monkeypatch) -> None:
    import churn_mlops

    observed = {}

    def fake_monitoring_main() -> None:
        observed["argv"] = importlib.sys.argv.copy()

    monitoring_module = importlib.import_module("churn_mlops.monitoring.core")
    monkeypatch.setattr(monitoring_module, "main", fake_monitoring_main)

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

    assert observed["argv"] == [
        "churn-mlops.monitoring",
        "--reference_csv",
        "reference.csv",
        "--analysis_csv",
        "analysis.csv",
        "--output_dir",
        "reports",
    ]


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
    capsys: pytest.CaptureFixture[str],
) -> None:
    import churn_mlops
    from churn_mlops.data import splitting

    observed: dict[str, object] = {}
    metadata = {"train_rows": 2, "test_rows": 1}

    def fake_create_time_based_split(**kwargs: object) -> dict[str, int]:
        observed.update(kwargs)
        return metadata

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
            "baseline",
        ]
    )

    assert observed == {
        "dataset_name": "partitioned",
        "timestamp_column": "reference_date",
        "start_date": "2026-01-01",
        "split_date": "2026-04-01",
        "end_date": "2026-06-30",
        "split_name": "baseline",
    }
    assert capsys.readouterr().out.strip() == str(metadata)


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
