import importlib

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
