import importlib

import pytest


@pytest.mark.smoke
def test_project_package_imports_successfully() -> None:
    import churn_mlops

    assert churn_mlops is not None


@pytest.mark.smoke
def test_package_main_dispatches_train_command(monkeypatch) -> None:
    import churn_mlops

    called = {"train": False}

    def fake_train_main() -> None:
        called["train"] = True

    train_module = importlib.import_module("churn_mlops.train")
    monkeypatch.setattr(train_module, "main", fake_train_main)

    churn_mlops.main(["train", "--config", "sample_training_config.yaml"])

    assert called["train"] is True


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
