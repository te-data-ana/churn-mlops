from collections.abc import Callable
from dataclasses import replace

import optuna
import pandas as pd
import pytest

from churn_mlops.config.schemas import TrainingConfig, TuningConfig
from churn_mlops.models import MODEL_CATALOG, create_model
from churn_mlops.training.tuning import (
    _build_expanding_window_folds,
    _suggest_parameters,
    optimize_hyperparameters,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_alias",
    [alias for alias in MODEL_CATALOG if alias != "dc"],
)
def test_catalog_search_space_suggestions_build_each_model(
    model_alias: str,
) -> None:
    search_space = MODEL_CATALOG[model_alias]["search_space"]
    trial = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=7),
    ).ask()

    suggested = _suggest_parameters(
        trial=trial,
        search_space=search_space,
    )
    model, _ = create_model(
        model_alias=model_alias,
        model_params=suggested,
    )

    assert model is not None
    assert set(suggested) == set(search_space)


@pytest.mark.unit
def test_expanding_window_folds_keep_equal_timestamps_together(
    sample_training_df: pd.DataFrame,
) -> None:
    train_df = sample_training_df.copy()
    train_df["churn"] = [0, 1] * 5
    train_df["reference_date"] = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-01",
            "2026-02-01",
            "2026-02-01",
            "2026-03-01",
            "2026-03-01",
            "2026-04-01",
            "2026-04-01",
            "2026-05-01",
            "2026-05-01",
        ]
    )

    folds = _build_expanding_window_folds(
        train_df,
        target_column="churn",
        time_column="reference_date",
        n_splits=2,
    )

    for train_indices, validation_indices in folds:
        train_dates = train_df.iloc[train_indices]["reference_date"]
        validation_dates = train_df.iloc[validation_indices]["reference_date"]
        assert train_dates.max() < validation_dates.min()
        assert set(train_dates).isdisjoint(validation_dates)
        assert train_df.iloc[train_indices]["churn"].nunique() == 2
        assert train_df.iloc[validation_indices]["churn"].nunique() == 2


@pytest.mark.unit
def test_expanding_window_folds_reject_missing_time_column(
    sample_training_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="requires time column"):
        _build_expanding_window_folds(
            sample_training_df,
            target_column="churn",
            time_column="reference_date",
            n_splits=2,
        )


@pytest.mark.unit
def test_optimize_hyperparameters_returns_reproducible_study_result(
    config_factory: Callable[..., TrainingConfig],
    sample_training_df: pd.DataFrame,
) -> None:
    config: TrainingConfig = config_factory(classifier="dt")
    config = replace(
        config,
        tuning=TuningConfig(
            enabled=True,
            n_trials=2,
            n_splits=2,
            time_column="reference_date",
            metric="roc_auc",
            random_state=13,
        ),
    )
    train_df = sample_training_df.copy()
    train_df["reference_date"] = pd.date_range("2026-01-01", periods=len(train_df))

    result = optimize_hyperparameters(config, train_df)

    assert set(result.best_params) == {
        "max_depth",
        "min_samples_leaf",
    }
    assert 0.0 <= result.best_score <= 1.0
    assert result.n_trials == 2
    assert result.n_splits == 2
    assert result.time_column == "reference_date"


@pytest.mark.unit
def test_optimize_hyperparameters_rejects_classifier_without_search_space(
    config_factory: Callable[..., TrainingConfig],
    sample_training_df: pd.DataFrame,
) -> None:
    config: TrainingConfig = config_factory(classifier="dc")
    config = replace(config, tuning=TuningConfig(enabled=True))

    with pytest.raises(ValueError, match="does not define an Optuna search space"):
        optimize_hyperparameters(config, sample_training_df)
