from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from churn_mlops.data import splitting, storage
from churn_mlops.data.splitting import create_time_based_split
from churn_mlops.data.storage import write_partitioned_dataset


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    settings = SimpleNamespace(data_dir=tmp_path)
    monkeypatch.setattr(storage, "RuntimeSettings", lambda: settings)
    monkeypatch.setattr(splitting, "RuntimeSettings", lambda: settings)
    return tmp_path


@pytest.mark.unit
def test_create_time_based_split_writes_datasets_and_returns_metadata(
    data_dir: Path,
) -> None:
    write_partitioned_dataset(
        df=pd.DataFrame(
            {
                "reference_date": pd.to_datetime(
                    ["2026-01-15", "2026-02-15", "2026-03-15", "2026-04-15"]
                ),
                "value": [1, 2, 3, 4],
            }
        ),
        dataset_name="events",
        timestamp_column="reference_date",
    )

    metadata = create_time_based_split(
        dataset_name="events",
        timestamp_column="reference_date",
        start_date="2026-01-01",
        split_date="2026-03-15",
        end_date="2026-04-30",
        split_name="baseline",
    )

    splits_dir = data_dir / "splits"
    train = pd.read_parquet(splits_dir / "baseline_train.parquet")
    test = pd.read_parquet(splits_dir / "baseline_test.parquet")

    assert train["value"].tolist() == [1, 2]
    assert test["value"].tolist() == [3, 4]
    assert metadata == {
        "train_rows": 2,
        "test_rows": 2,
        "train_start": pd.Timestamp("2026-01-15"),
        "train_end": pd.Timestamp("2026-02-15"),
        "test_start": pd.Timestamp("2026-03-15"),
        "test_end": pd.Timestamp("2026-04-15"),
    }


@pytest.mark.unit
def test_load_and_validate_training_splits_reads_and_prepares_parquet_files(
    data_dir: Path,
    generated_training_df: pd.DataFrame,
) -> None:
    split_dir = data_dir / "splits"
    split_dir.mkdir()
    df = generated_training_df.copy()
    df["Reference Date"] = pd.date_range("2026-01-01", periods=len(df))
    df.iloc[:6].to_parquet(split_dir / "baseline_train.parquet", index=False)
    df.iloc[6:].to_parquet(split_dir / "baseline_test.parquet", index=False)

    train_df, test_df = splitting.load_and_validate_training_splits(
        split_name="baseline",
        split_dir=split_dir,
    )

    assert len(train_df) == 6
    assert len(test_df) == len(df) - 6
    assert "reference_date" not in train_df.columns
    assert "reference_date" not in test_df.columns
    assert "usage_frequency" in train_df.columns
    assert "usage_frequency" in test_df.columns


@pytest.mark.unit
@pytest.mark.parametrize(
    ("start_date", "split_date", "end_date", "message"),
    [
        ("2026-02-01", "2026-02-01", "2026-04-30", "later than"),
        ("2026-01-01", "2026-05-01", "2026-04-30", "before or equal"),
    ],
)
def test_create_time_based_split_rejects_invalid_date_bounds(
    start_date: str,
    split_date: str,
    end_date: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        create_time_based_split(
            dataset_name="events",
            timestamp_column="reference_date",
            start_date=start_date,
            split_date=split_date,
            end_date=end_date,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("timestamps", "message"),
    [
        (["2026-03-01", "2026-04-01"], "Training split is empty"),
        (["2026-01-01", "2026-01-15"], "Test split is empty"),
    ],
)
def test_create_time_based_split_rejects_empty_sides(
    monkeypatch: pytest.MonkeyPatch,
    timestamps: list[str],
    message: str,
) -> None:
    df = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(timestamps),
            "value": range(len(timestamps)),
        }
    )
    monkeypatch.setattr(splitting, "read_partitioned_dataset", lambda **_: df)

    with pytest.raises(ValueError, match=message):
        create_time_based_split(
            dataset_name="events",
            timestamp_column="reference_date",
            start_date="2026-01-01",
            split_date="2026-02-01",
            end_date="2026-04-30",
        )
