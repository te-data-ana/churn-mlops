from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from churn_mlops.data.storage import (
    _partition_filters,
    read_partitioned_dataset,
    write_partitioned_dataset,
)


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        "churn_mlops.data.storage.RuntimeSettings",
        lambda: SimpleNamespace(data_dir=tmp_path),
    )
    return tmp_path


@pytest.mark.unit
def test_write_and_read_partitioned_dataset_filters_inclusive_dates(
    data_dir: Path,
) -> None:
    df = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(
                [
                    "2026-01-31",
                    "2026-02-01",
                    "2026-02-28",
                    "2026-02-28 23:59:59.999999",
                    "2026-03-01",
                ],
                format="mixed",
            ),
            "value": [1, 2, 3, 4, 5],
        }
    )

    write_partitioned_dataset(
        df=df,
        dataset_name="events",
        timestamp_column="reference_date",
    )

    dataset_dir = data_dir / "events"
    assert (dataset_dir / "year=2026" / "month=1").is_dir()
    assert (dataset_dir / "year=2026" / "month=2").is_dir()
    assert (dataset_dir / "year=2026" / "month=3").is_dir()

    result = read_partitioned_dataset(
        dataset_name="events",
        timestamp_column="reference_date",
        start="2026-02-01",
        end="2026-02-28",
    )

    assert result["value"].tolist() == [2, 3, 4]
    assert result["reference_date"].tolist() == list(
        pd.to_datetime(
            ["2026-02-01", "2026-02-28", "2026-02-28 23:59:59.999999"],
            format="mixed",
        )
    )


@pytest.mark.unit
def test_write_partitioned_dataset_replaces_only_matching_partitions(
    data_dir: Path,
) -> None:
    write_partitioned_dataset(
        df=pd.DataFrame(
            {
                "reference_date": pd.to_datetime(["2026-01-05", "2026-02-05"]),
                "value": ["old-january", "keep-february"],
            }
        ),
        dataset_name="events",
        timestamp_column="reference_date",
    )
    write_partitioned_dataset(
        df=pd.DataFrame(
            {
                "reference_date": pd.to_datetime(["2026-01-20"]),
                "value": ["new-january"],
            }
        ),
        dataset_name="events",
        timestamp_column="reference_date",
        overwrite_partitions=True,
    )

    result = read_partitioned_dataset(
        dataset_name="events",
        timestamp_column="reference_date",
    )

    assert set(result["value"]) == {"new-january", "keep-february"}


@pytest.mark.unit
def test_write_partitioned_dataset_skips_empty_dataframe(data_dir: Path) -> None:
    write_partitioned_dataset(
        df=pd.DataFrame(),
        dataset_name="empty",
        timestamp_column="reference_date",
    )

    assert not (data_dir / "empty").exists()


@pytest.mark.unit
def test_partition_filters_include_every_month_in_range() -> None:
    assert _partition_filters(
        pd.Timestamp("2025-12-15"),
        pd.Timestamp("2026-02-01"),
    ) == [
        [("year", "==", 2025), ("month", "==", 12)],
        [("year", "==", 2026), ("month", "==", 1)],
        [("year", "==", 2026), ("month", "==", 2)],
    ]


@pytest.mark.unit
def test_partition_filters_reject_reversed_dates() -> None:
    with pytest.raises(ValueError, match="before or equal"):
        _partition_filters(pd.Timestamp("2026-02-01"), pd.Timestamp("2026-01-01"))


@pytest.mark.unit
def test_read_partitioned_dataset_requires_both_date_bounds(data_dir: Path) -> None:
    (data_dir / "events").mkdir()

    with pytest.raises(ValueError, match="both be provided"):
        read_partitioned_dataset(
            dataset_name="events",
            timestamp_column="reference_date",
            start="2026-01-01",
        )


@pytest.mark.unit
def test_read_partitioned_dataset_rejects_missing_dataset(data_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset does not exist"):
        read_partitioned_dataset(
            dataset_name="missing",
            timestamp_column="reference_date",
        )
