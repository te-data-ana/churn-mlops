from churn_mlops.data import load_raw_data


def test_load_data():
    df = load_raw_data("customer_churn_dataset-training.csv")

    assert len(df) > 0
