from pathlib import Path

# globally accessible direcory names
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# globally accessible definition of data types
DTYPES = {
    "CustomerID": "Int64",
    "Age": "Int64",
    "Gender": "category",
    "Tenure": "Int64",
    "Usage Frequency": "Int64",
    "Support Calls": "Int64",
    "Payment Delay": "Int64",
    "Subscription Type": "category",
    "Contract Length": "category",
    "Total Spend": "Float64",
    "Last Interaction": "Int64",
    "Churn": "Int64",
}
