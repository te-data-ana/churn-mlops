from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# direcory names
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CONFIG_DIR = PROJECT_ROOT / "src/config"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
TRACKING_DIR = PROJECT_ROOT / "tracking"
TMP_DIR = PROJECT_ROOT / "tmp"


class ServingSettings(BaseSettings):
    model_name: str = Field(default="churn-propensity", alias="MODEL_NAME")
    model_alias: str = Field(default="champion", alias="MODEL_ALIAS")
