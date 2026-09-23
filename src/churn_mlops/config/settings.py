from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# direcory names
RAW_DATA_DIR = PROJECT_ROOT / "data/raw"
CONFIG_DIR = PROJECT_ROOT / "src/config"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts_local"
TRACKING_DIR = PROJECT_ROOT / "tracking_local"
TMP_DIR = PROJECT_ROOT / "tmp"


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    mlflow_tracking_uri: str | None = Field(default=None, alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(default="test", alias="MLFLOW_EXPERIMENT_NAME")
    raw_data_dir: Path = Field(default=RAW_DATA_DIR, alias="RAW_DATA_DIR")
    config_dir: Path = Field(default=CONFIG_DIR, alias="CONFIG_DIR")
    artifact_dir: Path = Field(default=ARTIFACT_DIR, alias="ARTIFACT_DIR")
    tracking_dir: Path = Field(default=TRACKING_DIR, alias="TRACKING_DIR")
    tmp_dir: Path = Field(default=TMP_DIR, alias="TMP_DIR")
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    def model_post_init(self, __context: object, /) -> None:
        if self.mlflow_tracking_uri is None:
            self.mlflow_tracking_uri = f"sqlite:///{self.tracking_dir / 'mlflow.db'}"


class ServingSettings(RuntimeSettings):
    model_name: str = Field(default="churn-propensity", alias="MODEL_NAME")
    model_alias: str = Field(default="champion", alias="MODEL_ALIAS")
