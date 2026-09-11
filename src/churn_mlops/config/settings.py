from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# direcory names
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CONFIG_DIR = PROJECT_ROOT / "src/config"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
