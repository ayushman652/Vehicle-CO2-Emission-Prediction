from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

WORKSPACE_DIR = BASE_DIR.parent

DATASET_PATH = (
    WORKSPACE_DIR/"datasets"/"co2"/"FuelConsumptionCo2.csv"
)

OUTPUTS_DIR = BASE_DIR/"outputs"