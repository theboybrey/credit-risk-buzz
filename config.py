# config.py
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "experiments" / "results"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset settings
DATASET_NAME = "lending_club_loan_data"
TARGET_COLUMN = "loan_status"  # Adjust based on actual column name

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Imbalance ratios to test
IMBALANCE_RATIOS = [
    "original",
    (0.8, 0.2),
    (0.5, 0.5),
]

# Models to compare
MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]

# Sampling strategies
SAMPLING_STRATEGIES = ["original", "smote", "undersample", "class_weight"]