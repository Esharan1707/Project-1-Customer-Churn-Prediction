"""Settings used by every file of the project."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

# Put the Telco Customer Churn CSV in the data folder. Any .csv file in that folder is accepted.
DATA_FILE_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

TARGET = "Churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
