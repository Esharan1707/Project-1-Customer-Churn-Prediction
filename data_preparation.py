"""Step 1: load the data and clean it."""
import pandas as pd

from config import DATA_DIR, DATA_FILE_NAME, BASE_DIR


def find_data_file():
    """Return the path of the dataset, or explain clearly what to do if it is missing."""
    preferred = DATA_DIR / DATA_FILE_NAME
    if preferred.exists():
        return preferred
    for folder in (DATA_DIR, BASE_DIR):
        csv_files = sorted(folder.glob("*.csv"))
        if csv_files:
            return csv_files[0]
    raise FileNotFoundError(
        "Dataset not found. Download the 'Telco Customer Churn' CSV from Kaggle and put it in the "
        f"folder: {DATA_DIR}"
    )


def load_data():
    path = find_data_file()
    print(f"Loading data from: {path.name}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    print("Rows and columns:", df.shape)
    return df


def clean_data(df):
    df = df.copy()

    # TotalCharges is stored as text because a few rows are blank. Convert it to numbers.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    print("Missing TotalCharges before filling:", int(df["TotalCharges"].isna().sum()))

    # Those blank rows are customers with tenure = 0 (not billed yet), so 0 is a sensible fill value.
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df = df.drop_duplicates()
    df = df.drop(columns=["customerID"])                      # an ID does not help prediction
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})        # target as 1 / 0
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    print("Missing values left:", int(df.isna().sum().sum()))
    print("Churn share:", df["Churn"].value_counts(normalize=True).round(3).to_dict())
    return df
