from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "credit_data.csv"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

TARGET = "creditworthy"

NUMERIC_FEATURES = [
    "age",
    "income",
    "employment_years",
    "debt",
    "loan_amount",
    "credit_history_years",
    "num_loans",
    "late_payments",
    "credit_utilization",
    "savings",
]

CATEGORICAL_FEATURES = [
    "employment_status",
    "payment_history",
]
