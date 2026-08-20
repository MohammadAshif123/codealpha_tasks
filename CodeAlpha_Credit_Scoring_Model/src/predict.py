import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from config import MODEL_DIR
from preprocess import add_engineered_features


def predict_creditworthiness(customer: dict):
    model = joblib.load(MODEL_DIR / "best_credit_scoring_model.joblib")
    df = pd.DataFrame([customer])

    # Apply the same feature engineering used during model training.
    df = add_engineered_features(df)

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0, 1])

    label = "Creditworthy" if prediction == 1 else "Not Creditworthy"
    return label, probability


if __name__ == "__main__":
    sample_customer = {
        "age": 32,
        "income": 65000,
        "employment_years": 7,
        "debt": 12000,
        "loan_amount": 18000,
        "credit_history_years": 8,
        "num_loans": 2,
        "late_payments": 0,
        "credit_utilization": 0.25,
        "savings": 30000,
        "employment_status": "Employed",
        "payment_history": "Excellent",
    }

    label, probability = predict_creditworthiness(sample_customer)
    print(f"Prediction: {label}")
    print(f"Probability of being creditworthy: {probability:.2%}")
