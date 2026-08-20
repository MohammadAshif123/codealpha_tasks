import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve() / "src"))

from config import MODEL_DIR
from preprocess import add_engineered_features

st.set_page_config(page_title="Credit Scoring Model", page_icon="💳", layout="centered")

st.title("💳 Credit Scoring Model")
st.caption("CodeAlpha Machine Learning Internship — Task 1")

model_path = MODEL_DIR / "best_credit_scoring_model.joblib"

if not model_path.exists():
    st.warning("Model not found. Run `python src/train.py` first.")
    st.stop()

model = joblib.load(model_path)

st.subheader("Customer Financial Information")

age = st.number_input("Age", 18, 100, 30)
income = st.number_input("Annual Income", 0.0, 500000.0, 60000.0)
employment_years = st.number_input("Employment Years", 0.0, 50.0, 5.0)
debt = st.number_input("Current Debt", 0.0, 500000.0, 10000.0)
loan_amount = st.number_input("Requested Loan Amount", 0.0, 500000.0, 15000.0)
credit_history_years = st.number_input("Credit History Years", 0.0, 50.0, 6.0)
num_loans = st.number_input("Number of Loans", 0, 30, 2)
late_payments = st.number_input("Late Payments", 0, 30, 0)
credit_utilization = st.slider("Credit Utilization", 0.0, 1.0, 0.30)
savings = st.number_input("Savings", 0.0, 500000.0, 25000.0)
employment_status = st.selectbox(
    "Employment Status",
    ["Employed", "Self-employed", "Unemployed"],
)
payment_history = st.selectbox(
    "Payment History",
    ["Excellent", "Good", "Average", "Poor"],
)

if st.button("Predict Creditworthiness"):
    customer = pd.DataFrame([{
        "age": age,
        "income": income,
        "employment_years": employment_years,
        "debt": debt,
        "loan_amount": loan_amount,
        "credit_history_years": credit_history_years,
        "num_loans": num_loans,
        "late_payments": late_payments,
        "credit_utilization": credit_utilization,
        "savings": savings,
        "employment_status": employment_status,
        "payment_history": payment_history,
    }])

    customer = add_engineered_features(customer)

    prediction = int(model.predict(customer)[0])
    probability = float(model.predict_proba(customer)[0, 1])

    if prediction == 1:
        st.success(f"Creditworthy — probability: {probability:.2%}")
    else:
        st.error(f"Not Creditworthy — probability: {probability:.2%}")

st.info(
    "Educational ML project only. The prediction is not a real-world financial "
    "or lending decision."
)
