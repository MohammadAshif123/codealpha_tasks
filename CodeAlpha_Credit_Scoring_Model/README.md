# CodeAlpha — Credit Scoring Model

## Task 1: Credit Scoring Model

### Objective
Predict an individual's creditworthiness using past financial data.

### CodeAlpha requirements covered
- Classification algorithms: Logistic Regression, Decision Tree, Random Forest
- Feature engineering from financial history
- Evaluation using Precision, Recall, F1-Score and ROC-AUC
- Financial features such as income, debt, loan amount and payment history

## Project Structure

```text
CodeAlpha_Credit_Scoring_Model/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── credit_data.csv
│   └── processed/
├── models/
├── notebooks/
│   └── credit_scoring_model.ipynb
├── reports/
│   └── figures/
└── src/
    ├── config.py
    ├── preprocess.py
    ├── train.py
    └── predict.py
```

## Dataset

This repository includes a **synthetic financial dataset** so the project can be executed immediately without requiring an external dataset download.

The data contains:
- Income
- Debt
- Loan amount
- Credit history
- Number of loans
- Late payments
- Credit utilization
- Savings
- Employment status
- Payment history
- Creditworthiness target

For a production/academic extension, the bundled CSV can be replaced with a real, appropriately licensed credit dataset while keeping the same column structure.

## Feature Engineering

The project creates:
- Debt-to-Income Ratio
- Loan-to-Income Ratio
- Savings-to-Income Ratio

These features are calculated from the customer's financial history.

## Models

Three classification models are trained and compared:
1. Logistic Regression
2. Decision Tree
3. Random Forest

## Evaluation Metrics

The project calculates:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

It also generates:
- Confusion matrices
- ROC-AUC comparison graph
- Classification reports
- Model comparison CSV

## Installation

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the models

Run from the project root:

```bash
python src/train.py
```

After training, you will get:

```text
models/best_credit_scoring_model.joblib
reports/model_comparison.csv
reports/training_summary.json
reports/figures/
```

## Make a sample prediction

```bash
python src/predict.py
```

## Run the optional web interface

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/credit_scoring_model.ipynb
```

## Important note

This is an educational machine-learning project. It is not intended to make actual lending, banking, or financial decisions.

## Suggested GitHub repository name

```text
CodeAlpha_Credit_Scoring_Model
```
