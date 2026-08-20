# CodeAlpha Task 1 — Credit Scoring Model

## 1. Objective

Predict an individual's creditworthiness using past financial data.

## 2. Approach

This project treats creditworthiness as a binary classification problem.

The following classification algorithms are implemented and compared:

- Logistic Regression
- Decision Tree
- Random Forest

## 3. Dataset

The repository contains a bundled synthetic financial dataset with 1,800 records.

The dataset includes financial-history features such as:

- Income
- Debt
- Loan amount
- Credit history
- Payment history
- Late payments
- Credit utilization
- Savings
- Employment information
- Number of loans

A synthetic dataset is included so the project is immediately executable without depending on an external download.

## 4. Feature Engineering

The following features are derived from financial history:

- Debt-to-Income Ratio
- Loan-to-Income Ratio
- Savings-to-Income Ratio

Missing numerical values are handled using median imputation, while missing categorical values are handled using the most frequent category.

Categorical features are one-hot encoded and numerical features are standardized.

## 5. Model Evaluation

The project evaluates the models using the metrics explicitly requested in the CodeAlpha task:

- Precision
- Recall
- F1-Score
- ROC-AUC

Additional evaluation includes:

- Accuracy
- Confusion Matrix
- Classification Report
- ROC curve comparison

## 6. Example Training Result

The bundled dataset produced the following test-set results in the prepared project environment:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8333 | 0.8297 | 0.8389 | 0.8343 | 0.9223 |
| Random Forest | 0.8333 | 0.8191 | 0.8556 | 0.8370 | 0.9009 |
| Decision Tree | 0.7833 | 0.7772 | 0.7944 | 0.7857 | 0.8372 |

The best model by ROC-AUC in this run was Logistic Regression.

These values may vary if the dataset, random seed, or model configuration is changed.

## 7. Conclusion

The project demonstrates how past financial information can be transformed into engineered features and used with classification algorithms to predict creditworthiness.

The implementation satisfies the CodeAlpha Task 1 requirements for:

1. Financial-history feature engineering
2. Classification algorithms
3. Precision
4. Recall
5. F1-Score
6. ROC-AUC
