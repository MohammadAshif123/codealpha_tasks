import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create financial-history features required for the project."""
    data = df.copy()

    # Avoid division-by-zero.
    safe_income = data["income"].replace(0, np.nan)

    data["debt_to_income"] = data["debt"] / safe_income
    data["loan_to_income"] = data["loan_amount"] / safe_income
    data["savings_to_income"] = data["savings"] / safe_income

    return data


def build_preprocessor():
    engineered_numeric = NUMERIC_FEATURES + [
        "debt_to_income",
        "loan_to_income",
        "savings_to_income",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, engineered_numeric),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
