import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

sys.path.append(str(Path(__file__).resolve().parent))

from config import DATA_PATH, MODEL_DIR, REPORT_DIR, FIGURE_DIR, TARGET
from preprocess import add_engineered_features, build_preprocessor


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = add_engineered_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    roc_data = {}

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
        }
        results.append(metrics)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, metrics["ROC-AUC"])

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cbar=False,
            xticklabels=["Not Creditworthy", "Creditworthy"],
            yticklabels=["Not Creditworthy", "Creditworthy"],
        )
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=160)
        plt.close()

        with open(REPORT_DIR / f"{name.lower().replace(' ', '_')}_classification_report.txt", "w") as f:
            f.write(classification_report(
                y_test,
                y_pred,
                target_names=["Not Creditworthy", "Creditworthy"],
                zero_division=0,
            ))

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    # ROC curve comparison
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "roc_auc_comparison.png", dpi=160)
    plt.close()

    # Train and save the best model on the full dataset.
    best_name = results_df.iloc[0]["Model"]
    best_estimator = models[best_name]

    best_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", best_estimator),
        ]
    )
    best_pipeline.fit(X, y)

    joblib.dump(best_pipeline, MODEL_DIR / "best_credit_scoring_model.joblib")

    summary = {
        "best_model": best_name,
        "dataset_rows": int(len(df)),
        "creditworthy_rate": float(y.mean()),
        "metrics": results_df.to_dict(orient="records"),
    }

    with open(REPORT_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete.")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'best_credit_scoring_model.joblib'}")
    print(f"Reports: {REPORT_DIR}")


if __name__ == "__main__":
    main()
