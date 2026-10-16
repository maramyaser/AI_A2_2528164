
"""Train and evaluate high-risk farm classification models."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "AI_A2_2528164.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
MODELS_DIR = ROOT / "models"

SEED = 8164
TARGET = "high_risk"

NUMERIC_FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "cooperative_support",
]
CATEGORICAL_FEATURES = ["season"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_preprocessor():
    """Create preprocessing that is fitted on training data only."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def evaluate_model(name, model, X_test, y_test):
    """Calculate classification metrics and return them as a dictionary."""
    predictions = model.predict(X_test)

    result = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
    }

    if hasattr(model, "predict_proba") and len(set(y_test)) == 2:
        probabilities = model.predict_proba(X_test)[:, 1]
        result["roc_auc"] = float(roc_auc_score(y_test, probabilities))
    else:
        result["roc_auc"] = None

    return result


def main():
    """Train models, compare results and save the selected classifier."""
    for directory in (ARTIFACTS_DIR, PLOTS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run data/generate_data.py first."
        )

    df = pd.read_csv(DATA_PATH)
    required_columns = FEATURES + [TARGET]
    missing_columns = sorted(set(required_columns) - set(df.columns))

    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    # Remove the deliberately added duplicate rows.
    # Missing feature values are handled inside the training pipeline.
    df = df.drop_duplicates().reset_index(drop=True)

    X = df[FEATURES].copy()
    y = pd.to_numeric(df[TARGET], errors="raise").astype(int)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("high_risk must contain only 0 and 1.")

    if y.nunique() != 2:
        raise ValueError("Classification requires both risk classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=SEED,
            class_weight="balanced",
        ),
    }

    results = []
    fitted_models = {}

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("classifier", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        results.append(evaluate_model(name, pipeline, X_test, y_test))

    # Select using test-set F1 for this assignment's simple comparison.
    # Discuss the limitations of test-based model selection in the report.
    candidates = [
        result for result in results
        if result["model"] != "majority_baseline"
    ]
    selected_result = max(candidates, key=lambda result: result["f1"])
    selected_name = selected_result["model"]
    selected_model = fitted_models[selected_name]

    output = {
        "student_id": "ug2528164@ines.ac.rw",
        "seed": SEED,
        "verification_code": "AI-A2-8164",
        "target": TARGET,
        "features": FEATURES,
        "rows_after_duplicate_removal": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "results": results,
        "selected_model": selected_name,
        "selection_reason": (
            "Selected the non-baseline classifier with the highest "
            "test-set F1 score among the compared candidates."
        ),
        "warning": (
            "For a stronger evaluation, model selection should use a "
            "validation split or cross-validation, with the test set "
            "reserved for final evaluation."
        ),
    }

    metrics_path = ARTIFACTS_DIR / "classification_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    model_path = MODELS_DIR / "classification_model.joblib"
    joblib.dump(selected_model, model_path)

    # Save a confusion matrix for the selected model.
    predictions = selected_model.predict(X_test)
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ),
        display_labels=["Not high risk", "High risk"],
    )
    display.plot()
    plt.title(f"Confusion Matrix — {selected_name}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    print("Verification code: AI-A2-8164")
    print(f"Rows after duplicate removal: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print("\nClassification results:")

    for result in results:
        print(
            f"{result['model']}: "
            f"accuracy={result['accuracy']:.4f}, "
            f"precision={result['precision']:.4f}, "
            f"recall={result['recall']:.4f}, "
            f"F1={result['f1']:.4f}, "
            f"ROC-AUC={result['roc_auc']}"
        )

    print(f"\nSelected model: {selected_name}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Model saved to: {model_path}")
    print(f"Confusion matrix saved to: {PLOTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
