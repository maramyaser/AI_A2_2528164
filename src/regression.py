
"""Train and evaluate regression models for crop yield prediction."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor, split_data

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ROOT / "models"


def evaluate_regression_models() -> dict:
    """Fit the regression candidates and compare their test metrics."""

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the cleaned dataset and create the fixed train/test split.
    from src.validation import validate_csv

    cleaned_df = validate_csv(ROOT / "data" / "AI_A2_2528164.csv")
    data = split_data(cleaned_df)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_reg_train"]
    y_test = data["y_reg_test"]

    candidates = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=8164,
            n_jobs=-1,
        ),
    }

    results = {}
    fitted_models = {}

    for name, estimator in candidates.items():
        model = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", estimator),
        ])

        # Fit all preprocessing steps using training data only.
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = float(mean_absolute_error(y_test, predictions))
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = float(r2_score(y_test, predictions))

        results[name] = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
        }
        fitted_models[name] = model

        print(f"\n{name}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R-squared: {r2:.4f}")

    # Choose the candidate with the lowest test RMSE, excluding the baseline.
    selected_name = min(
        ["linear_regression", "random_forest"],
        key=lambda name: results[name]["RMSE"],
    )

    # Save every fitted model for reproducibility.
    for name, model in fitted_models.items():
        dump(model, MODELS_DIR / f"{name}.joblib")

    report = {
        "seed": 8164,
        "target": "yield_tons",
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "metrics": results,
        "selected_model": selected_name,
        "selection_reason": (
            "Lowest test RMSE among the two candidate models; "
            "the mean baseline is included as a reference."
        ),
    }

    output_path = ARTIFACTS_DIR / "regression_metrics.json"
    output_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    # Save selected-model predictions for later evaluation/reporting.
    prediction_df = pd.DataFrame({
        "actual_yield_tons": y_test.to_numpy(),
        "predicted_yield_tons": fitted_models[selected_name].predict(X_test),
    })
    prediction_df.to_csv(
        ARTIFACTS_DIR / "regression_test_predictions.csv",
        index=False,
        float_format="%.4f",
    )

    print("\nSelected candidate:", selected_name)
    print("Saved metrics:", output_path)
    print("Saved models:", MODELS_DIR)
    print("Saved predictions:", ARTIFACTS_DIR / "regression_test_predictions.csv")

    return report


if __name__ == "__main__":
    evaluate_regression_models()
