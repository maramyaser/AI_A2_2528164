
"""Predict harvest yield, farm risk and cluster from one JSON record."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
]

NUMERIC_FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "cooperative_support",
]


def load_record(json_path):
    """Load and validate a single JSON input record."""
    with open(json_path, "r", encoding="utf-8") as file:
        record = json.load(file)

    if not isinstance(record, dict):
        raise ValueError("Input JSON must contain one object.")

    missing = [feature for feature in FEATURES if feature not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    if set(record.keys()) != set(FEATURES):
        extra = sorted(set(record.keys()) - set(FEATURES))
        if extra:
            raise ValueError(f"Unexpected fields: {extra}")

    for feature in NUMERIC_FEATURES:
        value = record[feature]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{feature} must be a number.")
        if not np.isfinite(value):
            raise ValueError(f"{feature} must be finite.")

    
    ranges = {
        "rainfall_mm": (600, 1400),
        "avg_temp_c": (12, 26),
        "soil_moisture_pct": (20, 80),
        "fertilizer_kg": (0, 250),
        "field_size_ha": (0.2, 4),
        "cooperative_support": (0, 1),
    }

    for feature, (minimum, maximum) in ranges.items():
        value = record[feature]
        if not minimum <= value <= maximum:
            raise ValueError(
                f"{feature} must be between {minimum} and {maximum}."
            )

    if record["cooperative_support"] not in (0, 1):
        raise ValueError("cooperative_support must be 0 or 1.")

    if not isinstance(record["season"], str):
        raise ValueError("season must be a string.")

    if record["season"] not in ("A", "B"):
        raise ValueError("season must be 'A' or 'B'.")

    return record



def predict_yield(record):
    """Predict harvest yield using the saved regression model."""
    import joblib
    import pandas as pd

    model_path = MODELS_DIR / "linear_regression.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Regression model not found: {model_path}"
        )

    model = joblib.load(model_path)

    # Use the exact feature columns recorded by the trained model.
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
        missing = [name for name in features if name not in record]

        if missing:
            raise ValueError(
                f"Missing regression input features: {missing}"
            )

        X = pd.DataFrame(
            [{name: record[name] for name in features}],
            columns=features,
        )
    else:
        # Fallback for models trained without feature-name metadata.
        X = pd.DataFrame([record])

    prediction = model.predict(X)
    return float(prediction[0])


def main():
    parser = argparse.ArgumentParser(
        description="Predict harvest yield, risk and farm cluster."
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to a JSON file containing one farm record.",
    )
    args = parser.parse_args()

    try:
        record = load_record(args.json)
        X = pd.DataFrame([record], columns=FEATURES)

        classification_path = MODELS_DIR / "classification_model.joblib"
        clustering_path = MODELS_DIR / "clustering_model.joblib"

        if not classification_path.exists():
            raise FileNotFoundError(
                f"Classification model not found: {classification_path}"
            )

        if not clustering_path.exists():
            raise FileNotFoundError(
                f"Clustering model not found: {clustering_path}. "
                "Rerun python src/clustering.py."
            )

        classifier = joblib.load(classification_path)
        clustering_bundle = joblib.load(clustering_path)

        yield_estimate = predict_yield(record)

        risk_prediction = int(classifier.predict(X)[0])
        risk_probability = float(classifier.predict_proba(X)[0][1])

        cluster_input = pd.DataFrame(
            [{key: record[key] for key in clustering_bundle["features"]}]
        )
        transformed = clustering_bundle["preprocessor"].transform(
            cluster_input
        )
        cluster_id = int(
            clustering_bundle["model"].predict(transformed)[0]
        )

        output = {
            "yield_estimate_tons": round(yield_estimate, 4),
            "high_risk_prediction": risk_prediction,
            "high_risk_label": (
                "high risk" if risk_prediction == 1 else "not high risk"
            ),
            "high_risk_probability": round(risk_probability, 4),
            "cluster": cluster_id,
            "verification_code": "AI-A2-8164",
            "note": (
                "Classroom decision-support prototype using synthetic data; "
                "not a substitute for an agronomist."
            ),
        }

        print(json.dumps(output, indent=4))

    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
