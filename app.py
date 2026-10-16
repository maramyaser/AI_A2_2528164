
"""FastAPI service for farm yield and risk predictions."""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

app = FastAPI(
    title="SWE 3513 Farm Prediction API",
    description="Farm yield, high-risk classification, and clustering.",
    version="1.0.0",
)

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


class FarmInput(BaseModel):
    rainfall_mm: float = Field(ge=0, le=500)
    avg_temp_c: float = Field(ge=-5, le=45)
    soil_moisture_pct: float = Field(ge=0, le=100)
    fertilizer_kg: float = Field(ge=0, le=500)
    field_size_ha: float = Field(gt=0, le=100)
    season: str
    cooperative_support: int = Field(ge=0, le=1)


def load_models():
    """Load the models saved by the training scripts."""
    regression_path = MODELS_DIR / "linear_regression.joblib"
    classification_path = MODELS_DIR / "classification_model.joblib"
    clustering_path = MODELS_DIR / "clustering_model.joblib"

    paths = [regression_path, classification_path, clustering_path]
    missing = [str(path) for path in paths if not path.exists()]

    if missing:
        raise RuntimeError(
            "Missing model files: " + ", ".join(missing)
        )

    return (
        joblib.load(regression_path),
        joblib.load(classification_path),
        joblib.load(clustering_path),
    )


@app.get("/health")
def health():
    """Report whether the API process is running."""
    return {
        "status": "ok",
        "verification_code": "AI-A2-8164",
    }


@app.get("/model-info")
def model_info():
    """Describe the models used by this API."""
    return {
        "regression_model": "linear_regression.joblib",
        "classification_model": "classification_model.joblib",
        "clustering_model": "clustering_model.joblib",
        "features": FEATURES,
        "verification_code": "AI-A2-8164",
    }


@app.post("/predict")
def predict(data: FarmInput):
    """Return yield, risk, probability, and cluster predictions."""
    if data.season not in ("A", "B"):
        raise HTTPException(
            status_code=422,
            detail="season must be 'A' or 'B'.",
        )

    record = data.model_dump()
    X = pd.DataFrame([record], columns=FEATURES)

    try:
        regression, classifier, clustering = load_models()

        regression_features = (
            list(regression.feature_names_in_)
            if hasattr(regression, "feature_names_in_")
            else FEATURES
        )
        regression_X = pd.DataFrame(
            [{key: record[key] for key in regression_features}],
            columns=regression_features,
        )
        yield_estimate = float(regression.predict(regression_X)[0])

        risk = int(classifier.predict(X)[0])
        probability = float(classifier.predict_proba(X)[0][1])

        cluster_X = pd.DataFrame(
            [{key: record[key] for key in clustering["features"]}]
        )
        transformed = clustering["preprocessor"].transform(cluster_X)
        cluster_id = int(clustering["model"].predict(transformed)[0])

        return {
            "yield_estimate_tons": round(yield_estimate, 4),
            "high_risk_prediction": risk,
            "high_risk_label": (
                "high risk" if risk == 1 else "not high risk"
            ),
            "high_risk_probability": round(probability, 4),
            "cluster": cluster_id,
            "verification_code": "AI-A2-8164",
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}: {exc}",
        ) from exc
