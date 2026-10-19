
"""FastAPI service for Musanze Harvest Intelligence."""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

VERIFICATION_CODE = "AI-A2-8164"

FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
]


app = FastAPI(
    title="Musanze Harvest Intelligence Service",
    description=(
        "Machine learning decision-support prototype "
        "using synthetic potato farm data."
    ),
    version="1.0.0",
)



class FarmInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rainfall_mm: float = Field(ge=600, le=1400)
    avg_temp_c: float = Field(ge=12, le=26)
    soil_moisture_pct: float = Field(ge=20, le=80)
    fertilizer_kg: float = Field(ge=0, le=250)
    field_size_ha: float = Field(ge=0.2, le=4)
    season: Literal["A", "B"]
    cooperative_support: Literal[0, 1]



def load_model(filename):
    """Load a saved model or model bundle."""
    path = MODELS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    return joblib.load(path)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "verification_code": VERIFICATION_CODE,
    }


@app.get("/model-info")
def model_info():
    return {
        "service": "Musanze Harvest Intelligence Service",
        "verification_code": VERIFICATION_CODE,
        "available_models": [
            path.name
            for path in sorted(MODELS_DIR.glob("*.joblib"))
        ],
        "features": FEATURES,
        "disclaimer": (
            "Synthetic data for classroom use only."
        ),
    }


@app.post("/predict")
def predict(farm: FarmInput):
    """Predict yield, risk classification and farm cluster."""

    record = farm.model_dump()

    X = pd.DataFrame(
        [record],
        columns=FEATURES,
    )

    # Load existing trained models.
    regression_model = load_model(
        "linear_regression.joblib"
    )

    classification_model = load_model(
        "classification_model.joblib"
    )

    clustering_bundle = load_model(
        "clustering_model.joblib"
    )

    # 1. Regression prediction
    if hasattr(regression_model, "feature_names_in_"):
        regression_features = list(
            regression_model.feature_names_in_
        )
        regression_input = X[regression_features]
    else:
        regression_input = X

    yield_estimate = float(
        regression_model.predict(regression_input)[0]
    )

    # 2. Classification prediction
    risk_prediction = int(
        classification_model.predict(X)[0]
    )

    risk_probability = float(
        classification_model.predict_proba(X)[0][1]
    )

    # 3. Clustering prediction
    cluster_features = clustering_bundle["features"]

    cluster_input = pd.DataFrame(
        [
            {
                feature: record[feature]
                for feature in cluster_features
            }
        ]
    )

    transformed = clustering_bundle[
        "preprocessor"
    ].transform(cluster_input)

    cluster_id = int(
        clustering_bundle["model"].predict(
            transformed
        )[0]
    )

    # 4. Return prediction
    return {
        "yield_estimate_tons": round(
            yield_estimate, 4
        ),
        "high_risk_prediction": risk_prediction,
        "high_risk_label": (
            "high risk"
            if risk_prediction == 1
            else "not high risk"
        ),
        "high_risk_probability": round(
            risk_probability, 4
        ),
        "cluster": cluster_id,
        "verification_code": VERIFICATION_CODE,
        "note": (
            "Classroom decision-support prototype "
            "using synthetic data; not a substitute "
            "for an agronomist."
        ),
    }
