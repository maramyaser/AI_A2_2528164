
"""Automated tests for Musanze Harvest Intelligence."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from predict import load_record

import json

client = TestClient(app)

VALID_RECORD = {
    "rainfall_mm": 950,
    "avg_temp_c": 18,
    "soil_moisture_pct": 55,
    "fertilizer_kg": 120,
    "field_size_ha": 1.5,
    "season": "A",
    "cooperative_support": 1,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_verification_code():
    response = client.get("/health")
    assert response.json()["verification_code"] == "AI-A2-8164"


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "available_models" in response.json()


def test_valid_prediction():
    response = client.post("/predict", json=VALID_RECORD)
    assert response.status_code == 200
    result = response.json()
    assert "yield_estimate_tons" in result
    assert "high_risk_prediction" in result
    assert "cluster" in result


def test_invalid_rainfall():
    record = VALID_RECORD.copy()
    record["rainfall_mm"] = 2000

    response = client.post("/predict", json=record)
    assert response.status_code == 422


def test_invalid_season():
    record = VALID_RECORD.copy()
    record["season"] = "C"

    response = client.post("/predict", json=record)
    assert response.status_code == 422


def test_missing_field():
    record = VALID_RECORD.copy()
    del record["fertilizer_kg"]

    response = client.post("/predict", json=record)
    assert response.status_code == 422


def test_cli_validation(tmp_path):
    file_path = tmp_path / "valid.json"
    file_path.write_text(
        json.dumps(VALID_RECORD),
        encoding="utf-8",
    )

    result = load_record(file_path)
    assert result == VALID_RECORD
