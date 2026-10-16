
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app

client = TestClient(app)

VALID_RECORD = {
    "rainfall_mm": 120.0,
    "avg_temp_c": 18.5,
    "soil_moisture_pct": 55.0,
    "fertilizer_kg": 30.0,
    "field_size_ha": 2.5,
    "season": "A",
    "cooperative_support": 1,
}


def test_valid_prediction_cli():
    """A valid JSON record should produce a prediction."""
    sample_path = ROOT / "samples" / "valid.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "predict.py"), "--json", str(sample_path)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert "yield_estimate_tons" in output
    assert "high_risk_prediction" in output
    assert "high_risk_probability" in output
    assert "cluster" in output
    assert output["verification_code"] == "AI-A2-8164"


def test_missing_field_rejected():
    record = VALID_RECORD.copy()
    del record["rainfall_mm"]
    response = client.post("/predict", json=record)
    assert response.status_code == 422


def test_wrong_type_rejected():
    record = VALID_RECORD.copy()
    record["rainfall_mm"] = "a lot"
    response = client.post("/predict", json=record)
    assert response.status_code == 422


def test_out_of_range_rejected():
    record = VALID_RECORD.copy()
    record["rainfall_mm"] = -20
    response = client.post("/predict", json=record)
    assert response.status_code == 422


def test_seed_reproducibility():
    """Generating data twice with the same seed should give identical bytes."""
    generator = ROOT / "data" / "generate_data.py"
    output_path = ROOT / "data" / "AI_A2_2528164.csv"

    first = subprocess.run(
        [sys.executable, str(generator)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert first.returncode == 0, first.stderr
    first_bytes = output_path.read_bytes()

    second = subprocess.run(
        [sys.executable, str(generator)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert second.returncode == 0, second.stderr
    second_bytes = output_path.read_bytes()

    assert first_bytes == second_bytes


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["verification_code"] == "AI-A2-8164"
