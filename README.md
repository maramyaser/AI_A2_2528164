# Musanze Harvest Intelligence Service

**Student:** Muram Yaser Basher  
**Registration number:** 25/28164  
**Institutional username:** ug2528164@ines.ac.rw  
**Verification code:** AI-A2-8164

## 1. Project Overview

This individual SWE 3513 Artificial Intelligence Assignment 2 implements a reproducible machine-learning pipeline and FastAPI service for a simulated potato cooperative in Musanze, Rwanda.

The system demonstrates:

1. Regression to estimate harvest quantity.
2. Classification to predict high production risk.
3. Clustering to group farms with similar recorded conditions.
4. Command-line and API inference using saved models.

**Dataset status:** Synthetic; not real farm observations.

## 2. Requirements

- Windows
- Python 3
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- joblib
- FastAPI
- Uvicorn
- pytest
- HTTPX

## 3. Environment Setup

Open PowerShell in the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use an appropriate local execution-policy setting or run the environment's Python executable directly.

## 4. Reproduce the Project

Run the complete pipeline using the student's identity and seed:

```powershell
python run_all.py --student-id ug2528164@ines.ac.rw --seed 8164
```

The pipeline should regenerate the dataset, train and evaluate the models, and recreate the required artifacts. Confirm that this command works from a clean project state before submission.

## 5. Command-Line Prediction

Run a prediction for a valid JSON record:

```powershell
python predict.py --json samples/valid.json
```

The expected response fields are:

- Estimated yield in tons.
- High-risk prediction.
- High-risk probability.
- Farm cluster.
- Verification code.

## 6. Automated Tests

Run the tests:

```powershell
pytest -q
```

The test suite covers valid prediction, missing fields, incorrect types, out-of-range inputs, reproducibility, and API responses. Record the actual test results before submission.

## 7. Start the API

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
```

Available endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/model-info` | Model and feature information |
| POST | `/predict` | Farm yield, risk, and cluster predictions |

Interactive documentation: `http://127.0.0.1:8000/docs`

## 8. Important Artifacts

- `artifacts/data_fingerprint.txt`
- `artifacts/regression_metrics.json`
- `artifacts/classification_metrics.json`
- `artifacts/clustering_report.json`
- `artifacts/test_predictions.csv` — create or verify this required artifact before submission.
- `artifacts/plots/residuals.png` — create or verify this required plot before submission.
- `artifacts/plots/confusion_matrix.png`
- `artifacts/plots/clusters.png`
- `models/` — saved models and preprocessing objects.

Only list artifacts as complete after confirming they exist and contain the expected information.

## 9. Responsible AI

The system is a classroom prototype using synthetic data. Its predictions are not validated agricultural advice and must not be the sole basis for financial, insurance, public-service, or other consequential decisions. Human review is required. See `docs/RESPONSIBLE_AI.md`.

## 10. Repository and Submission Information

- Repository URL: **Replace with your actual repository URL**
- Final commit hash: **Fill in after your final commit**
- Operating system tested: Windows
- Python version tested: **Fill in the version from `python --version`**

## 11. Known Limitations

Document any remaining differences between the assignment requirements and the implementation, including any missing artifacts or features. Do not claim full reproducibility until the complete pipeline has been tested twice and the generated dataset fingerprint and metrics have been checked.
