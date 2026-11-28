# MP010 Auto-Rickshaw Fare (Lucknow)

SCSE3040 MLOps Mini Project | **Group G001** | **Problem MP010** | Faculty: Dr. Amit Soni (Batch 1)

## Team

| Enrolment No. | Name |
|---|---|
| S24CSEU0004 | Kashish Pawar |
| S24CSEU0010 | Khushi Nain |
| S24CSEU0014 | Komal Pandey |
| S24CSEU0047 | Kriti Dogra |
| S24CSEU0098 | Udhav Sarin |

## About

Predicts the fare (in rupees) of a Lucknow auto-rickshaw ride for the Lucknow Auto Drivers' Cooperative booking app.
The pipeline cleans the data, tracks models with MLflow, serves predictions with FastAPI (plus a small web page),
runs in Docker and Kubernetes, and monitors drift and retrains on newer data.

## Run

```bash
pip install -r requirements.txt
pytest
uvicorn app.main:app --port 8000
```

Open http://localhost:8000 for the web page, or http://localhost:8000/docs for the API.

To rebuild everything (data checks, models, predictions): `./run_pipeline.sh`

## Docker and Kubernetes

```bash
docker build -t mp010-api:1.0 .
docker run -p 8000:8000 mp010-api:1.0
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
```

## Folders

- `app/` web page and API
- `src/fare_pipeline/` data checks, cleaning, training, drift, retraining
- `tests/` automatic tests
- `k8s/` Kubernetes files
- `reports/findings.yaml` data problems and monitoring results
- `predictions/` holdout predictions

## Extra features (added after the core pipeline)

| Feature | Where | What it does |
|---|---|---|
| "Why this fare?" | `POST /predict/explain`, `src/fare_pipeline/explain.py`, web page | Swaps each input for a typical ride's value, one at a time, and reports the Rs each input adds or removes. Works with any model. The typical ride is `explain.baseline` in `params.yaml`. |
| Batch prediction | `POST /predict/batch` | Up to 100 rides per call; every ride must pass the same client rules (distance <= 40, wait >= 0). Logs only the row count. |
| Model info | `GET /model-info` | Final estimator type, expected input columns and the client's two rules. |
| Service metrics | `GET /metrics` | Requests, errors, average latency and prediction count (in memory, per pod). |
| Drift severity | `monitoring.psi_status`, `categorical_psi` | Labels each PSI as stable (< 0.1), moderate (< 0.25) or significant, and measures drift of text columns (`zone`, `booking`). |
| Extended drift report | `python -m src.fare_pipeline.drift_report` | Writes `reports/drift_extended.json`: severity per column, columns needing attention, v1 error per booking type and zone. Needs the DVC data files; it does not touch `findings.yaml` or the registry. |

Tests for all of these are in `tests/test_extras.py` (they use the small in-memory model, no CSVs needed).
