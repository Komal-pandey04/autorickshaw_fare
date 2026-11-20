# MP010 Auto-Rickshaw Fare (Lucknow): SCSE3040 MLOps Mini Project

**Group:** G001  **Problem:** MP010  **Faculty:** Dr. Amit Soni (Batch 1)

| Enrolment no. | Name |
|---|---|
| S24CSEU0010 | Khushi Nain |
| S24CSEU0004 | Kashish Pawar |
| S24CSEU0098 | Udhav Sarin |
| S24CSEU0047 | Kriti Dogra |
| S24CSEU0014 | Komal Pandey |

Predict `fare_rs` (rupees; regression, scored by MAE) for the Lucknow Auto
Drivers' Cooperative booking app. Problem text: `PROBLEM.md`; rules: `STUDENT_HANDBOOK.md`.

## Pipeline at a glance

```
data/raw (DVC) -> run_data_checks -> cleaning -> train (v1, MLflow) -> retrain (drift, v2, promotion)
                                                                   -> models/model.joblib -> FastAPI -> Docker -> Kubernetes
```

| Step | Code | Output |
|---|---|---|
| Data versioning | DVC (`data/raw/*.csv.dvc`) | CSVs stay out of git |
| 7 data checks | `src/fare_pipeline/data_checks.py`, `run_data_checks.py` | `reports/findings.yaml` (`data_issues`) |
| Cleaning | `cleaning.py` | drop duplicates, wait_min/60 for Outskirts, 999 hour -> median |
| v1 training + registry | `train.py`, `modeling.py`, `registry.py` | `mlflow.db`, version 1 |
| Drift, v2, promotion | `retrain.py`, `monitoring.py` | `reports/monitoring.json`, version 2, `@champion` |
| Holdout predictions | `retrain.py` | `predictions/holdout_predictions.csv` |
| Service | `app/main.py`, `app/schemas.py` | `POST /predict`, `GET /health`, JSON logs |
| Deploy | `Dockerfile`, `k8s/`, `.github/workflows/ci.yml` | `mp010-api:1.0`, 4 replicas |

## Data problems found in `train.csv`

| Problem | Result |
|---|---|
| leaking_column | absent: every column except `fare_rs` is in the holdout file, none mirrors the target |
| unit_mixup | `wait_min` in zone `Outskirts`, factor 60 (seconds instead of minutes) |
| sentinel_value | `hour` = 999, 49 rows |
| duplicate_rows | 55 extra copies |
| stuck_sensor | absent |
| target_typos | absent |
| spelling_variants | absent (`booking`, `zone`: 3 clean levels each) |

## Results

* **v1** (ridge, chosen on a validation split; trained on cleaned train only): MAE on `later.csv` = 42.8.
* **Drift:** most drifted column is `distance_km` (PSI 3.43; later rides are longer, up to 34.5 km vs 25 km in train).
* **Worst-served zone:** Market (v1 MAE 93.5 vs 21.4 Outskirts, 25.9 Station Area): Market's rate per km rose most.
* **v2** (settings re-selected on the newest slice of the retraining data, trained on train + first 70% of later):
  MAE on the last 30% = 21.9 vs v1 45.8, i.e. 52% lower. The rule needs at least 12%, so **v2 is promoted** to `@champion`.

## Run it

```bash
pip install -r requirements.txt            # add requirements-dev.txt for DVC
dvc pull   # or place the original train.csv / later.csv in data/raw/
./run_pipeline.sh                          # checks -> v1 -> drift/v2/promotion -> holdout (rebuilds mlflow.db)
pytest                                     # tests need no CSVs
uvicorn app.main:app --port 8000           # serve locally
```

Example:

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d \
 '{"distance_km":12.5,"wait_min":5,"passengers":2,"hour":14,"luggage_bags":1,"fuel_price_rs":100,"booking":"app","zone":"Market"}'
```

Client rules are enforced by the pydantic schema (`app/schemas.py`): `distance_km` above 40 and
negative `wait_min` return 422. Every prediction writes one JSON line to stdout.

## Deploy

```bash
docker build -t mp010-api:1.0 .
docker run -p 8000:8000 mp010-api:1.0
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml   # Deployment mp010-api x4, Service mp010-svc 80 -> 8000
```

## Layout

```
app/                FastAPI service (main.py, schemas.py)
src/fare_pipeline/  config, data_checks, cleaning, modeling, registry, train, monitoring, retrain, findings, logging_config
tests/              25 pytest tests (incl. TestClient), no CSVs needed
k8s/                deployment.yaml, service.yaml
models/model.joblib champion model loaded by the service
reports/            findings.yaml (the form), monitoring.json
predictions/        holdout_predictions.csv
params.yaml         every setting, read with yaml.safe_load
```
