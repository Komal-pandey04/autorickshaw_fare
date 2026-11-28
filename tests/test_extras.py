"""Tests for the extra features: explain, batch, model-info, metrics, PSI status, categorical PSI."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.fare_pipeline.explain import explain_prediction
from src.fare_pipeline.monitoring import categorical_psi, psi_status

VALID = {"distance_km": 12.5, "wait_min": 5.0, "passengers": 2, "hour": 14,
         "luggage_bags": 1, "fuel_price_rs": 100.0, "booking": "app", "zone": "Market"}


@pytest.fixture
def client(tiny_model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tiny_model_path))
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_explain_effects_reconcile_and_distance_dominates(client):
    r = client.post("/predict/explain", json=VALID)
    assert r.status_code == 200
    d = r.json()
    total = d["baseline_fare"] + sum(d["effects_rs"].values()) + d["other_effects_rs"]
    assert total == pytest.approx(d["prediction"], abs=0.1)
    assert next(iter(d["effects_rs"])) == "distance_km"      # sorted by size; distance drives the tiny model


def test_explain_same_input_as_baseline_has_zero_effects(tiny_model_path, params):
    import joblib
    model = joblib.load(tiny_model_path)
    base = params["explain"]["baseline"]
    out = explain_prediction(model, base, base, params["model"]["feature_columns"])
    assert all(v == 0 for v in out["effects_rs"].values()) and out["prediction"] == out["baseline_fare"]


def test_explain_applies_client_rules(client):
    assert client.post("/predict/explain", json={**VALID, "distance_km": 41}).status_code == 422


def test_batch_matches_single_predictions(client):
    rides = [VALID, {**VALID, "distance_km": 3.0, "zone": "Outskirts"}]
    batch = client.post("/predict/batch", json={"rides": rides}).json()
    singles = [client.post("/predict", json=r).json()["prediction"] for r in rides]
    assert batch["count"] == 2 and batch["predictions"] == singles


def test_batch_rejects_bad_ride_empty_and_too_large(client):
    assert client.post("/predict/batch", json={"rides": [{**VALID, "wait_min": -1}]}).status_code == 422
    assert client.post("/predict/batch", json={"rides": []}).status_code == 422
    assert client.post("/predict/batch", json={"rides": [VALID] * 101}).status_code == 422


def test_model_info_and_metrics(client):
    info = client.get("/model-info").json()
    assert info["model_type"] == "Ridge" and "distance_km" in info["features"]
    client.post("/predict", json=VALID)
    m = client.get("/metrics").json()
    assert m["requests"] >= 1 and m["predictions"] >= 1 and m["avg_latency_ms"] >= 0


def test_psi_status_labels(params):
    th = params["explain"]["psi_thresholds"]
    assert [psi_status(v, th) for v in (0.03, 0.15, 3.4)] == ["stable", "moderate", "significant"]


def test_categorical_psi_zero_when_same_and_positive_when_mix_changes():
    a = pd.Series(["Market"] * 50 + ["Outskirts"] * 50)
    assert categorical_psi(a, a) == pytest.approx(0, abs=1e-9)
    b = pd.Series(["Market"] * 90 + ["Outskirts"] * 10)
    assert categorical_psi(a, b) > 0.25
