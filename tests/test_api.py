"""API tests with FastAPI's TestClient (a tiny model is built in conftest.py)."""
import pytest
from fastapi.testclient import TestClient

VALID = {"distance_km": 12.5, "wait_min": 5.0, "passengers": 2, "hour": 14,
         "luggage_bags": 1, "fuel_price_rs": 100.0, "booking": "app", "zone": "Market"}


@pytest.fixture
def client(tiny_model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tiny_model_path))
    from app.main import app
    with TestClient(app) as c:     # the `with` runs the startup that loads the model
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_frontend_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Estimate fare" in r.text


def test_predict_ok(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    assert isinstance(r.json()["prediction"], float)


def test_distance_above_40_rejected(client):
    assert client.post("/predict", json={**VALID, "distance_km": 40.1}).status_code == 422


def test_distance_exactly_40_accepted(client):
    assert client.post("/predict", json={**VALID, "distance_km": 40}).status_code == 200


def test_negative_wait_rejected(client):
    assert client.post("/predict", json={**VALID, "wait_min": -0.1}).status_code == 422


@pytest.mark.parametrize("bad", [{"zone": "Mars"}, {"booking": "phone"}, {"extra": 1}])
def test_bad_values_and_extra_fields_rejected(client, bad):
    assert client.post("/predict", json={**VALID, **bad}).status_code == 422


def test_missing_field_rejected(client):
    body = {k: v for k, v in VALID.items() if k != "hour"}
    assert client.post("/predict", json=body).status_code == 422


def test_prediction_is_logged_as_json(client):
    import io, json, logging
    from src.fare_pipeline.logging_config import JsonFormatter, get_logger
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    logger = get_logger()
    logger.addHandler(handler)
    try:
        client.post("/predict", json=VALID)
    finally:
        logger.removeHandler(handler)
    line = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert line["message"] == "prediction" and "prediction" in line and line["inputs"]["zone"] == "Market"
