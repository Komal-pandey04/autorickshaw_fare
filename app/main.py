"""Prediction service: POST /predict and GET /health."""
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from app.schemas import BatchRequest, BatchResponse, ExplainResponse, PredictRequest, PredictResponse
from src.fare_pipeline.config import load_params
from src.fare_pipeline.explain import explain_prediction
from src.fare_pipeline.logging_config import get_logger

logger = get_logger()
ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params()
FEATURES = PARAMS["model"]["feature_columns"]
STARTED = time.time()
STATS = {"requests": 0, "errors": 0, "latency_ms_total": 0.0, "predictions": 0}


def model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", ROOT / load_params()["paths"]["model_out"]))


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = model_path()
    app.state.model = joblib.load(path) if path.exists() else None
    logger.info("startup", extra={"extra_fields": {"model_path": str(path), "model_loaded": app.state.model is not None}})
    yield


app = FastAPI(title="MP010 Auto-Rickshaw Fare API", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def count_requests(request: Request, call_next):
    """Tiny in-memory counters for /metrics (per pod; Prometheus would scrape each replica)."""
    start = time.perf_counter()
    response = await call_next(request)
    STATS["requests"] += 1
    STATS["errors"] += int(response.status_code >= 400)
    STATS["latency_ms_total"] += (time.perf_counter() - start) * 1000
    return response


@app.get("/", include_in_schema=False)
def frontend():
    """The small web page (app/static/index.html); it calls POST /predict on this same server."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    start = time.perf_counter()
    row = pd.DataFrame([req.model_dump()])[FEATURES]
    fare = round(float(model.predict(row)[0]), 2)
    STATS["predictions"] += 1
    logger.info("prediction", extra={"extra_fields": {
        "inputs": req.model_dump(), "prediction": fare,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2)}})
    return PredictResponse(prediction=fare)


def _loaded_model():
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return app.state.model


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    """Predict up to 100 rides in one call; one log line per call (counts only, no ride details)."""
    model = _loaded_model()
    rows = pd.DataFrame([r.model_dump() for r in req.rides])[FEATURES]
    preds = [round(float(x), 2) for x in model.predict(rows)]
    STATS["predictions"] += len(preds)
    logger.info("batch_prediction", extra={"extra_fields": {"rows": len(preds)}})
    return BatchResponse(predictions=preds, count=len(preds))


@app.post("/predict/explain", response_model=ExplainResponse)
def predict_explain(req: PredictRequest):
    """Fare plus 'why': how much each input moves it versus a typical ride."""
    model = _loaded_model()
    out = explain_prediction(model, req.model_dump(), PARAMS["explain"]["baseline"], FEATURES)
    logger.info("explanation", extra={"extra_fields": {"prediction": out["prediction"]}})
    return ExplainResponse(**out)


@app.get("/model-info")
def model_info():
    """Which model is serving (final estimator type and the input columns it expects)."""
    model = _loaded_model()
    return {"model_type": type(model.steps[-1][1]).__name__, "features": FEATURES,
            "client_rules": {"distance_km_max": 40, "wait_min_min": 0}}


@app.get("/metrics")
def metrics():
    n = STATS["requests"]
    return {"uptime_s": round(time.time() - STARTED, 1), "requests": n, "errors": STATS["errors"],
            "avg_latency_ms": round(STATS["latency_ms_total"] / n, 2) if n else 0.0,
            "predictions": STATS["predictions"]}
