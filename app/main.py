"""Prediction service: POST /predict and GET /health."""
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.schemas import PredictRequest, PredictResponse
from src.fare_pipeline.config import load_params
from src.fare_pipeline.logging_config import get_logger

logger = get_logger()
ROOT = Path(__file__).resolve().parents[1]
FEATURES = load_params()["model"]["feature_columns"]


def model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", ROOT / load_params()["paths"]["model_out"]))


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = model_path()
    app.state.model = joblib.load(path) if path.exists() else None
    logger.info("startup", extra={"extra_fields": {"model_path": str(path), "model_loaded": app.state.model is not None}})
    yield


app = FastAPI(title="MP010 Auto-Rickshaw Fare API", version="1.0", lifespan=lifespan)


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
    logger.info("prediction", extra={"extra_fields": {
        "inputs": req.model_dump(), "prediction": fare,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2)}})
    return PredictResponse(prediction=fare)
