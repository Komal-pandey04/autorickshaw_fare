"""Shared fixtures: tiny in-memory data and model, so no test needs the (DVC-tracked) CSVs."""
import joblib
import numpy as np
import pandas as pd
import pytest

from src.fare_pipeline.config import load_params
from src.fare_pipeline.modeling import build_model


@pytest.fixture(scope="session")
def params():
    return load_params()


@pytest.fixture(scope="session")
def tiny_df():
    r = np.random.default_rng(1)
    n = 120
    d = pd.DataFrame({
        "distance_km": r.uniform(1, 25, n).round(1), "wait_min": r.uniform(0, 30, n).round(1),
        "passengers": r.integers(1, 5, n), "hour": r.integers(5, 23, n),
        "luggage_bags": r.integers(0, 5, n), "fuel_price_rs": r.uniform(90, 110, n).round(2),
        "booking": r.choice(["app", "stand", "street"], n),
        "zone": r.choice(["Market", "Outskirts", "Station Area"], n)})
    d["fare_rs"] = (20 + 7 * d.distance_km + 20 * d.luggage_bags + r.normal(0, 5, n)).round()
    return d


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory, tiny_df, params):
    cfg = {"name": "ridge", "type": "ridge", "alpha": 1.0}
    m = build_model(cfg, params, 0).fit(tiny_df[params["model"]["feature_columns"]], tiny_df["fare_rs"])
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(m, path)
    return path
