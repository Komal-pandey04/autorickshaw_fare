import numpy as np
import pandas as pd

from src.fare_pipeline.cleaning import clean_train
from src.fare_pipeline.logging_config import JsonFormatter
from src.fare_pipeline.monitoring import psi


def raw(params):
    d = pd.DataFrame({
        "record_id": ["a", "b", "c", "c"], "date": ["2026-01-01"] * 4,
        "distance_km": [5.0, 6.0, 7.0, 7.0], "wait_min": [10.0, 600.0, 12.0, 12.0],
        "passengers": [1, 2, 3, 3], "hour": [8, 10, 999, 999], "luggage_bags": [0, 1, 2, 2],
        "fuel_price_rs": [100.0] * 4, "booking": ["app"] * 4,
        "zone": ["Market", "Outskirts", "Market", "Market"], "fare_rs": [50, 60, 70, 70]})
    return d


def test_clean_removes_duplicates_fixes_unit_and_sentinel(params):
    out = clean_train(raw(params), params)
    assert len(out) == 3                                             # one duplicate dropped
    assert out.loc[out.zone == "Outskirts", "wait_min"].iloc[0] == 10.0   # 600 s -> 10 min
    assert (out.hour != 999).all()


def test_psi_zero_for_same_data_and_large_for_shift():
    x = pd.Series(np.random.default_rng(0).uniform(0, 25, 1000))
    assert psi(x, x) < 1e-6
    assert psi(x, x + 10) > 0.25


def test_json_formatter_outputs_valid_json():
    import json, logging
    rec = logging.LogRecord("t", logging.INFO, "f", 1, "hello", None, None)
    assert json.loads(JsonFormatter().format(rec))["message"] == "hello"
