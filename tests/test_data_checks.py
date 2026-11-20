"""Detectors must fire on planted problems and stay quiet on clean data."""
import numpy as np
import pandas as pd
import pytest

from src.fare_pipeline import data_checks as dc
from src.fare_pipeline.config import load_params


@pytest.fixture
def p():
    return load_params()


def make(n=300, seed=0):
    r = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=60).repeat(n // 60).astype(str)
    return pd.DataFrame({
        "record_id": [f"R{i}" for i in range(n)], "date": dates,
        "distance_km": r.uniform(1, 25, n).round(1), "wait_min": r.uniform(0, 30, n).round(1),
        "passengers": r.integers(1, 5, n), "hour": r.integers(5, 23, n),
        "luggage_bags": r.integers(0, 5, n), "fuel_price_rs": r.uniform(90, 110, n).round(2),
        "booking": r.choice(["app", "stand", "street"], n),
        "zone": r.choice(["Market", "Outskirts", "Station Area"], n),
        "fare_rs": r.integers(50, 350, n)})


def test_clean_data_flags_nothing(p):
    df = make()
    assert not dc.check_unit_mixup(df, p)["present"]
    assert not dc.check_sentinel_value(df, p)["present"]
    assert not dc.check_duplicate_rows(df)["present"]
    assert not dc.check_stuck_sensor(df, p)["present"]
    assert not dc.check_target_typos(df, p)["present"]
    assert not dc.check_spelling_variants(df, p)["present"]


def test_unit_mixup_found(p):
    df = make()
    df.loc[df.zone == "Outskirts", "wait_min"] *= 60
    r = dc.check_unit_mixup(df, p)
    assert (r["column"], r["group_value"], r["factor"]) == ("wait_min", "Outskirts", 60)


def test_sentinel_found(p):
    df = make()
    df.loc[:9, "hour"] = 999
    assert dc.check_sentinel_value(df, p) == {"present": True, "column": "hour", "value": 999, "count": 10}


def test_duplicates_counted_as_extra_copies():
    df = make()
    assert dc.check_duplicate_rows(pd.concat([df, df.iloc[:7]]))["count"] == 7


def test_stuck_sensor_dates(p):
    df = make()
    m = (df.date >= "2026-02-01") & (df.date <= "2026-02-20")
    df.loc[m, "fuel_price_rs"] = 99.5
    r = dc.check_stuck_sensor(df, p)
    assert (r["column"], r["start_date"], r["end_date"]) == ("fuel_price_rs", "2026-02-01", "2026-02-20")


def test_target_typos_found(p):
    df = make()
    df["fare_rs"] = np.linspace(50, 350, len(df)).round()
    df.loc[:4, "fare_rs"] *= 10
    assert dc.check_target_typos(df, p)["count"] == 5


def test_spelling_variants_found(p):
    df = make()
    df.loc[:5, "booking"] = "App "
    df.loc[6:9, "booking"] = "STAND."
    r = dc.check_spelling_variants(df, p)
    assert (r["column"], r["true_levels"]) == ("booking", 3)


def test_leak_found(p):
    df = make()
    hold = df.drop(columns=["fare_rs", "wait_min"])
    assert dc.check_leaking_column(df, hold, p) == {"present": True, "column": "wait_min"}

def test_detectors_do_not_modify_input(p):
    df = make()
    original = df.copy(deep=True)

    dc.check_unit_mixup(df, p)
    dc.check_sentinel_value(df, p)
    dc.check_duplicate_rows(df)
    dc.check_stuck_sensor(df, p)
    dc.check_target_typos(df, p)
    dc.check_spelling_variants(df, p)

    pd.testing.assert_frame_equal(df, original)
