"""Detectors for the seven kinds of data problem in train.csv.

Every function takes the raw dataframe (as given) plus the params dict and
returns a dict shaped like the matching block of findings.yaml.
"""
import numpy as np
import pandas as pd


def check_leaking_column(train, holdout, p):
    """A column absent from holdout_features.csv, or that mirrors the target."""
    d, target = p["data"], p["data"]["target"]
    missing = [c for c in train.columns if c not in holdout.columns and c != target]
    if missing:
        return {"present": True, "column": missing[0]}
    for c in d["numeric_columns"]:
        if abs(train[c].corr(train[target])) >= p["checks"]["leak_corr_threshold"]:
            return {"present": True, "column": c}
    return {"present": False, "column": None}


def check_unit_mixup(train, p):
    """One group whose column mean is an exact conversion factor off the others.

    Each group's mean is compared with the MEDIAN of all group means, so a single
    odd group cannot drag the reference towards itself.
    """
    d, ck = p["data"], p["checks"]
    g = d["group_column"]
    for col in d["numeric_columns"]:
        means = train.loc[train[col] > 0].groupby(g)[col].mean()
        ref = means.median()
        for grp, m in means.items():
            ratio = m / ref
            if max(ratio, 1 / ratio) >= ck["unit_ratio_threshold"]:
                # snap the rough ratio to the exact conversion factor
                est = ratio if ratio > 1 else 1 / ratio
                factor = min(ck["unit_known_factors"], key=lambda f: abs(f - est))
                return {"present": True, "column": col, "group_value": grp,
                        "factor": factor if ratio > 1 else 1 / factor}
    return {"present": False, "column": None, "group_value": None, "factor": None}


def check_sentinel_value(train, p):
    """An impossible number repeated many times (e.g. 999 in `hour`)."""
    ck = p["checks"]
    for col in p["data"]["numeric_columns"]:
        vc = train[col].value_counts()
        for v in ck["sentinel_candidates"]:
            if v in vc.index and vc[v] >= ck["sentinel_min_count"]:
                # a sentinel must stand far outside the rest of the column
                rest = train.loc[train[col] != v, col]
                if v > rest.max() * 2 or v < min(rest.min(), 0):
                    return {"present": True, "column": col, "value": v,
                            "count": int(vc[v])}
    return {"present": False, "column": None, "value": None, "count": None}


def check_duplicate_rows(train):
    n = int(train.duplicated().sum())
    return {"present": n > 0, "count": n if n else None}


def check_stuck_sensor(train, p):
    """A column with one value on every row for >= N consecutive days."""
    d, ck = p["data"], p["checks"]
    df = train.drop_duplicates().copy()
    df[d["date_column"]] = pd.to_datetime(df[d["date_column"]])
    for col in d["numeric_columns"]:
        daily = df.groupby(d["date_column"])[col].agg(["nunique", "first"])
        one = daily["nunique"] == 1
        # runs of consecutive days showing a single, unchanged value
        run_id = ((daily["first"] != daily["first"].shift()) | ~one).cumsum()
        for _, run in daily[one].groupby(run_id[one]):
            if len(run) >= ck["stuck_min_run_days"]:
                return {"present": True, "column": col,
                        "start_date": str(run.index.min().date()),
                        "end_date": str(run.index.max().date())}
    return {"present": False, "column": None, "start_date": None, "end_date": None}


def check_target_typos(train, p):
    """Targets typed 10x too big: a high cluster separated by a big empty gap."""
    ck = p["checks"]
    y = np.sort(train[p["data"]["target"]].astype(float).values)
    gaps = np.diff(y)
    i = int(gaps.argmax())
    n_high = len(y) - i - 1
    if gaps[i] > ck["typo_min_gap_ratio"] * y[i] and n_high <= ck["typo_max_share"] * len(y):
        return {"present": True, "count": int(n_high)}
    return {"present": False, "count": None}


def check_spelling_variants(train, p):
    """Text column where normalising case/space/punctuation merges categories."""
    for col in p["data"]["text_columns"]:
        raw = train[col].dropna().unique()
        norm = {"".join(ch for ch in str(v).lower() if ch.isalnum()) for v in raw}
        if len(norm) < len(raw):
            return {"present": True, "column": col, "true_levels": len(norm)}
    return {"present": False, "column": None, "true_levels": None}
