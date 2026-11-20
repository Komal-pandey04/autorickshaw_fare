"""Fix the data problems found in train.csv (see data_checks.py)."""
import numpy as np
import pandas as pd


def clean_train(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Return a cleaned copy of the raw train.csv.

    1. drop exact duplicate rows
    2. convert the mixed-up unit back (Outskirts wait_min was seconds)
    3. treat the sentinel as missing and fill with the column median
    """
    c = p["cleaning"]
    out = df.drop_duplicates().copy()

    rows = out[p["data"]["group_column"]] == c["unit_group_value"]
    out.loc[rows, c["unit_column"]] = out.loc[rows, c["unit_column"]] / c["unit_factor"]

    col = c["sentinel_column"]
    out[col] = out[col].replace(c["sentinel_value"], np.nan)
    out[col] = out[col].fillna(out[col].median()).round().astype(int)
    return out.reset_index(drop=True)


def split_xy(df: pd.DataFrame, p: dict):
    return df[p["model"]["feature_columns"]], df[p["data"]["target"]]
