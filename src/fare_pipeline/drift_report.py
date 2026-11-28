"""Extra monitoring report: severity label per column, drift of text columns, error per booking type.

Usage:  python -m src.fare_pipeline.drift_report      (after train.py; needs the DVC data files)
Writes reports/drift_extended.json. It does not change findings.yaml or any registry alias.
"""
import json

import mlflow
import pandas as pd

from .cleaning import clean_train, split_xy
from .config import ROOT, load_params
from .modeling import mae_by_group
from .monitoring import categorical_psi, psi_status, psi_table
from .registry import setup


def build_report(train, later, v1, p) -> dict:
    mon, th = p["monitoring"], p["explain"]["psi_thresholds"]
    numeric = psi_table(train, later, p["data"]["numeric_columns"], mon["psi_bins"], mon["psi_eps"])
    text = {c: categorical_psi(train[c], later[c], mon["psi_eps"]) for c in p["data"]["text_columns"]}
    allpsi = {**numeric, **text}
    Xl, _ = split_xy(later, p)
    pred = v1.predict(Xl)
    target = p["data"]["target"]
    return {
        "psi_status": {c: {"psi": v, "status": psi_status(v, th)} for c, v in allpsi.items()},
        "needs_attention": [c for c, v in allpsi.items() if psi_status(v, th) == "significant"],
        "v1_mae_by_booking": mae_by_group(later, pred, "booking", target),
        "v1_mae_by_zone": mae_by_group(later, pred, p["data"]["group_column"], target),
    }


def main():
    p = load_params()
    setup(p)
    train = clean_train(pd.read_csv(ROOT / p["data"]["train"]), p)
    later = pd.read_csv(ROOT / p["data"]["later"])
    v1 = mlflow.sklearn.load_model(f"models:/{p['mlflow']['registered_model_name']}/1")
    report = build_report(train, later, v1, p)
    (ROOT / "reports" / "drift_extended.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
