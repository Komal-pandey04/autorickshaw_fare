"""Stage 4: drift on later.csv, v2 retrain, promotion rule, holdout predictions.

Usage:  python -m src.fare_pipeline.retrain      (after train.py)
"""
import json

import joblib
import mlflow
import pandas as pd

from .cleaning import clean_train, split_xy
from .config import ROOT, load_params, sha256_of
from .findings import update_findings
from .modeling import build_model, mae, mae_by_group
from .monitoring import psi_table
from .registry import log_and_register, setup


def select_v2_config(train, first, p, seed):
    """Pick v2's settings on the NEWEST slice of the retraining data (time-ordered validation)."""
    k = int(len(first) * (1 - p["monitoring"]["v2_selection_share"]))
    fit, val = pd.concat([train, first.iloc[:k]], ignore_index=True), first.iloc[k:]
    Xf, yf = split_xy(fit, p)
    Xv, yv = split_xy(val, p)
    scores = {c["name"]: mae(yv, build_model(c, p, seed).fit(Xf, yf).predict(Xv)) for c in p["model"]["candidates"]}
    print("v2 candidate mae on newest validation slice:", {k_: round(v, 2) for k_, v in scores.items()})
    return next(c for c in p["model"]["candidates"] if c["name"] == min(scores, key=scores.get))


def main():
    p = load_params()
    client = setup(p)
    name, alias = p["mlflow"]["registered_model_name"], p["mlflow"]["champion_alias"]
    mon, seed = p["monitoring"], p["model"]["random_state"]
    target, group = p["data"]["target"], p["data"]["group_column"]
    sha = sha256_of(ROOT / p["data"]["train"])

    train = clean_train(pd.read_csv(ROOT / p["data"]["train"]), p)
    later = pd.read_csv(ROOT / p["data"]["later"]).sort_values(["date", "record_id"]).reset_index(drop=True)
    v1 = mlflow.sklearn.load_model(f"models:/{name}/1")

    # 1) drift: PSI per numeric column, cleaned train vs later (10 bins)
    psi = psi_table(train, later, p["data"]["numeric_columns"], mon["psi_bins"], mon["psi_eps"])
    most_drifted = max(psi, key=psi.get)

    # 2) v1 on all of later.csv, per zone
    Xl, yl = split_xy(later, p)
    by_group = mae_by_group(later, v1.predict(Xl), group, target)
    worst = max(by_group, key=by_group.get)

    # 3) v2 = same settings, trained on train + first 70% of later; checked on last 30%
    cut = int(len(later) * mon["retrain_share"])
    first, last = later.iloc[:cut], later.iloc[cut:]
    X1, y1 = split_xy(pd.concat([train, first], ignore_index=True), p)
    cfg = select_v2_config(train, first, p, seed)   # uses only train + first 70%, never the check rows
    Xc, yc = split_xy(last, p)
    v2 = build_model(cfg, p, seed).fit(X1, y1)
    mae_v1, mae_v2 = mae(yc, v1.predict(Xc)), mae(yc, v2.predict(Xc))
    log_and_register(
        v2, p, "v2-retrain",
        {"data_sha256": sha, "stage": "v2", "rows": len(X1), "later_rows_used": len(first),
         **{f"cfg_{k}": v for k, v in cfg.items()}},
        {"mae_check": mae_v2, "mae_check_v1": mae_v1})

    # 4) the client's promotion rule: v2 must be at least 12% better than v1
    promote = mae_v2 <= (1 - mon["promotion_min_improvement"]) * mae_v1
    champion = 2 if promote else 1
    client.set_registered_model_alias(name, alias, champion)
    final = v2 if promote else v1
    joblib.dump(final, ROOT / p["paths"]["model_out"])

    # 5) holdout predictions with the champion
    holdout = pd.read_csv(ROOT / p["data"]["holdout"])
    out = ROOT / p["paths"]["predictions_out"]
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({"record_id": holdout[p["data"]["id_column"]],
                  "prediction": final.predict(holdout[p["model"]["feature_columns"]]).round(2)}).to_csv(out, index=False)

    report = {"psi": psi, "most_drifted_feature": most_drifted, "mae_by_group": by_group,
              "worst_served_group": worst, "mae_v1_on_check": mae_v1, "mae_v2_on_check": mae_v2,
              "improvement": 1 - mae_v2 / mae_v1, "decision": "promote" if promote else "keep",
              "champion_version": champion, "check_rows": len(last), "retrain_extra_rows": len(first)}
    (ROOT / p["paths"]["monitoring_report"]).write_text(json.dumps(report, indent=2))
    update_findings("tracking", {"champion_version": champion})
    update_findings("monitoring", {k: report[k] for k in (
        "psi", "most_drifted_feature", "mae_by_group", "worst_served_group",
        "mae_v1_on_check", "mae_v2_on_check", "decision")})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
