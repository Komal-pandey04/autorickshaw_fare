"""Stage 2: train v1 on cleaned train.csv only, with MLflow tracking + registry.

Usage:  python -m src.fare_pipeline.train
Tries every candidate in params.yaml, picks the best on a validation split,
refits it on all cleaned train rows, registers it as version 1 with alias `champion`.
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .cleaning import clean_train, split_xy
from .config import ROOT, load_params, sha256_of
from .findings import update_findings
from .modeling import build_model, mae
from .registry import log_and_register, setup


def main():
    p = load_params()
    client = setup(p)
    sha = sha256_of(ROOT / p["data"]["train"])
    seed = p["model"]["random_state"]

    train = clean_train(pd.read_csv(ROOT / p["data"]["train"]), p)
    later = pd.read_csv(ROOT / p["data"]["later"])
    X, y = split_xy(train, p)
    Xl, yl = split_xy(later, p)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=p["model"]["validation_size"], random_state=seed)

    # 1) compare candidate settings on a validation split (one MLflow run each)
    scores = {}
    for cfg in p["model"]["candidates"]:
        m = build_model(cfg, p, seed).fit(Xtr, ytr)
        scores[cfg["name"]] = mae(yva, m.predict(Xva))
        log_and_register(
            m, p, f"candidate-{cfg['name']}",
            {"data_sha256": sha, "stage": "candidate", "rows": len(Xtr), **{f"cfg_{k}": v for k, v in cfg.items()}},
            {"mae_val": scores[cfg["name"]], "mae_later": mae(yl, m.predict(Xl))},
            register=False)
        print(f"candidate {cfg['name']:<18} mae_val={scores[cfg['name']]:.2f}")

    # 2) refit the best on ALL cleaned train rows: this is v1
    best = min(p["model"]["candidates"], key=lambda c: scores[c["name"]])
    v1 = build_model(best, p, seed).fit(X, y)
    mae_later = mae(yl, v1.predict(Xl))
    run_id, version = log_and_register(
        v1, p, "v1-final",
        {"data_sha256": sha, "stage": "v1", "rows": len(X), **{f"cfg_{k}": v for k, v in best.items()}},
        {"mae_val": scores[best["name"]], "mae_later": mae_later})
    client.set_registered_model_alias(p["mlflow"]["registered_model_name"], p["mlflow"]["champion_alias"], version)

    (ROOT / p["paths"]["model_out"]).parent.mkdir(exist_ok=True)
    joblib.dump(v1, ROOT / p["paths"]["model_out"])
    update_findings("tracking", {
        "experiment_name": p["mlflow"]["experiment_name"], "best_run_id": run_id,
        "registered_model_name": p["mlflow"]["registered_model_name"], "champion_version": version})
    print(f"v1 = {best['name']} (version {version}, run {run_id}); mae_later={mae_later:.2f}")


if __name__ == "__main__":
    main()
