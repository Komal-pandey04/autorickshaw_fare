"""Run all seven checks on the ORIGINAL train.csv and fill findings.yaml.

Usage (from repo root):  python -m src.fare_pipeline.run_data_checks
"""
import json

import pandas as pd
import yaml

from .config import ROOT, load_params, sha256_of
from . import data_checks as dc


def run():
    p = load_params()
    train = pd.read_csv(ROOT / p["data"]["train"])
    holdout = pd.read_csv(ROOT / p["data"]["holdout"])
    return {
        "leaking_column": dc.check_leaking_column(train, holdout, p),
        "unit_mixup": dc.check_unit_mixup(train, p),
        "sentinel_value": dc.check_sentinel_value(train, p),
        "duplicate_rows": dc.check_duplicate_rows(train),
        "stuck_sensor": dc.check_stuck_sensor(train, p),
        "target_typos": dc.check_target_typos(train, p),
        "spelling_variants": dc.check_spelling_variants(train, p),
    }, sha256_of(ROOT / p["data"]["train"])


def main():
    p = load_params()
    issues, sha = run()
    out = ROOT / p["paths"]["findings_out"]
    findings = yaml.safe_load(out.read_text())
    findings["data_issues"].update(issues)
    out.write_text(yaml.safe_dump(findings, sort_keys=False, allow_unicode=True))
    print(json.dumps(issues, indent=2, default=str))
    print("train.csv sha256:", sha)


if __name__ == "__main__":
    main()
