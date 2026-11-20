#!/usr/bin/env bash
# Rebuild everything from the DVC-tracked data: checks -> v1 -> drift/v2/promotion -> holdout.
set -euo pipefail
export MLFLOW_DISABLE_AGENT_HINT=1
rm -rf mlflow.db mlruns mlartifacts
python -m src.fare_pipeline.run_data_checks
python -m src.fare_pipeline.train
python -m src.fare_pipeline.retrain
