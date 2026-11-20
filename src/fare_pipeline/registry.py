"""Thin helpers around MLflow tracking and the model registry."""
import mlflow
from mlflow.tracking import MlflowClient

from .config import ROOT


def setup(p: dict) -> MlflowClient:
    # sqlite file in the repo root (committed); MLflow 3 refuses ./mlruns as a store
    mlflow.set_tracking_uri(p["mlflow"]["tracking_uri"].replace("sqlite:///", f"sqlite:///{ROOT}/"))
    mlflow.set_experiment(p["mlflow"]["experiment_name"])
    return MlflowClient()


def log_and_register(model, p: dict, name_for_run: str, params: dict, metrics: dict, register=True):
    """Log one run (params, metrics, model). Returns (run_id, registered version or None)."""
    with mlflow.start_run(run_name=name_for_run) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        info = mlflow.sklearn.log_model(
            model, name="model",
            registered_model_name=p["mlflow"]["registered_model_name"] if register else None,
            serialization_format="cloudpickle")
        version = int(info.registered_model_version) if register else None
        return run.info.run_id, version
