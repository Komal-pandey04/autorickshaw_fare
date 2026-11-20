import pytest

from src.fare_pipeline.modeling import build_model, mae


@pytest.mark.parametrize("cfg", [
    {"type": "ridge", "alpha": 1.0},
    {"type": "ridge_interact", "alpha": 10.0},
    {"type": "gradient_boosting", "n_estimators": 20, "learning_rate": 0.1, "max_depth": 2},
    {"type": "random_forest", "n_estimators": 10, "max_depth": 4}])
def test_every_candidate_fits_and_predicts(cfg, tiny_df, params):
    X, y = tiny_df[params["model"]["feature_columns"]], tiny_df["fare_rs"]
    assert mae(y, build_model(cfg, params, 0).fit(X, y).predict(X)) < 30
