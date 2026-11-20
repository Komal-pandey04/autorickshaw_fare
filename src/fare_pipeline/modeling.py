"""Model construction and evaluation helpers."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


def build_model(cfg: dict, p: dict, seed: int) -> Pipeline:
    """One sklearn Pipeline = preprocessing + regressor, so the API takes raw fields."""
    cat = p["model"]["categorical_columns"]
    num = [c for c in p["model"]["feature_columns"] if c not in cat]
    kind = cfg["type"]
    if kind == "ridge":
        reg = Ridge(alpha=cfg["alpha"])
        scale = StandardScaler()
    elif kind == "ridge_interact":
        # all pairwise interactions (e.g. zone x distance): lets each zone have its own rate per km
        reg = Ridge(alpha=cfg["alpha"])
        scale = StandardScaler()
    elif kind == "random_forest":
        reg = RandomForestRegressor(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                                    random_state=seed, n_jobs=-1)
        scale = "passthrough"
    elif kind == "gradient_boosting":
        reg = GradientBoostingRegressor(n_estimators=cfg["n_estimators"], learning_rate=cfg["learning_rate"],
                                        max_depth=cfg["max_depth"], random_state=seed)
        scale = "passthrough"
    else:
        raise ValueError(f"unknown model type: {kind}")
    pre = ColumnTransformer([
        ("num", scale, num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    steps = [("pre", pre)]
    if kind == "ridge_interact":
        steps.append(("interactions", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))
    return Pipeline(steps + [("reg", reg)])


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def mae_by_group(df, y_pred, group_col: str, target: str) -> dict:
    err = np.abs(df[target].to_numpy() - np.asarray(y_pred))
    return {g: float(err[(df[group_col] == g).to_numpy()].mean()) for g in sorted(df[group_col].unique())}
