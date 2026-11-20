"""Drift monitoring: Population Stability Index and per-group error."""
import numpy as np
import pandas as pd


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI of `actual` against `expected`, using `bins` quantile bins of `expected`.

    Outer edges are widened to +-inf so values outside the training range
    (e.g. longer rides arriving later) fall in the first/last bin.
    """
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_table(train: pd.DataFrame, later: pd.DataFrame, columns, bins=10, eps=1e-4) -> dict:
    return {c: psi(train[c], later[c], bins, eps) for c in columns}
