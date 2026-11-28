"""Model-agnostic "why this fare?" breakdown (works for ridge, forest and boosting alike).

Idea: start from a typical ride (the baseline in params.yaml). For each input, put the rider's
real value back and see how much the predicted fare moves. That move is the input's effect.
Effects are measured one input at a time, so with interactions they do not add up exactly to the
total; `other_effects_rs` holds the remainder so the numbers always reconcile.
"""
import pandas as pd


def explain_prediction(model, row: dict, baseline: dict, features: list) -> dict:
    actual = pd.DataFrame([row])[features]
    base = pd.DataFrame([baseline])[features]
    fare = float(model.predict(actual)[0])
    base_fare = float(model.predict(base)[0])

    effects = {}
    for col in features:
        swapped = actual.copy()
        swapped[col] = base[col].iloc[0]          # put the typical value back for this one input
        effects[col] = fare - float(model.predict(swapped)[0])

    total_change = fare - base_fare
    return {
        "prediction": round(fare, 2),
        "baseline_fare": round(base_fare, 2),
        "effects_rs": {k: round(v, 2) for k, v in sorted(effects.items(), key=lambda kv: -abs(kv[1]))},
        "other_effects_rs": round(total_change - sum(effects.values()), 2),
    }
