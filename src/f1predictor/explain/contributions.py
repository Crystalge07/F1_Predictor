"""SHAP contributions: break each prediction into "how much each feature moved it".

For one driver, the model's predicted position is split exactly into:

    predicted position = baseline + contribution(grid) + contribution(form) + ...

    baseline       the model's average prediction over its training data
                   (roughly "a typical driver finishes about here")
    contribution   how many places that feature moved THIS driver, given
                   their value. Negative = moved them FORWARD (smaller
                   position number = better). Positive = moved them back.

These are SHAP values (SHapley Additive exPlanations). The idea comes from
game theory: treat the features as players who share the "payout" (the
prediction), and give each one its fair share of credit. The useful property:
the contributions always add up exactly to the prediction, so the explanation
never hides anything.

We don't need the separate `shap` library: XGBoost computes exact SHAP values
for its own trees with predict(..., pred_contribs=True).

Scope: this explains the position MODEL. Post-quali probabilities also blend
in the grid (simulate/probabilities.py) and add randomness and DNF chances,
so on the dashboard these are "why the model rates this driver here", not
"why their win chance is exactly 31%".
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from f1predictor.models.position import shifted_inputs

# Plain-English names for the dashboard and printouts.
FEATURE_LABELS = {
    "grid": "starting position",
    "pit_lane_start": "pit-lane start",
    "quali_gap_pct": "qualifying gap to pole (%)",
    "driver_form_finish": "recent finishes (last 5)",
    "driver_form_grid": "recent starting positions (last 5)",
    "team_season_finish": "team's finishes this season",
    "team_season_quali": "team's qualifying this season",
    "driver_track_finish": "driver's record at this track",
    "team_track_finish": "team's record at this track",
    "long_run_gap_pct": "practice long-run pace gap (%)",
    "short_run_gap_pct": "best practice lap gap (%)",
    "driver_season_points": "championship points",
    "team_season_points": "team championship points",
    "driver_points_per_race": "points per race this season",
    "team_points_per_race": "team points per race this season",
    "prev_season_points": "points last season",
}


def explain_race(model: xgb.XGBRegressor, race: pd.DataFrame) -> pd.DataFrame:
    """SHAP contributions for every driver in one race, in "long" format.

    Returns one row per (driver, feature) with columns:
        driver_id, feature, label, value (the driver's feature value),
        contribution (places; negative = helped), baseline, predicted
    """
    columns = list(model.feature_names_in_)
    # The prediction is the model averaged over nearby paces
    # (models/position.PACE_SMOOTHING), so the contributions are averaged over
    # the same inputs. SHAP values add up, so the averages still sum exactly
    # to the (smoothed) prediction.
    # DMatrix is XGBoost's own data container; the low-level booster needs it.
    # missing=nan tells it NaN means "unknown" (the same as during training).
    # Shape (n_drivers, n_features + 1). The LAST column is the baseline (XGBoost
    # calls it the bias): the same number for every driver.
    contribs = np.mean([model.get_booster().predict(xgb.DMatrix(X, missing=float("nan")), pred_contribs=True)
                        for X in shifted_inputs(race, columns)], axis=0)
    baseline = contribs[:, -1]
    predicted = contribs.sum(axis=1)   # = the model's prediction, exactly

    rows = []
    # zip pairs each driver with their row of contributions and feature values.
    for i, (driver, values) in enumerate(zip(race["driver_id"], race[columns].itertuples(index=False))):
        for j, feature in enumerate(columns):
            rows.append({
                "driver_id": driver,
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "value": values[j],
                "contribution": float(contribs[i, j]),
                "baseline": float(baseline[i]),
                "predicted": float(predicted[i]),
            })
    return pd.DataFrame(rows)


def top_reasons(explanation: pd.DataFrame, driver_id: str, n: int = 3) -> list[str]:
    """The n features that moved this driver the most, as readable lines.

    Example: ["starting position = 2: 4.1 places forward",
              "recent finishes (last 5) = 3.4: 1.2 places forward",
              "team's record at this track = 9.5: 0.4 places back"]
    """
    mine = explanation[explanation["driver_id"] == driver_id]
    # Biggest effect first, whichever direction: sort by the absolute value.
    biggest = mine.reindex(mine["contribution"].abs().sort_values(ascending=False).index).head(n)
    lines = []
    for row in biggest.itertuples():
        direction = "forward" if row.contribution < 0 else "back"
        value = "unknown" if pd.isna(row.value) else f"{row.value:.3g}"
        lines.append(f"{row.label} = {value}: {abs(row.contribution):.1f} places {direction}")
    return lines
