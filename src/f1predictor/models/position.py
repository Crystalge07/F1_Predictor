"""The post-qualifying XGBoost model: predict finishing order from Saturday's info.

What it learns: a regressor that, given one driver's features (see
features/race_features.py), predicts their finishing position *if they finish*.
Retirements have no target and are skipped in training (DNF risk is the Monte
Carlo step's job in Phase 3).

How a race is predicted: score every driver in the race, then sort by the
predicted position. The model never has to output exact integers 1..20: we
only care about the ORDER, so predictions like 3.4, 3.9, 7.1 are fine.

Sample weights (the 2026 decision from the PRD): every training row gets a
weight; rows from the regulation-reset season ON get `weight_2026` (8.0 =
each such row counts like eight older rows). 2026 is a regulation reset, so
races under the new rules should matter more. The multiplier was tuned in
scripts/train_post_quali.py and is frozen.

In 2027 that means 2026 AND 2027 rows both count x8 (decision 2026-11-04,
docs/PRD.md): the weight belongs to the regulation era, not to "the latest
season", so 2027 races are not weighted above 2026 ones. A 2026/2027 split
can't be tuned without 2027 data; revisit after the season. Pinned by
tests/test_position_model.py::test_2027_rows_weigh_the_same_as_2026_rows.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from f1predictor import settings

from f1predictor.features.race_features import POST_QUALI_FEATURES

# The first season of the new regulations, whose rows get the extra weight.
REGULATION_RESET_SEASON = 2026

# The 2026 multiplier, chosen by scripts/train_post_quali.py and stored in
# tuned_settings.json (see settings.py). Every value from 1 to 12 scores
# within ~0.08 places on the calibration races, so the exact value (3 in
# Phase 2, 8 at the 2026-11-03 freeze) is effectively arbitrary.
WEIGHT_2026: float = settings.load()["weight_2026"]

# XGBoost settings. Deliberately conservative: we only have ~3,300 training
# rows, and a big, deep model would memorise noise (overfit).
#   n_estimators   how many small trees are added one after another
#   learning_rate  how much each tree is allowed to correct the previous ones
#                  (small rate + many trees = slower but steadier learning)
#   max_depth      how many questions each tree may ask in a row; 3 keeps
#                  each tree simple
#   subsample      each tree sees a random 80% of rows...
#   colsample_bytree ...and a random 80% of features, which reduces overfitting
#   random_state   fixed seed, so the same data always gives the same model
XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 0,
}


# Monotone constraints: the direction each feature may push the predicted
# position, whatever the training data's noise suggests.
#   +1  a higher value can only predict a finish further back (or the same)
#   -1  a higher value can only predict a finish further forward (or the same)
# Features not listed are free (0).
#
# Why: XGBoost on its own learns whatever the noise says. Before this, the
# practice model predicted Piastri better at 0.26% off the pace than at 0.25%
# (2026 Mexico), so a slower practice could RAISE a driver's podium chance by
# 12 points, and 0.01% of practice pace (about 9 ms) could move a chance by 16
# points. tests/test_monotonicity.py and tests/test_stability.py check this.
MONOTONE_DIRECTIONS = {
    "grid": 1,                # starting further back
    "quali_gap_pct": 1,       # slower in qualifying
    "short_run_gap_pct": 1,   # slower over one lap in practice
    "long_run_gap_pct": 1,    # slower on long runs in practice
}


def monotone_constraints(columns: list[str]) -> tuple[int, ...]:
    """MONOTONE_DIRECTIONS as XGBoost wants it: one number per column, in order."""
    return tuple(MONOTONE_DIRECTIONS.get(column, 0) for column in columns)


def sample_weights(train: pd.DataFrame, weight_2026: float) -> pd.Series:
    """1.0 for every row, `weight_2026` for rows from the regulation-reset season onward."""
    return (train["season"] >= REGULATION_RESET_SEASON).map({True: weight_2026, False: 1.0})


def train_model(train: pd.DataFrame, weight_2026: float = 1.0,
                columns: list[str] = POST_QUALI_FEATURES) -> xgb.XGBRegressor:
    """Fit the model on feature rows. Rows without a target (retirements) are dropped.

    `columns` picks the features: the post-quali ones by default; other stages
    pass their own list (features/stages.py). The model remembers which
    columns it was trained on, so predicting needs no list.
    """
    trainable = train[train["finish_position"].notna()]
    model = xgb.XGBRegressor(**XGB_PARAMS, monotone_constraints=monotone_constraints(columns))
    model.fit(
        trainable[columns],
        trainable["finish_position"],
        sample_weight=sample_weights(trainable, weight_2026),
    )
    return model


# Pace smoothing: the trees are step functions, so 9 ms of practice pace can
# cross a split and move a driver's prediction by a third of a place in one
# jump (2026-11-04: 2024 Miami, Sainz +0.33, which moved a win chance by 11
# points). So every prediction is the AVERAGE of the model at the driver's
# pace and at nearby paces: each step becomes a short ramp, and a tiny change
# makes a tiny difference.
#   PACE_FEATURES   the lap-time gaps (in % of lap time) that get smoothed.
#                   Not grid: it's a whole place, not a measurement.
#   PACE_SMOOTHING  half-width of the band, in % of lap time (0.05% = ~45 ms
#                   on a 90 s lap, about the size of timing/track noise)
#   PACE_OFFSETS    9 evenly spaced points from -1 to +1 times the half-width
# An average of monotone predictions is still monotone, so the direction
# rules above still hold. NaN (no lap) stays NaN at every offset. Chosen on
# stability alone (docs/evaluation-protocol.md, 2026-11-04); not a tuned
# setting, so tuned_settings.json stays frozen.
PACE_FEATURES = ("quali_gap_pct", "short_run_gap_pct", "long_run_gap_pct")
PACE_SMOOTHING = 0.05
PACE_OFFSETS = np.linspace(-1.0, 1.0, 9)


def shifted_inputs(race: pd.DataFrame, columns: list[str],
                   smoothing: float = PACE_SMOOTHING) -> list[pd.DataFrame]:
    """The model's inputs for `race`, once per pace offset (just once if the
    columns have no pace feature or smoothing is 0). The prediction is the
    mean of the model over these; explain/contributions.py averages over the
    same list, so its contributions still add up to the prediction."""
    X = race[columns].astype(float)
    pace = [c for c in columns if c in PACE_FEATURES]
    if not pace or smoothing <= 0:
        return [X]
    shifted = []
    for offset in PACE_OFFSETS:
        Xo = X.copy()
        Xo[pace] = Xo[pace] + offset * smoothing
        shifted.append(Xo)
    return shifted


def predict_positions(model: xgb.XGBRegressor, race: pd.DataFrame) -> pd.DataFrame:
    """The model's expected finishing position for each driver in one race.

    `race` is the feature rows for that race, ALL drivers (including those who
    will go on to retire: we don't know that on Saturday).

    Returns the race rows with a new `expected_position` column. These are the
    raw numbers the Monte Carlo simulator starts from (e.g. 3.4 vs 3.9 means
    "slightly ahead"), not whole places.
    """
    # feature_names_in_: the columns the model was trained on, in that order.
    columns = list(model.feature_names_in_)
    # Averaged over nearby paces (see PACE_SMOOTHING).
    predictions = [model.predict(X) for X in shifted_inputs(race, columns)]
    return race.assign(expected_position=np.mean(predictions, axis=0))


def predict_order(model: xgb.XGBRegressor, race: pd.DataFrame) -> list[str]:
    """Predicted finishing order for one race: driver_ids, predicted winner first."""
    scored = predict_positions(model, race).rename(columns={"expected_position": "predicted"})
    # Sort by predicted position; if two drivers tie exactly, the one starting
    # further forward goes first, then driver_id so the order is always the same.
    scored = scored.sort_values(["predicted", "grid", "driver_id"])
    return scored["driver_id"].tolist()
