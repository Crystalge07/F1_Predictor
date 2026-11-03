"""The post-qualifying XGBoost model: predict finishing order from Saturday's info.

What it learns: a regressor that, given one driver's features (see
features/post_quali.py), predicts their finishing position *if they finish*.
Retirements have no target and are skipped in training (DNF risk is the Monte
Carlo step's job in Phase 3).

How a race is predicted: score every driver in the race, then sort by the
predicted position. The model never has to output exact integers 1..20: we
only care about the ORDER, so predictions like 3.4, 3.9, 7.1 are fine.

Sample weights (the 2026 decision from the PRD): every training row gets a
weight; 2026 rows get `weight_2026` (e.g. 3.0 = each 2026 row counts like
three older rows). 2026 is a regulation reset, so recent races should matter
more. The multiplier is tuned in scripts/train_post_quali.py.
"""

import pandas as pd
import xgboost as xgb

from f1predictor.features.post_quali import FEATURE_COLUMNS

# The first season of the new regulations, whose rows get the extra weight.
REGULATION_RESET_SEASON = 2026

# The 2026 multiplier chosen by scripts/train_post_quali.py (2026-11-02). Every
# value from 1 to 12 scored within 0.04 places, so it barely matters yet.
WEIGHT_2026 = 3.0

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


def sample_weights(train: pd.DataFrame, weight_2026: float) -> pd.Series:
    """1.0 for every row, `weight_2026` for rows from the regulation-reset season onward."""
    return (train["season"] >= REGULATION_RESET_SEASON).map({True: weight_2026, False: 1.0})


def train_model(train: pd.DataFrame, weight_2026: float = 1.0) -> xgb.XGBRegressor:
    """Fit the model on feature rows. Rows without a target (retirements) are dropped."""
    trainable = train[train["finish_position"].notna()]
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        trainable[FEATURE_COLUMNS],
        trainable["finish_position"],
        sample_weight=sample_weights(trainable, weight_2026),
    )
    return model


def predict_positions(model: xgb.XGBRegressor, race: pd.DataFrame) -> pd.DataFrame:
    """The model's expected finishing position for each driver in one race.

    `race` is the feature rows for that race, ALL drivers (including those who
    will go on to retire: we don't know that on Saturday).

    Returns the race rows with a new `expected_position` column. These are the
    raw numbers the Monte Carlo simulator starts from (e.g. 3.4 vs 3.9 means
    "slightly ahead"), not whole places.
    """
    return race.assign(expected_position=model.predict(race[FEATURE_COLUMNS]))


def predict_order(model: xgb.XGBRegressor, race: pd.DataFrame) -> list[str]:
    """Predicted finishing order for one race: driver_ids, predicted winner first."""
    scored = predict_positions(model, race).rename(columns={"expected_position": "predicted"})
    # Sort by predicted position; if two drivers tie exactly, the one starting
    # further forward goes first, then driver_id so the order is always the same.
    scored = scored.sort_values(["predicted", "grid", "driver_id"])
    return scored["driver_id"].tolist()
