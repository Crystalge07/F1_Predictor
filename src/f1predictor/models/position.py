"""The post-qualifying XGBoost model: predict finishing order from Saturday's info.

What it learns: a regressor that, given one driver's features (see
features/race_features.py), predicts their finishing position *if they finish*.
Retirements have no target and are skipped in training (DNF risk is the Monte
Carlo step's job in Phase 3).

How a race is predicted: score every driver in the race, then sort by the
predicted position. The model never has to output exact integers 1..20: we
only care about the ORDER, so predictions like 3.4, 3.9, 7.1 are fine.

Sample weights (recency): every training row gets a weight that shrinks with
age, counted in seasons back from the newest season in the training data:

    weight = SEASON_DECAY ** (newest season - row's season)

With SEASON_DECAY 0.6: newest season 1.0, one season back 0.6, two back 0.36,
five back ~0.08. Recent races matter more, which helps after a regulation
reset like 2026, and the same rule keeps working for every future season.
(It replaced a fixed "2026 counts 3x" multiplier on 2026-11-03; the decay is
tuned by scripts/tune_recency.py.)
"""

import pandas as pd
import xgboost as xgb

from f1predictor import settings

from f1predictor.features.race_features import POST_QUALI_FEATURES

# How much each season back counts, relative to the one after it. Chosen by
# scripts/tune_recency.py and stored in tuned_settings.json (see settings.py).
SEASON_DECAY: float = settings.load()["season_decay"]

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


def sample_weights(train: pd.DataFrame, season_decay: float) -> pd.Series:
    """decay ** (seasons before the newest season in `train`): 1.0 for the newest.

    "Newest" is taken from the training data itself, so in walk-forward each
    model weights relative to the season it's about to predict (or the one
    just before it, at round 1).
    """
    seasons_back = train["season"].max() - train["season"]
    return season_decay ** seasons_back


def train_model(train: pd.DataFrame, season_decay: float = SEASON_DECAY,
                columns: list[str] = POST_QUALI_FEATURES) -> xgb.XGBRegressor:
    """Fit the model on feature rows. Rows without a target (retirements) are dropped.

    `columns` picks the features: the post-quali ones by default; other stages
    pass their own list (features/stages.py). The model remembers which
    columns it was trained on, so predicting needs no list.
    """
    trainable = train[train["finish_position"].notna()]
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        trainable[columns],
        trainable["finish_position"],
        sample_weight=sample_weights(trainable, season_decay),
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
    # feature_names_in_: the columns the model was trained on, in that order.
    columns = list(model.feature_names_in_)
    return race.assign(expected_position=model.predict(race[columns]))


def predict_order(model: xgb.XGBRegressor, race: pd.DataFrame) -> list[str]:
    """Predicted finishing order for one race: driver_ids, predicted winner first."""
    scored = predict_positions(model, race).rename(columns={"expected_position": "predicted"})
    # Sort by predicted position; if two drivers tie exactly, the one starting
    # further forward goes first, then driver_id so the order is always the same.
    scored = scored.sort_values(["predicted", "grid", "driver_id"])
    return scored["driver_id"].tolist()
