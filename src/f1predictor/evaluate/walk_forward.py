"""Walk-forward evaluation: test a model the way it will actually be used.

In real life, before each race we retrain on every race so far, then predict
the next one. Walk-forward does exactly that for each test race:

    test race = 2026 round 7
    train on  = every race before 2026 round 7 (2018 R1 ... 2026 R6)
    predict   = 2026 round 7, then score it

It's a time-based split repeated race by race, so the model never sees the
future (the PRD rule: never split randomly). Retraining ~20 times is cheap
here because the dataset is small (a model trains in well under a second).
"""

from collections.abc import Callable
from typing import Any

import pandas as pd

# A "trainer" takes the training rows and returns a function that turns one
# race's rows into a prediction. Usually that's a predicted order (a list of
# driver_ids), but it can be anything, e.g. a table of expected positions for
# the simulator. Passing it in (instead of hard-coding XGBoost) lets tests plug
# in a fake trainer and check what it was given.
Trainer = Callable[[pd.DataFrame], Callable[[pd.DataFrame], Any]]


def walk_forward(features: pd.DataFrame, test_session_ids: list[int],
                 trainer: Trainer) -> dict[int, Any]:
    """A prediction for every test race, each from a model trained only on earlier races.

    Returns {session_id: whatever the trainer's predict function returned},
    e.g. {session_id: [driver_ids in predicted order]}.
    """
    predictions = {}
    for session_id in test_session_ids:
        race = features[features["session_id"] == session_id]
        race_date = race["race_date"].iloc[0]
        # STRICTLY earlier races. race_date is ISO text ('2026-05-03'), and ISO
        # dates sort correctly as text, so < works.
        history = features[features["race_date"] < race_date]
        predict = trainer(history)
        predictions[session_id] = predict(race)
    return predictions
