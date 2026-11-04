"""Tests for the post-quali model helpers and walk-forward evaluation."""

import pandas as pd

from f1predictor.evaluate.walk_forward import walk_forward
from f1predictor.features.race_features import POST_QUALI_FEATURES
from f1predictor.models.position import predict_order, sample_weights, train_model


def fake_features(races: list[tuple[int, int, str]]) -> pd.DataFrame:
    """Two drivers per race. races: [(session_id, season, race_date)].

    Driver 'front' always starts and finishes 1st, 'back' 2nd, so any sensible
    model must learn "lower grid -> better finish".
    """
    rows = []
    for session_id, season, date in races:
        for driver, pos in (("front", 1), ("back", 2)):
            row = {f: 0.0 for f in POST_QUALI_FEATURES}
            row.update(session_id=session_id, season=season, race_date=date, driver_id=driver,
                       grid=pos, finish_position=float(pos), actual_position=pos)
            rows.append(row)
    return pd.DataFrame(rows)


def test_sample_weights_upweight_2026():
    df = pd.DataFrame({"season": [2024, 2025, 2026]})
    assert sample_weights(df, 3.0).tolist() == [1.0, 1.0, 3.0]


def test_2027_rows_weigh_the_same_as_2026_rows():
    # Decision 2026-11-04 (docs/PRD.md, "Decisions made"): the extra weight
    # belongs to the regulation ERA that started in 2026, not to "the latest
    # season". So in 2027, 2026 and 2027 rows both count x8 and pre-2026 rows
    # x1. Why: the settings stay frozen for the 2027 test (the split between
    # 2026 and 2027 can't be tuned without 2027 data), and every weight from
    # 1 to 12 scored within 0.08 places anyway. Revisit after 2027.
    # Pinned so that a change (e.g. favouring the newest season) is deliberate.
    from f1predictor.models.position import WEIGHT_2026
    assert WEIGHT_2026 == 8.0                       # the frozen value (tuned_settings.json)
    df = pd.DataFrame({"season": [2025, 2026, 2027, 2028]})
    assert sample_weights(df, WEIGHT_2026).tolist() == [1.0, 8.0, 8.0, 8.0]


def test_monotone_constraints_follow_the_column_order():
    # XGBoost takes the constraints as one number per column, in the order
    # the model is trained on; every stage trains on a different list.
    from f1predictor.models.position import monotone_constraints
    assert monotone_constraints(["driver_form_finish", "grid", "short_run_gap_pct"]) == (0, 1, 1)
    assert monotone_constraints(["driver_season_points"]) == (0,)


def test_model_learns_grid_order():
    df = fake_features([(i, 2024, f"2024-{i:02d}-01") for i in range(1, 11)])
    model = train_model(df)
    # Present the race "back" first, to check the output is really sorted.
    race = df[df.session_id == 1].iloc[::-1]
    assert predict_order(model, race) == ["front", "back"]


def test_retirements_are_not_trained_on():
    df = fake_features([(i, 2024, f"2024-{i:02d}-01") for i in range(1, 11)])
    df.loc[df.driver_id == "back", "finish_position"] = float("nan")   # 'back' always retires
    train_model(df)   # must not crash on NaN targets: those rows are dropped


def test_walk_forward_only_trains_on_earlier_races():
    df = fake_features([(1, 2025, "2025-03-01"), (2, 2025, "2025-04-01"), (3, 2026, "2026-03-01")])
    seen = {}

    def spy_trainer(history):
        # Record which races each model was trained on, then predict nothing clever.
        def predict(race):
            seen[race.session_id.iloc[0]] = sorted(history.session_id.unique())
            return race.driver_id.tolist()
        return predict

    walk_forward(df, [2, 3], spy_trainer)
    assert seen == {2: [1], 3: [1, 2]}      # never the race itself, never the future
