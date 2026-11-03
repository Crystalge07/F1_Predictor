"""Tests for championship features, the championship-order baseline, and
training on a stage's own feature list."""

import math

import pandas as pd
import pytest

from f1predictor.db.connection import get_connection, init_db
from f1predictor.features.championship import championship_features
from f1predictor.features.stages import PRE_WEEKEND_FEATURES
from f1predictor.models.baselines import championship_order
from f1predictor.models.post_quali import predict_order, train_model


@pytest.fixture
def conn():
    connection = get_connection(":memory:")
    init_db(connection)
    connection.execute("INSERT INTO circuits (circuit_id, name) VALUES ('c', 'C')")
    for team in ("red", "blue"):
        connection.execute("INSERT INTO constructors (constructor_id, name) VALUES (?, ?)", (team, team))
    for driver in ("a", "b", "c"):
        connection.execute("INSERT INTO drivers (driver_id, given_name, family_name) "
                           "VALUES (?, 'X', 'Y')", (driver,))
    yield connection
    connection.close()


def add_weekend(conn, season, rnd, race_points: dict, sprint_points: dict | None = None) -> int:
    """race_points: {driver: (team, points)}. Returns the race session_id."""
    event = conn.execute("INSERT INTO events (season, round, name, circuit_id, race_date) "
                         "VALUES (?, ?, 'GP', 'c', ?)", (season, rnd, f"{season}-{rnd:02d}-01")).lastrowid
    ids = {}
    for kind, pts in (("R", race_points), ("S", sprint_points)):
        if pts is None:
            continue
        ids[kind] = conn.execute("INSERT INTO sessions (event_id, session_type) VALUES (?, ?)",
                                 (event, kind)).lastrowid
        for pos, (driver, (team, p)) in enumerate(pts.items(), start=1):
            conn.execute("INSERT INTO results (session_id, driver_id, constructor_id, position, "
                         "position_text, points) VALUES (?, ?, ?, ?, ?, ?)",
                         (ids[kind], driver, team, pos, str(pos), p))
    return ids["R"]


def race_rows(conn, session_id, season, rnd, drivers: dict) -> pd.DataFrame:
    return pd.DataFrame([{"session_id": session_id, "season": season,
                          "race_date": f"{season}-{rnd:02d}-01",
                          "driver_id": d, "constructor_id": t} for d, t in drivers.items()])


def get(df, driver):
    return df[df.driver_id == driver].iloc[0]


def test_points_before_the_weekend_including_earlier_sprints(conn):
    add_weekend(conn, 2024, 1, {"a": ("red", 25), "b": ("red", 18), "c": ("blue", 15)},
                sprint_points={"a": ("red", 8), "c": ("blue", 7)})
    r2 = add_weekend(conn, 2024, 2, {"a": ("red", 0), "b": ("red", 25), "c": ("blue", 18)},
                     sprint_points={"b": ("red", 8)})       # same-weekend sprint: must NOT count
    out = championship_features(conn, race_rows(conn, r2, 2024, 2, {"a": "red", "b": "red", "c": "blue"}))
    assert get(out, "a")["driver_season_points"] == 33      # 25 + 8 sprint
    assert get(out, "b")["driver_season_points"] == 18      # weekend-2 sprint not counted
    assert get(out, "a")["team_season_points"] == 51        # red: 33 + 18
    assert get(out, "c")["driver_points_per_race"] == 22    # 22 points / 1 weekend


def test_round_one_uses_last_season(conn):
    add_weekend(conn, 2024, 1, {"a": ("red", 25), "b": ("red", 18)})
    r = add_weekend(conn, 2025, 1, {"a": ("red", 0), "b": ("red", 0), "c": ("blue", 0)})
    out = championship_features(conn, race_rows(conn, r, 2025, 1, {"a": "red", "b": "red", "c": "blue"}))
    assert get(out, "a")["driver_season_points"] == 0       # new season: back to zero
    assert math.isnan(get(out, "a")["driver_points_per_race"])
    assert get(out, "a")["prev_season_points"] == 25
    assert math.isnan(get(out, "c")["prev_season_points"])  # rookie


def test_championship_order_tie_breaks():
    race = pd.DataFrame({
        "driver_id":            ["rookie", "vet", "leader", "teammate"],
        "driver_season_points": [0, 0, 50, 0],
        "team_season_points":   [0, 0, 60, 60],
        "prev_season_points":   [float("nan"), 100, 80, 10],
    })
    # leader on points; teammate beats the 0-point pair on team points;
    # vet beats rookie on last season (rookie has none -> last).
    assert championship_order(race) == ["leader", "teammate", "vet", "rookie"]


def test_training_on_a_stage_feature_list():
    # Lower driver_form_finish -> better finish. The pre-weekend model must
    # learn it using only pre-weekend columns (no grid column exists at all).
    rows = []
    for i in range(10):
        for driver, form in (("good", 2.0), ("bad", 10.0)):
            row = {c: 0.0 for c in PRE_WEEKEND_FEATURES}
            row.update(season=2024, driver_id=driver, driver_form_finish=form,
                       finish_position=1.0 if driver == "good" else 2.0, grid=1)
            rows.append(row)
    df = pd.DataFrame(rows)
    model = train_model(df, columns=PRE_WEEKEND_FEATURES)
    assert list(model.feature_names_in_) == PRE_WEEKEND_FEATURES
    assert predict_order(model, df.iloc[[1, 0]]) == ["good", "bad"]
