"""Tests for as-of prediction rows (features/as_of.py).

The point of the module is that a prediction can't see its own race's
result. So the second race here is set up so that every shortcut would show:
    * a reserve driver ('e') runs FP1 in place of 'd'
    * penalties reshuffle the grid (the starting grid differs from qualifying)
    * on race day 'd' is ill and 'e' starts instead, from a different slot
Each stage must give the drivers and grid that were known at its cutoff.
"""

import math

import pytest

from f1predictor.db.connection import get_connection, init_db
from f1predictor.features.as_of import prediction_rows
from f1predictor.features.practice_pace import PRACTICE_PACE_COLUMNS
from f1predictor.features.race_features import build_race_features


@pytest.fixture
def conn():
    connection = get_connection(":memory:")
    init_db(connection)
    connection.execute("INSERT INTO circuits (circuit_id, name) VALUES ('monza', 'Monza')")
    for team in ("red", "blue"):
        connection.execute("INSERT INTO constructors (constructor_id, name) VALUES (?, ?)", (team, team))
    for driver in ("a", "b", "c", "d", "e"):
        connection.execute("INSERT INTO drivers (driver_id, given_name, family_name) "
                           "VALUES (?, 'X', 'Y')", (driver,))
    yield connection
    connection.close()


TEAMS = {"a": "red", "b": "red", "c": "blue", "d": "blue", "e": "blue"}
CARS = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 40}


def add_event(conn, rnd, race_date, sessions):
    """sessions: {session_type: start_time_utc}. Returns {session_type: session_id} plus 'event'."""
    event_id = conn.execute(
        "INSERT INTO events (season, round, name, circuit_id, race_date) VALUES (2024, ?, 'GP', 'monza', ?)",
        (rnd, race_date)).lastrowid
    ids = {"event": event_id}
    for session_type, start in sessions.items():
        ids[session_type] = conn.execute(
            "INSERT INTO sessions (event_id, session_type, start_time_utc) VALUES (?, ?, ?)",
            (event_id, session_type, start)).lastrowid
    return ids


def add_results(conn, race_id, finishes, grids):
    for driver, finish in finishes.items():
        conn.execute(
            "INSERT INTO results (session_id, driver_id, constructor_id, car_number, grid, position, "
            "position_text) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (race_id, driver, TEAMS[driver], CARS[driver], grids[driver], finish, str(finish)))


def add_entries(conn, session_id, drivers):
    for driver in drivers:
        conn.execute(
            "INSERT INTO session_entries (session_id, car_number, driver_id, constructor_id) VALUES (?, ?, ?, ?)",
            (session_id, CARS[driver], driver, TEAMS[driver]))


def add_qualifying(conn, quali_id, order):
    for position, driver in enumerate(order, start=1):
        conn.execute(
            "INSERT INTO qualifying (session_id, driver_id, constructor_id, position, q1_ms) "
            "VALUES (?, ?, ?, ?, ?)", (quali_id, driver, TEAMS[driver], position, 80_000 + position * 100))


@pytest.fixture
def weekend(conn):
    """Race 1 is history. Race 2 is the one being predicted (see the module docstring)."""
    r1 = add_event(conn, 1, "2024-03-03", {"Q": "2024-03-02T15:00:00Z", "R": "2024-03-03T15:00:00Z"})
    add_qualifying(conn, r1["Q"], ["a", "b", "c", "d"])
    add_results(conn, r1["R"], {"a": 1, "b": 2, "c": 3, "d": 4}, {"a": 1, "b": 2, "c": 3, "d": 4})

    r2 = add_event(conn, 2, "2024-03-17", {
        "FP1": "2024-03-15T11:00:00Z", "FP3": "2024-03-16T11:00:00Z",
        "Q": "2024-03-16T15:00:00Z", "R": "2024-03-17T15:00:00Z"})
    add_entries(conn, r2["FP1"], ["a", "b", "c", "e"])       # reserve 'e' in d's car for FP1
    add_entries(conn, r2["FP3"], ["a", "b", "c", "d"])
    add_entries(conn, r2["Q"], ["a", "b", "c", "d"])
    add_qualifying(conn, r2["Q"], ["a", "b", "c", "d"])
    # Penalties: 'a' (pole) drops to 3rd; everyone else moves up.
    for driver, slot in {"b": 1, "c": 2, "a": 3, "d": 4}.items():
        conn.execute("INSERT INTO starting_grid (session_id, car_number, position) VALUES (?, ?, ?)",
                     (r2["Q"], CARS[driver], slot))
    # Race day: 'd' is ill, 'e' starts from the pit lane (grid 0). This result
    # must not leak into any prediction.
    add_results(conn, r2["R"], {"e": 1, "c": 2, "b": 3, "a": 4}, {"b": 1, "c": 2, "a": 3, "e": 0})
    return r2


def by_driver(rows, column):
    return dict(zip(rows["driver_id"], rows[column]))


def test_pre_weekend_uses_previous_race_lineup_and_no_weekend_data(conn, weekend):
    rows, meta = prediction_rows(conn, weekend["event"], "pre_weekend")
    assert sorted(rows["driver_id"]) == ["a", "b", "c", "d"]      # race 1's starters, not 'e'
    assert meta["lineup_source"] == "previous_race"
    assert meta["as_of_utc"] == "2024-03-15T11:00:00Z"           # FP1 start
    # Nothing from this weekend: no grid, qualifying or practice values.
    for column in ["grid", "pit_lane_start", "quali_gap_pct"] + PRACTICE_PACE_COLUMNS:
        assert rows[column].isna().all(), column
    # No result either...
    assert rows["actual_position"].isna().all() and rows["finish_position"].isna().all()
    # ...but history works as in training: 'a' won race 1.
    assert by_driver(rows, "driver_form_finish")["a"] == 1.0


def test_practice_uses_last_session_before_qualifying(conn, weekend):
    rows, meta = prediction_rows(conn, weekend["event"], "practice")
    # FP3, not FP1: the reserve driver 'e' is gone again.
    assert sorted(rows["driver_id"]) == ["a", "b", "c", "d"]
    assert meta["lineup_source"] == "practice_entries"
    assert meta["as_of_utc"] == "2024-03-16T15:00:00Z"           # qualifying start
    assert rows["grid"].isna().all()


def test_post_quali_grid_comes_from_starting_grid_not_result(conn, weekend):
    rows, meta = prediction_rows(conn, weekend["event"], "post_quali")
    assert meta["lineup_source"] == "starting_grid"
    assert meta["as_of_utc"] == "2024-03-17T15:00:00Z"           # race start
    # The penalty-shuffled grid, with 'd' (who was ill on Sunday) still in it.
    assert by_driver(rows, "grid") == {"b": 1, "c": 2, "a": 3, "d": 4}
    # The qualifying gap still comes from qualifying itself: 'a' was on pole.
    assert by_driver(rows, "quali_gap_pct")["a"] == 0.0


def test_post_quali_falls_back_to_qualifying_order(conn, weekend):
    # No starting grid published (yet), and 'd' set no qualifying time.
    conn.execute("DELETE FROM starting_grid")
    conn.execute("DELETE FROM qualifying WHERE session_id = ? AND driver_id = 'd'", (weekend["Q"],))
    rows, meta = prediction_rows(conn, weekend["event"], "post_quali")
    assert meta["lineup_source"] == "qualifying"
    # Qualifying order, and 'd' (in FP3 but no qualifying row) at the back.
    assert by_driver(rows, "grid") == {"a": 1, "b": 2, "c": 3, "d": 4}
    assert math.isnan(by_driver(rows, "quali_gap_pct")["d"])


@pytest.mark.parametrize("stage", ["pre_weekend", "practice", "post_quali"])
def test_changing_the_result_changes_nothing(conn, weekend, stage):
    # The leakage test: rewrite race 2's result completely. No stage's rows
    # may change, because none of them may look at it.
    before, _ = prediction_rows(conn, weekend["event"], stage)
    conn.execute("UPDATE results SET position = 5 - position, grid = 9 WHERE session_id = ?",
                 (weekend["R"],))
    after, _ = prediction_rows(conn, weekend["event"], stage)
    assert before.equals(after)


def test_training_table_is_unchanged_without_upcoming(conn, weekend):
    # Training still sees race 2's real result (it's history for later races).
    table = build_race_features(conn)
    race2 = table[table["session_id"] == weekend["R"]]
    assert sorted(race2["driver_id"]) == ["a", "b", "c", "e"]


def test_upcoming_must_be_one_race(conn, weekend):
    rows, _ = prediction_rows(conn, weekend["event"], "practice")
    two_races = rows.assign(session_id=[1, 2, 1, 2])
    with pytest.raises(ValueError, match="exactly one race"):
        build_race_features(conn, upcoming=two_races)


def test_unknown_stage_rejected(conn, weekend):
    with pytest.raises(ValueError, match="Unknown stage"):
        prediction_rows(conn, weekend["event"], "after_fp1")


@pytest.mark.parametrize("stage", ["pre_weekend", "practice", "post_quali"])
def test_earlier_races_are_the_same_with_or_without_the_predicted_race_swapped_in(conn, weekend, stage):
    # predict/stage.make_prediction builds the feature table twice: once with
    # the predicted race swapped for its as-of rows (prediction_rows), once
    # plain, for the training history. That's only right if every EARLIER
    # race's rows are identical in both (every feature looks backwards), so
    # the model is trained on exactly the history its prediction rows were
    # computed from. Checked on the real database too (2026-11-03: 3,854
    # rows, identical; only actual_position's dtype differs, int vs float,
    # because the upcoming rows have no result).
    import pandas as pd
    from f1predictor.features.as_of import event_info, upcoming_rows

    info = event_info(conn, weekend["event"])
    upcoming, _, _ = upcoming_rows(conn, info, stage)
    plain = build_race_features(conn)
    swapped = build_race_features(conn, upcoming=upcoming)
    earlier = lambda table: table[table["race_date"] < info["race_date"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(earlier(plain), earlier(swapped), check_dtype=False)
