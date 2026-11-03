"""Tests for the post-qualifying feature table.

Small hand-built seasons, so every expected number can be checked by hand.
The most important test is the leakage one: changing a race's result must not
change any feature for that same race.
"""

import math

import pytest

from f1predictor.db.connection import get_connection, init_db
from f1predictor.features.race_features import POST_QUALI_FEATURES, build_race_features


@pytest.fixture
def conn():
    connection = get_connection(":memory:")
    init_db(connection)
    for circuit in ("monza", "spa"):
        connection.execute("INSERT INTO circuits (circuit_id, name) VALUES (?, ?)", (circuit, circuit))
    for team in ("red", "blue"):
        connection.execute("INSERT INTO constructors (constructor_id, name) VALUES (?, ?)", (team, team))
    for driver in ("a", "b", "c", "d"):
        connection.execute("INSERT INTO drivers (driver_id, given_name, family_name) "
                           "VALUES (?, 'X', 'Y')", (driver,))
    yield connection
    connection.close()


def add_race(conn, season, rnd, circuit, entries):
    """entries: {driver: (team, grid, finish_position, position_text, quali_pos, best_lap_ms)}."""
    event_id = conn.execute(
        "INSERT INTO events (season, round, name, circuit_id, race_date) VALUES (?, ?, 'GP', ?, ?)",
        (season, rnd, circuit, f"{season}-{rnd:02d}-01")).lastrowid
    race = conn.execute("INSERT INTO sessions (event_id, session_type) VALUES (?, 'R')",
                        (event_id,)).lastrowid
    quali = conn.execute("INSERT INTO sessions (event_id, session_type) VALUES (?, 'Q')",
                         (event_id,)).lastrowid
    for driver, (team, grid, finish, text, quali_pos, lap_ms) in entries.items():
        conn.execute("INSERT INTO results (session_id, driver_id, constructor_id, grid, position, "
                     "position_text) VALUES (?, ?, ?, ?, ?, ?)",
                     (race, driver, team, grid, finish, text))
        # Q1-only lap for everyone keeps it simple; the best-lap test covers Q2/Q3.
        conn.execute("INSERT INTO qualifying (session_id, driver_id, constructor_id, position, q1_ms) "
                     "VALUES (?, ?, ?, ?, ?)", (quali, driver, team, quali_pos, lap_ms))
    return race


def row(df, session_id, driver):
    """The single feature row for one driver in one race."""
    match = df[(df["session_id"] == session_id) & (df["driver_id"] == driver)]
    assert len(match) == 1
    return match.iloc[0]


def standard_season(conn):
    """Three races. Driver 'b' retires in race 2."""
    r1 = add_race(conn, 2024, 1, "monza", {
        "a": ("red", 1, 1, "1", 1, 80_000), "b": ("red", 2, 2, "2", 2, 80_800),
        "c": ("blue", 3, 3, "3", 3, 81_000), "d": ("blue", 4, 4, "4", 4, 82_000)})
    r2 = add_race(conn, 2024, 2, "spa", {
        "a": ("red", 2, 3, "3", 2, 100_000), "b": ("red", 1, 4, "R", 1, 99_000),
        "c": ("blue", 3, 1, "1", 3, 101_000), "d": ("blue", 4, 2, "2", 4, 102_000)})
    r3 = add_race(conn, 2025, 1, "monza", {
        "a": ("red", 1, 2, "2", 1, 80_000), "b": ("red", 2, 1, "1", 2, 80_400),
        "c": ("blue", 3, 3, "3", 3, 81_000), "d": ("blue", 0, 4, "4", 4, 82_000)})
    return r1, r2, r3


def test_one_row_per_driver_per_race_with_all_columns(conn):
    standard_season(conn)
    df = build_race_features(conn)
    assert len(df) == 12
    assert set(POST_QUALI_FEATURES) <= set(df.columns)


def test_retirements_have_no_target_but_keep_actual_position(conn):
    _, r2, _ = standard_season(conn)
    b = row(build_race_features(conn), r2, "b")
    assert math.isnan(b["finish_position"])     # not trained on
    assert b["actual_position"] == 4            # still scored


def test_quali_gap_to_pole(conn):
    r1, _, _ = standard_season(conn)
    df = build_race_features(conn)
    assert row(df, r1, "a")["quali_gap_pct"] == 0.0                     # pole
    assert row(df, r1, "b")["quali_gap_pct"] == pytest.approx(1.0)      # 80.8 / 80.0 = 1% slower


def test_best_quali_lap_uses_fastest_segment(conn):
    r1, _, _ = standard_season(conn)
    # Give 'c' a Q2 lap faster than everyone's Q1: c becomes the pole reference.
    conn.execute("UPDATE qualifying SET q2_ms = 79000 WHERE driver_id = 'c' "
                 "AND session_id = (SELECT session_id FROM sessions WHERE event_id = "
                 "(SELECT event_id FROM sessions WHERE session_id = ?) AND session_type = 'Q')", (r1,))
    df = build_race_features(conn)
    assert row(df, r1, "c")["quali_gap_pct"] == 0.0
    assert row(df, r1, "a")["quali_gap_pct"] == pytest.approx((80_000 / 79_000 - 1) * 100)


def test_pit_lane_start_goes_to_the_back(conn):
    _, _, r3 = standard_season(conn)
    d = row(build_race_features(conn), r3, "d")
    assert d["pit_lane_start"] == 1
    assert d["grid"] == 4                       # 4-car field: back of the grid, not "0 = pole"


def test_first_race_has_no_history(conn):
    r1, _, _ = standard_season(conn)
    a = row(build_race_features(conn), r1, "a")
    for feature in ("driver_form_finish", "team_season_finish", "driver_track_finish"):
        assert math.isnan(a[feature])


def test_driver_form_skips_retirements(conn):
    _, _, r3 = standard_season(conn)
    df = build_race_features(conn)
    # 'a' finished 1st and 3rd before race 3 -> 2.0.
    assert row(df, r3, "a")["driver_form_finish"] == 2.0
    # 'b' finished 2nd, then retired: only the finish counts -> 2.0.
    assert row(df, r3, "b")["driver_form_finish"] == 2.0
    # Grid form counts every start: b started 2nd and 1st -> 1.5.
    assert row(df, r3, "b")["driver_form_grid"] == 1.5


def test_team_season_pace_resets_each_season(conn):
    _, r2, r3 = standard_season(conn)
    df = build_race_features(conn)
    # Race 2: red's race-1 average finish was (1 + 2) / 2 = 1.5.
    assert row(df, r2, "a")["team_season_finish"] == 1.5
    assert row(df, r2, "b")["team_season_quali"] == 1.5
    # Race 3 is a new season -> no team pace yet.
    assert math.isnan(row(df, r3, "a")["team_season_finish"])


def test_track_history_spans_seasons(conn):
    _, _, r3 = standard_season(conn)
    df = build_race_features(conn)
    # Back at Monza in 2025: c finished 3rd there in 2024. Spa doesn't count.
    assert row(df, r3, "c")["driver_track_finish"] == 3.0
    # Blue at Monza 2024: (3 + 4) / 2 = 3.5.
    assert row(df, r3, "c")["team_track_finish"] == 3.5


def test_no_leakage_from_the_race_being_predicted(conn):
    """Shuffling race 3's finishing order must not change race 3's features."""
    _, _, r3 = standard_season(conn)
    before = build_race_features(conn)
    conn.execute("UPDATE results SET position = 5 - position, position_text = CAST(5 - position AS TEXT) "
                 "WHERE session_id = ?", (r3,))
    after = build_race_features(conn)

    for driver in ("a", "b", "c", "d"):
        for feature in POST_QUALI_FEATURES:
            x, y = row(before, r3, driver)[feature], row(after, r3, driver)[feature]
            assert (math.isnan(x) and math.isnan(y)) or x == y, f"{driver} {feature} leaked"
