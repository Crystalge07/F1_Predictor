"""Tests for the validation checks.

Each test builds a tiny database with one specific problem (or one specific
legitimate oddity, like a pit-lane start) and checks that the right issue is
raised (or that nothing is). Testing the "legitimate oddity" cases matters as
much as the broken ones: a check that cries wolf gets ignored.
"""

import pytest

from f1predictor.db.connection import get_connection, init_db
from f1predictor.validate import checks


@pytest.fixture
def conn():
    connection = get_connection(":memory:")
    init_db(connection)
    yield connection
    connection.close()


def make_race(conn, n_drivers=20, season=2024, round_number=1, overrides=None):
    """Insert an event + race session with a clean n-driver result. Returns session_id.

    overrides: {position: {column: value}} to break specific rows,
    e.g. {2: {"position": 1}} makes two drivers finish first.
    """
    overrides = overrides or {}
    conn.execute("INSERT INTO circuits (circuit_id, name) VALUES ('c', 'C') ON CONFLICT DO NOTHING")
    conn.execute("INSERT INTO constructors (constructor_id, name) VALUES ('t', 'T') "
                 "ON CONFLICT DO NOTHING")
    event_id = conn.execute(
        "INSERT INTO events (season, round, name, circuit_id, race_date) "
        "VALUES (?, ?, 'GP', 'c', '2024-01-01')", (season, round_number)).lastrowid
    session_id = conn.execute(
        "INSERT INTO sessions (event_id, session_type) VALUES (?, 'R')", (event_id,)).lastrowid

    for p in range(1, n_drivers + 1):
        row = {"driver_id": f"d{p}", "car_number": p, "grid": p, "position": p,
               "position_text": str(p), "laps_completed": 50}
        row.update(overrides.get(p, {}))
        conn.execute("INSERT INTO drivers (driver_id, code, given_name, family_name) "
                     "VALUES (?, ?, 'X', 'Y') ON CONFLICT DO NOTHING",
                     (row["driver_id"], f"D{p:02d}"))
        conn.execute(
            "INSERT INTO results (session_id, driver_id, constructor_id, car_number, grid, "
            "position, position_text, laps_completed) VALUES (?, ?, 't', ?, ?, ?, ?, ?)",
            (session_id, row["driver_id"], row["car_number"], row["grid"],
             row["position"], row["position_text"], row["laps_completed"]))
    return session_id


def check_names(issues):
    return {(i.check, i.severity) for i in issues}


# -----------------------------------------------------------------------------
# Race / sprint classification
# -----------------------------------------------------------------------------

def test_clean_race_has_no_issues(conn):
    assert checks.check_classification(conn, make_race(conn)) == []


def test_pit_lane_starts_are_not_duplicate_grid_slots(conn):
    # Two drivers starting from the pit lane both have grid = 0. That's real.
    race = make_race(conn, overrides={19: {"grid": 0}, 20: {"grid": 0}})
    assert checks.check_classification(conn, race) == []


def test_duplicate_positions_are_an_error(conn):
    race = make_race(conn, overrides={2: {"position": 1}})
    assert ("positions_not_sequential", "error") in check_names(
        checks.check_classification(conn, race))


def test_duplicate_grid_slots_are_an_error(conn):
    race = make_race(conn, overrides={2: {"grid": 1}})
    assert ("duplicate_grid_slots", "error") in check_names(
        checks.check_classification(conn, race))


def test_partial_result_is_an_error(conn):
    race = make_race(conn, n_drivers=8)
    assert ("too_few_results", "error") in check_names(checks.check_classification(conn, race))


def test_unusual_field_size_is_only_a_warning(conn):
    # Like Australia 2024: 19 starters when the season usually has 20.
    make_race(conn, round_number=1)
    make_race(conn, round_number=2)
    short_race = make_race(conn, n_drivers=19, round_number=3)
    assert check_names(checks.check_classification(conn, short_race)) == {
        ("unusual_entry_count", "warning")}


def test_quali_disqualification_duplicates_are_only_a_warning(conn):
    # Real case, Zandvoort 2024: Albon disqualified, but Jolpica lists him at
    # P10 AND moves Sainz up to P10. Positions: 1..10, 10, 11..19.
    race = make_race(conn)
    event_id = conn.execute("SELECT event_id FROM sessions WHERE session_id = ?",
                            (race,)).fetchone()[0]
    quali = conn.execute("INSERT INTO sessions (event_id, session_type) VALUES (?, 'Q')",
                         (event_id,)).lastrowid
    positions = list(range(1, 11)) + list(range(10, 20))
    for p, position in enumerate(positions, start=1):
        conn.execute("INSERT INTO qualifying (session_id, driver_id, constructor_id, position) "
                     "VALUES (?, ?, 't', ?)", (quali, f"d{p}", position))

    assert check_names(checks.check_qualifying(conn, quali)) == {
        ("positions_not_sequential", "warning")}


# -----------------------------------------------------------------------------
# Schedule
# -----------------------------------------------------------------------------

def test_schedule_with_no_events_is_an_error(conn):
    assert check_names(checks.check_schedule(conn, 2030)) == {("no_events", "error")}


def test_event_without_race_session_is_an_error(conn):
    conn.execute("INSERT INTO circuits (circuit_id, name) VALUES ('c', 'C')")
    conn.execute("INSERT INTO events (season, round, name, circuit_id, race_date) "
                 "VALUES (2024, 1, 'GP', 'c', '2024-01-01')")
    assert ("event_missing_race_session", "error") in check_names(checks.check_schedule(conn, 2024))


# -----------------------------------------------------------------------------
# OpenF1 session
# -----------------------------------------------------------------------------

def add_openf1_data(conn, session_id, laps_per_car, cars=range(1, 21), stints=True, weather=True):
    """Add entries (matched to d1..d20), laps, stints and weather to a race session."""
    for car in cars:
        conn.execute("INSERT INTO session_entries (session_id, car_number, acronym, driver_id) "
                     "VALUES (?, ?, ?, ?)", (session_id, car, f"D{car:02d}", f"d{car}"))
        for lap in range(1, laps_per_car + 1):
            conn.execute("INSERT INTO laps (session_id, car_number, lap_number, lap_time_ms) "
                         "VALUES (?, ?, ?, 90000)", (session_id, car, lap))
        if stints:
            conn.execute("INSERT INTO stints (session_id, car_number, stint_number, compound, "
                         "lap_start, lap_end, tyre_age_at_start) VALUES (?, ?, 1, 'HARD', 1, ?, 0)",
                         (session_id, car, laps_per_car))
    if weather:
        conn.execute("INSERT INTO weather (session_id, sample_time_utc) VALUES (?, '2024-01-01T13:00:00Z')",
                     (session_id,))


def test_clean_openf1_race_has_no_issues(conn):
    race = make_race(conn)                    # every driver: 50 laps completed
    add_openf1_data(conn, race, laps_per_car=50)
    assert checks.check_openf1_session(conn, race) == []


def test_retirement_lap_is_tolerated(conn):
    # OpenF1 records the lap a driver retired on; Jolpica doesn't count it.
    race = make_race(conn, overrides={20: {"laps_completed": 30}})
    add_openf1_data(conn, race, laps_per_car=50, cars=range(1, 20))
    add_openf1_data(conn, race, laps_per_car=31, cars=[20], weather=False)
    assert checks.check_openf1_session(conn, race) == []


def test_disqualified_drivers_are_left_out_of_lap_count_check(conn):
    # Real case (China 2025): a disqualified driver ran all 50 laps, but
    # Jolpica records 0 laps completed for them.
    race = make_race(conn, overrides={20: {"laps_completed": 0, "position_text": "D"}})
    add_openf1_data(conn, race, laps_per_car=50)
    assert checks.check_openf1_session(conn, race) == []


def test_unmatched_car_without_laps_is_ignored(conn):
    race = make_race(conn)
    add_openf1_data(conn, race, laps_per_car=50)
    conn.execute("INSERT INTO session_entries (session_id, car_number) VALUES (?, 236)", (race,))
    assert checks.check_openf1_session(conn, race) == []


def test_half_a_race_is_an_error(conn):
    race = make_race(conn)
    add_openf1_data(conn, race, laps_per_car=20)   # official result says 50 each
    assert ("race_laps_incomplete", "error") in check_names(checks.check_openf1_session(conn, race))


def test_missing_stints_and_weather_are_warnings(conn):
    race = make_race(conn)
    add_openf1_data(conn, race, laps_per_car=50, stints=False, weather=False)
    assert check_names(checks.check_openf1_session(conn, race)) == {
        ("laps_without_stint", "warning"), ("no_weather", "warning")}


# -----------------------------------------------------------------------------
# Plumbing
# -----------------------------------------------------------------------------

def test_raise_if_errors_ignores_warnings():
    checks.raise_if_errors([checks.Issue("x", "warning", "fine")])   # no exception
    with pytest.raises(checks.ValidationError, match="bad"):
        checks.raise_if_errors([checks.Issue("x", "error", "bad")])


def test_record_issues_replaces_previous_issues(conn):
    checks.record_issues(conn, "jolpica", "2024-results", [checks.Issue("a", "warning", "1"),
                                                            checks.Issue("b", "warning", "2")])
    checks.record_issues(conn, "jolpica", "2024-results", [])    # re-validated: now clean
    assert conn.execute("SELECT COUNT(*) FROM validation_issues").fetchone()[0] == 0
