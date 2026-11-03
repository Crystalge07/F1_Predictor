"""Tests for the practice pace features (long-run and short-run gaps).

One hand-built weekend, with lap times chosen so each expected number can be
worked out on paper.
"""

import pytest

from f1predictor.db.connection import get_connection, init_db
from f1predictor.features.practice_pace import practice_pace


@pytest.fixture(autouse=True)
def no_fuel_correction(monkeypatch):
    """These tests use hand-worked lap times without fuel correction, whatever
    value is currently tuned. (autouse = applied to every test in this file;
    the fuel test overrides it.)"""
    import f1predictor.features.practice_pace as pp
    monkeypatch.setattr(pp, "FUEL_EFFECT_MS_PER_LAP", 0.0)


@pytest.fixture
def conn():
    connection = get_connection(":memory:")
    init_db(connection)
    connection.execute("INSERT INTO circuits (circuit_id, name) VALUES ('c', 'C')")
    for driver in ("a", "b"):
        connection.execute("INSERT INTO drivers (driver_id, given_name, family_name) "
                           "VALUES (?, 'X', 'Y')", (driver,))
    yield connection
    connection.close()


def add_weekend(conn, session_starts: dict[str, str]) -> dict[str, int]:
    """An event with the given sessions and start times. Returns {session_type: session_id}."""
    event_id = conn.execute("INSERT INTO events (season, round, name, circuit_id, race_date) "
                            "VALUES (2024, 1, 'GP', 'c', '2024-03-03')").lastrowid
    ids = {}
    for session_type, start in session_starts.items():
        ids[session_type] = conn.execute(
            "INSERT INTO sessions (event_id, session_type, start_time_utc) VALUES (?, ?, ?)",
            (event_id, session_type, start)).lastrowid
    return ids


def add_stint(conn, session_id, car, driver, stint_number, first_lap, lap_times_ms, out_lap=True):
    """One stint: lap numbers first_lap.. in order. The first lap is flagged as a pit out-lap."""
    conn.execute("INSERT OR IGNORE INTO session_entries (session_id, car_number, driver_id) "
                 "VALUES (?, ?, ?)", (session_id, car, driver))
    last_lap = first_lap + len(lap_times_ms) - 1
    conn.execute("INSERT INTO stints (session_id, car_number, stint_number, lap_start, lap_end) "
                 "VALUES (?, ?, ?, ?, ?)", (session_id, car, stint_number, first_lap, last_lap))
    for i, ms in enumerate(lap_times_ms):
        conn.execute("INSERT INTO laps (session_id, car_number, lap_number, lap_time_ms, is_pit_out_lap) "
                     "VALUES (?, ?, ?, ?, ?)",
                     (session_id, car, first_lap + i, ms, int(out_lap and i == 0)))


def standard_weekend(conn):
    s = add_weekend(conn, {"FP2": "2024-03-01T15:00:00Z", "Q": "2024-03-02T15:00:00Z",
                           "R": "2024-03-03T15:00:00Z"})
    # Long runs: out-lap (120 s), 6 race-pace laps, in-lap (130 s).
    # 'a' runs 90.0 s, 'b' runs 91.8 s -> b is 2% slower.
    add_stint(conn, s["FP2"], 1, "a", 1, 1, [120_000] + [90_000] * 6 + [130_000])
    add_stint(conn, s["FP2"], 2, "b", 1, 1, [120_000] + [91_800] * 6 + [130_000])
    # Short runs: 3-lap stints with one fast lap each. a 80.0 s, b 80.4 s -> 0.5%.
    add_stint(conn, s["FP2"], 1, "a", 2, 9, [120_000, 80_000, 130_000])
    add_stint(conn, s["FP2"], 2, "b", 2, 9, [120_000, 80_400, 130_000])
    return s


def pace_of(pace, driver):
    return pace[pace["driver_id"] == driver].iloc[0]


def test_long_and_short_run_gaps(conn):
    s = standard_weekend(conn)
    pace = practice_pace(conn)
    assert set(pace["session_id"]) == {s["R"]}     # keyed by the RACE the features predict
    assert pace_of(pace, "a")["long_run_gap_pct"] == 0.0
    assert pace_of(pace, "b")["long_run_gap_pct"] == pytest.approx(2.0)
    assert pace_of(pace, "a")["short_run_gap_pct"] == 0.0
    assert pace_of(pace, "b")["short_run_gap_pct"] == pytest.approx(0.5)


def test_short_stints_are_not_long_runs(conn):
    s = standard_weekend(conn)
    # 'b' also does a 4-lap stint (2 clean laps after removing out/in laps) at
    # an unrealistically fast 70 s. Too short to count as a long run.
    add_stint(conn, s["FP2"], 2, "b", 3, 20, [120_000, 70_000, 70_000, 130_000])
    assert pace_of(practice_pace(conn), "b")["long_run_gap_pct"] == pytest.approx(2.0)


def test_slow_laps_in_a_long_run_are_dropped(conn):
    s = add_weekend(conn, {"FP2": "2024-03-01T15:00:00Z", "Q": "2024-03-02T15:00:00Z",
                           "R": "2024-03-03T15:00:00Z"})
    add_stint(conn, s["FP2"], 1, "a", 1, 1, [120_000] + [90_000] * 6 + [130_000])
    # 'b' has 6 laps at 91.8 s plus a 110 s lap stuck in traffic. 110 > 105% of
    # the median, so it's dropped and b is still exactly 2% off.
    add_stint(conn, s["FP2"], 2, "b", 1, 1, [120_000] + [91_800] * 3 + [110_000] + [91_800] * 3 + [130_000])
    assert pace_of(practice_pace(conn), "b")["long_run_gap_pct"] == pytest.approx(2.0)


def test_sessions_after_qualifying_are_ignored(conn):
    # 2023-style sprint weekend: the sprint race runs AFTER qualifying, so its
    # laps aren't known at the post-quali stage and must not be used.
    s = add_weekend(conn, {"FP1": "2024-03-01T10:00:00Z", "Q": "2024-03-01T15:00:00Z",
                           "S": "2024-03-02T15:00:00Z", "R": "2024-03-03T15:00:00Z"})
    add_stint(conn, s["FP1"], 1, "a", 1, 1, [120_000] + [90_000] * 6 + [130_000])
    add_stint(conn, s["FP1"], 2, "b", 1, 1, [120_000] + [91_800] * 6 + [130_000])
    # In the sprint, b is much faster. If the sprint leaked in, b's gap would change.
    add_stint(conn, s["S"], 1, "a", 1, 1, [100_000] + [95_000] * 10, out_lap=False)
    add_stint(conn, s["S"], 2, "b", 1, 1, [100_000] + [85_000] * 10, out_lap=False)
    assert pace_of(practice_pace(conn), "b")["long_run_gap_pct"] == pytest.approx(2.0)


def test_unmatched_practice_drivers_are_skipped(conn):
    s = standard_weekend(conn)
    # An FP1-style rookie we couldn't match to Jolpica: no driver_id.
    add_stint(conn, s["FP2"], 99, None, 1, 1, [120_000] + [85_000] * 6 + [130_000])
    pace = practice_pace(conn)
    assert set(pace["driver_id"]) == {"a", "b"}
    # The rookie's faster laps also don't become the reference "best car".
    assert pace_of(pace, "a")["long_run_gap_pct"] == 0.0



def test_fuel_correction_evens_out_burn_off(conn, monkeypatch):
    import f1predictor.features.practice_pace as pp
    s = add_weekend(conn, {"FP2": "2024-03-01T15:00:00Z", "Q": "2024-03-02T15:00:00Z",
                           "R": "2024-03-03T15:00:00Z"})
    # Both cars have the same underlying pace, but 'a' does a LONG stint (12
    # clean laps), getting 100 ms faster every lap purely from burning fuel,
    # while 'b' does a short 6-lap one. Uncorrected, a's later, lighter laps
    # flatter its median; corrected, the gap shrinks to what's left of the
    # shorter stint's own burn-off.
    add_stint(conn, s["FP2"], 1, "a", 1, 1, [120_000] + [90_000 - 100 * i for i in range(12)] + [130_000])
    add_stint(conn, s["FP2"], 2, "b", 1, 1, [120_000] + [90_000 - 100 * i for i in range(6)] + [130_000])

    monkeypatch.setattr(pp, "FUEL_EFFECT_MS_PER_LAP", 0.0)
    raw = pace_of(practice_pace(conn), "b")["long_run_gap_pct"]
    monkeypatch.setattr(pp, "FUEL_EFFECT_MS_PER_LAP", 100.0)
    corrected = pace_of(practice_pace(conn), "b")["long_run_gap_pct"]

    # Raw medians: a = 89.45 s (laps from 90.0 down to 88.9), b = 89.75 s.
    assert raw == pytest.approx((89_750 / 89_450 - 1) * 100)
    # Corrected, every lap of both cars becomes 90.1 s: identical pace.
    assert corrected == pytest.approx(0.0)
