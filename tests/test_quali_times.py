"""Qualifying times when Jolpica publishes the order first and the times later.

What went wrong (2026 R20 Brazil, live run 60): Jolpica's first qualifying
release had the order but no Q1/Q2/Q3 times. The planner counted qualifying
as "in" (it has rows), the post-quali prediction was logged with the gap to
pole blank for all 22 drivers, and nothing was alerted. The times only came
in ~49h later, by luck (another fetch of the season's qualifying).

The fix (features/quali_times.py, pipeline/plan.py, predict/stage.py):
    1. Jolpica has the order but no times -> each driver's best OpenF1
       qualifying lap (deleted laps and timing glitches removed), which is
       the official time for 99.7% of drivers in 2023-2026.
    2. The planner keeps fetching Jolpica's qualifying until its times are
       in, and the post-quali prediction waits for times from either source
       (or the last call before the race).
    3. Logged with the gap missing for most drivers anyway -> a
       'weak_inputs' alert, once.

The end-to-end tests run the real hourly tick through 2027 round 1 in the
fake world (tests/fake_f1.py), like tests/test_data_problems.py. Round 1:
Q Sat Mar 6 15:00, race Sun Mar 7 15:00 UTC.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from f1predictor.features.quali_times import (DELETED_TIME, SOURCE_JOLPICA, SOURCE_NONE, SOURCE_OPENF1,
                                              openf1_best_laps, qualifying_times)
from f1predictor.predict.stage import StagePrediction, input_warnings

from fake_f1 import session_key
from test_data_problems import FakeF1, clean_code, count, new_alerts, run_week  # noqa: F401 (fixture)

TIMES_ARRIVE = datetime(2027, 3, 7, 20, 0, tzinfo=timezone.utc)   # Jolpica's times: Sunday evening


def quali_session(conn):
    return conn.execute("SELECT s.session_id FROM sessions s JOIN events e ON e.event_id = s.event_id "
                        "WHERE e.season = 2027 AND e.round = 1 AND s.session_type = 'Q'").fetchone()[0]


def logged_gaps(conn) -> pd.Series:
    """quali_gap_pct of the live post-quali run of 2027 R1, per driver."""
    return pd.read_sql_query(
        "SELECT f.driver_id, f.value FROM prediction_features f "
        "JOIN prediction_runs r ON r.run_id = f.run_id JOIN events e ON e.event_id = r.event_id "
        "WHERE e.season = 2027 AND e.round = 1 AND r.stage = 'post_quali' AND r.mode = 'live' "
        "AND f.feature = 'quali_gap_pct'", conn).set_index("driver_id")["value"]


# --- The Brazil case, end to end ----------------------------------------------------

@pytest.fixture(scope="module")
def no_times_week():
    """Jolpica publishes round 1's qualifying without times; they arrive Sunday 20:00."""
    world = FakeF1(START_PLACEHOLDER, tamper={(2027, 1, "qualifying"): "no_times"})

    def times_arrive(now, conn):
        if now >= TIMES_ARRIVE:
            world.tamper.pop((2027, 1, "qualifying"), None)

    return run_week_clean(world, times_arrive)


# run_week needs the clean_code fixture's patch; module-scoped fixtures can't
# use function-scoped ones, so the patch is applied by hand here.
START_PLACEHOLDER = datetime(2027, 3, 1, tzinfo=timezone.utc)


def run_week_clean(world, before_tick=None):
    from f1predictor.predict import stage
    original = stage.model_version
    stage.model_version = lambda: "abc1234+settings.00000000"
    try:
        return run_week(world, before_tick=before_tick)
    finally:
        stage.model_version = original


def test_brazil_case_uses_openf1_laps_and_raises_no_alert(no_times_week):
    conn, reports = no_times_week
    gaps = logged_gaps(conn)
    assert len(gaps) == 20
    # Every driver has a gap (Brazil: 22 of 22 blank), from OpenF1's laps.
    assert gaps.notna().all()
    assert gaps.min() == 0                                   # someone is on pole
    predicted = [line for r in reports.values() for line in r.predicted if "post_quali" in line]
    assert len(predicted) == 1 and "qualifying times from openf1" in predicted[0]
    assert not [m for _, m in new_alerts(reports) if "weak inputs" in m]


def test_brazil_case_keeps_fetching_jolpica_until_the_times_arrive(no_times_week):
    conn, reports = no_times_week
    quali_end = datetime(2027, 3, 6, 16, 0, tzinfo=timezone.utc)
    # (Earlier in the week the whole season is fetched once, with the schedule.)
    fetched = sorted(now for now, r in reports.items() if "2027-qualifying" in r.fetched and now > quali_end)
    # Every hour from Q's end + 30 min (Sat 16:30 -> first tick 17:00) until
    # the times are in, then never again.
    assert fetched[0] == datetime(2027, 3, 6, 17, 0, tzinfo=timezone.utc)
    assert fetched[-1] == TIMES_ARRIVE
    assert len(fetched) == (TIMES_ARRIVE - fetched[0]).total_seconds() / 3600 + 1
    # ...and the times are saved for the history (training on this race later).
    q = quali_session(conn)
    assert count(conn, "SELECT COUNT(*) FROM qualifying WHERE session_id = ? AND q1_ms IS NOT NULL", q) == 20


# --- No times anywhere: wait for the last call, then log and alert -----------------------

def test_no_times_anywhere_waits_for_the_last_call_then_alerts():
    # OpenF1's qualifying session is never published either.
    world = FakeF1(START_PLACEHOLDER, broken={(2027, 1, "Q")}, tamper={(2027, 1, "qualifying"): "no_times"})
    conn, reports = run_week_clean(world)
    logged = [now for now, r in reports.items() for line in r.predicted if "post_quali" in line]
    # The last call: within plan.LAST_CALL (2h) of the race at 15:00.
    assert logged == [datetime(2027, 3, 7, 13, 0, tzinfo=timezone.utc)]
    assert logged_gaps(conn).isna().all()
    weak = [m for _, m in new_alerts(reports) if "weak inputs" in m]
    assert len(weak) == 1
    assert "qualifying gap missing for 20 of 20 drivers (qualifying times: none)" in weak[0]


# --- The pieces -------------------------------------------------------------------

def test_deleted_lap_message_parsing():
    def parse(message):
        match = DELETED_TIME.search(message)
        return match and (int(match[1]), match[2], match[3])

    assert parse("CAR 23 (ALB) TIME 1:35.457 DELETED - DOUBLE YELLOW AT TURN 12") == (23, "1", "35.457")
    assert parse("CAR 18 (STR) TIME 1:38.172 DELETED - TRACK LIMITS AT TURN 12 LAP 3 7:34:27") == (18, "1", "38.172")
    assert parse("CAR 4 (NOR) TIME 59.123 DELETED - TRACK LIMITS") == (4, None, "59.123")
    # A lap deleted without its time can't be matched to a lap: not used.
    assert parse("CAR 77 (BOT) LAP DELETED - DOUBLE YELLOW AT TURN 12") is None


def test_a_deleted_lap_and_a_glitch_are_not_anyones_best(no_times_week):
    conn, _ = no_times_week
    q = quali_session(conn)
    before = openf1_best_laps(conn, q).set_index("driver_id")["openf1_best_ms"]
    # The pole-sitter's best lap is deleted by the stewards...
    pole = before.idxmin()
    car = conn.execute("SELECT car_number FROM session_entries WHERE session_id = ? AND driver_id = ?",
                       (q, pole)).fetchone()[0]
    ms = int(before[pole])
    text = f"CAR {car} (XXX) TIME {ms // 60000}:{ms % 60000 / 1000:06.3f} DELETED - TRACK LIMITS AT TURN 4"
    # ...and another car gets a 20 s timing glitch.
    other = before.drop(pole).idxmax()
    other_car = conn.execute("SELECT car_number FROM session_entries WHERE session_id = ? AND driver_id = ?",
                             (q, other)).fetchone()[0]
    # Undone by the rollback below: the module's other tests share this database.
    conn.execute("INSERT INTO race_control (session_id, message_time_utc, category, message) "
                 "VALUES (?, '2027-03-06T15:30:00Z', 'Other', ?)", (q, text))
    conn.execute("INSERT INTO laps (session_id, car_number, lap_number, lap_time_ms) VALUES (?, ?, 99, 70000)",
                 (q, other_car))
    try:
        after = openf1_best_laps(conn, q).set_index("driver_id")["openf1_best_ms"]
        assert after[pole] > before[pole]                  # their next-best lap now
        assert after[other] == before[other]               # the glitch is ignored
        assert (after.drop(pole) == before.drop(pole)).all()
    finally:
        conn.rollback()


def test_jolpica_times_win_when_there_are_any(no_times_week):
    conn, _ = no_times_week
    # By the end of the week Jolpica's times are in: they're used, OpenF1 isn't.
    rows, source = qualifying_times(conn, quali_session(conn))
    assert source == SOURCE_JOLPICA
    assert rows["openf1_best_ms"].isna().all()
    # A driver with no Jolpica time (didn't set one) is NOT given an OpenF1
    # lap: training saw such drivers with no time too.
    q = quali_session(conn)
    conn.execute("UPDATE qualifying SET q1_ms = NULL WHERE session_id = ? AND position = 20", (q,))
    try:
        rows, source = qualifying_times(conn, q)
        assert source == SOURCE_JOLPICA
        assert rows.set_index("quali_position").loc[20, ["q1_ms", "openf1_best_ms"]].isna().all()
    finally:
        conn.rollback()


def test_no_qualifying_rows_at_all_is_source_none(no_times_week):
    conn, _ = no_times_week
    rows, source = qualifying_times(conn, 999_999)
    assert rows.empty and source == SOURCE_NONE


def test_input_warnings_only_when_most_gaps_are_missing():
    def prediction(stage, gaps, source=SOURCE_OPENF1):
        features = pd.DataFrame({"driver_id": [f"d{i}" for i in range(len(gaps))],
                                 "feature": "quali_gap_pct", "value": gaps, "contribution": 0.0})
        return StagePrediction(event_id=1, stage=stage, as_of_utc="", lineup_source="", n_training_races=1,
                               drivers=pd.DataFrame(), features=features, quali_times_source=source)

    assert input_warnings(prediction("post_quali", [0.0, 0.5, None, 1.0])) == []      # one no-time driver: normal
    assert input_warnings(prediction("post_quali", [0.0, 0.5, None, None])) == []     # half: still fine
    warned = input_warnings(prediction("post_quali", [None, None, None, 1.0], SOURCE_NONE))
    assert warned == ["qualifying gap missing for 3 of 4 drivers (qualifying times: none)"]
    # Before qualifying the gap is always blank: nothing to warn about.
    assert input_warnings(prediction("practice", [None] * 4, None)) == []


# --- On the real database ---------------------------------------------------------

def test_openf1_fallback_matches_the_official_times_on_real_data():
    """Every 2023+ qualifying in the real database: the fallback's best lap
    vs Jolpica's official best time. Measured 2026-11-10: 1,772 of 1,778 equal
    (99.7%); the rest are laps deleted without their time in the message."""
    from real_db import connect_read_only
    conn = connect_read_only()
    sessions = [r[0] for r in conn.execute(
        "SELECT DISTINCT l.session_id FROM laps l JOIN sessions s ON s.session_id = l.session_id "
        "WHERE s.session_type = 'Q'")]
    if not sessions:
        pytest.skip("no OpenF1 qualifying laps in this database")
    equal = total = 0
    for session_id in sessions:
        official = pd.read_sql_query("SELECT driver_id, q1_ms, q2_ms, q3_ms FROM qualifying WHERE session_id = ?",
                                     conn, params=(session_id,)).set_index("driver_id").min(axis=1).dropna()
        fallback = openf1_best_laps(conn, session_id).set_index("driver_id")["openf1_best_ms"]
        both = official.index.intersection(fallback.index)
        total += len(both)
        equal += int((official[both] == fallback[both]).sum())
    assert total > 1000
    assert equal / total >= 0.99, f"only {equal} of {total} equal"
