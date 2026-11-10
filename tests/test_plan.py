"""Tests for the pipeline planner (pipeline/plan.py), with a fake clock.

A two-round 2027 season. Round 1 (a week earlier) is over and has a result.
Round 2 is a normal weekend:
    FP1 Fri 12:00   FP2 Fri 16:00   FP3 Sat 11:00   Q Sat 15:00   R Sun 15:00   (UTC)
Each test sets the clock and the database state, and checks what plan() asks for.
"""

from datetime import datetime, timezone

import pytest

from f1predictor.db import ingestion_log
from f1predictor.db.connection import get_connection, init_db
from f1predictor.pipeline.plan import plan

WEEKEND = {"FP1": "2027-03-12T12:00:00Z", "FP2": "2027-03-12T16:00:00Z", "FP3": "2027-03-13T11:00:00Z",
           "Q": "2027-03-13T15:00:00Z", "R": "2027-03-14T15:00:00Z"}


def at(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


@pytest.fixture
def db():
    c = get_connection(":memory:")
    init_db(c)
    c.execute("INSERT INTO circuits (circuit_id, name) VALUES ('x', 'X')")
    c.execute("INSERT INTO drivers (driver_id, given_name, family_name) VALUES ('a', 'A', 'A')")
    c.execute("INSERT INTO constructors (constructor_id, name) VALUES ('t', 'T')")
    ids = {}
    for rnd, sessions in [(1, {"R": "2027-03-07T15:00:00Z"}), (2, WEEKEND)]:
        event = c.execute("INSERT INTO events (season, round, name, circuit_id, race_date) "
                          "VALUES (2027, ?, 'GP', 'x', ?)", (rnd, sessions["R"][:10])).lastrowid
        ids[rnd] = {"event": event}
        for kind, start in sessions.items():
            ids[rnd][kind] = c.execute("INSERT INTO sessions (event_id, session_type, start_time_utc) "
                                       "VALUES (?, ?, ?)", (event, kind, start)).lastrowid
    # Round 1 is fully done: result in, OpenF1 fetched.
    add_result(c, ids[1]["R"])
    ingestion_log.record(c, "openf1", "2027-01-R", "ok")
    yield c, ids
    c.close()


@pytest.fixture
def conn(db):
    return db[0]


@pytest.fixture
def ids(db):
    """{round: {'event': event_id, 'FP1': session_id, ...}}"""
    return db[1]


def add_result(c, session_id):
    c.execute("INSERT INTO results (session_id, driver_id, constructor_id, position, position_text) "
              "VALUES (?, 'a', 't', 1, '1')", (session_id,))


def add_live_run(c, event_id, stage, created="2027-03-01T00:00:00Z"):
    return c.execute("INSERT INTO prediction_runs (event_id, stage, mode, as_of_utc, created_at_utc, "
                     "model_version, settings_json) VALUES (?, ?, 'live', 'x', ?, 'v', '{}')",
                     (event_id, stage, created)).lastrowid


def fetched(c, *items):
    for item in items:
        ingestion_log.record(c, "openf1", f"2027-02-{item}", "ok")


def round2(c, now):
    """What's planned for round 2 (round 1's 'missed' notes are left out)."""
    return [(a.kind, a.target) for a in plan(c, now) if a.round == 2]


def test_monday_after_previous_race_predicts_pre_weekend(conn):
    assert round2(conn, at("2027-03-08T09:00")) == [("predict", "pre_weekend")]


def test_a_logged_stage_is_not_planned_again(conn, ids):
    add_live_run(conn, ids[2]["event"], "pre_weekend")
    assert round2(conn, at("2027-03-08T09:00")) == []


def test_pre_weekend_waits_for_the_previous_result(conn):
    conn.execute("DELETE FROM results")
    assert ("predict", "pre_weekend") not in round2(conn, at("2027-03-08T09:00"))


def test_fetch_a_session_once_it_has_ended(conn, ids):
    add_live_run(conn, ids[2]["event"], "pre_weekend")
    assert round2(conn, at("2027-03-12T13:00")) == []                      # FP1 just ended: too early
    assert round2(conn, at("2027-03-12T13:31")) == [("fetch_session", "FP1")]
    fetched(conn, "FP1")
    assert round2(conn, at("2027-03-12T13:31")) == []


def test_practice_stage_once_all_practice_is_in(conn, ids):
    add_live_run(conn, ids[2]["event"], "pre_weekend")
    fetched(conn, "FP1", "FP2", "FP3")
    assert round2(conn, at("2027-03-13T12:40")) == [("predict", "practice")]


def test_practice_stage_last_call_with_a_missing_session(conn, ids):
    # FP3 never came through (cancelled, or OpenF1 is broken). Practice must
    # not be blocked by it: at the last call, predict with what there is.
    add_live_run(conn, ids[2]["event"], "pre_weekend")
    fetched(conn, "FP1", "FP2")
    assert round2(conn, at("2027-03-13T12:40")) == [("fetch_session", "FP3")]
    assert round2(conn, at("2027-03-13T13:05")) == [("fetch_session", "FP3"), ("predict", "practice")]


def test_post_quali_needs_qualifying_then_the_grid_or_last_call(conn, ids):
    event = ids[2]["event"]
    for stage in ("pre_weekend", "practice"):
        add_live_run(conn, event, stage)
    fetched(conn, "FP1", "FP2", "FP3")
    # Saturday evening: qualifying over, nothing fetched yet.
    assert round2(conn, at("2027-03-13T16:31")) == [("fetch_session", "Q"), ("fetch_jolpica", "qualifying")]
    # Qualifying results in, but no starting grid yet: wait...
    fetched(conn, "Q")
    conn.execute("INSERT INTO qualifying (session_id, driver_id, constructor_id, position, q1_ms) "
                 "VALUES (?, 'a', 't', 1, 80000)", (ids[2]["Q"],))
    # (but keep asking OpenF1 for the grid)...
    assert round2(conn, at("2027-03-14T09:00")) == [("fetch_session", "grid")]
    # ...until the last call (falls back to qualifying order)...
    assert round2(conn, at("2027-03-14T13:05")) == [("fetch_session", "grid"), ("predict", "post_quali")]
    # ...or until the grid arrives.
    fetched(conn, "grid")
    assert round2(conn, at("2027-03-14T09:00")) == [
        ("fetch_session", "grid"), ("predict", "post_quali")]


def test_qualifying_order_without_times(conn, ids):
    # 2026 R20 Brazil: Jolpica published the qualifying ORDER, no times.
    event = ids[2]["event"]
    for stage in ("pre_weekend", "practice"):
        add_live_run(conn, event, stage)
    fetched(conn, "FP1", "FP2", "FP3", "grid")
    conn.execute("INSERT INTO qualifying (session_id, driver_id, constructor_id, position) "
                 "VALUES (?, 'a', 't', 1)", (ids[2]["Q"],))
    # No times anywhere (OpenF1's qualifying not fetched either): keep asking
    # Jolpica for them, and don't predict yet...
    # (The grid refresh waits for OpenF1's qualifying, as before.)
    assert round2(conn, at("2027-03-14T09:00")) == [("fetch_session", "Q"), ("fetch_jolpica", "qualifying")]
    # ...unless it's the last call: predict with what there is.
    assert ("predict", "post_quali") in round2(conn, at("2027-03-14T13:05"))
    # OpenF1's qualifying laps are in: predict (with those times), and keep
    # asking Jolpica for its own (needed later, for training).
    fetched(conn, "Q")
    assert round2(conn, at("2027-03-14T09:00")) == [
        ("fetch_session", "grid"), ("fetch_jolpica", "qualifying"), ("predict", "post_quali")]
    # Jolpica's times arrive: no more fetching it.
    conn.execute("UPDATE qualifying SET q1_ms = 80000 WHERE session_id = ?", (ids[2]["Q"],))
    assert ("fetch_jolpica", "qualifying") not in round2(conn, at("2027-03-14T09:00"))


def test_jolpica_qualifying_times_late_after_48h(conn, ids):
    for stage in ("pre_weekend", "practice", "post_quali"):
        add_live_run(conn, ids[2]["event"], stage)
    fetched(conn, "FP1", "FP2", "FP3", "Q", "grid", "R")
    conn.execute("INSERT INTO qualifying (session_id, driver_id, constructor_id, position) "
                 "VALUES (?, 'a', 't', 1)", (ids[2]["Q"],))
    add_result(conn, ids[2]["R"])
    planned = round2(conn, at("2027-03-15T17:00"))       # Q ended Sat 16:30 + 48h
    assert ("fetch_jolpica", "qualifying") in planned and ("late", "qualifying") in planned


def test_a_stage_whose_cutoff_passed_is_missed_not_predicted_late(conn, ids):
    # Live since round 1. Then the pipeline was down all of round 2's
    # weekend: Sunday 16:00, nothing logged for it.
    add_live_run(conn, ids[1]["event"], "post_quali", created="2027-03-07T12:00:00Z")
    planned = round2(conn, at("2027-03-14T16:00"))
    assert ("missed", "pre_weekend") in planned and ("missed", "post_quali") in planned
    assert not any(kind == "predict" for kind, _ in planned)


def test_score_once_the_result_is_in(conn, ids):
    event = ids[2]["event"]
    for stage in ("pre_weekend", "practice", "post_quali"):
        add_live_run(conn, event, stage)
    fetched(conn, "FP1", "FP2", "FP3", "Q", "grid")
    conn.execute("INSERT INTO qualifying (session_id, driver_id, constructor_id, position, q1_ms) "
                 "VALUES (?, 'a', 't', 1, 80000)", (ids[2]["Q"],))
    # Race over (+2h +30m): fetch OpenF1 and Jolpica; no result yet, so no scoring.
    assert round2(conn, at("2027-03-14T17:31")) == [("fetch_session", "R"), ("fetch_jolpica", "results")]
    fetched(conn, "R")
    add_result(conn, ids[2]["R"])
    assert round2(conn, at("2027-03-14T18:00")) == [("score", "race")]


def test_season_opener_is_predicted_within_two_weeks_of_fp1(conn):
    # Round 1 is the opener: no previous race this season.
    conn.execute("DELETE FROM prediction_runs")
    opener = [a for a in plan(conn, at("2027-02-15T09:00")) if a.round == 1]
    assert opener == []                                            # > 14 days before
    opener = [(a.kind, a.target) for a in plan(conn, at("2027-02-25T09:00")) if a.round == 1]
    assert ("predict", "pre_weekend") in opener


def test_fetches_come_before_predictions(conn, ids):
    # Both due at once: the practice prediction needs FP3's data, so fetching goes first.
    add_live_run(conn, ids[2]["event"], "pre_weekend")
    fetched(conn, "FP1", "FP2")
    kinds = [kind for kind, _ in round2(conn, at("2027-03-13T13:05"))]
    assert kinds == ["fetch_session", "predict"]


def test_races_before_going_live_are_never_missed(conn):
    # No live run has ever been made (e.g. the backtested past): nothing is "missed".
    assert not any(a.kind == "missed" for a in plan(conn, at("2027-03-14T16:00")))


def test_starting_grid_is_refreshed_until_post_quali_is_logged(conn, ids):
    for stage in ("pre_weekend", "practice"):
        add_live_run(conn, ids[2]["event"], stage)
    fetched(conn, "FP1", "FP2", "FP3", "Q")
    assert ("fetch_session", "grid") in round2(conn, at("2027-03-13T17:00"))
    fetched(conn, "grid")
    # A cached success may pre-date a penalty, so refresh immediately before
    # the immutable prediction instead of trusting ingestion_log forever.
    assert ("fetch_session", "grid") in round2(conn, at("2027-03-13T17:00"))
    add_live_run(conn, ids[2]["event"], "post_quali")
    assert ("fetch_session", "grid") not in round2(conn, at("2027-03-13T17:00"))


def test_grid_is_not_chased_after_the_race_starts(conn, ids):
    fetched(conn, "FP1", "FP2", "FP3", "Q")
    assert ("fetch_session", "grid") not in round2(conn, at("2027-03-14T15:30"))


def test_a_session_that_never_arrives_is_given_up_after_a_day(conn, ids):
    add_live_run(conn, ids[2]["event"], "pre_weekend")
    # FP1 ready at 13:30 Friday; still nothing on Saturday 14:00.
    planned = round2(conn, at("2027-03-13T14:00"))
    assert ("gave_up", "FP1") in planned and ("fetch_session", "FP1") not in planned


def test_a_missing_season_calendar_is_fetched(conn):
    # Our fixture only has 2027; on New Year's Day 2028 the planner asks for 2028's calendar.
    planned = [(a.kind, a.season, a.target) for a in plan(conn, at("2028-01-01T09:00"))]
    assert planned == [("fetch_jolpica", 2028, "schedule")]
    assert not any(a.target == "schedule" for a in plan(conn, at("2027-03-08T09:00")))


def test_jolpica_results_missing_two_days_are_late_but_still_fetched(conn, ids):
    for stage in ("pre_weekend", "practice", "post_quali"):
        add_live_run(conn, ids[2]["event"], stage)
    fetched(conn, "FP1", "FP2", "FP3", "Q", "grid", "R")
    conn.execute("INSERT INTO qualifying (session_id, driver_id, constructor_id, position, q1_ms) "
                 "VALUES (?, 'a', 't', 1, 80000)", (ids[2]["Q"],))
    planned = round2(conn, at("2027-03-16T18:00"))     # race over Sun 17:30; now Tue 18:00
    assert ("fetch_jolpica", "results") in planned and ("late", "results") in planned


# -----------------------------------------------------------------------------
# Results revised after the race (architecture review, 2026-11-03)
# -----------------------------------------------------------------------------

def scored_round2(c, ids):
    """Round 2 raced, its result in and its three live runs scored against it."""
    from f1predictor.evaluate.scorer import record_scored_snapshot, result_snapshot
    run_ids = [add_live_run(c, ids[2]["event"], stage) for stage in ("pre_weekend", "practice", "post_quali")]
    fetched(c, "FP1", "FP2", "FP3", "Q", "grid", "R")
    c.execute("INSERT INTO qualifying (session_id, driver_id, constructor_id, position, q1_ms) "
              "VALUES (?, 'a', 't', 1, 80000)", (ids[2]["Q"],))
    add_result(c, ids[2]["R"])
    for run_id in run_ids:
        c.execute("INSERT INTO scores (run_id, metric, model_value, scored_at_utc) "
                  "VALUES (?, 'position_mae', 0, 'x')", (run_id,))
    record_scored_snapshot(c, ids[2]["event"], result_snapshot(c, ids[2]["event"]))


def test_the_result_is_rechecked_24h_and_72h_after_the_race(conn, ids):
    from f1predictor.pipeline.plan import RESULT_RECHECKS, recheck_key
    scored_round2(conn, ids)
    # Race ready Sun 17:30. Nothing to do on Sunday evening...
    assert round2(conn, at("2027-03-14T18:00")) == []
    # ...a re-check from Monday 17:30...
    assert round2(conn, at("2027-03-15T17:31")) == [("fetch_jolpica", "recheck_24h")]
    # ...done once it worked (pipeline/fetch.py marks it)...
    ingestion_log.record(conn, "jolpica", recheck_key(2027, 2, RESULT_RECHECKS[0]), "ok")
    assert round2(conn, at("2027-03-15T18:00")) == []
    # ...and the second one from Wednesday 17:30.
    assert round2(conn, at("2027-03-17T17:31")) == [("fetch_jolpica", "recheck_72h")]


def test_a_failed_recheck_is_planned_again(conn, ids):
    from f1predictor.pipeline.plan import RESULT_RECHECKS, recheck_key
    scored_round2(conn, ids)
    ingestion_log.record(conn, "jolpica", recheck_key(2027, 2, RESULT_RECHECKS[0]), "failed", message="503")
    assert ("fetch_jolpica", "recheck_24h") in round2(conn, at("2027-03-15T19:00"))


def test_a_result_changed_after_scoring_is_rescored(conn, ids):
    scored_round2(conn, ids)
    assert round2(conn, at("2027-03-14T18:00")) == []
    # The stewards disqualify the winner on Sunday night.
    # (Jolpica keeps a classified position on a disqualified row; position_text says 'D'.)
    conn.execute("UPDATE results SET position_text = 'D' WHERE session_id = ?", (ids[2]["R"],))
    assert round2(conn, at("2027-03-14T23:00")) == [("rescore", "race")]


def test_a_race_scored_before_snapshots_existed_is_rescored_once(conn, ids):
    scored_round2(conn, ids)
    conn.execute("DELETE FROM scored_results")
    [action] = [a for a in plan(conn, at("2027-03-14T18:00")) if a.round == 2]
    assert action.kind == "rescore" and "no record" in action.reason
