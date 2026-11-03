"""Late, broken and cancelled data, run through the real hourly tick.

The PRD's rule for data problems is "retry, validate, alert and skip; never
crash or save bad data". This file checks each kind of problem end to end:
the fake world (tests/fake_f1.py) misbehaves in one specific way, the tick
runs every hour through 2027 round 1's week, and the tests check what was
saved, what was alerted, and that the weekend still got predicted and scored.

Two kinds of data problem are deliberately treated differently (decision
2026-11-03):

    LATE data     (not published yet, empty, a server error) fixes itself:
                  retry every tick, alert only if it lasts (24h give-up for
                  OpenF1, 48h 'late' for Jolpica).
    BROKEN data   (fails validation: a partial result, a missing driver)
                  won't fix itself in minutes: never saved, alerted at once
                  (and still retried, in case the source corrects it).

Round 1's weekend (all UTC): FP1 Fri Mar 5 12:00, FP2 Fri 16:00, FP3 Sat
11:00, Q Sat 15:00, race Sun Mar 7 15:00. Each session's data is fetched by
the first tick after it ends + 30 minutes (pipeline/plan.py), e.g. FP2 at
Fri 18:00 and the race at Sun 18:00.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest
import requests

from f1predictor.db.connection import get_connection, init_db
from f1predictor.ingest import jolpica
from f1predictor.ingest.http import JsonHttpClient
from f1predictor.pipeline import tick
from f1predictor.predict import stage

from fake_f1 import FakeF1, session_key

START = datetime(2027, 3, 1, 0, 0, tzinfo=timezone.utc)   # Monday before round 1
END = datetime(2027, 3, 8, 6, 0, tzinfo=timezone.utc)     # Monday morning after it
HOUR = timedelta(hours=1)
N_SIMS = 2_000

FP2_FIRST_FETCH = datetime(2027, 3, 5, 18, 0, tzinfo=timezone.utc)
RACE_FIRST_FETCH = datetime(2027, 3, 7, 18, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def clean_code(monkeypatch):
    # A dirty working tree blocks live predictions; pretend it's a clean commit.
    monkeypatch.setattr(stage, "model_version", lambda: "abc1234+settings.00000000")


def run_week(world, openf1_client=None, before_tick=None):
    """Tick hourly from START to END in `world`. Returns (conn, {tick time: report}).

    openf1_client: what the tick fetches OpenF1 through (default: the world itself).
    before_tick(now, conn): called before each tick, e.g. to check what's saved
    or to make the world fix its data at some point.
    """
    conn = get_connection(":memory:")
    init_db(conn)
    jolpica.ingest_season(conn, world, 2026)                # history from before go-live
    reports, now = {}, START
    while now <= END:
        if before_tick:
            before_tick(now, conn)
        world.now = now
        reports[now] = tick.run_tick(conn, now=now, jolpica_client=world,
                                     openf1_client=openf1_client or world, n_sims=N_SIMS)
        now += HOUR
    return conn, reports


def new_alerts(reports):
    """[(tick time, message)] for every alert raised, across the week."""
    return [(now, m) for now, report in reports.items() for m in report.new_alerts]


def count(conn, sql, *params):
    return conn.execute(sql, params).fetchone()[0]


def laps_in(conn, session_type):
    return count(conn, "SELECT COUNT(*) FROM laps l JOIN sessions s USING (session_id) JOIN events e "
                       "USING (event_id) WHERE e.season = 2027 AND e.round = 1 AND s.session_type = ?",
                 session_type)


def race_results(conn):
    return count(conn, "SELECT COUNT(*) FROM results r JOIN sessions s USING (session_id) JOIN events e "
                       "USING (event_id) WHERE e.season = 2027 AND e.round = 1 AND s.session_type = 'R'")


def scored_runs(conn):
    return count(conn, "SELECT COUNT(DISTINCT run_id) FROM scores")


def logged_stages(conn, rnd=1):
    return {r[0] for r in conn.execute(
        "SELECT r.stage FROM prediction_runs r JOIN events e USING (event_id) "
        "WHERE e.season = 2027 AND e.round = ? AND r.mode = 'live'", (rnd,))}


# -----------------------------------------------------------------------------
# Late data: retried, nothing empty saved, no alert
# -----------------------------------------------------------------------------

def test_empty_openf1_laps_are_retried_and_never_saved():
    # FP2 is "published" but has no laps for its first 3 fetch attempts
    # (Fri 18:00, 19:00, 20:00); the laps appear at 21:00.
    world = FakeF1(now=START, empty_until={(2027, 1, "FP2"): FP2_FIRST_FETCH + 3 * HOUR})
    seen = {}

    def watch(now, conn):
        seen[now] = laps_in(conn, "FP2")       # FP2 laps saved BEFORE this tick

    conn, reports = run_week(world, before_tick=watch)

    # Each of the 3 empty attempts failed (NoDataYet), nothing was saved...
    for attempt in range(3):
        now = FP2_FIRST_FETCH + attempt * HOUR
        assert reports[now].fetched["2027-01-FP2"] == "failed"
        assert seen[now + HOUR] == 0                           # still nothing after it
    # ...and the 4th attempt loaded it (20 cars x 8 laps).
    assert reports[FP2_FIRST_FETCH + 3 * HOUR].fetched["2027-01-FP2"] == "ok"
    assert laps_in(conn, "FP2") == 160
    # Late data isn't an alert (only data missing for 24h is), and the
    # weekend went on as normal: every stage logged, the race scored.
    assert new_alerts(reports) == []
    assert logged_stages(conn) == {"pre_weekend", "practice", "post_quali"}
    assert scored_runs(conn) == 3


def test_a_late_jolpica_result_is_retried_and_scoring_waits_for_it():
    # Jolpica's race result is missing for the first 3 attempts after the race.
    world = FakeF1(now=START, empty_until={(2027, 1, "results"): RACE_FIRST_FETCH + 3 * HOUR})
    conn, reports = run_week(world)
    for attempt in range(3):
        report = reports[RACE_FIRST_FETCH + attempt * HOUR]
        # The fetch itself "worked" (Jolpica just doesn't list the race yet),
        # but no rows landed, so the planner asks again next tick...
        assert "2027-results" in report.fetched and report.scored == []
    # ...and once it's in, the tick scores every logged run of the race.
    in_at = RACE_FIRST_FETCH + 3 * HOUR
    assert reports[in_at].scored == ["2027 R1: 3 run(s)"]
    assert race_results(conn) == 20
    assert new_alerts(reports) == []
    # Round 2's pre-weekend stage waited for the result, then was logged.
    assert any("R2 pre_weekend -> live run" in line for line in reports[in_at].predicted)


class FlakyServer:
    """Stands in for requests.Session in the REAL JsonHttpClient, serving the
    fake world over "HTTP": the first `failures` requests for FP2's laps get a
    503 (server overloaded), everything after that is answered normally."""

    def __init__(self, world, failures):
        self.world = world
        self.failures = failures
        self.fp2_laps_requests = 0

    def get(self, url, params=None, timeout=None):
        response = requests.Response()
        if url.endswith("/laps") and params.get("session_key") == session_key(2027, 1, "FP2"):
            self.fp2_laps_requests += 1
            if self.fp2_laps_requests <= self.failures:
                response.status_code = 503
                return response
        try:
            body = self.world.get_json(url, params)
        except requests.HTTPError as error:        # the world's "not published": a 404
            response.status_code = error.response.status_code
            return response
        response.status_code = 200
        response._content = json.dumps(body).encode()   # what .json() will parse
        return response


def test_server_errors_are_retried_with_backoff_inside_the_tick():
    # The other layer of retrying: inside a single request. The real HTTP
    # client retries 5xx errors with doubling waits (ingest/http.py). Its
    # sleep is replaced by a recorder, so the test doesn't actually wait.
    world = FakeF1(now=START)
    server = FlakyServer(world, failures=2)
    waits = []
    client = JsonHttpClient(min_interval_s=0, session=server, sleep=waits.append, clock=lambda: 0.0)
    conn, reports = run_week(world, openf1_client=client)

    # Two 503s, then success: waits of 2s then 4s (backoff_base 2, doubling),
    # and FP2 loaded in the SAME tick, not an hour later.
    assert waits == [2.0, 4.0]
    assert server.fp2_laps_requests == 3
    assert reports[FP2_FIRST_FETCH].fetched["2027-01-FP2"] == "ok"
    assert new_alerts(reports) == []


# -----------------------------------------------------------------------------
# Broken data: never saved, alerted at once, retried until the source fixes it
# -----------------------------------------------------------------------------

FIXED_AT = RACE_FIRST_FETCH + 3 * HOUR        # the source corrects its data 3 ticks later

BROKEN_CASES = {
    # name: (tamper key, kind, item that fails, words expected in the alert)
    "result missing two drivers": ((2027, 1, "results"), "drop_two", "2027-results",
                                   "2 drivers qualified but have no result"),
    "openf1 race missing a driver": ((2027, 1, "R"), "drop_driver", "2027-01-R",
                                     "Official result has laps for ['d20'], OpenF1 has none"),
    "openf1 race with half its laps": ((2027, 1, "R"), "half_laps", "2027-01-R",
                                       "OpenF1 has 200 laps, official result implies 400"),
}


@pytest.mark.parametrize("case", list(BROKEN_CASES))
def test_broken_data_is_not_saved_and_alerted_at_once(case):
    key, kind, item, expected_words = BROKEN_CASES[case]
    world = FakeF1(now=START, tamper={key: kind})
    saved_before_fix = {}

    def watch(now, conn):
        if now == FIXED_AT:
            # What the broken attempts left behind, just before the fix.
            saved_before_fix["results"] = race_results(conn)
            saved_before_fix["race laps"] = laps_in(conn, "R")
            saved_before_fix["scored"] = scored_runs(conn)
            world.tamper.clear()                  # the source fixes its data

    conn, reports = run_week(world, before_tick=watch)

    # 1. Every broken attempt failed validation and was rolled back...
    for attempt in range(3):
        assert reports[RACE_FIRST_FETCH + attempt * HOUR].fetched[item] == "failed"
    if item == "2027-results":
        assert saved_before_fix["results"] == 0    # no partial result saved...
        assert saved_before_fix["scored"] == 0     # ...so nothing scored against it
    else:
        assert saved_before_fix["race laps"] == 0  # no partial laps saved

    # 2. ...alerted in the tick it first arrived, exactly once (not every retry).
    alerts = new_alerts(reports)
    assert len(alerts) == 1
    when, message = alerts[0]
    assert when == RACE_FIRST_FETCH
    assert message.startswith(f"invalid data, not saved: {item}") and expected_words in message

    # 3. Once the source fixed it, the next tick loaded it and the weekend
    #    finished normally: results in, race laps in, every run scored.
    assert reports[FIXED_AT].fetched[item] == "ok"
    assert race_results(conn) == 20 and laps_in(conn, "R") == 400
    assert scored_runs(conn) == 3


def test_a_driver_missing_from_practice_is_not_broken_data():
    # Drivers do skip practice sessions (a rookie in the car, an illness), so
    # a practice session without one driver is saved as normal: no alert.
    world = FakeF1(now=START, tamper={(2027, 1, "FP1"): "drop_driver"})
    conn, reports = run_week(world)
    assert laps_in(conn, "FP1") == 19 * 8
    assert new_alerts(reports) == []
    assert logged_stages(conn) == {"pre_weekend", "practice", "post_quali"}


# -----------------------------------------------------------------------------
# Cancelled session
# -----------------------------------------------------------------------------

def test_a_cancelled_fp3_predicts_practice_from_fp1_and_fp2():
    # FP3 is cancelled before the weekend: it's not on the calendar at all.
    # (FP3 on the calendar but never published is test_simulated_weekend.py's
    # broken-data test: there the stage waits for the 2-hour last call.)
    world = FakeF1(now=START, cancelled={(2027, 1, "FP3")})
    conn, reports = run_week(world)

    assert count(conn, "SELECT COUNT(*) FROM sessions s JOIN events e USING (event_id) "
                       "WHERE e.season = 2027 AND e.round = 1 AND s.session_type = 'FP3'") == 0
    # The practice stage didn't wait for a session that isn't coming: it was
    # logged as soon as FP2 (the last practice left) was in, on Friday evening.
    assert any("R1 practice -> live run" in line for line in reports[FP2_FIRST_FETCH].predicted)
    # With practice pace for every driver, from the sessions that did run.
    paced = count(conn, "SELECT COUNT(*) FROM prediction_features f JOIN prediction_runs r USING (run_id) "
                        "WHERE r.stage = 'practice' AND f.feature = 'short_run_gap_pct' AND f.value IS NOT NULL")
    assert paced == 20
    # Nothing to alert: a cancellation isn't missing data.
    assert new_alerts(reports) == []
    assert logged_stages(conn) == {"pre_weekend", "practice", "post_quali"}
    assert scored_runs(conn) == 3
