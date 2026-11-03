"""Idempotency and crash recovery for the hourly tick (pipeline/tick.py).

The pipeline is meant to be safe to run "too often" and safe to crash:

    idempotent      running a tick twice does the work once: no duplicate
                    predictions, no double-counted scores.
    crash recovery  a tick that fails or dies partway leaves the database in
                    a state the next tick can carry on from, and ends up
                    exactly where an uninterrupted run would have.

How the tests check it: the fake world (tests/fake_f1.py) runs 2027 round 1
from the Monday before to the morning after the race, one tick per hour, with
nothing else. That uninterrupted run is the REFERENCE. Each test runs the same
week with something going wrong (or repeated), then compares its end state
with the reference: every logged run, prediction, feature and score, keyed by
race and stage, plus the data loaded. Wall-clock timestamps (created_at_utc,
scored_at_utc, ingested_at) are left out: they're the real clock, so they
always differ. Everything else must match exactly, which is possible because
predictions are deterministic (the simulator's seed is fixed, predict/stage.py).
"""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from f1predictor.db.connection import get_connection, init_db
from f1predictor.evaluate.scorer import score_runs
from f1predictor.ingest import jolpica
from f1predictor.pipeline import tick
from f1predictor.predict import stage

from fake_f1 import FakeF1

START = datetime(2027, 3, 1, 0, 0, tzinfo=timezone.utc)   # Monday before round 1 (race Sun Mar 7)
END = datetime(2027, 3, 8, 6, 0, tzinfo=timezone.utc)     # Monday morning: race scored, R2 pre-weekend logged
HOUR = timedelta(hours=1)
N_SIMS = 2_000                                            # enough for these checks, and fast
CLEAN_VERSION = "abc1234+settings.00000000"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def new_world():
    """An empty database with 2026 as history (as at go-live), and the fake APIs."""
    conn = get_connection(":memory:")
    init_db(conn)
    world = FakeF1(now=START)
    jolpica.ingest_season(conn, world, 2026)
    return conn, world


def run_tick(conn, world, now):
    world.now = now                      # the fake APIs publish data by this clock
    return tick.run_tick(conn, now=now, jolpica_client=world, openf1_client=world, n_sims=N_SIMS)


def run_week(ticks_per_hour=1):
    """Tick every hour from START to END. Returns (conn, reports)."""
    conn, world = new_world()
    reports, now = [], START
    while now <= END:
        for _ in range(ticks_per_hour):
            reports.append(run_tick(conn, world, now))
        now += HOUR
    return conn, reports


def fingerprint(conn) -> dict[str, list[tuple]]:
    """Everything that matters about the end state, without wall-clock times.

    Runs are identified by (season, round, stage, mode), not run_id: the ids
    are an implementation detail, and the question is "is the same prediction
    logged for the same race and stage", not "did it get the same number".
    """
    run_key = "e.season, e.round, r.stage, r.mode"
    joins = "JOIN prediction_runs r USING (run_id) JOIN events e ON e.event_id = r.event_id"
    queries = {
        "runs": f"SELECT {run_key}, r.as_of_utc, r.lineup_source, r.model_version, r.n_training_races "
                "FROM prediction_runs r JOIN events e USING (event_id)",
        "predictions": f"SELECT {run_key}, p.driver_id, p.predicted_rank, p.anchor_position, "
                       f"p.p_win, p.p_podium, p.p_points, p.p_dnf FROM predictions p {joins}",
        "features": f"SELECT {run_key}, f.driver_id, f.feature, f.value FROM prediction_features f {joins}",
        "scores": f"SELECT {run_key}, s.metric, s.model_value, s.baseline_value FROM scores s {joins}",
        # The data side: which fetches succeeded, and how many rows landed.
        "ingestion": "SELECT source, item_key, status FROM ingestion_log",
        "row_counts": "SELECT 'laps', COUNT(*) FROM laps UNION ALL SELECT 'results', COUNT(*) FROM results "
                      "UNION ALL SELECT 'qualifying', COUNT(*) FROM qualifying "
                      "UNION ALL SELECT 'starting_grid', COUNT(*) FROM starting_grid "
                      "UNION ALL SELECT 'session_entries', COUNT(*) FROM session_entries",
    }
    # sorted(): compare as sets of rows, whatever order they were written in.
    return {name: sorted(tuple(row) for row in conn.execute(sql)) for name, sql in queries.items()}


def alert_keys(conn) -> set[str]:
    return {row[0] for row in conn.execute("SELECT alert_key FROM pipeline_alerts")}


def count(conn, table, where="1=1"):
    return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0]


@pytest.fixture(autouse=True)
def clean_code(monkeypatch):
    # The working tree is dirty while developing; pretend it's a clean commit
    # (a dirty version blocks every live prediction, predict/stage.py).
    monkeypatch.setattr(stage, "model_version", lambda: CLEAN_VERSION)


@pytest.fixture(scope="module")
def reference():
    """The uninterrupted week every other test is compared with. Run once and
    shared (scope="module"), so tests must not change it."""
    mp = pytest.MonkeyPatch()            # the autouse fixture is per-test, so patch by hand here
    mp.setattr(stage, "model_version", lambda: CLEAN_VERSION)
    try:
        conn, reports = run_week()
        yield {"conn": conn, "reports": reports, "fingerprint": fingerprint(conn)}
    finally:
        mp.undo()


def test_the_reference_week_did_the_whole_weekend(reference):
    # A sanity check on the yardstick itself: if the reference were empty,
    # "matches the reference" would prove nothing.
    fp = reference["fingerprint"]
    assert [r[:4] for r in fp["runs"]] == [
        (2027, 1, "post_quali", "live"), (2027, 1, "practice", "live"),
        (2027, 1, "pre_weekend", "live"), (2027, 2, "pre_weekend", "live"),
    ]
    scored_runs = {s[:4] for s in fp["scores"]}
    assert scored_runs == {r[:4] for r in fp["runs"] if r[1] == 1}    # all of round 1 scored
    assert alert_keys(reference["conn"]) == set()


# -----------------------------------------------------------------------------
# Idempotency
# -----------------------------------------------------------------------------

def test_every_tick_running_twice_changes_nothing(reference):
    # As if the scheduler fired every job twice (a retried run, a manual
    # "Run workflow" on top of the cron). The second tick each hour must find
    # nothing new to do, so the week ends exactly like the reference: one run
    # per race and stage, each scored once.
    conn, reports = run_week(ticks_per_hour=2)
    assert fingerprint(conn) == reference["fingerprint"]
    assert alert_keys(conn) == set()
    # The second tick of each hour predicted, scored and alerted nothing...
    for second in reports[1::2]:
        assert (second.predicted, second.scored, second.new_alerts) == ([], [], [])
        # ...and loaded no new data. It may RETRY something still missing
        # (e.g. the starting grid before OpenF1 publishes it: 'skipped'), which
        # also refreshes OpenF1's session list ('2027-sessions'), but nothing
        # the first tick already got is fetched again.
        loaded = {key for key, status in second.fetched.items()
                  if status == "ok" and not key.endswith("-sessions")}
        assert loaded == set(), second.fetched


def test_rerunning_the_last_tick_does_nothing():
    conn, _ = run_week()
    before = fingerprint(conn)
    world = FakeF1(now=END)
    again = run_tick(conn, world, END)
    assert again.lines()[1:] == ["  nothing due"]
    assert world.requests == 0                       # not even a request to the APIs
    assert fingerprint(conn) == before


def test_scoring_again_replaces_scores_instead_of_adding_them():
    # Scores are upserts keyed by (run, metric): scoring every run again (as a
    # rerun after a crash might) rewrites the same rows with the same values.
    conn, _ = run_week()
    before = fingerprint(conn)["scores"]
    n_rows = count(conn, "scores")
    score_runs(conn)                                 # every logged run, again
    assert count(conn, "scores") == n_rows
    assert fingerprint(conn)["scores"] == before


# -----------------------------------------------------------------------------
# Crash recovery
# -----------------------------------------------------------------------------

def test_a_crash_while_scoring_is_alerted_and_finished_later(reference, monkeypatch):
    # The first time the tick scores, it scores ONE run and then crashes,
    # leaving the race half scored.
    real_score_runs = tick.score_runs
    crashes = []

    def crash_once(conn, run_ids):
        if not crashes:
            crashes.append(run_ids)
            real_score_runs(conn, run_ids[:1])
            raise RuntimeError("simulated crash while scoring")
        return real_score_runs(conn, run_ids)

    monkeypatch.setattr(tick, "score_runs", crash_once)
    conn, reports = run_week()

    assert len(crashes) == 1
    # The crashing tick still finished (alerts raised, tick recorded)...
    assert count(conn, "pipeline_ticks", "finished_at_utc IS NULL") == 0
    failed = [r for r in reports if any("score_failed" in m for m in r.new_alerts)]
    assert len(failed) == 1 and "simulated crash" in failed[0].new_alerts[0]
    # ...the next tick scored the rest, and the week ends like the reference,
    # with the one extra alert (raised once, not every tick).
    assert fingerprint(conn) == reference["fingerprint"]
    assert alert_keys(conn) == {"score_failed:2027:1"}


def test_a_crash_while_writing_a_prediction_leaves_no_half_run(reference, monkeypatch):
    # The first prediction write dies halfway: AFTER the prediction_runs row
    # is inserted, while the driver rows are being built (stage._number
    # converts each driver's values). write_run's transaction must roll the
    # run row back, so there's no run without predictions.
    real_number = stage._number
    crashed = []

    def crash_once(value):
        if not crashed:
            crashed.append(True)
            raise sqlite3.OperationalError("simulated disk I/O error")
        return real_number(value)

    monkeypatch.setattr(stage, "_number", crash_once)
    conn, reports = run_week()

    # The first tick reported it as blocked (not raised), logging nothing...
    assert reports[0].predicted == [] and "simulated disk I/O error" in reports[0].blocked[0]
    # ...and the next tick logged it, well before its cutoff. Same end state.
    assert reports[1].predicted and "pre_weekend -> live run" in reports[1].predicted[0]
    assert fingerprint(conn) == reference["fingerprint"]
    assert alert_keys(conn) == {"blocked:2027:1:pre_weekend"}


def test_a_tick_killed_halfway_is_rerun_cleanly(reference, monkeypatch):
    # The hardest case: the tick dies with some work already committed. Here
    # it dies while predicting the practice stage, AFTER that tick's fetches
    # (the FP3 data that made the stage ready) were saved.
    # KeyboardInterrupt is a BaseException, not an Exception, so the tick's
    # "never crash" handlers don't catch it: like a real kill, the tick
    # stops on the spot. (When GitHub kills a job outright, the database isn't
    # uploaded at all and the next run starts from the copy before the tick,
    # which is just a missed tick. This is the case where work WAS saved.)
    real_make_prediction = tick.make_prediction
    killed_at = []

    def killed_once(conn, event_id, stage_name, **kwargs):
        if stage_name == "practice" and not killed_at:
            killed_at.append(True)
            raise KeyboardInterrupt("simulated kill")
        return real_make_prediction(conn, event_id, stage_name, **kwargs)

    monkeypatch.setattr(tick, "make_prediction", killed_once)

    conn, world = new_world()
    now, reruns = START, []
    while now <= END:
        try:
            run_tick(conn, world, now)
        except KeyboardInterrupt:
            # The premise: the dead tick had already committed its fetch.
            assert conn.execute("SELECT status FROM ingestion_log WHERE item_key = '2027-01-FP3'"
                                ).fetchone()[0] == "ok"
            assert count(conn, "prediction_runs", "stage = 'practice'") == 0
            reruns.append(run_tick(conn, world, now))   # rerun the same hour
        now += HOUR

    assert len(reruns) == 1
    # The rerun didn't fetch FP3 again (it was saved) and logged the stage.
    assert "2027-01-FP3" not in reruns[0].fetched
    assert any("practice -> live run" in line for line in reruns[0].predicted)
    # End state: exactly the reference, with no alerts...
    assert fingerprint(conn) == reference["fingerprint"]
    assert alert_keys(conn) == set()
    # ...and the dead tick is still on record as one that never finished.
    assert count(conn, "pipeline_ticks", "finished_at_utc IS NULL") == 1
