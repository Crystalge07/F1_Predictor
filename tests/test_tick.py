"""Tests for the pipeline tick (pipeline/tick.py).

Reuses the weekend from test_as_of.py (2024: race 1 on Mar 3, race 2 on
Mar 17 with FP1 Fri 11:00, FP3 Sat 11:00, Q Sat 15:00, R Sun 15:00). Fetching
is faked (it has its own tests); these tests are about what the tick does
with the plan: predict live, block bad predictions, score, alert once.
"""

from datetime import datetime, timezone

import pytest

from f1predictor.pipeline import tick
from f1predictor.predict import stage

from test_as_of import conn, weekend  # noqa: F401  (fixtures)

MONDAY_BEFORE = datetime(2024, 3, 11, 9, 0, tzinfo=timezone.utc)   # race 2's pre-weekend window


@pytest.fixture(autouse=True)
def fake_world(monkeypatch):
    """No network, a clean code version, and a 4-car field allowed."""
    monkeypatch.setattr(tick, "run_fetches", lambda conn, actions, now, j, o: {})
    monkeypatch.setattr(stage, "model_version", lambda: "abc1234+settings.00000000")
    monkeypatch.setattr(stage, "MIN_DRIVERS", 2)


@pytest.fixture(autouse=True)
def previous_result_rechecked(conn):
    """Race 1's result re-checks are done, so race 2's pre-weekend stage is
    due on the Monday before (plan._previous_result_settled)."""
    from f1predictor.db import ingestion_log
    for after in ("24h", "72h"):
        ingestion_log.record(conn, "jolpica", f"2024-01-results-recheck-{after}", "ok")


def run(conn, now=MONDAY_BEFORE, dry_run=False):
    return tick.run_tick(conn, now=now, dry_run=dry_run, jolpica_client=object(),
                         openf1_client=object(), n_sims=2_000)


def count(conn, table, where="1=1"):
    return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0]


def test_tick_logs_the_due_stage_live_once(conn, weekend):
    report = run(conn)
    assert len(report.predicted) == 1 and "pre_weekend -> live run" in report.predicted[0]
    [run_row] = conn.execute("SELECT * FROM prediction_runs").fetchall()
    assert run_row["mode"] == "live" and run_row["stage"] == "pre_weekend"
    assert run_row["event_id"] == weekend["event"]
    # The next tick finds it done and does nothing new.
    again = run(conn)
    assert again.predicted == [] and count(conn, "prediction_runs") == 1


def test_dry_run_logs_nothing(conn, weekend):
    report = run(conn, dry_run=True)
    assert "not logged: dry run" in report.predicted[0]
    for table in ("prediction_runs", "scores", "pipeline_alerts"):
        assert count(conn, table) == 0, table
    assert count(conn, "pipeline_ticks", "dry_run = 1 AND finished_at_utc IS NOT NULL") == 1


def test_a_prediction_failing_a_check_is_blocked_and_alerted_once(conn, weekend, monkeypatch):
    monkeypatch.setattr(stage, "MIN_DRIVERS", 18)        # the real rule: 4 cars is broken data
    report = run(conn)
    assert count(conn, "prediction_runs") == 0
    assert "4 drivers" in report.blocked[0]
    blocked_alerts = [m for m in report.new_alerts if m.startswith("not logged")]
    assert len(blocked_alerts) == 1
    # Still blocked an hour later, but the alert isn't sent again.
    later = run(conn, now=MONDAY_BEFORE.replace(hour=10))
    assert later.blocked and later.new_alerts == []


def test_uncommitted_code_blocks_live_runs(conn, weekend, monkeypatch):
    monkeypatch.setattr(stage, "model_version", lambda: "abc1234-dirty+settings.00000000")
    report = run(conn)
    assert count(conn, "prediction_runs") == 0
    assert "uncommitted changes" in report.blocked[0]


def test_a_crashing_prediction_is_reported_not_raised(conn, weekend, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("model file missing")
    monkeypatch.setattr(tick, "make_prediction", broken)
    report = run(conn)
    assert "model file missing" in report.blocked[0]
    assert count(conn, "pipeline_ticks", "finished_at_utc IS NOT NULL") == 1


def test_results_are_scored_by_the_tick(conn, weekend):
    # In this fixture race 2's result is already in the database, so the
    # tick scores the run it has just logged (live, a result arriving later
    # would be scored by a later tick, the same way).
    report = run(conn)
    assert report.scored and count(conn, "scores") > 0


def test_a_failed_fetch_is_reported_but_not_alerted(conn, weekend, monkeypatch):
    # Data a few minutes late is normal: the next tick retries. Problems that
    # last are alerted by the planner instead ('gave_up', 'late').
    monkeypatch.setattr(tick, "run_fetches", lambda conn, actions, now, j, o: {"2024-02-FP1": "failed"})
    report = run(conn)
    assert report.fetched == {"2024-02-FP1": "failed"}
    assert not any("FP1" in m for m in report.new_alerts)


def test_a_result_revised_after_scoring_is_rescored_and_alerted_once(conn, weekend):
    # Architecture review, 2026-11-03: a late penalty changes the official
    # result after the race was scored. The tick re-scores every run of that
    # race against the new result and alerts once.
    first = run(conn)
    assert first.scored and not any("re-scored" in line for line in first.scored)
    before = {r["metric"]: r["model_value"] for r in conn.execute("SELECT metric, model_value FROM scores")}

    # The two leading finishers swap places (a time penalty for the winner).
    race = conn.execute("SELECT session_id FROM sessions WHERE event_id = ? AND session_type = 'R'",
                        (weekend["event"],)).fetchone()[0]
    top = conn.execute("SELECT driver_id FROM results WHERE session_id = ? AND position IN (1, 2) "
                       "ORDER BY position", (race,)).fetchall()
    conn.execute("UPDATE results SET position = 99 WHERE session_id = ? AND driver_id = ?", (race, top[0][0]))
    conn.execute("UPDATE results SET position = 1, position_text = '1' WHERE session_id = ? AND driver_id = ?",
                 (race, top[1][0]))
    conn.execute("UPDATE results SET position = 2, position_text = '2' WHERE session_id = ? AND driver_id = ?",
                 (race, top[0][0]))

    second = run(conn)
    assert any("re-scored" in line for line in second.scored)
    [alert] = [m for m in second.new_alerts if m.startswith("result_revised")]
    assert f"{top[0][0]}: 1 -> 2" in alert and f"{top[1][0]}: 2 -> 1" in alert
    after = {r["metric"]: r["model_value"] for r in conn.execute("SELECT metric, model_value FROM scores")}
    assert after != before                        # the scores follow the new result

    third = run(conn)                             # nothing changed since: nothing to do
    assert third.scored == [] and third.new_alerts == []


def test_a_prediction_finished_after_its_cutoff_is_not_logged(conn, weekend, monkeypatch):
    # Review 2026-11-10: the cutoff used to be checked against the time the
    # tick STARTED. A tick starting at 10:58, two minutes before FP1 (11:00,
    # the pre-weekend cutoff), whose fetching and simulating took 5 minutes,
    # would have logged a "live" run after FP1 began. Now the check uses the
    # tick's clock at the moment of the check: start + elapsed time.
    fake_seconds = [0.0]
    monkeypatch.setattr(tick.time, "monotonic", lambda: fake_seconds[0])
    real_make_prediction = tick.make_prediction

    def slow_make_prediction(*args, **kwargs):
        prediction = real_make_prediction(*args, **kwargs)
        fake_seconds[0] += 5 * 60                       # it took 5 minutes
        return prediction
    monkeypatch.setattr(tick, "make_prediction", slow_make_prediction)

    report = run(conn, now=datetime(2024, 3, 15, 10, 58, tzinfo=timezone.utc))
    assert count(conn, "prediction_runs") == 0
    assert "has passed" in report.blocked[0]
