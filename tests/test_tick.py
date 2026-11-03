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


def test_fetch_failures_alert_once_a_day(conn, weekend, monkeypatch):
    monkeypatch.setattr(tick, "run_fetches", lambda conn, actions, now, j, o: {"2024-02-FP1": "failed"})
    first = run(conn)
    assert any("2024-02-FP1 failed" in m for m in first.new_alerts)
    same_day = run(conn, now=MONDAY_BEFORE.replace(hour=15))
    assert not any("FP1" in m for m in same_day.new_alerts)
    next_day = run(conn, now=MONDAY_BEFORE.replace(day=12))
    assert any("FP1" in m for m in next_day.new_alerts)
