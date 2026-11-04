"""The gap alert (pipeline/tick.py, MAX_GAP): a run that starts more than 3
hours after the previous one says so.

Why: the pipeline should run every hour. If it doesn't (GitHub's scheduler
skipping runs, the outside trigger broken), a stage can pass its cutoff
unpredicted. The pipeline can't notice while it isn't running, so this is
the after-the-fact alert, once runs resume; the real-time one is the
outside dead-man's switch the workflow pings (test_scheduler_utc.py).

The gap is measured on the real clock (pipeline_ticks.started_at_utc), as
in production: each test plants a previous run some time ago, then runs a
tick with nothing else to do (an empty plan).
"""

from datetime import datetime, timedelta, timezone

import pytest

from f1predictor.db.connection import get_connection, init_db
from f1predictor.pipeline import tick
from f1predictor.pipeline.plan import to_utc_text

from fake_f1 import FakeF1


@pytest.fixture
def conn():
    connection = get_connection(":memory:")
    init_db(connection)
    return connection


@pytest.fixture
def run(monkeypatch):
    monkeypatch.setattr(tick, "plan", lambda conn, now: [])     # nothing to fetch, predict or score
    world = FakeF1(now=None)                                     # (never called)
    return lambda conn, dry_run=False: tick.run_tick(conn, dry_run=dry_run, jolpica_client=world, openf1_client=world)


def previous_run(conn, ago: timedelta, dry_run=False):
    conn.execute("INSERT INTO pipeline_ticks (started_at_utc, finished_at_utc, dry_run) VALUES (?, ?, ?)",
                 (to_utc_text(datetime.now(timezone.utc) - ago), to_utc_text(datetime.now(timezone.utc) - ago),
                  int(dry_run)))
    conn.commit()


def test_a_run_after_more_than_3_hours_of_nothing_alerts_once(conn, run):
    previous_run(conn, timedelta(hours=4))
    first = run(conn)
    assert len(first.new_alerts) == 1
    assert first.new_alerts[0].startswith("gap: no pipeline run for 4.0 hours")
    assert "hourly trigger" in first.new_alerts[0]
    # The next run comes an hour (here: a moment) later: no gap, no alert.
    assert run(conn).new_alerts == []


def test_under_3_hours_is_normal(conn, run):
    # GitHub may start a run late, or skip one: 2 h 50 min is not a gap.
    previous_run(conn, timedelta(hours=2, minutes=50))
    assert run(conn).new_alerts == []


def test_the_very_first_run_has_nothing_to_compare_with(conn, run):
    assert run(conn).new_alerts == []


def test_a_dry_run_counts_as_a_run(conn, run):
    # A dry run started by hand is the pipeline running too.
    previous_run(conn, timedelta(hours=5))
    previous_run(conn, timedelta(minutes=30), dry_run=True)
    assert run(conn).new_alerts == []


def test_the_gap_is_measured_on_the_real_clock_not_the_ticks_now(conn, monkeypatch):
    # `now` can be any time in a test or a --now dry run; it must not
    # create (or hide) a gap. A tick "at" 2030 a moment after a real run: none.
    monkeypatch.setattr(tick, "plan", lambda conn, now: [])
    world = FakeF1(now=None)
    previous_run(conn, timedelta(minutes=5))
    report = tick.run_tick(conn, now=datetime(2030, 1, 1, tzinfo=timezone.utc), jolpica_client=world, openf1_client=world)
    assert report.new_alerts == []
