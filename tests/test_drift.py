"""The guardrail (pipeline/drift.py): alert when the live model falls behind.

Rule: per stage, over the last 8 scored LIVE races, alert if the model is
more than 15% worse than its baseline on places off or podium Brier.

These tests write scores straight into the database (no model, no
simulation): the rule only reads the `scores` table, so that's all it needs,
and every number in a test is then chosen by hand and easy to check. Most
use places off with a baseline of exactly 1.0, so "model 1.16" reads as
"16% worse than the baseline".
"""

from datetime import date, timedelta

import pytest

from f1predictor.db.connection import get_connection, init_db
from f1predictor.pipeline import drift, tick
from f1predictor.pipeline.drift import MARGIN, WINDOW, drift_alerts

from fake_f1 import FakeF1


@pytest.fixture
def conn():
    connection = get_connection(":memory:")
    init_db(connection)
    connection.execute("INSERT INTO circuits (circuit_id, name) VALUES ('c', 'C')")
    return connection


def add_race(conn, season, rnd, scores, stage="post_quali", mode="live"):
    """One scored run: an event (created if new), a run of `stage`, and its
    scores. scores = {metric: (model_value, baseline_value)}.
    Race dates follow the round, so newer rounds are newer races."""
    race_date = date(season, 3, 1) + timedelta(weeks=rnd)
    event = conn.execute("SELECT event_id FROM events WHERE season = ? AND round = ?", (season, rnd)).fetchone()
    event_id = event[0] if event else conn.execute(
        "INSERT INTO events (season, round, name, circuit_id, race_date) VALUES (?, ?, 'GP', 'c', ?)",
        (season, rnd, race_date.isoformat())).lastrowid
    run_id = conn.execute(
        "INSERT INTO prediction_runs (event_id, stage, mode, as_of_utc, created_at_utc, model_version, "
        "settings_json) VALUES (?, ?, ?, '2027-01-01T00:00:00Z', '2027-01-01T00:00:00Z', 'v', '{}')",
        (event_id, stage, mode)).lastrowid
    conn.executemany(
        "INSERT INTO scores (run_id, metric, model_value, baseline_value, baseline_name, scored_at_utc) "
        "VALUES (?, ?, ?, ?, 'grid_order', '2027-01-01T00:00:00Z')",
        [(run_id, metric, model, base) for metric, (model, base) in scores.items()])
    conn.commit()


def races(conn, model, rounds, season=2027, metric="position_mae", **kwargs):
    """The same score for several rounds, against a baseline of 1.0."""
    for rnd in rounds:
        add_race(conn, season, rnd, {metric: (model, 1.0)}, **kwargs)


def keys(conn):
    return [key for key, _ in drift_alerts(conn)]


# -----------------------------------------------------------------------------
# The threshold and the window
# -----------------------------------------------------------------------------

def test_the_agreed_numbers():
    # Guard against a quiet edit: these were chosen together (docs/PRD.md);
    # changing them is a decision, so the test has to be changed with it.
    assert (WINDOW, MARGIN, drift.METRICS) == (8, 0.15, ("position_mae", "brier_podium"))


def test_more_than_15_percent_worse_alerts(conn):
    races(conn, 1.16, range(1, 9))
    alerts = drift_alerts(conn)
    # One alert, keyed by stage, metric and the NEWEST race in the window.
    assert [k for k, _ in alerts] == ["drift:post_quali:position_mae:2027:8"]
    message = alerts[0][1]
    assert "16% worse" in message and "2027 R1 to 2027 R8" in message
    assert "Nothing was changed automatically" in message      # alert only, by decision


def test_less_than_15_percent_worse_is_left_alone(conn):
    # +14%: within ordinary bad luck (the worst normal 8-race stretch in
    # 2026 was +10%; 15% leaves headroom above it).
    races(conn, 1.14, range(1, 9))
    assert keys(conn) == []


def test_seven_races_are_not_enough_however_bad(conn):
    # +50% over 7 races: still quiet. Short stretches are too noisy
    # (some stage was 13-29% behind over 3-5 races in 2026, by luck alone).
    races(conn, 1.50, range(1, 8))
    assert keys(conn) == []
    races(conn, 1.50, [8])                    # the 8th race completes the window
    assert keys(conn) == ["drift:post_quali:position_mae:2027:8"]


def test_better_than_the_baseline_never_alerts(conn):
    # Direction check: both metrics are lower-is-better, so a model far
    # BELOW the baseline is far better. A sign error would alert here.
    races(conn, 0.50, range(1, 9))
    add_race(conn, 2027, 9, {"brier_podium": (0.01, 0.10)})   # podium Brier too
    assert keys(conn) == []


def test_only_the_last_8_races_count(conn):
    # A terrible start, then 8 normal races: the old ones have left the window.
    races(conn, 2.00, range(1, 5))
    races(conn, 1.00, range(5, 13))
    assert keys(conn) == []


def test_a_sudden_break_is_caught_within_three_races(conn):
    # 5 normal races, then the model breaks and runs 45% worse. The 8-race
    # average moves 45/8 = ~5.6 points per bad race: 2 bad races give +11%
    # (quiet), the 3rd gives +17% (alert). That's the reaction time the
    # window buys: a third of a season of memory, ~3 races to notice.
    races(conn, 1.00, range(1, 6))
    races(conn, 1.45, [6, 7, 8])
    assert keys(conn) == ["drift:post_quali:position_mae:2027:8"]   # (5 + 3*1.45) / 8 = +17%


def test_two_bad_races_in_a_full_window_are_still_quiet(conn):
    races(conn, 1.00, range(1, 7))
    races(conn, 1.45, [7, 8])                 # (6*1.00 + 2*1.45) / 8 = 1.1125: +11%
    assert keys(conn) == []


# -----------------------------------------------------------------------------
# What counts
# -----------------------------------------------------------------------------

def test_backtest_runs_are_never_counted(conn):
    # Backtest replays are in-sample and not what the live model did. 8
    # terrible backtest races: quiet. And they don't dilute the live ones.
    races(conn, 2.00, range(1, 9), season=2026, mode="backtest")
    assert keys(conn) == []
    races(conn, 1.20, range(1, 9), season=2027)
    races(conn, 0.50, range(9, 17), season=2026, mode="backtest")
    assert keys(conn) == ["drift:post_quali:position_mae:2027:8"]


def test_the_window_runs_across_seasons(conn):
    # 2026's 4 rehearsal races + 2027's first 4: one window of 8. So the
    # first alert is possible at 2027 round 4, not round 8.
    races(conn, 1.20, range(20, 24), season=2026)
    races(conn, 1.20, range(1, 5), season=2027)
    alerts = drift_alerts(conn)
    assert [k for k, _ in alerts] == ["drift:post_quali:position_mae:2027:4"]
    assert "2026 R20 to 2027 R4" in alerts[0][1]


def test_only_places_off_and_podium_brier_are_watched(conn):
    # The other metrics move in big steps (winner_hit: one race = 0 or 1)
    # or have a tiny baseline (brier_win), so they'd raise false alarms.
    # Terrible on all of them, fine on the two watched ones: quiet.
    for rnd in range(1, 9):
        add_race(conn, 2027, rnd, {"winner_hit": (0.0, 1.0), "podium_hits": (0.0, 1.0),
                                   "spearman": (-1.0, 1.0), "brier_win": (0.20, 0.02),
                                   "position_mae": (1.0, 1.0), "brier_podium": (0.05, 0.05),
                                   "lineup_missing": (5.0, None)})
    assert keys(conn) == []


def test_podium_brier_is_watched_too(conn):
    for rnd in range(1, 9):
        add_race(conn, 2027, rnd, {"position_mae": (1.0, 1.0), "brier_podium": (0.070, 0.060)})  # +17%
    assert keys(conn) == ["drift:post_quali:brier_podium:2027:8"]


def test_each_stage_is_judged_on_its_own(conn):
    # Each stage has its own baseline; one stage being behind says nothing
    # about the others. Post-quali bad, practice fine: one alert, post-quali.
    for rnd in range(1, 9):
        add_race(conn, 2027, rnd, {"position_mae": (1.30, 1.0)}, stage="post_quali")
        add_race(conn, 2027, rnd, {"position_mae": (1.00, 1.0)}, stage="practice")
    assert keys(conn) == ["drift:post_quali:position_mae:2027:8"]


# -----------------------------------------------------------------------------
# In the tick
# -----------------------------------------------------------------------------

@pytest.fixture
def quiet_tick(monkeypatch):
    """run_tick with nothing to fetch, predict or score (an empty plan), so
    only the guardrail does anything. The FakeF1 client is never called."""
    monkeypatch.setattr(tick, "plan", lambda conn, now: [])
    world = FakeF1(now=None)

    def run(conn, hour):
        now = tick.datetime(2027, 6, 1, hour, 7, tzinfo=tick.timezone.utc)
        return tick.run_tick(conn, now=now, jolpica_client=world, openf1_client=world)
    return run


def test_the_tick_alerts_once_per_race_not_every_hour(conn, quiet_tick):
    races(conn, 1.20, range(1, 9))
    first, second = quiet_tick(conn, 0), quiet_tick(conn, 1)
    assert len(first.new_alerts) == 1 and "drift: post_quali position_mae" in first.new_alerts[0]
    assert second.new_alerts == []            # same newest race: already sent
    # The next race is scored and the model is still behind: a new alert,
    # because a person should hear it's still going on.
    races(conn, 1.20, [9])
    assert len(quiet_tick(conn, 2).new_alerts) == 1


def test_a_crashing_guardrail_does_not_take_the_tick_down(conn, quiet_tick, monkeypatch):
    def broken(conn):
        raise RuntimeError("boom")
    monkeypatch.setattr(tick, "drift_alerts", broken)
    report = quiet_tick(conn, 0)
    # The tick finished and recorded itself, and said the check is broken.
    assert report.new_alerts == ["drift_check_failed: the guardrail check crashed: boom"]
    assert conn.execute("SELECT COUNT(*) FROM pipeline_ticks WHERE finished_at_utc IS NULL").fetchone()[0] == 0
    # Once a day, not every hour.
    assert quiet_tick(conn, 1).new_alerts == []
