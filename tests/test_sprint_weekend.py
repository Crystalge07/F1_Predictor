"""A sprint weekend, end to end, run only by the hourly tick.

Same idea as test_simulated_weekend.py (the Phase 5 gate), with 2027 round 2
turned into a sprint weekend in the 2024+ format (all UTC):

    FP1                Fri Mar 12 12:00
    Sprint Qualifying  Fri 16:00          (SQ)
    Sprint             Sat Mar 13 11:00   (S)
    Qualifying         Sat 15:00
    Race               Sun Mar 14 15:00

What a sprint changes for the pipeline:
    * two extra sessions to fetch and validate, and Jolpica's sprint results
      (stored under the sprint session in `results`);
    * the practice stage means "everything before qualifying", which on a
      sprint weekend is FP1 + SQ + the Sprint, so it can't be predicted until
      the sprint is in (Saturday), not after FP1;
    * sprint points count toward the championship standings, but only from
      the NEXT weekend on: the pre-weekend prediction is made before this
      weekend's sprint has run (features/championship.py).

The last point is a leakage trap, so the step-4 leakage checks are repeated
here on the sprint weekend.
"""

from datetime import datetime, timezone

import pytest

from f1predictor.pipeline.plan import parse_utc
from f1predictor.predict import stage

from test_leakage import live_runs, logged, rebuilt, run_of, same, scrambled
from test_simulated_weekend import logged_at, simulate

CLEAN_VERSION = "abc1234+settings.00000000"
SPRINT_ROUND = 2


@pytest.fixture(scope="module")
def season():
    """Round 1 normal, round 2 a sprint weekend, ticked hourly from the Monday
    before round 1 to round 3's pre-weekend. Shared; tests mustn't change it
    (the leakage ones only change data inside scrambled(), which rolls back)."""
    mp = pytest.MonkeyPatch()
    mp.setattr(stage, "model_version", lambda: CLEAN_VERSION)   # pretend it's a clean commit
    try:
        yield simulate(sprint_rounds={(2027, SPRINT_ROUND)})
    finally:
        mp.undo()


def utc(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


# -----------------------------------------------------------------------------
# The dry run: zero manual steps
# -----------------------------------------------------------------------------

def test_every_stage_is_logged_live_once_before_its_cutoff(season):
    conn, reports = season
    runs = live_runs(conn)
    assert [(r["round"], r["stage"]) for r in runs] == [
        (1, "pre_weekend"), (1, "practice"), (1, "post_quali"),
        (2, "pre_weekend"), (2, "practice"), (2, "post_quali"),
        (3, "pre_weekend")]
    times = logged_at(reports)                      # also fails if anything was logged twice
    for r in runs:
        label = f"2027 R{r['round']} {r['stage']}"
        assert times[label] < parse_utc(r["as_of_utc"]), f"{label} logged after its cutoff"


def test_both_races_are_scored_and_nothing_is_alerted(season):
    conn, reports = season
    unscored = conn.execute(
        "SELECT e.round, r.stage FROM prediction_runs r JOIN events e USING (event_id) "
        "WHERE e.round <= 2 AND NOT EXISTS (SELECT 1 FROM scores s WHERE s.run_id = r.run_id)").fetchall()
    assert unscored == []
    assert [m for report in reports for m in report.new_alerts] == []
    assert conn.execute("SELECT COUNT(*) FROM pipeline_ticks WHERE finished_at_utc IS NULL").fetchone()[0] == 0


def test_sprint_sessions_and_results_are_loaded_and_validated(season):
    conn, _ = season
    sessions = {r["session_type"]: r["session_id"] for r in conn.execute(
        "SELECT s.session_type, s.session_id FROM sessions s JOIN events e USING (event_id) "
        "WHERE e.season = 2027 AND e.round = ?", (SPRINT_ROUND,))}
    assert set(sessions) == {"FP1", "SQ", "S", "Q", "R"}         # no FP2/FP3 on a sprint weekend
    # Both sprint sessions' OpenF1 data, and Jolpica's sprint results, fetched OK.
    status = dict(conn.execute("SELECT item_key, status FROM ingestion_log WHERE item_key IN "
                               "('2027-02-SQ', '2027-02-S', '2027-sprint')").fetchall())
    assert status == {"2027-02-SQ": "ok", "2027-02-S": "ok", "2027-sprint": "ok"}
    # The sprint result: 20 drivers under the SPRINT session (not the race), with points.
    sprint = conn.execute("SELECT COUNT(*), SUM(points) FROM results WHERE session_id = ?",
                          (sessions["S"],)).fetchone()
    assert tuple(sprint) == (20, 36)                             # 8+7+6+5+4+3+2+1
    # Validation ran on the sprint's laps against its result and found no errors.
    assert conn.execute("SELECT COUNT(*) FROM validation_issues WHERE severity = 'error'").fetchone()[0] == 0


def test_the_practice_stage_waits_for_the_sprint(season):
    conn, reports = season
    run = run_of(conn, SPRINT_ROUND, "practice")
    # Practice on a sprint weekend = FP1 + SQ + Sprint. The sprint ends Sat
    # 12:00 and is fetched at 13:00 (end + 30 min, next hourly tick); only
    # then is the stage predicted. Not on Friday after FP1.
    assert logged_at(reports)[f"2027 R{SPRINT_ROUND} practice"] == utc("2027-03-13T13:00:00")
    # Who's racing comes from the last session before qualifying: the sprint.
    assert run["lineup_source"] == "practice_entries"
    assert run["as_of_utc"] == "2027-03-13T15:00:00Z"            # cutoff: qualifying's start


# -----------------------------------------------------------------------------
# Retraining
# -----------------------------------------------------------------------------

def test_training_grows_by_one_race_per_weekend_not_per_sprint(season):
    # "Retrain after the race" = each prediction trains on every race before
    # it (pipeline/tick.py). 2 history races, then +1 per 2027 weekend: the
    # sprint weekend adds one race, not two.
    conn, _ = season
    assert {(r["round"], r["n_training_races"]) for r in live_runs(conn)} == {(1, 2), (2, 3), (3, 4)}


def test_sprint_points_count_from_the_next_weekend_on(season):
    conn, _ = season

    def season_points(rnd):
        """driver_season_points per driver, as the pre-weekend run for `rnd` used it."""
        run_id = run_of(conn, rnd, "pre_weekend")["run_id"]
        return dict(conn.execute("SELECT driver_id, value FROM prediction_features WHERE run_id = ? "
                                 "AND feature = 'driver_season_points'", (run_id,)).fetchall())

    def points_scored(rnd, session_type):
        return dict(conn.execute(
            "SELECT r.driver_id, r.points FROM results r JOIN sessions s USING (session_id) "
            "JOIN events e USING (event_id) WHERE e.season = 2027 AND e.round = ? AND s.session_type = ?",
            (rnd, session_type)).fetchall())

    r1_race, r2_race, r2_sprint = points_scored(1, "R"), points_scored(2, "R"), points_scored(2, "S")
    # Before the sprint weekend: round 1's race points only.
    assert season_points(2) == {d: r1_race[d] for d in r1_race}
    # After it: round 1 + round 2's race AND sprint.
    assert season_points(3) == {d: r1_race[d] + r2_race[d] + r2_sprint[d] for d in r1_race}


# -----------------------------------------------------------------------------
# Leakage on the sprint weekend (the method of test_leakage.py)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("rnd, stage_name", [
    (SPRINT_ROUND, "pre_weekend"), (SPRINT_ROUND, "practice"), (SPRINT_ROUND, "post_quali"),
    (3, "pre_weekend")])
def test_rebuilding_with_all_later_data_changes_nothing(season, rnd, stage_name):
    conn, _ = season
    run = run_of(conn, rnd, stage_name)
    log_drivers, log_features = logged(conn, run["run_id"])
    _, drivers, features = rebuilt(conn, run["event_id"], stage_name)
    assert same(drivers, log_drivers) and same(features, log_features)


def test_the_sprint_never_reaches_the_pre_weekend_prediction(season):
    # The trap: this weekend's sprint result and points exist by the end of
    # the season, but the pre-weekend prediction was made before FP1.
    conn, _ = season
    event_id = run_of(conn, SPRINT_ROUND, "pre_weekend")["event_id"]
    _, drivers_before, features_before = rebuilt(conn, event_id, "pre_weekend")
    with scrambled(conn, SPRINT_ROUND, ["FP1", "SQ", "S"]):
        _, drivers_after, features_after = rebuilt(conn, event_id, "pre_weekend")
    assert same(drivers_after, drivers_before) and same(features_after, features_before)


def test_control_the_sprint_does_reach_the_practice_prediction(season):
    # The sprint runs before qualifying, so the practice stage SHOULD use it
    # (long-run pace, and the lineup). Scrambling it must change that prediction.
    conn, _ = season
    event_id = run_of(conn, SPRINT_ROUND, "practice")["event_id"]
    _, drivers_before, features_before = rebuilt(conn, event_id, "practice")
    with scrambled(conn, SPRINT_ROUND, ["SQ", "S"]):
        _, drivers_after, features_after = rebuilt(conn, event_id, "practice")
    assert not same(features_after, features_before)
    assert not same(drivers_after, drivers_before)


def test_qualifying_and_the_race_never_reach_the_practice_prediction(season):
    conn, _ = season
    event_id = run_of(conn, SPRINT_ROUND, "practice")["event_id"]
    _, drivers_before, features_before = rebuilt(conn, event_id, "practice")
    with scrambled(conn, SPRINT_ROUND, ["Q", "R"]):
        _, drivers_after, features_after = rebuilt(conn, event_id, "practice")
    assert same(drivers_after, drivers_before) and same(features_after, features_before)
