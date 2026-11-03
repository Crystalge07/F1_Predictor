"""The Phase 5 gate: whole race weekends, run only by the hourly tick.

PRD gate: "a full simulated weekend runs with zero manual steps". Here the
start of a fake 2027 season (rounds 1 and 2, then round 3's pre-weekend)
runs by calling run_tick once an hour, and NOTHING else: no backfill, no
manual prediction, no scoring call. The world (tests/fake_f1.py) publishes
data on a fake clock the way the real APIs do. Everything except the network
is the real pipeline: planner, loaders + validation, live checks,
predictions, scorer.

The only setup is what exists before the pipeline is switched on: an empty
database plus one past season (2026, two races) as training history.
"""

from datetime import datetime, timedelta, timezone

import pytest

from f1predictor.db.connection import get_connection, init_db
from f1predictor.ingest import jolpica
from f1predictor.pipeline.plan import LAST_CALL, parse_utc
from f1predictor.pipeline.tick import run_tick
from f1predictor.predict import stage

from fake_f1 import FakeF1, starting_grid

START = datetime(2027, 3, 1, 0, 0, tzinfo=timezone.utc)      # Monday before round 1
END = datetime(2027, 3, 16, 12, 0, tzinfo=timezone.utc)      # Tuesday after round 2


def simulate(broken=None, **world_options):
    """Run the tick every hour from START to END. Returns (conn, reports).
    world_options go to FakeF1 (e.g. sprint_rounds={(2027, 2)})."""
    conn = get_connection(":memory:")
    init_db(conn)
    world = FakeF1(now=START, broken=broken, **world_options)
    jolpica.ingest_season(conn, world, 2026)               # history from before go-live

    reports = []
    now = START
    while now <= END:
        world.now = now
        reports.append(run_tick(conn, now=now, jolpica_client=world, openf1_client=world, n_sims=2_000))
        now += timedelta(hours=1)
    return conn, reports


@pytest.fixture(autouse=True)
def clean_code(monkeypatch):
    # The working tree is dirty while developing; pretend it's a clean commit.
    monkeypatch.setattr(stage, "model_version", lambda: "abc1234+settings.00000000")


def logged_at(reports):
    """{'2027 R2 practice': tick time it was logged} from the tick reports."""
    out = {}
    for report in reports:
        for line in report.predicted:
            label = line.split(" -> ")[0]
            assert label not in out, f"{label} logged twice"
            out[label] = parse_utc(report.now)
    return out


@pytest.fixture(scope="module")
def clean_run():
    # scope="module": one 370-tick simulation shared by the tests below.
    # (The model_version patch is applied by hand here, since the autouse
    # fixture is function-scoped.)
    mp = pytest.MonkeyPatch()
    mp.setattr(stage, "model_version", lambda: "abc1234+settings.00000000")
    try:
        yield simulate()
    finally:
        mp.undo()


def test_every_stage_is_logged_live_once_and_before_its_cutoff(clean_run):
    conn, reports = clean_run
    runs = conn.execute(
        "SELECT e.round, r.stage, r.mode, r.as_of_utc, r.lineup_source FROM prediction_runs r "
        "JOIN events e USING (event_id) ORDER BY e.round, r.run_id").fetchall()
    assert [(r["round"], r["stage"]) for r in runs] == [
        (1, "pre_weekend"), (1, "practice"), (1, "post_quali"),
        (2, "pre_weekend"), (2, "practice"), (2, "post_quali"),
        (3, "pre_weekend"),                       # the next GP starts after round 2's result
    ]
    assert all(r["mode"] == "live" for r in runs)
    times = logged_at(reports)
    for r in runs:
        label = f"2027 R{r['round']} {r['stage']}"
        assert times[label] < parse_utc(r["as_of_utc"]), f"{label} logged after its cutoff"
    sources = {(r["round"], r["stage"]): r["lineup_source"] for r in runs}
    assert sources[(2, "practice")] == "practice_entries"
    assert sources[(2, "post_quali")] == "starting_grid"


def test_post_quali_uses_the_official_starting_grid(clean_run):
    conn, _ = clean_run
    grid = conn.execute(
        "SELECT p.driver_id FROM predictions p JOIN prediction_runs r USING (run_id) "
        "JOIN events e USING (event_id) WHERE e.season = 2027 AND e.round = 2 AND r.stage = 'post_quali' "
        "ORDER BY p.anchor_position").fetchall()
    # The penalty-shuffled grid, not qualifying order.
    assert [g["driver_id"] for g in grid] == starting_grid(2027, 2)


def test_practice_stage_saw_practice_pace(clean_run):
    conn, _ = clean_run
    known = conn.execute(
        "SELECT COUNT(*) FROM prediction_features f JOIN prediction_runs r USING (run_id) "
        "JOIN events e USING (event_id) WHERE e.round = 2 AND r.stage = 'practice' "
        "AND f.feature = 'short_run_gap_pct' AND f.value IS NOT NULL").fetchone()[0]
    assert known == 20


def test_finished_races_are_scored_by_the_tick(clean_run):
    conn, reports = clean_run
    unscored = conn.execute(
        "SELECT e.round, r.stage FROM prediction_runs r JOIN events e USING (event_id) "
        "WHERE e.round <= 2 AND NOT EXISTS (SELECT 1 FROM scores s WHERE s.run_id = r.run_id)").fetchall()
    assert unscored == []
    assert any(report.scored for report in reports)


def test_a_clean_season_start_raises_no_alerts(clean_run):
    conn, reports = clean_run
    assert [m for report in reports for m in report.new_alerts] == []
    assert conn.execute("SELECT COUNT(*) FROM pipeline_ticks WHERE finished_at_utc IS NULL").fetchone()[0] == 0


def test_broken_data_still_needs_no_manual_step():
    # Round 2: FP3 is never published and OpenF1 never posts a starting grid.
    conn, reports = simulate(broken={(2027, 2, "FP3"), (2027, 2, "grid")})
    runs = {r["stage"]: r for r in conn.execute(
        "SELECT r.stage, r.lineup_source, r.as_of_utc FROM prediction_runs r JOIN events e USING (event_id) "
        "WHERE e.round = 2").fetchall()}
    # Every stage still logged, with the fallbacks: practice at the last call
    # (FP1 + FP2 only), post-quali from qualifying order at the last call.
    assert set(runs) == {"pre_weekend", "practice", "post_quali"}
    assert runs["post_quali"]["lineup_source"] == "qualifying"
    times = logged_at(reports)
    assert times["2027 R2 practice"] >= parse_utc(runs["practice"]["as_of_utc"]) - LAST_CALL
    # And the problem is reported exactly once: the lost session. Nothing
    # was missed, and the grid falling back to qualifying order is by design.
    alerts = [m for report in reports for m in report.new_alerts]
    assert len(alerts) == 1 and "gave_up" in alerts[0] and "FP3" in alerts[0]


# -----------------------------------------------------------------------------
# Retraining (test-suite step 7)
# -----------------------------------------------------------------------------
#
# There is no retraining job to test: every prediction fits its stage's
# model from scratch on every race before it (predict/stage.py). So
# "retraining works" means three things, checked on the same clean run:
#   1. each new result really reaches the next prediction (the training set
#      grows by exactly one race per race),
#   2. every prediction is graded once its race is over (all metrics),
#   3. every prediction records which code made it.

def live_runs(conn):
    return conn.execute(
        "SELECT e.round, r.stage, r.run_id, r.n_training_races, r.model_version FROM prediction_runs r "
        "JOIN events e USING (event_id) WHERE r.mode = 'live' ORDER BY e.round, r.run_id").fetchall()


def test_training_grows_by_exactly_one_race_per_race(clean_run):
    conn, _ = clean_run
    n = {(r["round"], r["stage"]): r["n_training_races"] for r in live_runs(conn)}
    # Before go-live the database holds 2026's two races, so round 1 trains
    # on 2; round 1's result is fetched on race night, so round 2 trains on
    # 3; round 2's result makes it 4 for round 3. All three stages of a
    # weekend see the same races: the race being predicted is never in its
    # own training set (cutoff = before its race), and nothing else is
    # added between Friday and Sunday.
    assert n == {(1, "pre_weekend"): 2, (1, "practice"): 2, (1, "post_quali"): 2,
                 (2, "pre_weekend"): 3, (2, "practice"): 3, (2, "post_quali"): 3,
                 (3, "pre_weekend"): 4}


# What the scorer writes for every run (evaluate/scorer.py): six metrics
# compared with the stage's baseline, and two driver-list counts.
METRICS_WITH_BASELINE = {"position_mae", "spearman", "winner_hit", "podium_hits", "brier_win", "brier_podium"}
LINEUP_COUNTS = {"lineup_missing", "lineup_extra"}


def test_every_finished_race_is_scored_on_every_metric_at_every_stage(clean_run):
    conn, _ = clean_run
    for r in live_runs(conn):
        rows = {s["metric"]: s for s in conn.execute(
            "SELECT metric, model_value, baseline_value, baseline_name FROM scores WHERE run_id = ?",
            (r["run_id"],))}
        if r["round"] == 3:
            # Round 3 hasn't been raced when the simulation ends: nothing to grade yet.
            assert rows == {}
            continue
        assert set(rows) == METRICS_WITH_BASELINE | LINEUP_COUNTS, f"R{r['round']} {r['stage']}"
        # Every comparable metric has its baseline next to it (or "% better
        # than baseline" can't be computed), and it's the right baseline.
        expected = {"pre_weekend": "championship_order", "practice": "practice_order",
                    "post_quali": "grid_order"}[r["stage"]]
        for metric in METRICS_WITH_BASELINE:
            assert rows[metric]["baseline_value"] is not None
            assert rows[metric]["baseline_name"] == expected
        # The fake world is clean: every starter was predicted, nobody extra.
        assert rows["lineup_missing"]["model_value"] == 0
        assert rows["lineup_extra"]["model_value"] == 0


def test_every_run_records_the_code_that_made_it(clean_run):
    conn, _ = clean_run
    # model_version = commit + settings fingerprint (patched to a clean
    # commit for the test). Without it, a score can't be traced back to the
    # model that earned it, e.g. to split a season at a model change.
    assert {r["model_version"] for r in live_runs(conn)} == {"abc1234+settings.00000000"}
