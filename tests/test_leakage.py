"""No data leakage: every prediction uses only what was known at its cutoff.

The cutoffs (features/as_of.stage_cutoff):
    pre_weekend  before the weekend's first session
    practice     before qualifying          (all practice known, no quali)
    post_quali   before the race            (quali + starting grid known, no result)
and training only ever uses races strictly BEFORE the one being predicted.

How it's tested: the fake world (tests/fake_f1.py) runs two whole weekends
and round 3's pre-weekend on the hourly tick (7 live runs). Each live run was
made BEFORE its cutoff, when later data simply didn't exist yet. At the end of
the simulation the database holds everything: qualifying, grids, both race
results, every session's laps. So:

    1. Rebuilding each logged prediction NOW, with all that later data
       present, must give exactly what was logged. Any feature or training
       row that peeked past its cutoff would come out different.
    2. Scrambling everything after a stage's cutoff (reversing results,
       qualifying and grids, reshuffling lap times) must change nothing.
    3-4. Two controls, so 1 and 2 can't pass by accident: scrambling data
       BEFORE the cutoff does change the prediction, and a deliberately
       planted leak is caught by test 1's comparison.

Comparisons are exact (no tolerance): predictions are deterministic (fixed
simulator seed), and SQLite stores the same 64-bit floats Python computed.
"""

from contextlib import contextmanager

import pandas as pd
import pytest

from f1predictor.predict import stage
from f1predictor.predict.stage import make_prediction

from test_simulated_weekend import simulate

N_SIMS = 2_000     # what simulate() logged with: a rebuild must use the same
CLEAN_VERSION = "abc1234+settings.00000000"

PREDICTION_COLUMNS = ["driver_id", "constructor_id", "predicted_rank", "anchor_position",
                      "expected_position", "p_win", "p_podium", "p_points", "p_dnf", "mean_position"]
FEATURE_COLUMNS = ["driver_id", "feature", "value", "contribution"]

# Which of a round's sessions happen AFTER each stage's cutoff (fake weekend:
# FP1, FP2, FP3, Q, R; no sprints).
AFTER_CUTOFF = {
    "pre_weekend": ["FP1", "FP2", "FP3", "Q", "R"],
    "practice": ["Q", "R"],
    "post_quali": ["R"],
}


@pytest.fixture(scope="module")
def world():
    """The finished two-weekend simulation, shared by every test here.
    Tests that change data do it inside scrambled(), which rolls back."""
    mp = pytest.MonkeyPatch()
    mp.setattr(stage, "model_version", lambda: CLEAN_VERSION)   # pretend it's a clean commit
    try:
        conn, _ = simulate()
        yield conn
    finally:
        mp.undo()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def live_runs(conn):
    return conn.execute(
        "SELECT r.run_id, r.event_id, r.stage, r.as_of_utc, r.lineup_source, r.n_training_races, "
        "e.round FROM prediction_runs r JOIN events e USING (event_id) "
        "WHERE r.mode = 'live' ORDER BY e.round, r.run_id").fetchall()


def run_of(conn, rnd, stage_name):
    return next(r for r in live_runs(conn) if r["round"] == rnd and r["stage"] == stage_name)


def logged(conn, run_id):
    """(predictions, features) exactly as the log holds them, in a fixed order."""
    drivers = pd.read_sql_query(f"SELECT {', '.join(PREDICTION_COLUMNS)} FROM predictions "
                                "WHERE run_id = ? ORDER BY predicted_rank", conn, params=(run_id,))
    features = pd.read_sql_query(f"SELECT {', '.join(FEATURE_COLUMNS)} FROM prediction_features "
                                 "WHERE run_id = ? ORDER BY driver_id, feature", conn, params=(run_id,))
    return drivers, features


def rebuilt(conn, event_id, stage_name):
    """(prediction object, predictions, features) made right now, in the log's shape."""
    p = make_prediction(conn, event_id, stage_name, n_sims=N_SIMS)
    drivers = p.drivers[PREDICTION_COLUMNS].sort_values("predicted_rank").reset_index(drop=True)
    features = (p.features[FEATURE_COLUMNS].sort_values(["driver_id", "feature"])
                .reset_index(drop=True))
    return p, drivers, features


def same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Exactly equal values, ignoring dtypes (the log reads ints back as int64,
    the rebuild may hold them as float) and treating NaN/None as equal."""
    a = a.reset_index(drop=True).astype(object).where(a.notna().reset_index(drop=True), None)
    b = b.reset_index(drop=True).astype(object).where(b.notna().reset_index(drop=True), None)
    return a.shape == b.shape and a.values.tolist() == b.values.tolist()


def session_ids(conn, rnd):
    return {r["session_type"]: r["session_id"] for r in conn.execute(
        "SELECT s.session_type, s.session_id FROM sessions s JOIN events e USING (event_id) "
        "WHERE e.season = 2027 AND e.round = ?", (rnd,))}


@contextmanager
def scrambled(conn, rnd, session_types):
    """Change every kind of data from these sessions of round `rnd`, then undo it.

    Reverses race and sprint results (positions, grid, and who scored points), qualifying and the starting
    grid, and makes each car's laps FASTER by a different amount: car 20 by
    8s, car 1 by 0.4s. In the fake world car 1 is the fastest, so this
    REVERSES the pace order, and these become every driver's best laps of the
    weekend. (A uniform shift would cancel out of the %-gap pace features;
    slower laps wouldn't replace anyone's best lap; and keeping the order
    wouldn't move the practice order, the practice stage's anchor.)
    It all happens in one open
    transaction: make_prediction reads through the same connection, so it
    sees the changes, and the rollback afterwards puts everything back.
    """
    ids = session_ids(conn, rnd)
    try:
        for session_type in session_types:
            sid = ids[session_type]
            conn.execute("UPDATE laps SET lap_time_ms = lap_time_ms - car_number * 400 "
                         "WHERE session_id = ?",
                         (sid,))
            if session_type == "Q":
                conn.execute("UPDATE qualifying SET position = 21 - position WHERE session_id = ?", (sid,))
                conn.execute("UPDATE starting_grid SET position = 21 - position WHERE session_id = ?", (sid,))
            if session_type in ("R", "S"):            # race or sprint result
                # Points too: the championship features are built from the
                # points column, not positions. Scorers get 0, the rest 10.
                conn.execute("UPDATE results SET position = 21 - position, grid = 21 - grid, "
                             "position_text = CAST(21 - position AS TEXT), "
                             "points = CASE WHEN points > 0 THEN 0 ELSE 10 END WHERE session_id = ?", (sid,))
        yield
    finally:
        conn.rollback()


# -----------------------------------------------------------------------------
# 1. Rebuilt with everything known, every live prediction is unchanged
# -----------------------------------------------------------------------------

def test_the_simulation_logged_every_stage(world):
    # The yardstick: 2 full weekends + round 3's pre-weekend, all live.
    assert [(r["round"], r["stage"]) for r in live_runs(world)] == [
        (1, "pre_weekend"), (1, "practice"), (1, "post_quali"),
        (2, "pre_weekend"), (2, "practice"), (2, "post_quali"), (3, "pre_weekend")]


@pytest.mark.parametrize("rnd, stage_name", [
    (1, "pre_weekend"), (1, "practice"), (1, "post_quali"),
    (2, "pre_weekend"), (2, "practice"), (2, "post_quali"), (3, "pre_weekend")])
def test_rebuilding_a_live_prediction_with_all_later_data_changes_nothing(world, rnd, stage_name):
    run = run_of(world, rnd, stage_name)
    log_drivers, log_features = logged(world, run["run_id"])
    p, drivers, features = rebuilt(world, run["event_id"], stage_name)

    # Same cutoff, same lineup source, and trained on the same races: only
    # the 2 history races + earlier 2027 rounds, never this race or a later one.
    assert (p.as_of_utc, p.lineup_source) == (run["as_of_utc"], run["lineup_source"])
    assert p.n_training_races == run["n_training_races"] == 2 + (rnd - 1)
    # Every driver's prediction and every feature value / SHAP contribution.
    assert same(drivers, log_drivers), f"R{rnd} {stage_name}: predictions differ from the log"
    assert same(features, log_features), f"R{rnd} {stage_name}: features differ from the log"


# -----------------------------------------------------------------------------
# 2. Data after the cutoff can be anything: the prediction doesn't move
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("stage_name", list(AFTER_CUTOFF))
def test_scrambling_data_after_the_cutoff_changes_nothing(world, stage_name):
    event_id = run_of(world, 1, stage_name)["event_id"]
    _, drivers_before, features_before = rebuilt(world, event_id, stage_name)
    # E.g. for practice: qualifying, the starting grid and the race result
    # all reversed, Q and race laps reshuffled. A practice prediction that
    # "saw" any of it would come out different.
    with scrambled(world, rnd=1, session_types=AFTER_CUTOFF[stage_name]):
        _, drivers_after, features_after = rebuilt(world, event_id, stage_name)
    assert same(drivers_after, drivers_before)
    assert same(features_after, features_before)


# -----------------------------------------------------------------------------
# 3-4. Controls: the tests above would notice a leak
# -----------------------------------------------------------------------------

def test_control_scrambling_data_before_the_cutoff_does_change_it(world):
    # FP2 is before the practice cutoff, so the practice prediction SHOULD
    # react to it. If it didn't, test 2 could pass with a scramble that never
    # reached the model at all.
    event_id = run_of(world, 1, "practice")["event_id"]
    _, drivers_before, features_before = rebuilt(world, event_id, "practice")
    with scrambled(world, rnd=1, session_types=["FP2"]):
        _, drivers_after, features_after = rebuilt(world, event_id, "practice")
    assert not same(features_after, features_before)
    assert not same(drivers_after, drivers_before)


def test_control_a_planted_leak_is_caught(world, monkeypatch):
    # Plant the classic leak: training that includes the race being predicted
    # (as if history were `race_date <= this race` instead of `<`). It's done
    # by making the race's own rows look a century old in the training table.
    # When the prediction was logged live the race had no result, so it had
    # no training rows to leak; rebuilt now, its result is in, and leaks.
    run = run_of(world, 1, "post_quali")
    race_session = session_ids(world, 1)["R"]
    real_build = stage.build_race_features

    def leaky_build(conn, *args, **kwargs):
        table = real_build(conn, *args, **kwargs)
        own_race = table["session_id"] == race_session
        return table.assign(race_date=table["race_date"].where(~own_race, "1927-01-01"))

    monkeypatch.setattr(stage, "build_race_features", leaky_build)
    p, drivers, _ = rebuilt(world, run["event_id"], "post_quali")
    log_drivers, _ = logged(world, run["run_id"])
    # The leak shows up two ways: one race too many in training, and a
    # different prediction. Test 1 checks both, so it would fail.
    assert p.n_training_races == run["n_training_races"] + 1
    assert not same(drivers, log_drivers)
