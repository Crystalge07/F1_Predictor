"""Probability invariants: rules every prediction run must obey, whatever the data.

Why these rules hold: the Monte Carlo simulator (simulate/monte_carlo.py)
turns each simulated race into a complete finishing order 1..N. So in EVERY
simulated race:
    exactly 1 driver wins            -> p_win summed over drivers    = 1
    exactly 3 drivers are on podium  -> p_podium summed              = 3
    exactly 10 drivers score points  -> p_points summed              = 10
(or N, if fewer than 10 / 3 drivers are in the race). Averaging over
simulations keeps those totals, so each run's probabilities must add up to
them exactly (up to floating-point rounding).

Extra rules that follow from the same fact:
    p_win <= p_podium <= p_points for each driver (a win is a podium is points)
    average positions add up to 1 + 2 + ... + N (each simulated race hands out
        every place once)
    predicted_rank is exactly 1..N (the schema blocks duplicates, not gaps)

If one of these breaks, rows were lost, duplicated or mismatched somewhere
between the simulator and the log, and every Brier score built on them is off.

Three layers:
    1. the checker catches a deliberately broken run (so a pass means something)
    2. fresh predictions on the small fake weekend from test_as_of.py, every stage
    3. every run already in the real prediction log, read-only (skipped if the
       database file isn't there; set F1_LOG_DB to check another copy, e.g. the
       release asset downloaded with `gh release download database`)
"""

import os
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from f1predictor import config
from f1predictor.predict.stage import make_prediction

from test_as_of import conn, weekend  # noqa: F401  (fixtures)

# Sums of 10,000 averaged True/False values are exact to ~1e-12; this leaves
# room for rounding without hiding a real error (a lost driver moves a sum by
# far more than 1e-6).
TOLERANCE = 1e-6

PROBABILITY_COLUMNS = ["p_win", "p_podium", "p_points", "p_dnf"]


def invariant_problems(run: pd.DataFrame) -> list[str]:
    """Every rule `run` breaks, as readable messages (empty list = all good).

    `run` is ONE run's predictions: one row per driver with predicted_rank,
    p_win, p_podium, p_points, p_dnf, mean_position. Returning messages
    instead of asserting means a failing test lists every broken rule at once.
    """
    problems = []
    n = len(run)

    # --- Every probability is a probability --------------------------------
    for column in PROBABILITY_COLUMNS:
        outside = run[(run[column] < 0) | (run[column] > 1) | run[column].isna()]
        if not outside.empty:
            problems.append(f"{column} outside 0..1 (or missing) for {outside['driver_id'].tolist()}")

    # --- The totals ---------------------------------------------------------
    # min(...): with fewer drivers than places, every driver gets the place.
    expected_totals = {"p_win": min(1, n), "p_podium": min(3, n), "p_points": min(10, n)}
    for column, expected in expected_totals.items():
        total = run[column].sum()
        if abs(total - expected) > TOLERANCE:
            problems.append(f"{column} sums to {total:.6f}, expected {expected} ({n} drivers)")

    # --- Nesting: win is a podium is points -----------------------------------
    # The tolerance stops float noise from flagging equal values.
    not_nested = run[(run["p_win"] > run["p_podium"] + TOLERANCE)
                     | (run["p_podium"] > run["p_points"] + TOLERANCE)]
    if not not_nested.empty:
        problems.append(f"p_win <= p_podium <= p_points broken for {not_nested['driver_id'].tolist()}")

    # --- Average positions: each simulated race hands out places 1..N once ----
    expected_position_total = n * (n + 1) / 2
    position_total = run["mean_position"].sum()
    if abs(position_total - expected_position_total) > TOLERANCE * n:
        problems.append(f"mean_position sums to {position_total:.4f}, "
                        f"expected {expected_position_total} ({n} drivers)")

    # --- predicted_rank is exactly 1..N ---------------------------------------
    ranks = sorted(run["predicted_rank"].astype(int).tolist())
    if ranks != list(range(1, n + 1)):
        problems.append(f"predicted_rank is {ranks}, expected 1..{n}")

    return problems


# -----------------------------------------------------------------------------
# 1. The checker itself: a correct run passes, a broken one fails
# -----------------------------------------------------------------------------

def valid_run() -> pd.DataFrame:
    """A hand-made 4-driver run that obeys every rule.

    Totals: win 1.0, podium 3.0, points 4.0 (all 4 score: fewer than 10 drivers),
    mean positions 1.2 + 2.1 + 2.9 + 3.8 = 10 = 1+2+3+4.
    """
    return pd.DataFrame({
        "driver_id": ["a", "b", "c", "d"],
        "predicted_rank": [1, 2, 3, 4],
        "p_win": [0.6, 0.3, 0.1, 0.0],
        "p_podium": [0.95, 0.9, 0.8, 0.35],
        "p_points": [1.0, 1.0, 1.0, 1.0],
        "p_dnf": [0.05, 0.1, 0.1, 0.2],
        "mean_position": [1.2, 2.1, 2.9, 3.8],
    })


def test_checker_accepts_a_valid_run():
    assert invariant_problems(valid_run()) == []


def test_checker_catches_a_lost_driver():
    # Dropping driver 'b' (the kind of thing a bad merge does) breaks the
    # win total, the podium total, the position total and the ranks at once.
    run = valid_run()
    run = run[run["driver_id"] != "b"]
    problems = invariant_problems(run)
    assert any(p.startswith("p_win sums") for p in problems)
    assert any(p.startswith("p_podium sums") for p in problems)
    assert any(p.startswith("mean_position sums") for p in problems)
    assert any(p.startswith("predicted_rank") for p in problems)


def test_checker_catches_bad_values_and_nesting():
    run = valid_run()
    run.loc[0, "p_dnf"] = 1.5    # not a probability
    run.loc[3, "p_win"] = 0.5    # 'd' wins more often than they make the podium (0.35)
    # (This also breaks the win total; we only check the two messages we caused on purpose.)
    problems = invariant_problems(run)
    assert any("p_dnf outside" in p for p in problems)
    assert any("p_win <= p_podium" in p and "'d'" in p for p in problems)


# -----------------------------------------------------------------------------
# 2. Fresh predictions: every stage, on the fake weekend
# -----------------------------------------------------------------------------

# No "practice" here: this fake weekend has practice entries but no laps, so
# the practice stage (rightly) refuses to predict it (see
# test_predict_stage.test_practice_stage_without_practice_laps_is_refused).
# The practice stage is checked on REAL laps instead, in
# test_real_weekend_replay.test_logged_runs_obey_probability_invariants:
# all three stages of Mexico City 2026, a 22-car field with a debutant.
@pytest.mark.parametrize("stage", ["pre_weekend", "post_quali"])
def test_fresh_predictions_obey_invariants(conn, weekend, stage):
    # make_prediction only reads: nothing is written to the log. A smaller
    # n_sims keeps the test quick; the rules hold for any number of sims.
    prediction = make_prediction(conn, weekend["event"], stage, n_sims=2_000)
    assert invariant_problems(prediction.drivers) == []


# -----------------------------------------------------------------------------
# 3. Every run in the real prediction log
# -----------------------------------------------------------------------------

def log_db_path() -> Path:
    """The database whose log is checked: $F1_LOG_DB if set, else data/f1.sqlite."""
    return Path(os.environ.get("F1_LOG_DB", config.DB_PATH))


def test_every_logged_run_obeys_invariants():
    path = log_db_path()
    if not path.exists():
        pytest.skip(f"no prediction log at {path}")

    # mode=ro: opened read-only, so this test can never change the log.
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as log:
        rows = pd.read_sql_query(
            """SELECT p.*, r.event_id, r.stage, r.mode
               FROM predictions p JOIN prediction_runs r USING (run_id)""",
            log)
        runs_without_rows = log.execute(
            "SELECT run_id FROM prediction_runs WHERE run_id NOT IN (SELECT run_id FROM predictions)"
        ).fetchall()

    if rows.empty:
        pytest.skip(f"prediction log at {path} is empty")

    # A run with no prediction rows at all would be invisible to the loop below.
    assert runs_without_rows == [], f"runs logged with no predictions: {runs_without_rows}"

    # Collect every broken run before failing, so one test run shows them all.
    failures = {}
    for run_id, run in rows.groupby("run_id"):
        problems = invariant_problems(run)
        if problems:
            first = run.iloc[0]
            failures[f"run {run_id} ({first['mode']} {first['stage']}, event {first['event_id']})"] = problems
    assert failures == {}, "\n".join(f"{k}: {v}" for k, v in failures.items())
