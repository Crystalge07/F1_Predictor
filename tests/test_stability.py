"""Stability: a tiny change to one driver's input should only cause a tiny
change in the output.

"Tiny" here = 0.01% of lap time on a pace feature (about 9 ms on a 90 s lap,
well inside timing noise), or 0.01 places on a form feature. "Small output
change" = no driver's win or podium chance moves by more than 3 points.

Two levels, as in test_monotonicity.py:

1. The simulator alone (always runs): nudging one driver's blended start by
   0.01 places moves chances by far less than a point. With the seed fixed,
   only simulations where two cars were within 0.01 of each other change.

2. The real trained model, 2026 R19 (skipped without the database). Two steps
   are NOT smooth:
     * XGBoost is piecewise constant: crossing a split threshold makes the
       raw prediction jump.
     * the blend uses the model's ORDER (rank), so a jump that passes another
       driver moves the start by a whole place times the model weight.
   Before the fix (was BUG 7), 0.01% more practice gap for Piastri moved
   someone's chance by 16 points: Piastri crossed a split the WRONG way
   (better prediction for a slower lap, 7.6 -> 6.1) and passed two drivers
   in the model's order. With monotone constraints
   (models/position.MONOTONE_DIRECTIONS) the jumps can only go the right
   way and are smaller: worst move now 1.9 points (practice) and 0
   (post-quali qualifying gap).
"""

import numpy as np
import pandas as pd
import pytest

from f1predictor.simulate.monte_carlo import simulate_race

import real_db

MAX_MOVE = 0.03     # 3 points of probability


# -----------------------------------------------------------------------------
# 1. The simulator alone
# -----------------------------------------------------------------------------

def test_simulator_tiny_change_tiny_effect():
    drivers = [f"d{i}" for i in range(20)]
    start = np.arange(1, 21, dtype=float)
    dnf = np.full(20, 0.1)
    before = simulate_race(drivers, start, dnf, noise_sd=2.0, n_sims=10_000).table
    for i in range(20):
        nudged = start.copy()
        nudged[i] += 0.01
        after = simulate_race(drivers, nudged, dnf, noise_sd=2.0, n_sims=10_000).table
        move = (after[["p_win", "p_podium", "p_points"]] - before[["p_win", "p_podium", "p_points"]]).abs()
        assert move.to_numpy().max() < 0.01, f"nudging d{i} moved a chance by {move.to_numpy().max():.3f}"


# -----------------------------------------------------------------------------
# 2. The real trained model
# -----------------------------------------------------------------------------

def worst_move(stage: str, feature: str, nudge: float) -> tuple[float, str | None]:
    """Nudge each driver's `feature` by `nudge`, one driver at a time; return
    the biggest change in anyone's p_win / p_podium, and whose nudge caused it.
    Drivers with no value for the feature (NaN) are skipped."""
    model, race = real_db.stage_setup(stage)
    base = real_db.probabilities(model, race, stage)
    worst = (0.0, None)
    for driver in race["driver_id"]:
        mask = race["driver_id"] == driver
        if race.loc[mask, feature].isna().all():
            continue
        nudged = race.copy()
        nudged.loc[mask, feature] += nudge
        move = float((real_db.probabilities(model, nudged, stage) - base).abs().to_numpy().max())
        if move > worst[0]:
            worst = (move, driver)
    return worst


@pytest.mark.parametrize("stage, feature, nudge", [
    ("post_quali", "quali_gap_pct", 0.01),
    ("post_quali", "driver_form_finish", 0.01),
    ("practice", "long_run_gap_pct", 0.01),
    ("practice", "driver_form_finish", 0.01),
    ("pre_weekend", "driver_form_finish", 0.01),
    ("pre_weekend", "team_season_finish", 0.01),
])
def test_tiny_change_tiny_effect(stage, feature, nudge):
    move, driver = worst_move(stage, feature, nudge)
    assert move <= MAX_MOVE, f"nudging {driver}'s {feature} by {nudge} moved a chance by {move:.3f}"


def test_practice_tiny_pace_change_tiny_effect():
    move, driver = worst_move("practice", "short_run_gap_pct", 0.01)
    assert move <= MAX_MOVE, f"nudging {driver}'s short_run_gap_pct by 0.01 moved a chance by {move:.3f}"
