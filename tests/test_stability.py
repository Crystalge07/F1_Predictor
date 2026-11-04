"""Stability: a tiny change to one driver's input should only cause a tiny
change in the output.

"Tiny" here = 0.01% of lap time on a pace feature (about 9 ms on a 90 s lap,
well inside timing noise), or 0.01 places on a form feature. "Small output
change" = no driver's win or podium chance moves by more than 3 points.

Two levels, as in test_monotonicity.py:

1. The simulator alone (always runs): nudging one driver's blended start by
   0.01 places moves chances by far less than a point. With the seed fixed,
   only simulations where two cars were within 0.01 of each other change.

2. The real trained model (skipped without the database): 2026 R19, plus
   the races where a survey of every calibration and held-out race found
   the biggest jumps (2026-11-04). Three steps used to be steps, not slopes:
     * XGBoost is piecewise constant: crossing a split threshold makes the
       raw prediction jump (2024 Miami: Sainz +0.33 places for 9 ms).
       Fix: predictions are averaged over nearby paces
       (models/position.PACE_SMOOTHING).
     * the blend used the model's ORDER as a plain rank, so a jump that
       passed another driver moved the start by a whole place times the
       model weight (2026 Mexico with 2021 Spa left out of training: 8.1
       points). Fix: a smooth order (simulate/probabilities.model_order).
     * the practice stage's anchor, the practice order, is a plain rank of
       best laps too: two laps 5 ms apart swapped places (2026 R12: 5.8
       points). Fix: smooth near-ties in the blend
       (simulate/probabilities.PRACTICE_ORDER_SMOOTHING).
   Before any of this (was BUG 7), 0.01% more practice gap for Piastri
   moved someone's chance by 16 points: Piastri crossed a split the WRONG
   way (better prediction for a slower lap). Monotone constraints
   (models/position.MONOTONE_DIRECTIONS) fixed the direction, not the size:
   the survey still found 14 points (2024 Miami) before the fixes above.
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

def worst_move(stage: str, feature: str, nudge: float,
               season: int = real_db.SEASON, rnd: int = real_db.ROUND) -> tuple[float, str | None]:
    """Nudge each driver's `feature` by `nudge`, one driver at a time; return
    the biggest change in anyone's p_win / p_podium, and whose nudge caused it.
    Drivers with no value for the feature (NaN) are skipped."""
    model, race = real_db.stage_setup(stage, season, rnd)
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


# The worst races in the 2026-11-04 survey of all 71 calibration and held-out
# races, worst move before -> after the smoothing fixes (points):
# 2024 R6 13.9 -> 0.6, 2024 R10 2.6 -> 0.7, 2026 R8 5.3 -> 0.6,
# 2026 R10 7.3 -> 0.4, 2026 R12 2.6 -> 1.3, 2026 R19 1.9 -> 0.6.
# (docs/evaluation-protocol.md, "Stability fixes and races stopped early".)
@pytest.mark.parametrize("season, rnd", [(2026, 19), (2024, 6), (2024, 10), (2026, 8), (2026, 10), (2026, 12)])
def test_practice_tiny_pace_change_tiny_effect(season, rnd):
    move, driver = worst_move("practice", "short_run_gap_pct", 0.01, season, rnd)
    assert move <= MAX_MOVE, (f"{season} R{rnd}: nudging {driver}'s short_run_gap_pct by 0.01 "
                              f"moved a chance by {move:.3f}")
