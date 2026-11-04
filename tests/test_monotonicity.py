"""Monotonicity: with everything else equal, a BETTER grid slot or FASTER
practice pace must never LOWER a driver's win or podium chance.

Two levels:

1. The simulator alone (always runs). Moving one driver's anchor slot back,
   with the model's view unchanged, raises their blended start position. The
   seed is fixed, so every simulated race uses the same random numbers: the
   driver's score is worse in EVERY simulation and nobody else's changes, so
   their chances can only go down. This holds exactly, not just on average.

2. The real trained model, 2026 R19 (tests/real_db.py; skipped without the
   database). XGBoost isn't monotone by construction: a tree can learn
   "0.26% off in practice -> better finish than 0.25%" from noise. Before the
   fix (was BUG 6) that happened for every driver in the practice model (up
   to 2.3 places), and with the model's order at 0.75 of the blend it reached
   the probabilities: Piastri's podium chance was 15% at 0.25% off the pace
   and 27% at 0.5% off. Fixed with monotone constraints on grid, qualifying
   gap and both practice gaps (models/position.MONOTONE_DIRECTIONS). Checked
   on the raw model output and on the final probabilities.
"""

import numpy as np
import pandas as pd
import pytest

from f1predictor.models.position import MONOTONE_DIRECTIONS
from f1predictor.simulate.probabilities import race_probabilities

import real_db

# Allowed "reversal" in the real-model tests: a chance may go UP by at most
# this much when the input gets worse. Leaves room for nothing but float noise
# (with the fixed seed, a truly monotone pipeline gives exactly 0).
TOLERANCE = 0.005


# -----------------------------------------------------------------------------
# 1. The simulator alone
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("stage_weight, noise_sd", [(0.9, 2.0), (0.25, 4.0)])
def test_simulator_worse_anchor_never_raises_chances(stage_weight, noise_sd):
    # 10 drivers; the model's order is fixed (0..9). Driver d3 is moved from
    # slot 1 to slot 10 while everyone else keeps their slot.
    n = 10
    curve = []      # one row of d3's chances per slot
    for slot in range(1, n + 1):
        race = pd.DataFrame({
            "driver_id": [f"d{i}" for i in range(n)],
            "grid": [float(i + 1) for i in range(n)],
            "expected_position": np.arange(n, dtype=float),
            "dnf_prob": 0.1,
        })
        race.loc[3, "grid"] = slot
        out = race_probabilities(race, noise_sd=noise_sd, grid_weight=stage_weight,
                                 anchor_column="grid", n_sims=4_000).set_index("driver_id")
        curve.append(out.loc["d3", ["p_win", "p_podium", "p_points"]])
    curve = pd.DataFrame(curve)
    # Going down the slots (worse), no chance may ever go up.
    assert (curve.diff().dropna() <= 1e-12).all().all(), curve


# -----------------------------------------------------------------------------
# 2. The real trained model
# -----------------------------------------------------------------------------

def reversals(stage: str, feature: str, values: list[float]) -> dict[str, float]:
    """For each driver: set `feature` to each of `values` (worse and worse),
    everything else unchanged, and record the biggest RISE in p_win / p_podium
    along the way. Returns {driver: biggest rise} for rises above TOLERANCE."""
    model, race = real_db.stage_setup(stage)
    found = {}
    for driver in race["driver_id"]:
        curve = []
        for value in values:
            changed = race.copy()
            changed.loc[changed["driver_id"] == driver, feature] = value
            curve.append(real_db.probabilities(model, changed, stage).loc[driver])
        rise = pd.DataFrame(curve).diff().max().max()
        if rise > TOLERANCE:
            found[driver] = round(float(rise), 3)
    return found


@pytest.mark.parametrize("stage, feature, values", [
    ("post_quali", "grid", [float(s) for s in range(1, 23)]),
    ("post_quali", "quali_gap_pct", [i * 0.1 for i in range(31)]),
    ("practice", "short_run_gap_pct", [i * 0.1 for i in range(31)]),
    ("practice", "long_run_gap_pct", [i * 0.1 for i in range(41)]),
])
def test_raw_model_never_predicts_better_for_a_worse_input(stage, feature, values):
    # The model's own expected position, before any blending: for every
    # driver, a worse value must never give a better (smaller) prediction.
    assert MONOTONE_DIRECTIONS[feature] == 1
    model, race = real_db.stage_setup(stage)
    columns = list(model.feature_names_in_)
    for _, row in race.iterrows():
        grid = pd.DataFrame([row[columns]] * len(values)).astype(float)
        grid[feature] = values
        steps = np.diff(model.predict(grid))
        assert (steps >= -1e-6).all(), f"{row['driver_id']}: predicted better at a worse {feature}"


def test_post_quali_better_grid_never_lowers_chances():
    assert reversals("post_quali", "grid", [float(s) for s in range(1, 23)]) == {}


def test_practice_faster_pace_never_lowers_chances():
    gaps = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]    # % off the fastest lap
    assert reversals("practice", "short_run_gap_pct", gaps) == {}
