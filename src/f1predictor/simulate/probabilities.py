"""Race probabilities: model + DNF chances + Monte Carlo, glued together.

Inputs for one race (the feature rows, plus two extra columns):
    expected_position   from the position model (models/position.predict_positions)
    dnf_prob            from the DNF model (models/dnf.add_dnf_probability)

Before simulating, the model's prediction is blended with the starting grid
(see blend_with_grid). Output: the rows with p_win, p_podium, p_points,
mean_position added.

Why blend (found in scripts/evaluate_probabilities.py, 2026-11-03): simulating
from the model's raw numbers was WORSE than the grid-only baseline. Two reasons:
    1. The model squeezes predictions towards the middle (the favourite gets
       ~2.4, not 1), so the front-runners mix too much and favourites end up
       underconfident. Using the model's predicted ORDER (1, 2, 3...) fixes the
       squeeze.
    2. As Phase 2 found, the grid is very hard to beat after qualifying. Mostly
       trusting it and letting the model nudge it worked best on the
       calibration races: 75% grid + 25% model order.

Also home to the scoring used for probabilities (Brier score per outcome).
"""

import numpy as np
import pandas as pd

from f1predictor import settings

from f1predictor.models.dnf import brier
from f1predictor.simulate.monte_carlo import N_SIMS, simulate_race

# Both chosen together by scripts/evaluate_probabilities.py on the calibration
# races (2024 + 2026 rounds 1-9).
# GRID_WEIGHT: how much of the simulator's starting point is the grid (1.0 =
#   ignore the model entirely, 0.0 = ignore the grid).
# NOISE_SD: how big the simulated race-day swings are, in finishing places.
# The values live in tuned_settings.json (see settings.py).
_post_quali = settings.load()["probabilities"]["post_quali"]
GRID_WEIGHT: float = _post_quali["anchor_weight"]
NOISE_SD: float = _post_quali["noise_sd"]
NOISE_POWER: float = _post_quali["noise_power"]

# Noise shape. One noise size for the whole field made favourites
# underconfident (drivers given ~43% to win actually won ~58%): gaps at the
# front are usually bigger and steadier than in the midfield pack. So each
# driver's noise scales with where they're expected to run:
#
#     driver's noise = noise_sd * (expected position / MIDFIELD) ** noise_power
#
# noise_sd is then "the noise for a midfielder"; power 0 = the same for
# everyone (the original model); power 0.5 = a P1 runner gets ~32% of the
# midfield noise, a P20 runner ~141%. Power is tuned with the other settings.
MIDFIELD = 10.0

# Settings per stage: (anchor column, anchor weight, noise_sd, noise_power). Post-quali is
# the pair above; the other two were chosen by
# scripts/evaluate_stage_probabilities.py on the same calibration races.
# Before qualifying there's more uncertainty, so the noise is larger (4.0 vs
# 2.0), and the anchor gets less weight because the baseline orders are much
# weaker guides than the real grid.
STAGE_SETTINGS = {
    stage: (s["anchor_column"], s["anchor_weight"], s["noise_sd"], s["noise_power"])
    for stage, s in settings.load()["probabilities"].items()
}


def blend_with_grid(race: pd.DataFrame, grid_weight: float = GRID_WEIGHT,
                    anchor_column: str = "grid") -> pd.DataFrame:
    """Replace expected_position with a grid/model blend, on a 1..N scale.

    The "grid" can be any baseline order given as 1..N places in
    `anchor_column`. Post-quali uses the real grid; stages before qualifying
    have no grid, so they anchor on their own baseline order instead
    (champ_rank or practice_rank, see models/baselines.add_baseline_ranks).
    grid_weight is then the weight of that anchor.

    The model's raw numbers are turned into its predicted ORDER first
    (rank: lowest expected position -> 1), so both ingredients use the same
    1..N scale before mixing. rank(method="first") breaks exact ties by row
    order, so every driver gets a distinct place.

    Example, grid_weight 0.75: a driver starting 4th whom the model ranks 2nd
    gets 0.75 * 4 + 0.25 * 2 = 3.5.
    """
    model_order = race["expected_position"].rank(method="first")
    return race.assign(expected_position=grid_weight * race[anchor_column] + (1 - grid_weight) * model_order)


def driver_noise(expected_position, noise_sd: float, noise_power: float):
    """Each driver's noise: noise_sd * (expected position / MIDFIELD) ** noise_power.

    Example, noise_sd 2, power 0.5: expected P1 -> 2 * 0.1 ** 0.5 = 0.63 places,
    P10 -> 2.0, P20 -> 2.83. Power 0 gives every driver exactly noise_sd.
    """
    return noise_sd * (np.asarray(expected_position, dtype=float) / MIDFIELD) ** noise_power


def race_probabilities(race: pd.DataFrame, noise_sd: float = NOISE_SD,
                       grid_weight: float = GRID_WEIGHT, anchor_column: str = "grid",
                       noise_power: float = NOISE_POWER,
                       n_sims: int = N_SIMS, seed: int | None = 0) -> pd.DataFrame:
    """Blend with the grid (or another anchor), simulate one race, and attach probabilities."""
    race = blend_with_grid(race, grid_weight, anchor_column)
    expected = race["expected_position"].to_numpy()
    result = simulate_race(
        race["driver_id"].tolist(),
        expected,
        race["dnf_prob"].to_numpy(),
        noise_sd=driver_noise(expected, noise_sd, noise_power),
        n_sims=n_sims,
        seed=seed,
    )
    # p_dnf in the simulation is just dnf_prob re-measured; drop the simulated
    # copy so there's one source of truth.
    return race.merge(result.table.drop(columns="p_dnf"), on="driver_id", how="left")


def score_probabilities(rows: pd.DataFrame) -> dict[str, float]:
    """Brier scores for win and podium over many driver-race rows (lower = better).

    Needs p_win, p_podium and actual_position (retirements included).
    """
    won = (rows["actual_position"] == 1).astype(int)
    podium = (rows["actual_position"] <= 3).astype(int)
    return {"brier_win": brier(rows["p_win"], won),
            "brier_podium": brier(rows["p_podium"], podium)}
