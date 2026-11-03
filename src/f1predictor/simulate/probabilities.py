"""Race probabilities: model + DNF chances + Monte Carlo, glued together.

Inputs for one race (the feature rows, plus two extra columns):
    expected_position   from the position model (models/post_quali.predict_positions)
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

import pandas as pd

from f1predictor.models.dnf import brier
from f1predictor.simulate.monte_carlo import N_SIMS, simulate_race

# Both chosen together by scripts/evaluate_probabilities.py on the calibration
# races (2024 + 2026 rounds 1-9).
# GRID_WEIGHT: how much of the simulator's starting point is the grid (1.0 =
#   ignore the model entirely, 0.0 = ignore the grid).
# NOISE_SD: how big the simulated race-day swings are, in finishing places.
GRID_WEIGHT = 0.75
NOISE_SD = 2.0


def blend_with_grid(race: pd.DataFrame, grid_weight: float = GRID_WEIGHT) -> pd.DataFrame:
    """Replace expected_position with a grid/model blend, on a 1..N scale.

    The model's raw numbers are turned into its predicted ORDER first
    (rank: lowest expected position -> 1), so both ingredients use the same
    1..N scale before mixing. rank(method="first") breaks exact ties by row
    order, so every driver gets a distinct place.

    Example, grid_weight 0.75: a driver starting 4th whom the model ranks 2nd
    gets 0.75 * 4 + 0.25 * 2 = 3.5.
    """
    model_order = race["expected_position"].rank(method="first")
    return race.assign(expected_position=grid_weight * race["grid"] + (1 - grid_weight) * model_order)


def race_probabilities(race: pd.DataFrame, noise_sd: float = NOISE_SD,
                       grid_weight: float = GRID_WEIGHT,
                       n_sims: int = N_SIMS, seed: int | None = 0) -> pd.DataFrame:
    """Blend with the grid, simulate one race, and attach each driver's probabilities."""
    race = blend_with_grid(race, grid_weight)
    result = simulate_race(
        race["driver_id"].tolist(),
        race["expected_position"].to_numpy(),
        race["dnf_prob"].to_numpy(),
        noise_sd=noise_sd,
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
