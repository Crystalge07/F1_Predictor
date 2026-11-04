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
       calibration races (75% grid in Phase 3; 90% at the 2026-11-03 freeze,
       see tuned_settings.json).

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

# Settings per stage: (anchor column, anchor weight, noise_sd). Post-quali is
# the pair above; the other two were chosen by
# scripts/evaluate_stage_probabilities.py on the same calibration races.
# Before qualifying there's more uncertainty, so the noise is larger (4.0 vs
# 2.0), and the anchor gets less weight because the baseline orders are much
# weaker guides than the real grid.
STAGE_SETTINGS = {
    stage: (s["anchor_column"], s["anchor_weight"], s["noise_sd"])
    for stage, s in settings.load()["probabilities"].items()
}


# How gradually two drivers swap places in the model's order, in units of the
# model's raw expected position. See model_order. Chosen 2026-11-04 on
# stability alone: every value from 0 to 0.3 scored the same on the
# calibration races (within 0.1% Brier), so it isn't a tuned setting and
# tuned_settings.json stays frozen. docs/evaluation-protocol.md has the numbers.
RANK_SMOOTHING = 0.2


def model_order(expected: pd.Series, smoothing: float = RANK_SMOOTHING) -> np.ndarray:
    """The model's ORDER on a 1..N scale, with near-ties blended smoothly.

    A plain rank jumps: if two drivers' raw numbers are 6.20 and 6.21, the
    tiniest change that swaps them moves each by a whole place, and with the
    practice blend (model weight 0.75) that alone could move a win chance by
    8 points (2026 Mexico, see tests/test_stability.py). So instead of
    "how many drivers are ahead of me" (a count of 0s and 1s), each other
    driver counts as ahead by a smooth amount between 0 and 1:

        order_i = 1 + sum over j != i of  S((e_i - e_j) / smoothing)

    where S is the logistic curve (0 when j is far behind, 0.5 when level,
    1 when j is far ahead). Drivers more than ~0.5 apart (5 x smoothing) get
    their plain rank to within 0.01 place; drivers level share the place
    (two drivers tied for 2nd both get 2.5). The places still add up to
    1 + 2 + ... + N. smoothing = 0 gives the old plain rank.
    """
    if smoothing <= 0:
        return expected.rank(method="first").to_numpy()
    values = expected.to_numpy(dtype=float)
    gaps = (values[:, None] - values[None, :]) / smoothing
    # Logistic written with tanh, which can't overflow for large gaps.
    ahead = 0.5 * (1.0 + np.tanh(gaps / 2.0))
    # The diagonal (a driver vs themself) is S(0) = 0.5; the 0.5 + ... keeps
    # the leader at 1.
    return 0.5 + ahead.sum(axis=1)


# The practice stage's anchor is the practice order (fastest practice lap
# first), which is also a plain rank: two laps 5 ms apart swap a whole place
# on a 9 ms change (2026 R12 Norris/Russell moved a chance by 5.8 points).
# So in the BLEND, drivers with a lap get their place in a smooth order of
# their short-run gap (model_order again, with this width in % of lap time:
# ~45 ms on a 90 s lap). Drivers with no lap keep their integer places behind
# them. The integer practice_rank itself is unchanged (it's still the
# baseline's slot, and what the log stores as anchor_position).
PRACTICE_ORDER_SMOOTHING = 0.05


def anchor_places(race: pd.DataFrame, anchor_column: str) -> pd.Series:
    """The anchor's places for the blend: `anchor_column` as is, except the
    practice order, whose lap-time near-ties are smoothed (see above)."""
    places = race[anchor_column].astype(float)
    if anchor_column != "practice_rank" or "short_run_gap_pct" not in race:
        return places
    has_lap = race["short_run_gap_pct"].notna()
    if has_lap.any():
        # practice_rank puts drivers with a lap at 1..k (k = how many), so
        # their smooth order covers the same 1..k and the rest stay at k+1..N.
        places = places.copy()
        places[has_lap] = model_order(race.loc[has_lap, "short_run_gap_pct"], PRACTICE_ORDER_SMOOTHING)
    return places


def blend_with_grid(race: pd.DataFrame, grid_weight: float = GRID_WEIGHT,
                    anchor_column: str = "grid") -> pd.DataFrame:
    """Replace expected_position with a grid/model blend, on a 1..N scale.

    The "grid" can be any baseline order given as 1..N places in
    `anchor_column`. Post-quali uses the real grid; stages before qualifying
    have no grid, so they anchor on their own baseline order instead
    (champ_rank or practice_rank, see models/baselines.add_baseline_ranks).
    grid_weight is then the weight of that anchor.

    The model's raw numbers are turned into its predicted ORDER first
    (lowest expected position -> 1, see model_order), so both ingredients use
    the same 1..N scale before mixing. Near-ties share places smoothly, so a
    tiny change in the model's numbers only makes a tiny change here.

    Example, grid_weight 0.75: a driver starting 4th whom the model ranks 2nd
    gets 0.75 * 4 + 0.25 * 2 = 3.5.
    """
    order = model_order(race["expected_position"])
    anchor = anchor_places(race, anchor_column)
    return race.assign(expected_position=grid_weight * anchor + (1 - grid_weight) * order)


def race_probabilities(race: pd.DataFrame, noise_sd: float = NOISE_SD,
                       grid_weight: float = GRID_WEIGHT, anchor_column: str = "grid",
                       n_sims: int = N_SIMS, seed: int | None = 0) -> pd.DataFrame:
    """Blend with the grid (or another anchor), simulate one race, and attach probabilities."""
    race = blend_with_grid(race, grid_weight, anchor_column)
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
