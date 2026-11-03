"""Monte Carlo race simulation: from "predicted 3rd" to "28% chance to win".

The model gives one number per driver: an expected finishing position, if
they finish. That can't say how LIKELY anything is. So we replay the race many
times (10,000 by default) with randomness added, and count outcomes:

    for each simulated race:
        score    = expected position + random noise     (luck, strategy, mistakes)
        retire?  = random draw against the driver's DNF probability
        order    = sort by score, retirements at the back
    P(win)    = share of simulated races the driver finished 1st
    P(podium) = share they finished in the top 3
    P(points) = share they finished in the top 10

Why "Monte Carlo": the casino. Instead of working probabilities out with
formulas (very hard for 20 interacting drivers), we just roll the dice many
times and count.

The two knobs:
    noise_sd   how big the random swings are, in finishing places. Small -> the
               predicted order almost always happens; big -> anything goes. It
               gets calibrated from the model's real past errors (Phase 3 step 3).
               Wet races could later get a larger value (PRD edge case).
    dnf_prob   each driver's chance of not finishing (Phase 3 step 2).

All the work is done with numpy arrays of shape (n_sims, n_drivers): one row
per simulated race, one column per driver, so all 10,000 races run at once
with no Python loop.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_SIMS = 10_000
POINTS_POSITIONS = 10   # top 10 score points (2010+ system)
RETIRED_PENALTY = 1_000.0   # added to a retired driver's score: far behind any finisher


@dataclass
class SimulationResult:
    """Per-driver probabilities from a simulated race, one row per driver."""
    table: pd.DataFrame   # columns: driver_id, p_win, p_podium, p_points, p_dnf, mean_position

    def order(self) -> list[str]:
        """The most likely order: drivers sorted by average simulated position."""
        return self.table.sort_values("mean_position")["driver_id"].tolist()


def simulate_race(
    driver_ids: list[str],
    expected_position: np.ndarray | list[float],
    dnf_prob: np.ndarray | list[float],
    noise_sd: float,
    n_sims: int = N_SIMS,
    seed: int | None = 0,
) -> SimulationResult:
    """Simulate one race n_sims times and return each driver's probabilities.

    Args:
        driver_ids:        the drivers, in any order.
        expected_position: the model's predicted position for each driver
                           (same order as driver_ids). Only the relative
                           values matter: 3.2 vs 3.9 means "slightly ahead".
        dnf_prob:          each driver's chance of retiring, 0..1.
        noise_sd:          standard deviation of the per-race random swing, in places.
        n_sims:            how many races to simulate.
        seed:              fixed by default so the same inputs always give the
                           same probabilities (reproducible tests and logs).
                           Pass None for fresh randomness.
    """
    expected = np.asarray(expected_position, dtype=float)
    dnf = np.asarray(dnf_prob, dtype=float)
    n_drivers = len(driver_ids)
    if not (len(expected) == len(dnf) == n_drivers):
        raise ValueError("driver_ids, expected_position and dnf_prob must have the same length")
    if np.any((dnf < 0) | (dnf > 1)):
        raise ValueError("dnf_prob values must be between 0 and 1")

    # A random number generator with its own seed: doesn't affect (and isn't
    # affected by) any other randomness in the program.
    rng = np.random.default_rng(seed)

    # Shape (n_sims, n_drivers). Adding a 1-D array of length n_drivers to a
    # 2-D array "broadcasts": the same expected positions are added to every row.
    scores = expected + rng.normal(0.0, noise_sd, size=(n_sims, n_drivers))

    # Who retires in each simulated race: a uniform draw in [0, 1) below the
    # driver's DNF probability. E.g. dnf 0.1 -> True in about 10% of races.
    retired = rng.random(size=(n_sims, n_drivers)) < dnf
    # Retired drivers go behind every finisher. They keep their noisy score on
    # top of the penalty, so retirees are still ordered among themselves.
    scores = scores + retired * RETIRED_PENALTY

    # Scores -> finishing positions. argsort twice is the standard trick:
    #   argsort(scores)       = which driver is 1st, 2nd, 3rd... (an ORDER)
    #   argsort(that order)   = what place each driver came (a RANK), 0-based
    # Example: scores [2.5, 0.3, 1.1] -> order [1, 2, 0] -> ranks [2, 0, 1].
    positions = scores.argsort(axis=1).argsort(axis=1) + 1   # 1-based places

    # Averaging a True/False array gives the share of True: that IS the probability.
    table = pd.DataFrame({
        "driver_id": driver_ids,
        "p_win": (positions == 1).mean(axis=0),
        "p_podium": (positions <= 3).mean(axis=0),
        "p_points": (positions <= POINTS_POSITIONS).mean(axis=0),
        "p_dnf": retired.mean(axis=0),
        "mean_position": positions.mean(axis=0),
    })
    return SimulationResult(table)
