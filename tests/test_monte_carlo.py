"""Tests for the Monte Carlo race simulator.

Probabilities from a simulation are random, so tests either use extreme
settings (where the answer is certain) or check a known value with a tolerance
wide enough for 10,000 samples.
"""

import numpy as np
import pytest

from f1predictor.simulate.monte_carlo import simulate_race

DRIVERS = ["a", "b", "c", "d"]


def probs(result, driver):
    return result.table.set_index("driver_id").loc[driver]


def test_no_noise_no_dnfs_follows_the_prediction_exactly():
    result = simulate_race(DRIVERS, [1, 2, 3, 4], [0, 0, 0, 0], noise_sd=0.0)
    assert probs(result, "a")["p_win"] == 1.0
    assert probs(result, "d")["p_podium"] == 0.0
    assert result.order() == DRIVERS


def test_probabilities_add_up():
    result = simulate_race(DRIVERS, [1, 2, 3, 4], [0.1] * 4, noise_sd=2.0)
    t = result.table
    # Exactly one winner and three podium places in every simulated race.
    assert t["p_win"].sum() == pytest.approx(1.0)
    assert t["p_podium"].sum() == pytest.approx(3.0)
    # With 4 drivers, everyone is "in the points" (top 10) every time.
    assert (t["p_points"] == 1.0).all()


def test_equal_drivers_get_equal_chances():
    result = simulate_race(DRIVERS, [5, 5, 5, 5], [0, 0, 0, 0], noise_sd=1.0, n_sims=40_000)
    # Each should win about 25%. With 40k sims, random error is ~0.2%.
    assert np.allclose(result.table["p_win"], 0.25, atol=0.02)


def test_more_noise_spreads_the_chances():
    calm = simulate_race(DRIVERS, [1, 2, 3, 4], [0] * 4, noise_sd=0.5)
    chaos = simulate_race(DRIVERS, [1, 2, 3, 4], [0] * 4, noise_sd=5.0)
    assert probs(calm, "a")["p_win"] > probs(chaos, "a")["p_win"]
    assert probs(calm, "d")["p_win"] < probs(chaos, "d")["p_win"]


def test_certain_dnf_never_wins_and_goes_to_the_back():
    result = simulate_race(DRIVERS, [1, 2, 3, 4], [1.0, 0, 0, 0], noise_sd=0.0)
    a = probs(result, "a")
    assert a["p_win"] == 0.0
    assert a["p_dnf"] == 1.0
    assert a["mean_position"] == 4.0
    # 'b' inherits the win.
    assert probs(result, "b")["p_win"] == 1.0


def test_dnf_rate_matches_the_input():
    result = simulate_race(DRIVERS, [1, 2, 3, 4], [0.2] * 4, noise_sd=1.0)
    assert np.allclose(result.table["p_dnf"], 0.2, atol=0.02)


def test_same_seed_same_answer():
    a = simulate_race(DRIVERS, [1, 2, 3, 4], [0.1] * 4, noise_sd=2.0, seed=7)
    b = simulate_race(DRIVERS, [1, 2, 3, 4], [0.1] * 4, noise_sd=2.0, seed=7)
    assert a.table.equals(b.table)


def test_bad_inputs_raise():
    with pytest.raises(ValueError, match="same length"):
        simulate_race(DRIVERS, [1, 2, 3], [0] * 4, noise_sd=1.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        simulate_race(DRIVERS, [1, 2, 3, 4], [0, 0, 0, 1.5], noise_sd=1.0)


def test_per_driver_noise():
    # a, b, c have no noise; only d (predicted last) is noisy. So a (predicted
    # 2nd) can never beat b (1st), and the only winners possible are b or d.
    result = simulate_race(DRIVERS, [2, 1, 3, 4], [0] * 4, noise_sd=[0.0, 0.0, 0.0, 3.0])
    assert probs(result, "a")["p_win"] == 0.0
    assert probs(result, "d")["p_win"] > 0          # d's noise lets it jump ahead sometimes
    assert probs(result, "b")["p_win"] + probs(result, "d")["p_win"] == pytest.approx(1.0)


def test_noise_per_driver_must_match_the_field():
    with pytest.raises(ValueError, match="one value per driver"):
        simulate_race(DRIVERS, [1, 2, 3, 4], [0] * 4, noise_sd=[1.0, 2.0])
