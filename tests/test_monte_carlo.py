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
    # Nobody retires: exactly one winner, three podium places and (with 4
    # drivers) everyone in the points, in every simulated race.
    t = simulate_race(DRIVERS, [1, 2, 3, 4], [0.0] * 4, noise_sd=2.0).table
    assert t["p_win"].sum() == pytest.approx(1.0)
    assert t["p_podium"].sum() == pytest.approx(3.0)
    assert (t["p_points"] == 1.0).all()
    # With retirements, a place goes unfilled when too few cars are
    # classified, so totals sit between max - E[unclassified] and max.
    # With 4 drivers every CLASSIFIED driver is in the points.
    t = simulate_race(DRIVERS, [1, 2, 3, 4], [0.1] * 4, noise_sd=2.0).table
    for column, places in {"p_win": 1, "p_podium": 3}.items():
        assert places - t["p_dnf"].sum() - 1e-9 <= t[column].sum() <= places + 1e-9
    assert np.allclose(t["p_points"] + t["p_dnf"], 1.0)


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


def test_unclassified_retirees_do_not_score_when_fewer_than_ten_finish():
    drivers = [f"d{i}" for i in range(12)]
    # In the project, dnf_prob means "not classified". Classified late
    # retirees (the 90% rule) have numeric position_text and belong in the
    # eight retired=False drivers below, so remain eligible for points.
    result = simulate_race(
        drivers,
        list(range(1, 13)),
        [0.0] * 8 + [1.0] * 4,
        noise_sd=0.0,
    )
    table = result.table.set_index("driver_id")
    assert (table.loc[drivers[:8], "p_points"] == 1.0).all()
    assert (table.loc[drivers[8:], "p_points"] == 0.0).all()
    assert table["p_points"].sum() == 8.0


def test_unclassified_cars_never_win_or_podium_and_nesting_holds():
    # 4 cars, 2 certain to be unclassified: only 2 classified, so in every
    # race the 3rd place goes to an unclassified car. It must not count as
    # a podium (it didn't count as points either: same mask on all three).
    result = simulate_race(DRIVERS, [1, 2, 3, 4], [0.0, 0.0, 1.0, 1.0], noise_sd=0.0)
    t = result.table.set_index("driver_id")
    assert (t.loc[["c", "d"], ["p_win", "p_podium", "p_points"]] == 0.0).all().all()
    assert t["p_win"].sum() == 1.0 and t["p_podium"].sum() == 2.0
    assert ((t["p_win"] <= t["p_podium"]) & (t["p_podium"] <= t["p_points"])).all()

    # Everyone unclassified: nobody wins, nobody scores.
    t = simulate_race(DRIVERS, [1, 2, 3, 4], [1.0] * 4, noise_sd=1.0).table
    assert t[["p_win", "p_podium", "p_points"]].to_numpy().sum() == 0.0


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
