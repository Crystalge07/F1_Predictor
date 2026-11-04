"""Grid gaps, shared slots and pit-lane starters, from the grid to the blend.

The post-quali simulator starts from 0.9 x grid slot + 0.1 x model order
(simulate/probabilities.blend_with_grid). Three odd grids happen in real life:

    gap          a car that qualified doesn't start: its slot stays EMPTY on
                 the real grid, so the cars behind really do start one place
                 further back. Keeping the gap is faithful.
    shared slot  in training, pit-lane starters (grid 0 in the result) all go
                 to slot N (field size), so two of them share it.
    pit lane     live, the grid comes from OpenF1's starting grid instead.

Passing tests: the blend copes with gaps and shared slots (probabilities still
add up, order follows the blend), and training puts pit-lane starters at the
back with the flag set.

KNOWN BUGS (xfail, strict: they will start failing once fixed, as a reminder
to drop the marker). Found in the real data:
    1. OpenF1's grid sometimes LEAVES OUT pit-lane starters (2023 R3: Perez,
       Bottas; 2023 R13: Magnussen; 2 of 88 stored grids). grid_lineup only
       falls back to qualifying when the grid is completely empty, so those
       drivers vanish from the post-quali prediction.
    2. When OpenF1 does list a pit-lane starter, it gives a normal slot (20),
       not 0, so live rows have pit_lane_start = 0 while training rows for the
       same kind of start have 1 (train/serve mismatch).
"""

import pandas as pd
import pytest

from f1predictor.features.as_of import prediction_rows
from f1predictor.features.race_features import build_race_features
from f1predictor.simulate.probabilities import race_probabilities

from test_as_of import CARS, conn, weekend  # noqa: F401  (fixtures)
from test_probability_invariants import invariant_problems


def simulate(grid: list[float], model_expected: list[float]) -> pd.DataFrame:
    """Post-quali-style probabilities for a made-up race, one driver per grid entry."""
    race = pd.DataFrame({
        "driver_id": [f"d{i}" for i in range(len(grid))],
        "grid": grid,
        "expected_position": model_expected,
        "dnf_prob": 0.1,
    })
    out = race_probabilities(race, noise_sd=2.0, grid_weight=0.9, anchor_column="grid", n_sims=4_000)
    # invariant_problems also wants a predicted order (the model's) and the
    # log's name for the DNF chance.
    return out.assign(predicted_rank=out["expected_position"].rank(method="first").astype(int),
                      p_dnf=out["dnf_prob"])


def test_blend_with_a_gap_and_a_shared_pit_lane_slot():
    # 6 cars: slot 4 empty (a non-starter), two pit-lane starters both on 7.
    # The model agrees with the grid order, and prefers d4 over d5 at the back.
    grid = [1, 2, 3, 5, 7, 7]
    probs = simulate(grid, model_expected=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # The totals still hold: one winner, three on the podium, ...
    assert invariant_problems(probs) == []
    by_driver = probs.set_index("driver_id")
    # Chances fall as the start gets worse, for cars on clearly different
    # slots (1, 2, 3, 5, then the pit lane)...
    p_podium = by_driver.loc[["d0", "d1", "d2", "d3", "d4"], "p_podium"].tolist()
    assert p_podium == sorted(p_podium, reverse=True)
    assert by_driver.loc["d3", "p_podium"] > by_driver.loc["d5", "p_podium"]
    # ...and between the two cars sharing slot 7, the model breaks the tie in
    # the blend: 0.9*7 + 0.1*5 = 6.8 vs 0.9*7 + 0.1*6 = 6.9. (Their simulated
    # chances are NOT compared: 0.1 places is smaller than the Monte Carlo's
    # own sampling noise, so either could come out ahead.)
    assert by_driver.loc["d4", "expected_position"] == pytest.approx(6.8)
    assert by_driver.loc["d5", "expected_position"] == pytest.approx(6.9)


def test_training_puts_pit_lane_starters_at_the_back_with_the_flag(conn, weekend):
    # In the weekend fixture, 'e' started race 2 from the pit lane (grid 0).
    # Make 'a' a pit-lane starter too: two cars on the same slot.
    conn.execute("UPDATE results SET grid = 0 WHERE session_id = ? AND driver_id = 'a'", (weekend["R"],))
    table = build_race_features(conn)
    race = table[table["session_id"] == weekend["R"]].set_index("driver_id")
    # 4 starters -> slot 4 for both, flag on for both, nobody else flagged.
    assert race.loc["a", "grid"] == race.loc["e", "grid"] == 4
    assert race["pit_lane_start"].to_dict() == {"a": 1, "b": 0, "c": 0, "e": 1}


@pytest.mark.xfail(strict=True, reason="BUG 1: a partly published starting grid drops the missing drivers")
def test_partial_starting_grid_keeps_every_driver(conn, weekend):
    # OpenF1 publishes the grid but leaves out d (as it did with real
    # pit-lane starters in 2023). d qualified and is in the entry lists, so
    # the post-quali prediction must still include them, at the back.
    conn.execute("DELETE FROM starting_grid WHERE session_id = ? AND car_number = ?",
                 (weekend["Q"], CARS["d"]))
    rows, meta = prediction_rows(conn, weekend["event"], "post_quali")
    assert meta["lineup_source"] == "starting_grid"
    assert set(rows["driver_id"]) == {"a", "b", "c", "d"}


@pytest.mark.xfail(strict=True, reason="BUG 2: pit-lane flag is 1 in training but 0 live for the same start")
def test_pit_lane_flag_matches_between_training_and_live(conn, weekend):
    # 'a' starts race 2 from the pit lane. The official result says grid 0
    # (-> training flag 1). OpenF1's starting grid lists the car in the last
    # slot instead, which is what it did for real pit-lane starters (2023
    # R15, R19; 2024 R20), so the live row gets flag 0.
    conn.execute("UPDATE results SET grid = 0 WHERE session_id = ? AND driver_id = 'a'", (weekend["R"],))
    conn.execute("UPDATE starting_grid SET position = 3 WHERE session_id = ? AND car_number = ?",
                 (weekend["Q"], CARS["b"]))     # free the old slots up the order...
    conn.execute("UPDATE starting_grid SET position = 1 WHERE session_id = ? AND car_number = ?",
                 (weekend["Q"], CARS["c"]))
    conn.execute("UPDATE starting_grid SET position = 2 WHERE session_id = ? AND car_number = ?",
                 (weekend["Q"], CARS["d"]))
    conn.execute("UPDATE starting_grid SET position = 4 WHERE session_id = ? AND car_number = ?",
                 (weekend["Q"], CARS["a"]))     # ...and put 'a' last, as OpenF1 does

    trained_on = build_race_features(conn)
    training_flag = trained_on[(trained_on["session_id"] == weekend["R"])
                               & (trained_on["driver_id"] == "a")]["pit_lane_start"].iloc[0]
    live, _ = prediction_rows(conn, weekend["event"], "post_quali")
    live_flag = live.set_index("driver_id").loc["a", "pit_lane_start"]
    assert training_flag == 1
    assert live_flag == training_flag
