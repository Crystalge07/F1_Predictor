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

FIXED: OpenF1's grid sometimes LEAVES OUT pit-lane starters (2023 R3:
Perez, Bottas; 2023 R13: Magnussen; 2 of 88 stored grids). They used to vanish
from the post-quali prediction (was BUG 1); now features/as_of.
add_missing_starters adds every qualifier whose car isn't on the grid as a
pit-lane starter.

FIXED: when OpenF1 does list a pit-lane starter, it gives a normal slot
(20), not 0, so live rows had pit_lane_start = 0 while training rows for the
same kind of start had 1 (a train/serve mismatch, was BUG 2). The flag is no
longer a model input (features/race_features.py); grid carries the
information in both.
"""

import pandas as pd
import pytest

from f1predictor.features.as_of import prediction_rows
from f1predictor.ingest import openf1
from f1predictor.features.race_features import POST_QUALI_FEATURES, build_race_features
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
    # own sampling noise, so either could come out ahead.) The model's order
    # is smooth (simulate/probabilities.model_order), so the places are
    # within a hair of 5 and 6, not exactly.
    assert by_driver.loc["d4", "expected_position"] == pytest.approx(6.8, abs=1e-3)
    assert by_driver.loc["d5", "expected_position"] == pytest.approx(6.9, abs=1e-3)


def test_training_puts_pit_lane_starters_at_the_back_with_the_flag(conn, weekend):
    # In the weekend fixture, 'e' started race 2 from the pit lane (grid 0).
    # Make 'a' a pit-lane starter too: two cars on the same slot.
    conn.execute("UPDATE results SET grid = 0 WHERE session_id = ? AND driver_id = 'a'", (weekend["R"],))
    table = build_race_features(conn)
    race = table[table["session_id"] == weekend["R"]].set_index("driver_id")
    # 4 starters -> slot 4 for both, flag on for both, nobody else flagged.
    assert race.loc["a", "grid"] == race.loc["e", "grid"] == 4
    assert race["pit_lane_start"].to_dict() == {"a": 1, "b": 0, "c": 0, "e": 1}


def test_partial_starting_grid_keeps_every_driver(conn, weekend):
    # Was BUG 1. OpenF1 publishes the grid but leaves out d (as it did with
    # real pit-lane starters in 2023). d qualified, so the post-quali
    # prediction must still include them: as a pit-lane starter, at the back.
    conn.execute("DELETE FROM starting_grid WHERE session_id = ? AND car_number = ?",
                 (weekend["Q"], CARS["d"]))
    rows, meta = prediction_rows(conn, weekend["event"], "post_quali")
    assert meta["lineup_source"] == "starting_grid"
    assert set(rows["driver_id"]) == {"a", "b", "c", "d"}
    d = rows.set_index("driver_id").loc["d"]
    assert d["grid"] == 4                       # 4 cars: the back of the field
    assert d["pit_lane_start"] == 1


def test_complete_starting_grid_adds_nobody(conn, weekend):
    # Every qualifier is on the grid: the lineup is exactly the grid.
    rows, _ = prediction_rows(conn, weekend["event"], "post_quali")
    assert sorted(rows["grid"]) == [1, 2, 3, 4]
    assert rows["pit_lane_start"].sum() == 0


def test_an_unmatched_car_on_the_grid_is_not_added_twice(conn, weekend):
    # d's car IS on the grid, but its qualifying entry couldn't be matched
    # to a driver. d is missing by name, not by car, so d must not be added
    # again at the back (that would put one car on the grid twice).
    conn.execute("UPDATE session_entries SET driver_id = NULL WHERE session_id = ? AND car_number = ?",
                 (weekend["Q"], CARS["d"]))
    rows, _ = prediction_rows(conn, weekend["event"], "post_quali")
    assert "d" not in set(rows["driver_id"])


def test_a_debutant_without_a_code_keeps_his_grid_slot(conn, weekend):
    # Real case, Brazil 2026: Camara qualified and started 11th, but Jolpica
    # gave him no code, so none of his car's entries could be matched to his
    # id before the race. add_missing_starters then saw "a qualifier with no
    # car on the grid" and added him as a pit-lane starter: grid 22 in the
    # live post-quali prediction. Here 'd' plays him, starting from pole.
    #
    # Everyone else is matched by code (rule 3), as real drivers are.
    for driver in ("a", "b", "c"):
        code = driver.upper() * 3
        conn.execute("UPDATE drivers SET code = ? WHERE driver_id = ?", (code, driver))
        conn.execute("UPDATE session_entries SET acronym = ? WHERE session_id = ? AND car_number = ?",
                     (code, weekend["Q"], CARS[driver]))
    # 'd': no code, a real family name, and no matched entry anywhere this weekend.
    conn.execute("UPDATE drivers SET code = NULL, given_name = 'Dan', family_name = 'Debut' "
                 "WHERE driver_id = 'd'")
    conn.execute("UPDATE session_entries SET driver_id = NULL, constructor_id = NULL, "
                 "acronym = 'DEB', full_name = 'Dan DEBUT' WHERE car_number = ? "
                 "AND session_id IN (?, ?)", (CARS["d"], weekend["FP3"], weekend["Q"]))
    # Penalties put the debutant on pole and 'b' at the back.
    conn.execute("UPDATE starting_grid SET position = 5 WHERE session_id = ? AND car_number = ?",
                 (weekend["Q"], CARS["b"]))
    conn.execute("UPDATE starting_grid SET position = 1 WHERE session_id = ? AND car_number = ?",
                 (weekend["Q"], CARS["d"]))

    # What the pipeline does once Jolpica's qualifying is in (pipeline/fetch.py).
    openf1.match_entries(conn, weekend["Q"])

    rows, meta = prediction_rows(conn, weekend["event"], "post_quali")
    assert meta["lineup_source"] == "starting_grid"
    d = rows.set_index("driver_id").loc["d"]
    assert d["grid"] == 1                    # his real slot, not the back of the field
    assert d["pit_lane_start"] == 0


def test_pit_lane_start_is_the_same_input_in_training_and_live(conn, weekend):
    # Was BUG 2. 'a' starts race 2 from the pit lane. The official result
    # says grid 0 (-> training flag 1). OpenF1's starting grid lists the car
    # in the last slot instead, as it did for real pit-lane starters (2023
    # R15, R19; 2024 R20), so live the flag is 0. Fix: the flag is no longer
    # a model input; `grid`, which IS an input, puts the car at the back in
    # both.
    conn.execute("UPDATE results SET grid = 0 WHERE session_id = ? AND driver_id = 'a'", (weekend["R"],))
    for driver, slot in {"c": 1, "d": 2, "b": 3, "a": 4}.items():   # 'a' last, as OpenF1 does
        conn.execute("UPDATE starting_grid SET position = ? WHERE session_id = ? AND car_number = ?",
                     (slot, weekend["Q"], CARS[driver]))

    assert "pit_lane_start" not in POST_QUALI_FEATURES
    trained_on = build_race_features(conn)
    training = trained_on[(trained_on["session_id"] == weekend["R"])].set_index("driver_id").loc["a"]
    live = prediction_rows(conn, weekend["event"], "post_quali")[0].set_index("driver_id").loc["a"]
    # The flag still differs (kept in the table for information)...
    assert (training["pit_lane_start"], live["pit_lane_start"]) == (1, 0)
    # ...but every model input agrees: the back of the field.
    assert training["grid"] == live["grid"] == 4
