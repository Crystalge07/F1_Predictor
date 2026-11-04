"""Tests for the probability glue (grid blend, race_probabilities, scoring)
and the grid-slot probability baseline."""

import pandas as pd
import pytest

from f1predictor.models.baselines import grid_probabilities
from f1predictor.simulate.probabilities import (anchor_places, blend_with_grid, model_order,
                                                race_probabilities, score_probabilities)


def race(expected, grid, dnf=0.0):
    drivers = [f"d{i}" for i in range(len(grid))]
    return pd.DataFrame({"driver_id": drivers, "grid": grid,
                         "expected_position": expected, "dnf_prob": dnf})


def test_blend_uses_model_order_not_raw_values():
    # The model's raw numbers are on their own scale (10 .. 14), but its ORDER is d1, d0, d2.
    r = blend_with_grid(race([12.0, 10.0, 14.0], grid=[1, 2, 3]), grid_weight=0.75)
    # d0: 0.75 * 1 + 0.25 * 2 = 1.25;  d1: 0.75 * 2 + 0.25 * 1 = 1.75;  d2: 3.
    assert r["expected_position"].tolist() == pytest.approx([1.25, 1.75, 3.0], abs=1e-3)


def test_grid_weight_extremes():
    r = race([6.0, 2.0, 4.0], grid=[1, 2, 3])
    assert blend_with_grid(r, 1.0)["expected_position"].tolist() == [1, 2, 3]    # grid only
    assert blend_with_grid(r, 0.0)["expected_position"].tolist() == pytest.approx([3, 1, 2], abs=1e-3)


def test_model_order_is_the_plain_rank_when_drivers_are_apart():
    expected = pd.Series([4.0, 1.0, 9.0, 2.5])
    assert model_order(expected).tolist() == pytest.approx([3, 1, 4, 2], abs=0.01)
    assert model_order(expected, smoothing=0).tolist() == [3, 1, 4, 2]


def test_model_order_shares_places_between_level_drivers():
    order = model_order(pd.Series([1.0, 5.0, 5.0, 9.0]))
    assert order.tolist() == pytest.approx([1, 2.5, 2.5, 4], abs=1e-6)
    assert order.sum() == pytest.approx(1 + 2 + 3 + 4)


def test_practice_anchor_smooths_near_tied_laps_and_keeps_no_lap_drivers_last():
    r = race([1.0, 5.0, 9.0, 13.0], grid=[1, 2, 3, 4]).assign(
        short_run_gap_pct=[0.0, 0.300, 0.305, float("nan")],     # d1 / d2 5 ms apart; d3 no lap
        practice_rank=[1.0, 2.0, 3.0, 4.0])
    places = anchor_places(r, "practice_rank")
    assert places.iloc[0] == pytest.approx(1, abs=0.01)           # 0.3% clear: plain 1st
    assert 2 < places.iloc[1] < places.iloc[2] < 3                # near-tied: they share 2nd/3rd
    assert places.iloc[1] + places.iloc[2] == pytest.approx(5, abs=0.01)
    assert places.iloc[3] == 4                                   # no lap: stays last
    # Other anchors (the grid, the championship order) are used as they are.
    assert anchor_places(r, "grid").tolist() == [1, 2, 3, 4]


def test_a_tiny_change_in_the_model_makes_a_tiny_change_in_the_order():
    # Two drivers level-ish on 6.20 / 6.21: nudging one past the other by
    # 0.02 used to swap them by a whole place; now it moves them a little.
    before = model_order(pd.Series([3.0, 6.20, 6.21, 9.0]))
    after = model_order(pd.Series([3.0, 6.22, 6.21, 9.0]))
    assert abs(after - before).max() < 0.1
    # Plain rank (the old behaviour) jumps by a whole place.
    jump = model_order(pd.Series([3.0, 6.22, 6.21, 9.0]), 0) - model_order(pd.Series([3.0, 6.20, 6.21, 9.0]), 0)
    assert abs(jump).max() == 1


def test_race_probabilities_attach_to_rows():
    out = race_probabilities(race([1.0, 2.0, 3.0, 4.0], grid=[1, 2, 3, 4]), noise_sd=1.0)
    assert {"p_win", "p_podium", "p_points", "mean_position"} <= set(out.columns)
    assert out["p_win"].sum() == pytest.approx(1.0)
    assert out.set_index("driver_id").loc["d0", "p_win"] > out.set_index("driver_id").loc["d3", "p_win"]


def test_score_probabilities():
    rows = pd.DataFrame({"p_win": [1.0, 0.0, 0.0], "p_podium": [1.0, 1.0, 0.5],
                         "actual_position": [1, 2, 4]})
    s = score_probabilities(rows)
    assert s["brier_win"] == 0.0
    # Podium outcomes: 1, 1, 0. Only d2 is off: (0.5 - 0)^2 / 3.
    assert s["brier_podium"] == pytest.approx(0.25 / 3)


def test_grid_probabilities_from_history():
    # 4 past races: pole won 3, P2 won 1. P3 was on the podium 2 times.
    history = pd.DataFrame({
        "grid":            [1, 2, 3] * 4,
        "actual_position": [1, 2, 3,  1, 3, 2,  2, 1, 4,  1, 2, 5],
    })
    upcoming = pd.DataFrame({"driver_id": ["a", "b", "c", "d"], "grid": [1, 2, 3, 20]})
    p = grid_probabilities(history, upcoming).set_index("driver_id")
    assert p.loc["a", "p_win"] == 0.75
    assert p.loc["b", "p_win"] == 0.25
    assert p.loc["c", "p_podium"] == 0.5
    assert p.loc["d", "p_win"] == 0.0        # slot never seen before -> 0


def test_blend_with_another_anchor():
    # Before qualifying the anchor is a baseline order (e.g. champ_rank), not the grid.
    r = race([1.0, 2.0, 3.0], grid=[3, 2, 1]).assign(champ_rank=[2, 1, 3])
    out = blend_with_grid(r, grid_weight=1.0, anchor_column="champ_rank")
    assert out["expected_position"].tolist() == [2, 1, 3]


def test_slot_probabilities_on_another_column_skip_unknown_slots():
    history = pd.DataFrame({
        "practice_rank":   [1, 2, 1, 2, float("nan"), float("nan")],
        "actual_position": [1, 2, 2, 1, 1, 2],
    })
    upcoming = pd.DataFrame({"driver_id": ["a", "b"], "practice_rank": [1.0, 2.0]})
    p = grid_probabilities(history, upcoming, slot_column="practice_rank").set_index("driver_id")
    # Only the 4 rows with a known rank count: rank 1 won 1 of 2.
    assert p.loc["a", "p_win"] == 0.5


def test_add_baseline_ranks():
    from f1predictor.models.baselines import add_baseline_ranks
    features = pd.DataFrame({
        "session_id":           [1, 1, 2, 2],
        "driver_id":            ["a", "b", "a", "b"],
        "driver_season_points": [10, 20, 0, 0],
        "team_season_points":   [10, 20, 0, 0],
        "prev_season_points":   [0, 0, 5, 1],
        "short_run_gap_pct":    [0.0, 0.5, float("nan"), float("nan")],   # race 2: no practice data
    })
    out = add_baseline_ranks(features)
    assert out["champ_rank"].tolist() == [2, 1, 1, 2]
    assert out["practice_rank"].iloc[:2].tolist() == [1.0, 2.0]
    assert out["practice_rank"].iloc[2:].isna().all()
