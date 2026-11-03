"""Tests for the probability glue (grid blend, race_probabilities, scoring)
and the grid-slot probability baseline."""

import pandas as pd
import pytest

from f1predictor.models.baselines import grid_probabilities
from f1predictor.simulate.probabilities import blend_with_grid, race_probabilities, score_probabilities


def race(expected, grid, dnf=0.0):
    drivers = [f"d{i}" for i in range(len(grid))]
    return pd.DataFrame({"driver_id": drivers, "grid": grid,
                         "expected_position": expected, "dnf_prob": dnf})


def test_blend_uses_model_order_not_raw_values():
    # The model's raw numbers are squeezed (2.4 .. 2.6), but its ORDER is d1, d0, d2.
    r = blend_with_grid(race([2.5, 2.4, 2.6], grid=[1, 2, 3]), grid_weight=0.75)
    # d0: 0.75 * 1 + 0.25 * 2 = 1.25;  d1: 0.75 * 2 + 0.25 * 1 = 1.75;  d2: 3.
    assert r["expected_position"].tolist() == [1.25, 1.75, 3.0]


def test_grid_weight_extremes():
    r = race([3.0, 1.0, 2.0], grid=[1, 2, 3])
    assert blend_with_grid(r, 1.0)["expected_position"].tolist() == [1, 2, 3]    # grid only
    assert blend_with_grid(r, 0.0)["expected_position"].tolist() == [3, 1, 2]    # model order only


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


def test_wet_noise():
    from f1predictor.simulate.probabilities import wet_noise
    assert wet_noise(2.0, rain_prob=0.0, wet_multiplier=2.0) == 2.0     # dry: unchanged
    assert wet_noise(2.0, rain_prob=1.0, wet_multiplier=2.0) == 4.0     # certainly wet: doubled
    assert wet_noise(2.0, rain_prob=0.5, wet_multiplier=2.0) == 3.0     # 50%: halfway
