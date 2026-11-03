"""Tests for the DNF probability model.

Uses small hand-made feature tables (no database needed: add_dnf_probability
only needs a DataFrame), so the expected numbers can be worked out by hand.
"""

import pandas as pd
import pytest

from f1predictor.models.dnf import DEFAULT_DNF_RATE, add_dnf_probability, brier


def races(results: list[dict[str, tuple[str, bool]]]) -> pd.DataFrame:
    """results[i] = {driver: (team, retired?)} for race i+1, one race per day."""
    rows = []
    for i, race in enumerate(results, start=1):
        for driver, (team, retired) in race.items():
            rows.append({"session_id": i, "race_date": f"2024-01-{i:02d}", "season": 2024,
                         "driver_id": driver, "constructor_id": team,
                         "finish_position": float("nan") if retired else 1.0})
    return pd.DataFrame(rows)


def get(df, session_id, driver):
    return df[(df.session_id == session_id) & (df.driver_id == driver)].iloc[0]


def test_first_race_uses_the_default_rate():
    df = add_dnf_probability(races([{"a": ("t", False), "b": ("t", True)}]))
    a = get(df, 1, "a")
    assert a["field_dnf_rate"] == DEFAULT_DNF_RATE
    # No history at all -> shrinkage gives exactly the field rate.
    assert a["dnf_prob"] == pytest.approx(DEFAULT_DNF_RATE)
    assert a["dnf"] == 0 and get(df, 1, "b")["dnf"] == 1


def test_field_rate_from_previous_races_only():
    # Race 1: 1 of 4 retires. Race 2: everyone retires (must not leak into race 2).
    df = add_dnf_probability(races([
        {"a": ("t", True), "b": ("t", False), "c": ("u", False), "d": ("u", False)},
        {"a": ("t", True), "b": ("t", True), "c": ("u", True), "d": ("u", True)},
    ]))
    assert get(df, 2, "c")["field_dnf_rate"] == 0.25


def test_shrinkage_by_hand():
    # Race 1: a retires, 3 others finish -> field rate 25% going into race 2.
    two_races = races([
        {"a": ("t", True), "b": ("t", False), "c": ("u", False), "d": ("u", False)},
        {"a": ("t", False), "b": ("t", False), "c": ("u", False), "d": ("u", False)},
    ])
    df = add_dnf_probability(two_races, prior_strength=2, driver_weight=1.0)
    # Driver a: 1 DNF in 1 race, plus 2 pretend races at 25%:
    #   (1 + 2 * 0.25) / (1 + 2) = 0.5
    assert get(df, 2, "a")["dnf_prob"] == pytest.approx(0.5)
    # Driver c: 0 in 1: (0 + 0.5) / 3 = 1/6
    assert get(df, 2, "c")["dnf_prob"] == pytest.approx(1 / 6)


def test_team_rate_counts_both_cars():
    two_races = races([
        {"a": ("t", True), "b": ("t", False), "c": ("u", False), "d": ("u", False)},
        {"a": ("t", False), "b": ("t", False), "c": ("u", False), "d": ("u", False)},
    ])
    df = add_dnf_probability(two_races, prior_strength=2, driver_weight=0.0)
    # Team t: 1 DNF in 2 starts: (1 + 0.5) / (2 + 2) = 0.375, the same for BOTH
    # of its drivers, even b who finished.
    assert get(df, 2, "b")["dnf_prob"] == pytest.approx(0.375)
    assert get(df, 2, "a")["dnf_prob"] == pytest.approx(0.375)


def test_more_shrinkage_pulls_towards_the_field():
    history = races([{"a": ("t", True), "b": ("u", False)}] * 5 + [{"a": ("t", False), "b": ("u", False)}])
    weak = get(add_dnf_probability(history, prior_strength=1), 6, "a")["dnf_prob"]
    strong = get(add_dnf_probability(history, prior_strength=100), 6, "a")["dnf_prob"]
    field = get(add_dnf_probability(history), 6, "a")["field_dnf_rate"]
    # a retired 5 races running: little shrinkage -> near 100%, lots -> near the field.
    assert weak > strong
    assert abs(strong - field) < abs(weak - field)


def test_brier():
    outcome = pd.Series([1, 0])
    assert brier(pd.Series([1.0, 0.0]), outcome) == 0.0      # perfect
    assert brier(pd.Series([0.5, 0.5]), outcome) == 0.25     # coin flip
    assert brier(pd.Series([0.0, 1.0]), outcome) == 1.0      # perfectly wrong
