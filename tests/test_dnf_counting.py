"""How disqualifications and withdrawals count in the DNF model (models/dnf.py).

Jolpica's position_text for a driver with no classified finish:
    'R' retired (crash, failure)    'D' disqualified    'W' withdrew / did not start
The DNF model counts ALL of these as a DNF (finish_position is NaN), and so
does the position model (no target). That's documented in models/dnf.py.

Is that right? It depends on what the scorer treats as "the race". The scorer
(evaluate/scorer.py) takes the field from the results table, which has a row
for D and W drivers too. So a driver who later doesn't start, or is
disqualified, is SCORED as part of the field with no podium. A probability of
"not being classified" is therefore exactly the right thing to feed the
simulator: the definition is consistent end to end. These tests pin that down,
so changing one side without the other fails loudly.

What it costs (real data, 2018-2026): D and W are 38 of 530 unclassified rows
(7%); they add 0.0-1.9 points to a season's field DNF rate (2026: 18.9% vs
17.2% retirements only). They count against the DRIVER rate (75% of the
blend), though a disqualification for plank wear or a car that won't start is
usually the team's doing: e.g. Piastri carries a 2025 DSQ and two 2026
non-starts. That's a modelling choice, not a bug: moving D/W to the team
rate only is a possible refinement.
"""

import pandas as pd
import pytest

from f1predictor.features.race_features import build_race_features
from f1predictor.models.baselines import actual_result
from f1predictor.models.dnf import add_dnf_probability

from test_as_of import CARS, TEAMS, add_event, conn  # noqa: F401  (fixtures)


def add_result(conn, race_id, driver, position, position_text, grid):
    """One results row. Jolpica gives unclassified drivers a position too
    (the back of the order), but position_text is a letter."""
    conn.execute(
        "INSERT INTO results (session_id, driver_id, constructor_id, car_number, grid, position, "
        "position_text) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (race_id, driver, TEAMS[driver], CARS[driver], grid, position, position_text))


@pytest.fixture
def two_races(conn):
    """Race 1: d wins; a retires (R), b is disqualified (D), c doesn't start (W).
    Race 2: everyone finishes. Teams: a, b = red; c, d = blue."""
    r1 = add_event(conn, 1, "2024-03-03", {"R": "2024-03-03T15:00:00Z"})
    add_result(conn, r1["R"], "d", 1, "1", 4)
    add_result(conn, r1["R"], "a", 2, "R", 1)
    add_result(conn, r1["R"], "b", 3, "D", 2)
    add_result(conn, r1["R"], "c", 4, "W", 3)
    r2 = add_event(conn, 2, "2024-03-17", {"R": "2024-03-17T15:00:00Z"})
    for position, driver in enumerate(["a", "b", "c", "d"], start=1):
        add_result(conn, r2["R"], driver, position, str(position), position)
    return r1["R"], r2["R"]


def test_retired_disqualified_and_withdrawn_all_have_no_finish(conn, two_races):
    # The position model's view: none of R / D / W is a finishing position to learn from.
    race1, _ = two_races
    table = build_race_features(conn)
    finish = table[table["session_id"] == race1].set_index("driver_id")["finish_position"]
    assert finish[["a", "b", "c"]].isna().all()
    assert finish["d"] == 1


def test_disqualified_and_withdrawn_count_exactly_like_a_retirement(conn, two_races):
    # The DNF model's view, in race 2: a (R), b (D) and c (W) each have
    # exactly one DNF in one start, so the same driver rate...
    _, race2 = two_races
    with_dnf = add_dnf_probability(build_race_features(conn))
    rates = with_dnf[with_dnf["session_id"] == race2].set_index("driver_id")
    assert rates.loc["a", "driver_dnf_rate"] == pytest.approx(rates.loc["b", "driver_dnf_rate"])
    assert rates.loc["a", "driver_dnf_rate"] == pytest.approx(rates.loc["c", "driver_dnf_rate"])
    assert rates.loc["a", "driver_dnf_rate"] > rates.loc["d", "driver_dnf_rate"]
    # ...and team red (a R + b D: 2 of 2) is rated worse than blue (c W: 1 of 2).
    assert rates.loc["a", "team_dnf_rate"] > rates.loc["c", "team_dnf_rate"]


def test_scorer_field_includes_disqualified_and_withdrawn_drivers(conn, two_races):
    # The scorer's view: D and W drivers are part of the scored field (with
    # no podium), which is why counting them as "DNF" in the probability is
    # the consistent choice. If the scorer ever starts dropping them, the DNF
    # definition above must change with it.
    race1, _ = two_races
    assert set(actual_result(conn, race1)) == {"a", "b", "c", "d"}
