"""Driver form after a run of DNFs, compared with a true rookie.

driver_form_finish = average finish over the driver's last FORM_WINDOW (5)
races, DNFs skipped (features/race_features._add_driver_form). The window
counts RACES, not finishes, so:

    5 DNFs in a row      -> no finish in the window -> NaN
    rookie               -> no races at all          -> NaN

Both look the same in that one column. Is that a bug? No, by design:
    * the position model predicts "where you finish IF you finish", and a
      DNF says nothing about that (DNF risk is the DNF model's job, which
      does count the streak)
    * the two drivers are still told apart by driver_form_grid (the DNF
      driver has 5 grid slots, the rookie none) and by the DNF model
These tests pin that down. One real weakness, noted for later: a driver who
won before the streak has that history forgotten after 5 DNFs, instead of
falling back to an older finish (e.g. "last 5 FINISHES").
"""

import pandas as pd
import pytest

from f1predictor.features.race_features import FORM_WINDOW, build_race_features
from f1predictor.models.dnf import add_dnf_probability

from test_as_of import CARS, TEAMS, add_event, conn  # noqa: F401  (fixtures)


def add_result(conn, race_id, driver, position, classified, grid):
    conn.execute(
        "INSERT INTO results (session_id, driver_id, constructor_id, car_number, grid, position, "
        "position_text) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (race_id, driver, TEAMS[driver], CARS[driver], grid,
         position, str(position) if classified else "R"))


@pytest.fixture
def streak(conn):
    """7 races (FORM_WINDOW + 2):
        a  wins race 1, then retires in races 2-6 (5 DNFs), races again in 7
        b  4 DNFs (races 2-5) and a 3rd place in race 6
        c  finishes 2nd every time (the steady reference)
        e  a rookie: first race is race 7
    Returns race 7's session_id.
    """
    assert FORM_WINDOW == 5     # the fixture is built around a 5-race window
    race_ids = []
    for rnd in range(1, 8):
        race = add_event(conn, rnd, f"2024-{rnd + 2:02d}-01", {"R": f"2024-{rnd + 2:02d}-01T15:00:00Z"})["R"]
        race_ids.append(race)
        add_result(conn, race, "c", 2, True, 2)
        if rnd == 7:
            add_result(conn, race, "a", 1, True, 1)
            add_result(conn, race, "b", 3, True, 3)
            add_result(conn, race, "e", 4, True, 4)
            continue
        add_result(conn, race, "a", 1, rnd == 1, 1)          # wins race 1, DNF 2-6
        add_result(conn, race, "b", 3, rnd in (1, 6), 3)     # DNF in 2-5
    return race_ids[-1]


def race7(conn, race_id) -> pd.DataFrame:
    table = add_dnf_probability(build_race_features(conn))
    return table[table["session_id"] == race_id].set_index("driver_id")


def test_five_dnfs_and_a_rookie_both_have_no_finish_form(conn, streak):
    rows = race7(conn, streak)
    assert pd.isna(rows.loc["a", "driver_form_finish"])
    assert pd.isna(rows.loc["e", "driver_form_finish"])


def test_the_dnf_driver_and_the_rookie_are_still_told_apart(conn, streak):
    rows = race7(conn, streak)
    # Grid form: the DNF driver started from pole 5 times; the rookie has nothing.
    assert rows.loc["a", "driver_form_grid"] == 1.0
    assert pd.isna(rows.loc["e", "driver_form_grid"])
    # DNF model: the streak raises a's chance well above the rookie's
    # (who gets the plain field rate, having no history).
    assert rows.loc["a", "dnf_prob"] > rows.loc["e", "dnf_prob"]


def test_one_finish_in_the_window_is_the_whole_form(conn, streak):
    # b: DNF, DNF, DNF, DNF, 3rd -> form = 3.0, from a single race.
    rows = race7(conn, streak)
    assert rows.loc["b", "driver_form_finish"] == 3.0
    assert rows.loc["c", "driver_form_finish"] == 2.0


def test_a_win_before_the_streak_is_forgotten(conn, streak):
    # The weakness from the docstring, pinned so a change is deliberate: a's
    # race-1 win sits 6 races back, outside the window, so it's gone (NaN),
    # although it's the best evidence of a's pace there is. (driver_track_finish
    # still has it here, but only because all 7 races are at one circuit.)
    rows = race7(conn, streak)
    assert pd.isna(rows.loc["a", "driver_form_finish"])
    assert rows.loc["a", "driver_track_finish"] == 1.0
