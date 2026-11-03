"""Tests for the scoring metrics and the grid-order baseline.

The metric tests use races small enough to check by hand, so each expected
number can be verified with pencil and paper.
"""

import pytest

from f1predictor.db.connection import get_connection, init_db
from f1predictor.evaluate.metrics import RaceScore, average_scores, score_race
from f1predictor.models.baselines import actual_result, grid_order, race_sessions


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def test_perfect_prediction():
    actual = {"a": 1, "b": 2, "c": 3, "d": 4}
    assert score_race(["a", "b", "c", "d"], actual) == RaceScore(0.0, 1.0, 1, 1.0)


def test_fully_reversed_prediction():
    actual = {"a": 1, "b": 2, "c": 3, "d": 4}
    score = score_race(["d", "c", "b", "a"], actual)
    # Errors: d 3, c 1, b 1, a 3 -> MAE 8/4 = 2.0. Spearman exactly -1.
    assert score.position_mae == 2.0
    assert score.spearman == -1.0
    assert score.winner_hit == 0
    # Our top 3 (d, c, b) vs real top 3 (a, b, c): b and c overlap -> 2/3.
    assert score.podium_hits == pytest.approx(2 / 3)


def test_two_leaders_swapped():
    actual = {"nor": 1, "ver": 2, "lec": 3, "ham": 4}
    score = score_race(["ver", "nor", "lec", "ham"], actual)
    assert score.position_mae == 0.5             # two drivers, 1 place off each: 2/4
    assert score.winner_hit == 0
    assert score.podium_hits == 1.0              # same three drivers, different order
    # sum(d^2) = 2, n = 4: 1 - 6*2 / (4*15) = 0.8
    assert score.spearman == pytest.approx(0.8)


def test_mismatched_drivers_raise():
    with pytest.raises(ValueError, match="differ"):
        score_race(["a", "b"], {"a": 1, "c": 2})


def test_average_scores():
    scores = [RaceScore(1.0, 0.5, 1, 1.0), RaceScore(3.0, 0.7, 0, 1 / 3)]
    averages = average_scores(scores)
    assert averages["position_mae"] == 2.0
    assert averages["winner_hit"] == 0.5         # won 1 of 2 races = 50%
    assert averages["spearman"] == pytest.approx(0.6)


# -----------------------------------------------------------------------------
# Grid-order baseline
# -----------------------------------------------------------------------------

@pytest.fixture
def conn():
    connection = get_connection(":memory:")
    init_db(connection)
    yield connection
    connection.close()


def make_race(conn, grid_and_finish: dict[str, tuple[int, int]], quali: dict[str, int] | None = None):
    """A race where each driver has (grid, finishing position). Returns session_id."""
    conn.execute("INSERT INTO circuits (circuit_id, name) VALUES ('c', 'C')")
    conn.execute("INSERT INTO constructors (constructor_id, name) VALUES ('t', 'T')")
    event_id = conn.execute("INSERT INTO events (season, round, name, circuit_id, race_date) "
                            "VALUES (2024, 1, 'GP', 'c', '2024-03-02')").lastrowid
    race = conn.execute("INSERT INTO sessions (event_id, session_type) VALUES (?, 'R')",
                        (event_id,)).lastrowid
    for driver, (grid, finish) in grid_and_finish.items():
        conn.execute("INSERT INTO drivers (driver_id, given_name, family_name) VALUES (?, 'X', 'Y')",
                     (driver,))
        conn.execute("INSERT INTO results (session_id, driver_id, constructor_id, grid, position, "
                     "position_text) VALUES (?, ?, 't', ?, ?, ?)",
                     (race, driver, grid, finish, str(finish)))
    if quali:
        q = conn.execute("INSERT INTO sessions (event_id, session_type) VALUES (?, 'Q')",
                         (event_id,)).lastrowid
        for driver, position in quali.items():
            conn.execute("INSERT INTO qualifying (session_id, driver_id, constructor_id, position) "
                         "VALUES (?, ?, 't', ?)", (q, driver, position))
    return race


def test_grid_order_follows_the_grid(conn):
    race = make_race(conn, {"a": (2, 1), "b": (1, 2), "c": (3, 3)})
    assert grid_order(conn, race) == ["b", "a", "c"]
    assert actual_result(conn, race) == {"a": 1, "b": 2, "c": 3}


def test_pit_lane_starters_are_predicted_last(conn):
    # 'x' and 'y' start from the pit lane (grid 0). They must go to the back,
    # ordered by where they qualified: y qualified 1st, x 4th.
    race = make_race(
        conn,
        {"a": (1, 1), "x": (0, 2), "b": (2, 3), "y": (0, 4)},
        quali={"y": 1, "a": 2, "b": 3, "x": 4},
    )
    assert grid_order(conn, race) == ["a", "b", "y", "x"]


def test_race_sessions_only_lists_races_with_results(conn):
    race = make_race(conn, {"a": (1, 1)})
    event_id = conn.execute("SELECT event_id FROM sessions WHERE session_id = ?", (race,)).fetchone()[0]
    conn.execute("INSERT INTO events (season, round, name, circuit_id, race_date) "
                 "VALUES (2024, 2, 'Future GP', 'c', '2030-01-01')")
    conn.execute("INSERT INTO sessions (event_id, session_type) VALUES (?, 'R')", (event_id + 1,))
    assert [r["session_id"] for r in race_sessions(conn)] == [race]
