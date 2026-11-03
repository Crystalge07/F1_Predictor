"""Baseline "models": simple rules every real model has to beat.

A baseline answers: "how good is a prediction that needs no machine learning
at all?" If the trained model can't beat it, the model isn't adding anything.

grid_order: predict that the race finishes in starting-grid order.
    This is the PRD's headline baseline for the post-qualifying stage. It's
    surprisingly strong: in this data the pole-sitter wins 38-68% of races
    depending on the season.

(The PRD's second baseline, championship order for the pre-weekend stage,
comes with the pre-weekend model in Phase 3.)
"""

import sqlite3

import pandas as pd


def race_sessions(conn: sqlite3.Connection, seasons: list[int] | None = None) -> list[sqlite3.Row]:
    """All race sessions that have results, in date order.

    Returns rows with session_id, season, round, name. Pass `seasons` to limit
    the list, e.g. [2024, 2025].
    """
    query = """
        SELECT s.session_id, e.season, e.round, e.name
        FROM sessions s
        JOIN events e ON e.event_id = s.event_id
        WHERE s.session_type = 'R'
          AND EXISTS (SELECT 1 FROM results r WHERE r.session_id = s.session_id)
    """
    params: list = []
    if seasons is not None:
        # One ? per season: "AND e.season IN (?, ?, ?)". Building the
        # placeholders (not the values) into the string keeps it safe.
        query += f" AND e.season IN ({', '.join('?' * len(seasons))})"
        params.extend(seasons)
    query += " ORDER BY e.season, e.round"
    return conn.execute(query, params).fetchall()


def actual_result(conn: sqlite3.Connection, session_id: int) -> dict[str, int]:
    """{driver_id: finishing position} for one race, including retirements."""
    return {
        row["driver_id"]: row["position"]
        for row in conn.execute(
            "SELECT driver_id, position FROM results WHERE session_id = ?", (session_id,))
    }


def grid_order(conn: sqlite3.Connection, session_id: int) -> list[str]:
    """Predict the finishing order = the official starting grid.

    Returns driver_ids, predicted winner first.

    Pit-lane starters have grid = 0. They start BEHIND everyone, so they're
    predicted last. Sorting 0 as a normal number would put them on pole,
    which would be a silent, nasty bug. Several pit-lane starters are ordered
    among themselves by qualifying position (where they'd have started), then
    by driver_id so the output is always the same.
    """
    rows = conn.execute(
        """
        SELECT r.driver_id, r.grid, q.position AS quali_position
        FROM results r
        JOIN sessions race ON race.session_id = r.session_id
        -- LEFT JOIN: keep the driver even if they have no qualifying row.
        LEFT JOIN sessions qs ON qs.event_id = race.event_id AND qs.session_type = 'Q'
        LEFT JOIN qualifying q ON q.session_id = qs.session_id AND q.driver_id = r.driver_id
        WHERE r.session_id = ?
        """,
        (session_id,),
    ).fetchall()

    def sort_key(row):
        # Python sorts tuples element by element, so this means:
        #   1. pit-lane starters (grid 0) after everyone else  (False < True)
        #   2. then by grid slot
        #   3. then by qualifying position (missing quali -> after everyone)
        #   4. then by driver_id, so ties always break the same way
        starts_from_pit_lane = row["grid"] == 0
        quali = row["quali_position"] if row["quali_position"] is not None else 999
        return (starts_from_pit_lane, row["grid"], quali, row["driver_id"])

    return [row["driver_id"] for row in sorted(rows, key=sort_key)]


def grid_probabilities(history: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
    """Grid-only probability baseline: P(win) and P(podium) from the starting slot alone.

    The probability version of grid_order. Each driver gets the share of past
    races in which the driver starting from their slot went on to win (or
    finish on the podium). E.g. if pole won 104 of 192 earlier races, the
    pole-sitter gets P(win) = 0.54.

    Both arguments are post-quali feature tables (see features/post_quali.py):
        history: rows from races BEFORE this one (the baseline learns from them)
        race:    the rows of the race to forecast
    They need `grid` (pit-lane starts already moved to the back) and
    `actual_position` (retirements included: a DNF is "not a podium").

    A slot never seen in history (e.g. 23rd in a big field) gets 0. Returns the
    race rows with p_win and p_podium columns added.
    """
    # Group past rows by grid slot; the mean of a True/False column is the
    # share of True, i.e. the historical rate for that slot.
    by_slot = history.assign(won=history["actual_position"] == 1,
                             podium=history["actual_position"] <= 3) \
                     .groupby("grid")[["won", "podium"]].mean()
    rates = race["grid"].map(by_slot["won"]).fillna(0.0), race["grid"].map(by_slot["podium"]).fillna(0.0)
    return race.assign(p_win=rates[0].values, p_podium=rates[1].values)


def championship_order(race: pd.DataFrame) -> list[str]:
    """Pre-weekend baseline: predict the race finishes in championship order.

    `race` is one race's rows from the feature table (needs the championship
    columns, see features/championship.py). Sort key, best first:
        1. driver's points so far this season
        2. their team's points so far this season (separates drivers on 0)
        3. driver's points last season (this is what decides round 1, when
           everyone is on 0; rookies have none and go last)
        4. driver_id, so the order is always the same
    """
    ranked = race.assign(prev=race["prev_season_points"].fillna(-1)).sort_values(
        ["driver_season_points", "team_season_points", "prev", "driver_id"],
        ascending=[False, False, False, True])
    return ranked["driver_id"].tolist()
