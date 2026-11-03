"""Championship features: the points table as it stood before each race weekend.

Used by the pre-weekend stage (and its "championship order" baseline). All
numbers count points from EARLIER weekends only: races (R) and sprints (S).
The current weekend's sprint is excluded too, because the pre-weekend
prediction is made before anything has run.

Per driver per race:
    driver_season_points      driver's points so far this season
    team_season_points        their team's points so far this season (both cars)
    driver_points_per_race    driver_season_points / weekends so far (NaN at round 1)
    team_points_per_race      team_season_points / weekends so far (NaN at round 1)
    prev_season_points        driver's total points last season (NaN for rookies).
                              Gives round 1 something to go on.
"""

import sqlite3

import pandas as pd

CHAMPIONSHIP_COLUMNS = [
    "driver_season_points",
    "team_season_points",
    "driver_points_per_race",
    "team_points_per_race",
    "prev_season_points",
]


def _points_per_weekend(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per (weekend, driver): points from that weekend's race + sprint."""
    query = """
        SELECT e.event_id, e.season, e.race_date, r.driver_id, r.constructor_id,
               SUM(r.points) AS points
        FROM results r
        JOIN sessions s ON s.session_id = r.session_id
        JOIN events e   ON e.event_id = s.event_id
        WHERE s.session_type IN ('R', 'S')
        -- A driver could (in theory) switch teams between sprint and race; we
        -- keep the pair, so team points always go to the team that scored them.
        GROUP BY e.event_id, e.season, e.race_date, r.driver_id, r.constructor_id
    """
    return pd.read_sql_query(query, conn)


def championship_features(conn: sqlite3.Connection, races: pd.DataFrame) -> pd.DataFrame:
    """Add CHAMPIONSHIP_COLUMNS to `races` (rows with session_id, season, race_date,
    driver_id, constructor_id), using only points from earlier weekends."""
    weekend = _points_per_weekend(conn)

    out = races.copy()
    # For each row, we need "sum of this driver's points this season, from
    # weekends before this race_date". merge + filter keeps it readable:
    #   1. pair every race row with every weekend of the same season...
    #   2. ...keep only weekends strictly before the race...
    #   3. ...and add up.
    # (A few thousand rows x ~24 weekends each: small enough to do this way.)
    earlier = out[["session_id", "season", "race_date"]].drop_duplicates().merge(
        weekend, on="season", suffixes=("", "_scored"))
    earlier = earlier[earlier["race_date_scored"] < earlier["race_date"]]

    driver_pts = (earlier.groupby(["session_id", "driver_id"], as_index=False)["points"].sum()
                         .rename(columns={"points": "driver_season_points"}))
    team_pts = (earlier.groupby(["session_id", "constructor_id"], as_index=False)["points"].sum()
                       .rename(columns={"points": "team_season_points"}))
    # How many weekends of this season ran before each race.
    weekends_before = (earlier.groupby("session_id")["event_id"].nunique()
                              .rename("weekends_before").reset_index())

    out = (out.merge(driver_pts, on=["session_id", "driver_id"], how="left")
              .merge(team_pts, on=["session_id", "constructor_id"], how="left")
              .merge(weekends_before, on="session_id", how="left"))
    # No row = no points yet (or round 1): that's 0 points, not "unknown".
    out[["driver_season_points", "team_season_points", "weekends_before"]] = \
        out[["driver_season_points", "team_season_points", "weekends_before"]].fillna(0)

    # Per-weekend rates. At round 1 there are 0 weekends: 0 / 0 -> NaN, which
    # is right: there is no rate yet. where() makes that explicit.
    has_history = out["weekends_before"] > 0
    out["driver_points_per_race"] = (out["driver_season_points"] / out["weekends_before"]).where(has_history)
    out["team_points_per_race"] = (out["team_season_points"] / out["weekends_before"]).where(has_history)

    # Last season's total, joined on (driver, season - 1).
    season_totals = (weekend.groupby(["season", "driver_id"], as_index=False)["points"].sum()
                            .rename(columns={"points": "prev_season_points"}))
    season_totals["season"] += 1   # last season's total, labelled with the season it's used in
    out = out.merge(season_totals, on=["season", "driver_id"], how="left")

    return out.drop(columns="weekends_before")
