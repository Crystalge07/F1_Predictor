"""Prediction rows "as of" a stage: only what was known at that moment.

The feature table (race_features.build_race_features) is built from race
RESULTS, which is fine for training: those races are over. But a prediction
is made BEFORE its race, so the race's own rows can't come from its result.
Two things in particular would leak (decision 2026-11-03, strict backtest):

    * who is racing: the result lists exactly who started, including
      last-minute replacements nobody knew about on Monday;
    * the grid: results.grid is the official grid after penalties, which
      Jolpica only publishes together with the result. In 2023-2026, 22-28%
      of drivers started somewhere other than their qualifying position.

So for the race being predicted, this module builds the rows from sources
that existed at the stage's cutoff, and hands them to build_race_features as
`upcoming` rows. Every HISTORY feature is then computed by exactly the same
code as in training (those already only look at earlier races).

    stage        cutoff               who's racing                    grid
    -----------  -------------------  ------------------------------  -----------------------
    pre_weekend  first session start  starters of the previous race   (none)
    practice     qualifying start     entries of the last session     (none)
                                      before qualifying
    post_quali   race start           OpenF1 starting grid            OpenF1 starting grid
                                      (fallback: Jolpica qualifying)  (fallback: quali order)

Measured on 2023-2026 (2026-11-03):
    * previous-race lineup: wrong at every round 1 (3-5 drivers missing:
      new teams, rookies) and ~1 driver missing in ~25% of other races
      (mid-season swaps). The cost of being strict before the weekend.
    * last-session-before-qualifying lineup: equal to the race starters in
      88 of 89 races.
    * OpenF1 starting grid: equal to the official grid for ~98% of drivers;
      the misses are late penalty reshuffles at the back, usually 1 place.

Known simplifications:
    * Pit-lane starts aren't known from the starting grid (OpenF1 lists most
      in a normal slot at the back), so pit_lane_start is usually 0 for
      upcoming rows. That's why it isn't a model input (race_features.py).
      Starters OpenF1 leaves out entirely are known: add_missing_starters.
    * session_entries matches car numbers to drivers using the same weekend's
      result when it exists (ingest/openf1.match_entries). That only tells us
      WHO drove car #16, not how they did, and live the acronym rule covers it.
"""

import sqlite3

import numpy as np
import pandas as pd

from f1predictor.features.practice_pace import PRACTICE_PACE_COLUMNS
from f1predictor.features.quali_times import qualifying_times
from f1predictor.features.race_features import build_race_features

STAGES = ("pre_weekend", "practice", "post_quali")

# Same-weekend columns, and the first stage at which each is known.
# Earlier stages get NaN in them, even if the database already has the data
# (in a backtest it always does), so no model can accidentally peek.
QUALI_COLUMNS = ["grid", "pit_lane_start", "quali_gap_pct"]
UNKNOWN_BEFORE = {
    "pre_weekend": QUALI_COLUMNS + PRACTICE_PACE_COLUMNS,
    "practice": QUALI_COLUMNS,
    "post_quali": [],
}

# Sessions that run before qualifying and list the race drivers.
PRE_QUALI_SESSION_TYPES = ("FP1", "FP2", "FP3", "SQ", "S")


# -----------------------------------------------------------------------------
# The event and its cutoffs
# -----------------------------------------------------------------------------

def event_info(conn: sqlite3.Connection, event_id: int) -> dict:
    """The race session and calendar facts for one event, plus its session start times.

    Returns e.g. {"race_session_id": 412, "season": 2026, "round": 7,
                  "race_date": "2026-05-24", "circuit_id": "monaco",
                  "starts": {"FP1": "2026-05-22T11:30:00Z", ..., "R": "..."}}
    """
    event = conn.execute(
        "SELECT event_id, season, round, race_date, circuit_id FROM events WHERE event_id = ?",
        (event_id,)).fetchone()
    if event is None:
        raise ValueError(f"No event {event_id}")
    sessions = conn.execute(
        "SELECT session_id, session_type, start_time_utc FROM sessions WHERE event_id = ?",
        (event_id,)).fetchall()
    ids = {s["session_type"]: s["session_id"] for s in sessions}
    if "R" not in ids:
        raise ValueError(f"Event {event_id} has no race session")
    return {
        "event_id": event_id,
        "race_session_id": ids["R"],
        "quali_session_id": ids.get("Q"),
        "season": event["season"],
        "round": event["round"],
        "race_date": event["race_date"],
        "circuit_id": event["circuit_id"],
        "starts": {s["session_type"]: s["start_time_utc"] for s in sessions},
        "session_ids": ids,
    }


def stage_cutoff(info: dict, stage: str) -> str:
    """The information cutoff for a stage (UTC ISO text): data from sessions
    that STARTED before it may be used, nothing else.

    pre_weekend: before the first session of the weekend
    practice:    before qualifying (on 2024+ sprint weekends this includes
                 the sprint, which runs before qualifying). If the weekend
                 has NO qualifying (cancelled and taken off the calendar),
                 before the race: everything that ran is "practice" then
                 (architecture review, 2026-11-03; before, the stage
                 couldn't be predicted at all).
    post_quali:  before the race. Needs a qualifying session: without one
                 there's no grid to predict from (stage_lineup refuses).
    """
    starts = {t: s for t, s in info["starts"].items() if s is not None}
    if stage == "pre_weekend":
        if not starts:
            raise ValueError(f"Event {info['event_id']} has no session start times")
        return min(starts.values())   # ISO text sorts in time order
    key = {"practice": "Q", "post_quali": "R"}.get(stage)
    if key is None:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGES}")
    if stage == "practice" and "Q" not in info["session_ids"]:
        key = "R"   # no qualifying this weekend (see above)
    if key not in starts:
        raise ValueError(f"Event {info['event_id']} has no {key} start time")
    return starts[key]


# -----------------------------------------------------------------------------
# Who is racing (and from where), per stage
# -----------------------------------------------------------------------------

def previous_race_lineup(conn: sqlite3.Connection, race_date: str) -> pd.DataFrame:
    """driver_id, constructor_id of every starter in the last race before race_date."""
    return pd.read_sql_query(
        """
        SELECT r.driver_id, r.constructor_id
        FROM results r
        JOIN sessions s ON s.session_id = r.session_id AND s.session_type = 'R'
        JOIN events e   ON e.event_id = s.event_id
        WHERE e.race_date = (
            -- The most recent race date before this one that has a result.
            SELECT MAX(e2.race_date)
            FROM events e2
            JOIN sessions s2 ON s2.event_id = e2.event_id AND s2.session_type = 'R'
            WHERE e2.race_date < ?
              AND EXISTS (SELECT 1 FROM results x WHERE x.session_id = s2.session_id)
        )
        ORDER BY r.driver_id
        """,
        conn, params=(race_date,))


def last_pre_quali_lineup(conn: sqlite3.Connection, info: dict) -> pd.DataFrame:
    """driver_id, constructor_id from the last session before qualifying that has entries.

    Why the LAST one: FP1 often has reserve/rookie drivers in place of race
    drivers; by FP3 (or the sprint sessions) the race drivers are back.
    Empty if there's no such session with entries (e.g. before 2023, when
    there's no OpenF1 data).

    Debutants: a car that ran in that session but can't be matched to a
    driver yet (no Jolpica id before his first result) is in the race too.
    It's added under a temporary id (features/provisional.py).
    """
    from f1predictor.features.provisional import provisional_cars

    quali_start = info["starts"].get("Q")
    candidates = sorted(
        ((start, info["session_ids"][t]) for t, start in info["starts"].items()
         if t in PRE_QUALI_SESSION_TYPES and start is not None
         and (quali_start is None or start < quali_start)),
        reverse=True)   # latest first
    for _start, session_id in candidates:
        lineup = pd.read_sql_query(
            "SELECT driver_id, constructor_id FROM session_entries "
            # Unmatched cars (no driver_id) can't get features; they're
            # practice-only drivers in practice anyway.
            "WHERE session_id = ? AND driver_id IS NOT NULL ORDER BY driver_id",
            conn, params=(session_id,))
        if not lineup.empty:
            debutants = provisional_cars(conn, info["event_id"])
            debutants = debutants[debutants["session_id"] == session_id]   # the same session
            # Only with a known team: without one his DNF chance can't be
            # computed, and that would block the WHOLE prediction. Left out,
            # he's just missing, as before this rule existed.
            debutants = debutants[debutants["constructor_id"].notna()]
            if not debutants.empty:
                lineup = pd.concat([lineup, debutants[["driver_id", "constructor_id"]]],
                                   ignore_index=True)
            return lineup
    return pd.DataFrame(columns=["driver_id", "constructor_id"])


def add_missing_starters(conn: sqlite3.Connection, info: dict, grid: pd.DataFrame) -> pd.DataFrame:
    """`grid` plus every driver who qualified but whose car isn't on it, as a
    pit-lane starter (grid 0).

    Why: OpenF1's starting grid sometimes leaves pit-lane starters out
    altogether (2023 Australia: Perez and Bottas; 2023 Netherlands:
    Magnussen). Without this they'd vanish from the post-quali prediction.
    Grid 0 is how the official result records a pit-lane start, so
    build_race_features treats them exactly as in training: flagged, and
    moved to the back of the field.

    Who "qualified": Jolpica's qualifying, or the last pre-qualifying lineup
    if that isn't loaded yet (same sources as the qualifying fallback below).
    A driver is only added if his car number is NOT on the grid, so a car
    that is on the grid but couldn't be matched to a driver isn't added twice.
    """
    quali_id = info["quali_session_id"]
    qualified = pd.read_sql_query(
        "SELECT driver_id, constructor_id FROM qualifying WHERE session_id = ?",
        conn, params=(quali_id,))
    if qualified.empty:
        qualified = last_pre_quali_lineup(conn, info)
        qualified = qualified[~qualified["driver_id"].str.startswith("provisional_")]

    cars_on_grid = {row["car_number"] for row in conn.execute(
        "SELECT car_number FROM starting_grid WHERE session_id = ?", (quali_id,))}
    # Each driver's car number this weekend, from ANY of the weekend's entry
    # lists (car numbers are unique within a weekend): if the qualifying entry
    # wasn't matched to a driver, practice usually was.
    car_of = {row["driver_id"]: row["car_number"] for row in conn.execute(
        "SELECT se.driver_id, se.car_number FROM session_entries se "
        "JOIN sessions s ON s.session_id = se.session_id "
        "WHERE s.event_id = ? AND se.driver_id IS NOT NULL", (info["event_id"],))}

    missing = qualified[~qualified["driver_id"].isin(grid["driver_id"])]
    missing = missing[~missing["driver_id"].map(car_of).isin(cars_on_grid)]
    if missing.empty:
        return grid
    return pd.concat([grid, missing.assign(grid=0)], ignore_index=True)


def grid_lineup(conn: sqlite3.Connection, info: dict) -> tuple[pd.DataFrame, str]:
    """driver_id, constructor_id, grid for the post-quali stage, plus which source was used.

    1. 'starting_grid': OpenF1's official grid (after penalties), stored by the
       OpenF1 loader against the qualifying session. Qualified drivers it
       leaves out are added as pit-lane starters (add_missing_starters).
    2. 'qualifying': Jolpica's qualifying order, if there's no starting grid
       (not published yet, or OpenF1 never had it). Drivers in the
       pre-qualifying lineup who didn't set a qualifying time go to the back.
    """
    quali_id = info["quali_session_id"]
    grid = pd.read_sql_query(
        """
        SELECT se.driver_id, se.constructor_id, g.position AS grid
        FROM starting_grid g
        JOIN session_entries se ON se.session_id = g.session_id AND se.car_number = g.car_number
        WHERE g.session_id = ? AND se.driver_id IS NOT NULL
        ORDER BY g.position
        """,
        conn, params=(quali_id,))
    if not grid.empty:
        return add_missing_starters(conn, info, grid), "starting_grid"

    grid = pd.read_sql_query(
        "SELECT driver_id, constructor_id, position AS grid FROM qualifying "
        "WHERE session_id = ? ORDER BY position",
        conn, params=(quali_id,))
    # Anyone who was in practice but has no qualifying row (no time set, or
    # missing data) still starts the race: put them at the back, in a fixed order.
    missing = last_pre_quali_lineup(conn, info)
    missing = missing[~missing["driver_id"].isin(grid["driver_id"])]
    # Not a debutant's temporary id: by now he's in Jolpica's qualifying
    # under his real id, and adding both would put him on the grid twice.
    missing = missing[~missing["driver_id"].str.startswith("provisional_")]
    if not missing.empty:
        missing = missing.assign(grid=range(len(grid) + 1, len(grid) + 1 + len(missing)))
        grid = pd.concat([grid, missing], ignore_index=True)
    return grid, "qualifying"


def stage_lineup(conn: sqlite3.Connection, info: dict, stage: str) -> tuple[pd.DataFrame, str]:
    """(driver_id, constructor_id[, grid]) for the race at this stage, and the source name."""
    if stage == "pre_weekend":
        return previous_race_lineup(conn, info["race_date"]), "previous_race"
    if stage == "practice":
        lineup = last_pre_quali_lineup(conn, info)
        if lineup.empty:
            # No lap data (before 2023): the best we know is last race's lineup.
            return previous_race_lineup(conn, info["race_date"]), "previous_race"
        return lineup, "practice_entries"
    if stage == "post_quali":
        if info["quali_session_id"] is None:
            # No qualifying this weekend: no grid, and nothing a post-quali
            # model could add to the practice prediction. Refused, not guessed.
            raise ValueError(f"Event {info['event_id']} has no qualifying session: "
                             "no post-quali prediction (the practice prediction covers it)")
        return grid_lineup(conn, info)
    raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGES}")


# -----------------------------------------------------------------------------
# Putting it together
# -----------------------------------------------------------------------------

def upcoming_rows(conn: sqlite3.Connection, info: dict, stage: str) -> tuple[pd.DataFrame, str, str | None]:
    """The race's rows in load_race_rows() format, with no result, for one stage.

    Returns (rows, lineup source, qualifying times source). The times source
    is None before qualifying, else quali_times.SOURCE_* ('jolpica',
    'openf1' or 'none').
    """
    lineup, source = stage_lineup(conn, info, stage)
    if lineup.empty:
        raise ValueError(f"No lineup for event {info['event_id']} at stage {stage}")
    rows = lineup.drop_duplicates("driver_id").copy()
    if "grid" not in rows:
        rows["grid"] = np.nan

    # Qualifying times, only once qualifying has happened (post_quali).
    # Jolpica's, or OpenF1's best laps if Jolpica has published the order
    # but no times yet (features/quali_times.py).
    quali = pd.DataFrame(columns=["driver_id", "quali_position", "q1_ms", "q2_ms", "q3_ms", "openf1_best_ms"])
    times_source = None
    if stage == "post_quali":
        quali, times_source = qualifying_times(conn, info["quali_session_id"])
    rows = rows.merge(quali, on="driver_id", how="left")

    rows = rows.assign(
        session_id=info["race_session_id"],
        season=info["season"],
        round=info["round"],
        race_date=info["race_date"],
        circuit_id=info["circuit_id"],
        # No result: the race hasn't happened. build_race_features turns these
        # into "no target" (finish_position NaN).
        actual_position=np.nan,
        position_text=None,
    )
    # Numeric columns as floats, so NaN works everywhere.
    for column in ["grid", "quali_position", "q1_ms", "q2_ms", "q3_ms", "openf1_best_ms"]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce").astype(float)
    return rows, source, times_source


def prediction_rows(conn: sqlite3.Connection, event_id: int, stage: str) -> tuple[pd.DataFrame, dict]:
    """Feature rows for one race at one stage, using only what was known then.

    Returns (rows, meta):
        rows  one row per driver, same columns as build_race_features() (the
              targets are NaN). Columns not known yet at this stage are NaN.
        meta  {"as_of_utc": cutoff, "lineup_source": "starting_grid" / ...,
               "race_session_id": ..., "quali_times_source": ...}: what
              the prediction log records, plus where the qualifying times
              came from (features/quali_times.py; None before qualifying).
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGES}")
    info = event_info(conn, event_id)
    cutoff = stage_cutoff(info, stage)
    upcoming, source, times_source = upcoming_rows(conn, info, stage)

    table = build_race_features(conn, upcoming=upcoming)
    rows = table[table["session_id"] == info["race_session_id"]].copy()
    # Blank out what this stage can't know yet (see UNKNOWN_BEFORE).
    rows[UNKNOWN_BEFORE[stage]] = np.nan

    meta = {"as_of_utc": cutoff, "lineup_source": source,
            "race_session_id": info["race_session_id"],
            # Where the qualifying times came from (post_quali only, else None).
            "quali_times_source": times_source}
    return rows.reset_index(drop=True), meta
