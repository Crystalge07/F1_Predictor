"""Practice pace: how fast each car looked on track before qualifying.

The grid already tells the model who was fastest over ONE lap in qualifying.
What it can't see is RACE pace: some cars are quick on low fuel and new tyres
but chew through their tyres over a stint, others are the reverse. Teams test
exactly that in practice with "long runs" (many laps in a row on one set of
tyres), so those laps are the best free clue to Sunday pace we have.

Two features per driver per race weekend (lap data exists from 2023 only;
earlier seasons get NaN, which XGBoost handles):

    long_run_gap_pct   typical lap on long runs, % slower than the best car's
                       long-run pace. Averaged over the weekend's sessions.
    short_run_gap_pct  best single practice lap, % slower than the fastest lap
                       of the weekend's practice sessions.

Which sessions count: every practice session (FP1/FP2/FP3) and sprint race (S)
that STARTS BEFORE QUALIFYING. Read from the calendar, never hard-coded,
because the order changes: in 2023 the sprint came after qualifying, from 2024
it comes before. (Sprint races are only used for long runs: a sprint race IS a
long run.) Sprint qualifying (SQ) is skipped: it's single-lap stuff that
qualifying already covers better.

What makes a long-run lap "clean":
    * not the first lap of a stint (out-lap, or the standing start of a sprint)
    * not the last lap of a stint (the in-lap, slowed by entering the pits)
    * within 105% of that stint's median lap: drops laps stuck in traffic,
      cool-down laps, and safety-car laps
    * the stint has at least MIN_LONG_RUN_LAPS clean laps left (otherwise it's a
      short qualifying-style run, not a long run)

Simplification (written down, not hidden): no fuel correction. Teams run
different, secret fuel loads, so a car running heavy looks a bit slow. The
PRD mentions fuel-correcting; it can be added later if the feature proves useful.
"""

import sqlite3

import pandas as pd

MIN_LONG_RUN_LAPS = 5
CLEAN_LAP_LIMIT = 1.05  # laps slower than 105% of the stint median are dropped

PRACTICE_PACE_COLUMNS = ["long_run_gap_pct", "short_run_gap_pct"]


def load_pre_quali_laps(conn: sqlite3.Connection) -> pd.DataFrame:
    """Every timed lap from sessions that ran before that weekend's qualifying.

    One row per lap, tagged with the race session it helps predict, the driver
    (via session_entries), and the tyre stint the lap belongs to.
    """
    query = """
        SELECT
            race.session_id   AS race_session_id,
            s.session_id,
            s.session_type,
            se.driver_id,
            l.lap_number,
            l.lap_time_ms,
            l.is_pit_out_lap,
            st.stint_number,
            st.lap_start      AS stint_first_lap,
            st.lap_end        AS stint_last_lap
        FROM laps l
        JOIN sessions s          ON s.session_id = l.session_id
        JOIN session_entries se  ON se.session_id = l.session_id AND se.car_number = l.car_number
        -- The same weekend's qualifying and race.
        JOIN sessions q          ON q.event_id = s.event_id AND q.session_type = 'Q'
        JOIN sessions race       ON race.event_id = s.event_id AND race.session_type = 'R'
        -- Which stint the lap is in: the stint whose lap range contains it.
        -- LEFT JOIN: keep laps with no stint data (they just can't be long-run laps).
        LEFT JOIN stints st      ON st.session_id = l.session_id AND st.car_number = l.car_number
                                AND l.lap_number BETWEEN st.lap_start AND st.lap_end
        WHERE s.session_type IN ('FP1', 'FP2', 'FP3', 'S')
          AND s.start_time_utc < q.start_time_utc   -- only what's known before qualifying
          AND l.lap_time_ms IS NOT NULL
          -- Practice-only drivers we couldn't match have no driver_id; they
          -- aren't in the race, so they can't get a feature row anyway.
          AND se.driver_id IS NOT NULL
    """
    return pd.read_sql_query(query, conn)


def _long_run_pace(laps: pd.DataFrame) -> pd.DataFrame:
    """long_run_gap_pct per (race_session_id, driver_id)."""
    # Step 1: drop out-laps and in-laps (first and last lap of each stint).
    in_a_stint = laps["stint_number"].notna()
    not_out_lap = (laps["is_pit_out_lap"] != 1) & (laps["lap_number"] != laps["stint_first_lap"])
    not_in_lap = laps["lap_number"] != laps["stint_last_lap"]
    candidates = laps[in_a_stint & not_out_lap & not_in_lap].copy()

    stint_keys = ["session_id", "driver_id", "stint_number"]

    # Step 2: drop slow laps (traffic, cool-down, safety car): anything over
    # 105% of the stint's median. transform("median") puts the stint median
    # on every lap row so we can compare row by row.
    stint_median = candidates.groupby(stint_keys)["lap_time_ms"].transform("median")
    clean = candidates[candidates["lap_time_ms"] <= stint_median * CLEAN_LAP_LIMIT].copy()

    # Step 3: keep only real long runs (enough clean laps left in the stint).
    clean_laps_in_stint = clean.groupby(stint_keys)["lap_time_ms"].transform("count")
    long_runs = clean[clean_laps_in_stint >= MIN_LONG_RUN_LAPS]

    # Step 4: each driver's pace in each session = median of their long-run
    # laps. Median rather than mean, so one odd lap can't drag it around.
    per_session = (long_runs.groupby(["race_session_id", "session_id", "driver_id"], as_index=False)
                   ["lap_time_ms"].median())

    # Step 5: compare drivers WITHIN a session (same track, same conditions):
    # % slower than the best long-run pace in that session.
    best = per_session.groupby("session_id")["lap_time_ms"].transform("min")
    per_session["gap"] = (per_session["lap_time_ms"] / best - 1) * 100

    # Step 6: one number per weekend: the average over the sessions where the
    # driver did a long run.
    return (per_session.groupby(["race_session_id", "driver_id"], as_index=False)["gap"].mean()
            .rename(columns={"gap": "long_run_gap_pct"}))


def _short_run_pace(laps: pd.DataFrame) -> pd.DataFrame:
    """short_run_gap_pct per (race_session_id, driver_id), from practice sessions only."""
    practice = laps[laps["session_type"].isin(["FP1", "FP2", "FP3"])]
    # Each driver's single best practice lap of the weekend...
    best = practice.groupby(["race_session_id", "driver_id"], as_index=False)["lap_time_ms"].min()
    # ...compared with the fastest practice lap anyone set that weekend.
    fastest = best.groupby("race_session_id")["lap_time_ms"].transform("min")
    best["short_run_gap_pct"] = (best["lap_time_ms"] / fastest - 1) * 100
    return best[["race_session_id", "driver_id", "short_run_gap_pct"]]


def practice_pace(conn: sqlite3.Connection) -> pd.DataFrame:
    """Both features, one row per (race session_id, driver_id) that has any practice laps."""
    laps = load_pre_quali_laps(conn)
    # outer merge: a driver might have a best lap but no long run, or vice versa.
    pace = _short_run_pace(laps).merge(_long_run_pace(laps),
                                       on=["race_session_id", "driver_id"], how="outer")
    return pace.rename(columns={"race_session_id": "session_id"})
