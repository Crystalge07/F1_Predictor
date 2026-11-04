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

Timing glitches (both features): a lap more than 7% FASTER than both the
best lap of every other driver in the session and the driver's own next-best
lap is not a real lap, and is dropped before anything else. (Real case: 2026 Miami FP1, a 60.4 s lap when
the next best was 89.3 s, which used to put 21 drivers ~52% off the pace.)

What makes a long-run lap "clean":
    * not the first lap of a stint (out-lap, or the standing start of a sprint)
    * not the last lap of a stint (the in-lap, slowed by entering the pits)
    * within 107% of that stint's FASTEST lap: drops laps stuck in traffic,
      cool-down laps, and safety-car laps. (It used to be 105% of the stint's
      median, but when most laps of a stint are cool-down laps the median IS
      a cool-down lap, and the filter kept them: 2026 Singapore FP1, Perez,
      a "long run" 52.7% off the pace. 107% rather than 105%: measured from
      the fastest lap, 105% also threw away real long runs whose tyres faded,
      9 more of them than 107% did, without catching any more junk. Chosen
      on data quality only, not on prediction scores.)
    * the stint has at least MIN_LONG_RUN_LAPS clean laps left (otherwise it's a
      short qualifying-style run, not a long run)

Short runs: a driver whose best lap of the weekend is outside 107% of the
weekend's typical best lap (F1's own 107% rule) never did a representative
flying lap, so their short-run gap is unknown (NaN) rather than huge. (Real
case: 2023 Belgium FP1, Verstappen's only timed lap spanned a red flag:
1,631 s, a 1,224% gap.) Their long runs, if any, still count.

Simplification (written down, not hidden): no fuel correction. Teams run
different, secret fuel loads, so a car running heavy looks a bit slow. The
PRD mentions fuel-correcting; it can be added later if the feature proves useful.
"""

import sqlite3

import pandas as pd

MIN_LONG_RUN_LAPS = 5
CLEAN_LAP_LIMIT = 1.07  # laps slower than 107% of the stint's fastest lap are dropped
GLITCH_LIMIT = 0.93     # laps this much faster than every other driver's best AND their own next-best are glitches
REPRESENTATIVE_LIMIT = 1.07   # the 107% rule for a driver's best practice lap

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
            s.event_id,
            l.car_number,
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
    """
    laps = pd.read_sql_query(query, conn)
    # Cars we couldn't match to a driver (no driver_id) are of two kinds:
    #   * debutants in the race (still driving in the last session before
    #     qualifying): their laps count, under the temporary id the
    #     prediction uses for them (features/provisional.py). Car numbers
    #     are unique within a weekend, so all their sessions' laps are found;
    #   * FP1-only drivers: not in the race, so they can't have a feature
    #     row; their laps are dropped (as they always were).
    # In history every race driver is matched, so this only changes live
    # weekends with a debutant (2023-2026: no change).
    from f1predictor.features.provisional import provisional_cars
    debutants = provisional_cars(conn)[["event_id", "car_number", "driver_id"]].rename(
        columns={"driver_id": "provisional_id"})
    if not debutants.empty:     # (merging on an empty table trips over its column types)
        laps = laps.merge(debutants.astype({"event_id": "int64", "car_number": "int64"}),
                          on=["event_id", "car_number"], how="left")
        laps["driver_id"] = laps["driver_id"].fillna(laps["provisional_id"])
        laps = laps.drop(columns=["provisional_id"])
    return laps[laps["driver_id"].notna()].drop(columns=["event_id", "car_number"])


def _typical_best(laps: pd.DataFrame, keys: list[str]) -> pd.Series:
    """For every lap row: the median, over drivers, of each driver's best lap
    within `keys` (e.g. one session). A robust "how fast is a good lap here",
    which one odd lap or one odd driver can't move much."""
    best = laps.groupby(keys + ["driver_id"], as_index=False)["lap_time_ms"].min()
    typical = best.groupby(keys)["lap_time_ms"].median().rename("typical_best_ms")
    return laps[keys].merge(typical, left_on=keys, right_index=True, how="left")["typical_best_ms"].values


def _drop_glitches(laps: pd.DataFrame) -> pd.DataFrame:
    """Remove timing glitches: laps that are BOTH
        * faster than GLITCH_LIMIT x the best lap any OTHER driver set in the
          session, and
        * faster than GLITCH_LIMIT x the same driver's own next-best lap.

    A real lap is never 7% faster than the whole rest of the field AND than
    everything else its own driver did. The 2026 Miami glitch (60.4 s, when
    the next best car was 89.3 s and that driver's other laps ~90 s) fails both.
    Each test alone would also catch real laps:
        * compared with a "typical" lap instead of the best other lap, real
          dry laps in a rain-hit session look too fast (2024 Japan FP2:
          Hamilton's 95.2 s, slower than Piastri's 94.7 s)
        * a driver's one push lap among slow laps is far faster than their
          own next-best
    A driver with a single timed lap, or alone in a session, can't be tested,
    so nothing is dropped there.
    """
    keys = ["session_id", "driver_id"]
    # Each driver's best lap in each session, and the fastest best of the
    # OTHER drivers: the session's fastest best, or the second-fastest for
    # the driver who set the fastest.
    best = laps.groupby(keys, as_index=False)["lap_time_ms"].min().rename(columns={"lap_time_ms": "own_best"})
    best = best.sort_values(["session_id", "own_best"])
    rank = best.groupby("session_id").cumcount()                  # 0 = fastest driver
    fastest = best.groupby("session_id")["own_best"].transform("first")
    second = best.groupby("session_id")["own_best"].transform(lambda b: b.iloc[1] if len(b) > 1 else float("nan"))
    best["others_best"] = second.where(rank == 0, fastest)

    # Each driver's second-fastest lap in the session. nsmallest(2).max() =
    # the 2nd smallest; with only one lap it's that lap itself, which never
    # counts as "much faster than itself".
    laps = laps.merge(best[keys + ["others_best"]], on=keys, how="left")
    own_next_best = laps.groupby(keys)["lap_time_ms"].transform(lambda t: t.nsmallest(2).max())
    glitch = ((laps["lap_time_ms"] < GLITCH_LIMIT * laps["others_best"])
              & (laps["lap_time_ms"] < GLITCH_LIMIT * own_next_best))
    return laps[~glitch].drop(columns="others_best")


def _long_run_pace(laps: pd.DataFrame) -> pd.DataFrame:
    """long_run_gap_pct per (race_session_id, driver_id)."""
    laps = _drop_glitches(laps)
    # Step 1: drop out-laps and in-laps (first and last lap of each stint).
    in_a_stint = laps["stint_number"].notna()
    not_out_lap = (laps["is_pit_out_lap"] != 1) & (laps["lap_number"] != laps["stint_first_lap"])
    not_in_lap = laps["lap_number"] != laps["stint_last_lap"]
    candidates = laps[in_a_stint & not_out_lap & not_in_lap].copy()

    stint_keys = ["session_id", "driver_id", "stint_number"]

    # Step 2: drop slow laps (traffic, cool-down, safety car): anything over
    # 107% of the stint's FASTEST lap (not its median: see the module
    # docstring). transform("min") puts the stint's fastest lap on every lap
    # row so we can compare row by row. Glitches are already gone, so the
    # fastest lap is a real one.
    stint_best = candidates.groupby(stint_keys)["lap_time_ms"].transform("min")
    clean = candidates[candidates["lap_time_ms"] <= stint_best * CLEAN_LAP_LIMIT].copy()

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
    laps = _drop_glitches(laps)
    practice = laps[laps["session_type"].isin(["FP1", "FP2", "FP3"])]
    # Each driver's single best practice lap of the weekend...
    best = practice.groupby(["race_session_id", "driver_id"], as_index=False)["lap_time_ms"].min()
    # ...kept only if it's a representative flying lap: within 107% of the
    # weekend's typical best (median over drivers). Otherwise: unknown (NaN).
    typical = best.groupby("race_session_id")["lap_time_ms"].transform("median")
    best["lap_time_ms"] = best["lap_time_ms"].where(best["lap_time_ms"] <= REPRESENTATIVE_LIMIT * typical)
    # ...compared with the fastest practice lap anyone set that weekend.
    # (min() skips the NaNs, so an unknown driver can't be the reference.)
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
