"""Post-qualifying features: one row per driver per race.

"Post-qualifying" means we pretend it's Saturday evening: qualifying is done,
the starting grid is known, the race hasn't happened. Every feature must use
ONLY information that existed at that moment. Using anything later (even by
accident) is called *leakage*: the model looks brilliant in testing and then
fails for real, because it was secretly peeking at the answer.

How leakage is prevented here:
    * Same-weekend data: only the starting grid and qualifying, which really
      are known before the race.
    * History features (form, team pace, track history) are computed from
      PREVIOUS races only. The trick is pandas' `.shift(1)`: compute a running
      average that includes this race, then shift it down one row so each race
      sees the average as it stood *before* it. There's a test for this.

The target (what the model learns to predict):
    finish_position = the official finishing position, but ONLY for classified
    finishers. Retirements get NaN (missing) and are left out of training.
    Decision (2026-11-02): this model answers "where does this driver finish,
    if they finish?". The chance of a DNF is handled later by the Monte Carlo
    simulation in Phase 3, not by this model.

    We still keep `actual_position` (retirements included) on every row, so
    models are scored on the full grid, exactly like the grid-order baseline.

Features (all numeric; NaN = "unknown", which XGBoost handles natively, so we
don't invent fake values for rookies or round 1):

    grid                     official starting slot (pit-lane starts moved to the back)
    pit_lane_start           1 if they start from the pit lane, else 0
    quali_gap_pct            % slower than pole in qualifying (0 = on pole)
    driver_form_finish       avg finish over the driver's last 5 races (finishes only)
    driver_form_grid         avg starting slot over the driver's last 5 races
    team_season_finish       avg finish of the team's cars so far this season
    team_season_quali        avg qualifying position of the team's cars so far this season
    driver_track_finish      driver's avg finish at this circuit in past seasons
    team_track_finish        team's avg finish at this circuit in past seasons

Also on every row, but NOT in FEATURE_COLUMNS (see the note there):
    long_run_gap_pct         practice race pace, % off the best car (2023+; see practice_pace.py)
    short_run_gap_pct        best practice lap, % off the fastest (2023+)

Known limitation (deliberately left for later): team rebrands get new ids in
Jolpica ('alphatauri' -> 'rb', 'alfa' -> 'sauber', 'renault' -> 'alpine'...).
For now team history restarts after a rebrand. Linking rebrands to one team is
a possible later improvement.
"""

import sqlite3

import pandas as pd

from f1predictor.features.practice_pace import PRACTICE_PACE_COLUMNS, practice_pace

# The model's input columns, in a fixed order. The training code imports this
# list, so adding a feature here is the only change needed to feed it in.
FEATURE_COLUMNS = [
    "grid",
    "pit_lane_start",
    "quali_gap_pct",
    "driver_form_finish",
    "driver_form_grid",
    "team_season_finish",
    "team_season_quali",
    "driver_track_finish",
    "team_track_finish",
]
# Practice pace (PRACTICE_PACE_COLUMNS) is computed and kept in the table, but
# deliberately left out of the post-quali model: on validation races it made
# predictions slightly worse (the grid already captures most of what practice
# shows). It's kept for the pre-weekend/practice stages in Phase 3, where there
# is no grid yet. Decision recorded in docs/PRD.md, 2026-11-02.

# How many past races count as "recent form". The PRD says 3-5; 5 smooths
# out one-off bad weekends a little more. A tunable knob for later.
FORM_WINDOW = 5


def load_race_rows(conn: sqlite3.Connection) -> pd.DataFrame:
    """Every driver in every race, joined with that weekend's qualifying.

    One row per (race, driver). Plain SQL does the joining, then pandas takes
    over for the running averages (which are painful to write in SQL).
    """
    query = """
        SELECT
            r.session_id,
            e.season,
            e.round,
            e.race_date,
            e.circuit_id,
            r.driver_id,
            r.constructor_id,
            r.grid,
            r.position       AS actual_position,
            r.position_text,
            q.position       AS quali_position,
            -- The best lap (fastest of Q1/Q2/Q3) is worked out in pandas below.
            -- SQLite's MIN(a, b, c) returns NULL if ANY of them is NULL, which
            -- would wipe out every driver knocked out in Q1 or Q2.
            q.q1_ms, q.q2_ms, q.q3_ms
        FROM results r
        JOIN sessions race ON race.session_id = r.session_id
        JOIN events e      ON e.event_id = race.event_id
        -- LEFT JOINs: keep the driver even if qualifying data is missing.
        LEFT JOIN sessions qs ON qs.event_id = race.event_id AND qs.session_type = 'Q'
        LEFT JOIN qualifying q ON q.session_id = qs.session_id AND q.driver_id = r.driver_id
        WHERE race.session_type = 'R'
        ORDER BY e.race_date, r.session_id, r.driver_id
    """
    # pd.read_sql_query runs the SQL and returns a DataFrame with the same columns.
    return pd.read_sql_query(query, conn)


def _add_same_weekend_features(df: pd.DataFrame) -> pd.DataFrame:
    """grid, pit_lane_start, quali_gap_pct: things known on Saturday night."""
    # --- Grid -----------------------------------------------------------------
    # grid = 0 means "starts from the pit lane", which is BEHIND everyone. Left
    # as 0, the model would read it as better than pole. So: flag it, then move
    # it to the back of the field (field size = number of drivers in the race).
    df["pit_lane_start"] = (df["grid"] == 0).astype(int)
    field_size = df.groupby("session_id")["driver_id"].transform("count")
    df["grid"] = df["grid"].where(df["grid"] != 0, field_size)

    # --- Qualifying gap to pole -----------------------------------------------
    # Best lap = fastest of Q1/Q2/Q3. min(axis=1) works across the columns of
    # each row and skips NaNs, so a Q1 knockout just uses their Q1 time.
    df["best_quali_ms"] = df[["q1_ms", "q2_ms", "q3_ms"]].min(axis=1)
    # The fastest lap anyone set that weekend. transform("min") gives every
    # row in the race the same value, so we can divide row by row.
    pole_ms = df.groupby("session_id")["best_quali_ms"].transform("min")
    # As a percentage, so tracks of different lengths are comparable:
    # 1.0 means 1% slower than the fastest lap. No time set -> NaN.
    # (Small bias: Q1 knockouts never ran on the faster, rubbered-in track of Q3.
    # Acceptable for a first version; their gap is large anyway.)
    df["quali_gap_pct"] = (df["best_quali_ms"] / pole_ms - 1) * 100
    return df


def _add_driver_form(df: pd.DataFrame) -> pd.DataFrame:
    """driver_form_finish / driver_form_grid: last FORM_WINDOW races, before this one."""
    # Group each driver's races in date order (df is already sorted by date).
    by_driver = df.groupby("driver_id")

    # rolling(5).mean() = the average of this row and the 4 before it.
    # min_periods=1: average whatever exists (a 2nd-race rookie gets 1 race).
    # NaN finishes (DNFs) are skipped by mean(), so this is "avg finish when
    # they finished". shift(1) then moves each value down one row, so a race
    # only sees races BEFORE it. A driver's first race becomes NaN (no history).
    df["driver_form_finish"] = by_driver["finish_position"].transform(
        lambda s: s.rolling(FORM_WINDOW, min_periods=1).mean().shift(1))
    df["driver_form_grid"] = by_driver["grid"].transform(
        lambda s: s.rolling(FORM_WINDOW, min_periods=1).mean().shift(1))
    return df


def _team_race_averages(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (race, team): the average of that team's cars in that race.

    Team features count each RACE equally, not each car, so first squash the
    two cars into one number per race.
    """
    return (
        df.groupby(["session_id", "season", "race_date", "circuit_id", "constructor_id"],
                   as_index=False)
          .agg(team_finish=("finish_position", "mean"),
               team_quali=("quali_position", "mean"))
          .sort_values(["race_date", "session_id"])
    )


def _add_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """team_season_* and team_track_finish, from previous races only."""
    team = _team_race_averages(df)

    # Team pace this season: running average over this season's races so far.
    # expanding().mean() = average of ALL rows up to and including this one;
    # shift(1) excludes the current race. Grouping by season as well means
    # round 1 starts fresh (NaN): last year's car tells us little about this
    # year's, especially across a regulation reset like 2026.
    by_team_season = team.groupby(["constructor_id", "season"])
    team["team_season_finish"] = by_team_season["team_finish"].transform(
        lambda s: s.expanding().mean().shift(1))
    team["team_season_quali"] = by_team_season["team_quali"].transform(
        lambda s: s.expanding().mean().shift(1))

    # Team history at this circuit: every previous visit, any season.
    team["team_track_finish"] = team.groupby(["constructor_id", "circuit_id"])["team_finish"] \
        .transform(lambda s: s.expanding().mean().shift(1))

    # Attach the team numbers back onto each driver's row (both cars get them).
    return df.merge(
        team[["session_id", "constructor_id",
              "team_season_finish", "team_season_quali", "team_track_finish"]],
        on=["session_id", "constructor_id"],
        how="left",
    )


def _add_driver_track_history(df: pd.DataFrame) -> pd.DataFrame:
    """driver_track_finish: the driver's avg finish at this circuit, previous visits only."""
    df["driver_track_finish"] = df.groupby(["driver_id", "circuit_id"])["finish_position"] \
        .transform(lambda s: s.expanding().mean().shift(1))
    return df


def build_post_quali_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """The full training/prediction table for the post-qualifying stage.

    Returns one row per driver per race with:
        id columns:  session_id, season, round, race_date, circuit_id,
                     driver_id, constructor_id
        features:    FEATURE_COLUMNS
        targets:     finish_position (NaN for retirements: not trained on)
                     actual_position (every driver: used for scoring)
    """
    df = load_race_rows(conn)

    # The target. position_text is '1', '2', ... for classified finishers and a
    # letter ('R', 'D', 'W', ...) otherwise. str.isdigit() tells them apart.
    is_classified = df["position_text"].str.isdigit()
    # where(cond) keeps values where cond is True and puts NaN elsewhere.
    df["finish_position"] = df["actual_position"].where(is_classified)

    df = _add_same_weekend_features(df)
    df = _add_driver_form(df)
    df = _add_team_features(df)
    df = _add_driver_track_history(df)
    # Practice pace: left merge, so drivers with no practice laps (and every
    # race before 2023) simply get NaN.
    df = df.merge(practice_pace(conn), on=["session_id", "driver_id"], how="left")

    id_columns = ["session_id", "season", "round", "race_date", "circuit_id",
                  "driver_id", "constructor_id"]
    return (df[id_columns + FEATURE_COLUMNS + PRACTICE_PACE_COLUMNS
               + ["finish_position", "actual_position"]]
            .sort_values(["race_date", "session_id", "grid"])
            .reset_index(drop=True))
