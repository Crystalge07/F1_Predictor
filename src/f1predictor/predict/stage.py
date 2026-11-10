"""Predict one race at one stage, and write the prediction to the log.

This is the function the 2026 replay (Phase 4) and the live 2027 pipeline
(Phase 5) both call, so a backtest runs exactly the code that will run live:

    predict_stage(conn, event_id, "post_quali", mode="backtest")  -> run_id

What happens, in order:

    1. rows      features/as_of.prediction_rows: one row per driver, built
                 only from what was known at the stage's cutoff (the driver
                 list and grid never come from the result).
    2. train     the stage's position model (features/stages.py), trained on
                 every race BEFORE this one, with the frozen 2026 weight.
    3. predict   expected finishing position per driver (if they finish).
    4. DNF       each driver's chance of not finishing (models/dnf.py).
    5. simulate  Monte Carlo from the anchor/model blend (simulate/
                 probabilities.py, frozen settings): p_win, p_podium, ...
                 The blend's order is `predicted_rank`.
    6. explain   SHAP contribution of every feature (explain/contributions.py).
    7. log       one prediction_runs row, one predictions row per driver, one
                 prediction_features row per driver per feature, all in ONE
                 transaction: either the whole run is logged or none of it.

Steps 1-6 are make_prediction() (reads only); step 7 is write_run(). Keeping
them apart lets tests check a prediction without touching the log, and lets
the replay script print a run before deciding to store it.

Which order is `predicted_rank`? The SIMULATOR'S order (decision 2026-11-03,
docs/evaluation-protocol.md): drivers sorted by the blended score the Monte
Carlo starts from (anchor_weight x anchor slot + the rest x model order,
simulate/probabilities.blend_with_grid). It is the order the win / podium /
points chances are built from, so the table and its odds agree: the driver
ranked 1st is the favourite to win.

Before that date it was the position model's raw order (lowest expected
position first). That order could contradict the odds shown next to it: at
2026 R2 after qualifying, Russell was ranked 1st with a 31% win chance and
Antonelli 5th with 40%, because the odds lean 90% on the grid (Antonelli was
on pole) and the raw order didn't at all. Runs logged before the change
keep the old meaning (the log can't be edited). On the 58 runs logged up to
then, the blended order put the favourite first in all of them (vs 50) and
scored the same on places off (3.44 vs 3.47 after qualifying, in-sample).

The model's raw order isn't lost: `expected_position` (the model's own
number) is stored for every driver, so ranking it gives that order back
(see model_order below). SHAP explains `expected_position`.
"""

import hashlib
import json
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from f1predictor import config, environment, settings
from f1predictor.explain.contributions import explain_race
from f1predictor.features.as_of import prediction_rows
from f1predictor.features.provisional import add_placeholder_drivers
from f1predictor.features.race_features import build_race_features
from f1predictor.features.stages import STAGE_FEATURES
from f1predictor.models.baselines import add_baseline_ranks
from f1predictor.models.dnf import add_dnf_probability
from f1predictor.models.position import WEIGHT_2026, predict_positions, train_model
from f1predictor.simulate.monte_carlo import N_SIMS
from f1predictor.simulate.probabilities import STAGE_SETTINGS, race_probabilities

MODES = ("backtest", "live")


@dataclass
class StagePrediction:
    """Everything one run will write to the log (see write_run)."""
    event_id: int
    stage: str
    as_of_utc: str          # the information cutoff (features/as_of.stage_cutoff)
    lineup_source: str      # where the driver list / grid came from
    n_training_races: int
    # One row per driver: driver_id, constructor_id, predicted_rank,
    # anchor_position, expected_position, p_win, p_podium, p_points, p_dnf,
    # mean_position. Sorted by predicted_rank.
    drivers: pd.DataFrame
    # One row per driver per feature: driver_id, feature, value, contribution.
    features: pd.DataFrame
    # Where the qualifying times came from (post_quali only; None before
    # qualifying): 'jolpica', 'openf1' (Jolpica had the order but no times
    # yet) or 'none'. Not written to the log; the tick reports it, and
    # input_warnings() raises an alert when the gaps are missing.
    quali_times_source: str | None = None


# -----------------------------------------------------------------------------
# Which code and settings made a prediction
# -----------------------------------------------------------------------------

def settings_text() -> str:
    """The tuned settings as JSON text, keys sorted so the same settings always
    give the same text (and the same hash)."""
    return json.dumps(settings.load(), sort_keys=True)


def model_version() -> str:
    """'<git commit>[-dirty]+settings.<hash>+<library versions>', e.g.
    'a1b2c3d+settings.9f8e7d6c+xgboost-3.2.0+pandas-2.3.3+numpy-2.2.6+scikit-learn-1.7.2+scipy-1.15.3'.

    git commit  which code made the prediction. '-dirty' means there were
                uncommitted changes, so the commit alone doesn't fully
                describe the code (fine in a backtest; a live run should
                always come from a clean checkout).
    settings    the first 8 characters of a SHA-256 hash of the tuned
                settings, so two runs with the same commit but different
                settings files can be told apart.
    libraries   the installed version of each library that can change what
                the model predicts (environment.MODEL_LIBRARIES; added
                2026-11-03). The model is retrained for every prediction,
                so these are as much a part of it as the code.
    'nogit' replaces the commit if git isn't available (e.g. a bare copy).
    """
    settings_hash = hashlib.sha256(settings_text().encode("utf-8")).hexdigest()[:8]
    try:
        # cwd=PROJECT_ROOT: ask about this repo, wherever the script was run from.
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.PROJECT_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        # --porcelain prints one line per changed file, nothing if the tree is clean.
        changes = subprocess.run(["git", "status", "--porcelain"], cwd=config.PROJECT_ROOT,
                                 capture_output=True, text=True, check=True).stdout.strip()
        if changes:
            commit += "-dirty"
    except (OSError, subprocess.CalledProcessError):
        commit = "nogit"
    libraries = "".join(f"+{name}-{version}" for name, version in environment.library_versions().items())
    return f"{commit}+settings.{settings_hash}{libraries}"


def model_order(drivers: pd.DataFrame) -> pd.Series:
    """The position model's raw order (1 = lowest expected_position), from a
    run's driver rows: what `predicted_rank` meant before 2026-11-03, and
    still the order to look at when asking "what did the model itself say,
    before the anchor blend?". Ties: better anchor slot, then driver_id."""
    ordered = drivers.sort_values(["expected_position", "anchor_position", "driver_id"])
    return pd.Series(np.arange(1, len(ordered) + 1), index=ordered.index).reindex(drivers.index)


# -----------------------------------------------------------------------------
# Steps 1-6: the prediction itself (reads the database, writes nothing)
# -----------------------------------------------------------------------------

def make_prediction(conn: sqlite3.Connection, event_id: int, stage: str,
                    n_sims: int = N_SIMS, seed: int | None = 0) -> StagePrediction:
    """Predict one event at one stage, using only what was known at that stage.

    seed fixes the simulator's random numbers, so the same data and settings
    always give the same probabilities (0 = the seed every evaluation used).
    """
    # 1. The race's rows, as known at this stage's cutoff.
    race, meta = prediction_rows(conn, event_id, stage)
    race_date = race["race_date"].iloc[0]

    # 2. Training history: every race strictly before this one, from the
    # normal (result-based) feature table. Those races are over, so their
    # results are exactly what the model would have known. (This builds the
    # feature table a second time; prediction_rows built one with the race
    # swapped in. Rows of earlier races are identical in both, and a build
    # takes a few seconds, so clarity wins over saving it: measured
    # 2026-11-03, 2.6s of a 5.6s prediction, at most a few predictions an
    # hour. The "identical" is pinned by tests/test_as_of.py.)
    table = build_race_features(conn)
    history = table[table["race_date"] < race_date]
    if history.empty:
        raise ValueError(f"No earlier races to train on for event {event_id}")
    model = train_model(history, WEIGHT_2026, columns=STAGE_FEATURES[stage])

    # 3. Expected positions: the model's own number for each driver.
    anchor_column, anchor_weight, noise_sd = STAGE_SETTINGS[stage]
    race = add_baseline_ranks(race)          # champ_rank / practice_rank for this race
    if race[anchor_column].isna().any():
        # E.g. the practice stage with no practice laps at all (pre-2023):
        # there's no practice order to anchor on.
        raise ValueError(f"Event {event_id}, stage {stage}: no {anchor_column} for every driver")
    race = predict_positions(model, race)
    # A fixed row order before blending. The blend's smooth order gives exact
    # ties a shared place, so row order no longer decides anything there; it
    # keeps the simulator's driver order (and so its random draws) the same
    # from run to run.
    race = race.sort_values(["expected_position", anchor_column, "driver_id"]).reset_index(drop=True)

    # 4. DNF chances: computed over the history plus this race's rows. Every
    # rate only looks at EARLIER races (models/dnf._past_rolling_sum), so the
    # race's own missing result plays no part in its own probability.
    with_dnf = add_dnf_probability(pd.concat([history, race[history.columns]], ignore_index=True))
    this_race = with_dnf[with_dnf["session_id"] == meta["race_session_id"]]
    race = race.merge(this_race[["driver_id", "dnf_prob"]], on="driver_id", how="left")

    # 5. Monte Carlo with the frozen settings for this stage.
    race = race.assign(anchor_position=race[anchor_column])
    simulated = race_probabilities(race, noise_sd=noise_sd, grid_weight=anchor_weight,
                                   anchor_column=anchor_column, n_sims=n_sims, seed=seed)
    # race_probabilities overwrites expected_position with the blend. The
    # blend is the simulator's starting score, so ITS order is the predicted
    # order (module docstring): lowest score first; exact ties go to the
    # better anchor slot, then driver_id, so the order is always the same.
    simulated = simulated.rename(columns={"expected_position": "blended_score"})
    simulated = simulated.sort_values(["blended_score", anchor_column, "driver_id"]).reset_index(drop=True)
    simulated["predicted_rank"] = np.arange(1, len(simulated) + 1)
    # Keep the model's raw number in expected_position (that's what the log's
    # expected_position means, and what SHAP explains).
    drivers = (race[["driver_id", "constructor_id", "anchor_position", "expected_position", "dnf_prob"]]
               .rename(columns={"dnf_prob": "p_dnf"})
               .merge(simulated[["driver_id", "predicted_rank", "p_win", "p_podium", "p_points",
                                 "mean_position"]], on="driver_id"))

    # 6. SHAP: value and contribution of each feature the model used.
    explanation = explain_race(model, race)[["driver_id", "feature", "value", "contribution"]]

    return StagePrediction(
        event_id=event_id,
        stage=stage,
        as_of_utc=meta["as_of_utc"],
        lineup_source=meta["lineup_source"],
        n_training_races=int(history["session_id"].nunique()),
        drivers=drivers.sort_values("predicted_rank").reset_index(drop=True),
        features=explanation,
        quali_times_source=meta.get("quali_times_source"),
    )


# -----------------------------------------------------------------------------
# Step 7: writing it to the log
# -----------------------------------------------------------------------------

def _number(value):
    """A value as SQLite expects it: plain Python float/None, never numpy or NaN."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def write_run(conn: sqlite3.Connection, prediction: StagePrediction, mode: str) -> int:
    """Store one prediction in the log and return its run_id.

    All inserts happen inside one transaction (`with conn:` commits at the
    end, or rolls everything back if any insert fails), so the log never
    holds half a run. Once written, the database's triggers stop anyone
    editing it (and deleting it, for live runs): see db/schema.sql.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with conn:
        run_id = conn.execute(
            "INSERT INTO prediction_runs (event_id, stage, mode, as_of_utc, lineup_source, "
            "created_at_utc, model_version, settings_json, n_training_races) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (prediction.event_id, prediction.stage, mode, prediction.as_of_utc,
             prediction.lineup_source, created, model_version(), settings_text(),
             prediction.n_training_races),
        ).lastrowid
        # A debutant predicted under a temporary id needs a placeholder
        # `drivers` row for the log's foreign key (features/provisional.py).
        # Same transaction: no run, no placeholder.
        add_placeholder_drivers(conn, prediction.event_id, prediction.drivers["driver_id"])
        # executemany runs the same INSERT once per tuple in the list.
        conn.executemany(
            "INSERT INTO predictions (run_id, driver_id, constructor_id, predicted_rank, "
            "anchor_position, expected_position, p_win, p_podium, p_points, p_dnf, mean_position) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(run_id, row.driver_id, row.constructor_id, int(row.predicted_rank),
              _number(row.anchor_position), _number(row.expected_position),
              _number(row.p_win), _number(row.p_podium), _number(row.p_points),
              _number(row.p_dnf), _number(row.mean_position))
             for row in prediction.drivers.itertuples()],
        )
        conn.executemany(
            "INSERT INTO prediction_features (run_id, driver_id, feature, value, contribution) "
            "VALUES (?, ?, ?, ?, ?)",
            [(run_id, row.driver_id, row.feature, _number(row.value), _number(row.contribution))
             for row in prediction.features.itertuples()],
        )
    return run_id


def predict_stage(conn: sqlite3.Connection, event_id: int, stage: str, mode: str = "backtest",
                  n_sims: int = N_SIMS, seed: int | None = 0) -> int:
    """Predict one event at one stage and log it. Returns the new run_id."""
    if mode not in MODES:   # check before the (slow) prediction, not after
        raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
    return write_run(conn, make_prediction(conn, event_id, stage, n_sims=n_sims, seed=seed), mode)


# -----------------------------------------------------------------------------
# Checks before a LIVE run is written (shared by scripts/predict_event.py and
# the pipeline tick). Live runs can never be edited or deleted, so anything
# that looks like a bad input blocks the write; the caller reports it and
# tries again later (or a person looks at it).
# -----------------------------------------------------------------------------

# A real F1 field: 20 cars until 2025, 22 from 2026. Outside this range the
# driver list is almost certainly broken (e.g. a failed match of car numbers).
MIN_DRIVERS, MAX_DRIVERS = 18, 26
# More lineup changes than this versus the last race is suspicious: a normal
# weekend has 0-2 (a stand-in, a mid-season swap).
MAX_LINEUP_CHANGES = 4


def check_prediction(conn: sqlite3.Connection, prediction: StagePrediction) -> list[str]:
    """Problems with a prediction's INPUTS or outputs, as readable lines (empty = fine)."""
    from f1predictor.features.as_of import event_info, previous_race_lineup

    problems = []
    drivers = prediction.drivers
    if not MIN_DRIVERS <= len(drivers) <= MAX_DRIVERS:
        problems.append(f"{len(drivers)} drivers (expected {MIN_DRIVERS}-{MAX_DRIVERS})")
    if drivers["driver_id"].duplicated().any():
        problems.append("a driver appears twice")
    probability_columns = ["p_win", "p_podium", "p_points", "p_dnf"]
    if drivers[probability_columns].isna().any().any():
        problems.append("missing probabilities")
    # Exactly one winner and three podium places per simulated race (unless
    # nearly everyone retires, which never happens), so the sums are ~1 and ~3.
    if abs(drivers["p_win"].sum() - 1) > 0.02:
        problems.append(f"win chances add up to {drivers['p_win'].sum():.3f}, not 1")
    if abs(drivers["p_podium"].sum() - 3) > 0.05:
        problems.append(f"podium chances add up to {drivers['p_podium'].sum():.3f}, not 3")
    if drivers["anchor_position"].isna().any():
        problems.append("a driver has no anchor position (grid / championship / practice slot)")

    info = event_info(conn, prediction.event_id)
    previous = set(previous_race_lineup(conn, info["race_date"])["driver_id"])
    if previous and info["round"] > 1:     # round 1: big changes are normal (new season)
        changes = len(set(drivers["driver_id"]) ^ previous)   # ^ = in one set but not both
        if changes > MAX_LINEUP_CHANGES:
            problems.append(f"{changes} driver changes vs the last race (more than {MAX_LINEUP_CHANGES})")
    return problems


# A post-quali prediction with the qualifying gap missing for more than this
# share of the drivers gets an alert (input_warnings). A driver or two with no
# time is normal (didn't set one); most of the field is not (2026 R20 Brazil:
# all 22, because Jolpica published the order before the times).
MAX_MISSING_QUALI_GAP_SHARE = 0.5


def input_warnings(prediction: StagePrediction) -> list[str]:
    """Things worth an alert about a prediction that is still fine to log.

    Unlike check_prediction's problems these don't block the run: a post-quali
    prediction without qualifying gaps is weaker, but much better than none
    (the grid still carries most of it), and waiting is already done by the
    planner (no prediction before LAST_CALL while no times are in anywhere).
    """
    warnings = []
    if prediction.stage == "post_quali":
        gap = prediction.features[prediction.features["feature"] == "quali_gap_pct"]["value"]
        missing = int(gap.isna().sum())
        if len(gap) and missing / len(gap) > MAX_MISSING_QUALI_GAP_SHARE:
            warnings.append(f"qualifying gap missing for {missing} of {len(gap)} drivers "
                            f"(qualifying times: {prediction.quali_times_source})")
    return warnings


def live_problems(conn: sqlite3.Connection, prediction: StagePrediction, now: datetime) -> list[str]:
    """Reasons a prediction can't be logged LIVE right now (empty = go ahead).

    On top of check_prediction: the cutoff hasn't passed (a "live" prediction
    made after the session started wouldn't be honest), this race and stage
    has no live run yet, the code is a clean commit (so model_version
    names exactly the code that ran), and the model libraries are the
    versions in requirements.lock.
    """
    from f1predictor.pipeline.plan import parse_utc

    problems = check_prediction(conn, prediction)
    if now >= parse_utc(prediction.as_of_utc):
        problems.append(f"the cutoff {prediction.as_of_utc} has passed")
    if conn.execute("SELECT 1 FROM prediction_runs WHERE event_id = ? AND stage = ? AND mode = 'live'",
                    (prediction.event_id, prediction.stage)).fetchone():
        problems.append("a live run for this race and stage already exists")
    version = model_version()
    if version.startswith("nogit") or "-dirty" in version.split("+", 1)[0]:
        problems.append(f"code version {version}: uncommitted changes")
    # The model libraries must be the locked, tested versions
    # (environment.py): otherwise the "frozen" model isn't the tested one.
    problems += [f"library not as locked in requirements.lock: {p}" for p in environment.lock_problems()]
    return problems
