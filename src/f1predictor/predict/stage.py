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
    3. predict   expected finishing position per driver (if they finish),
                 and the model's order: that's `predicted_rank`.
    4. DNF       each driver's chance of not finishing (models/dnf.py).
    5. simulate  Monte Carlo from the anchor/model blend (simulate/
                 probabilities.py, frozen settings): p_win, p_podium, ...
    6. explain   SHAP contribution of every feature (explain/contributions.py).
    7. log       one prediction_runs row, one predictions row per driver, one
                 prediction_features row per driver per feature, all in ONE
                 transaction: either the whole run is logged or none of it.

Steps 1-6 are make_prediction() (reads only); step 7 is write_run(). Keeping
them apart lets tests check a prediction without touching the log, and lets
the replay script print a run before deciding to store it.

Which order is `predicted_rank`? The position model's own order (lowest
expected position first), the same order the Phase 2/3 finishing-order
numbers (MAE, Spearman) were measured on. The simulator's view is stored
alongside it: `anchor_position` (the grid / championship / practice slot it
blends with) and `mean_position` (the average simulated place).
"""

import hashlib
import json
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from f1predictor import config, settings
from f1predictor.explain.contributions import explain_race
from f1predictor.features.as_of import prediction_rows
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


# -----------------------------------------------------------------------------
# Which code and settings made a prediction
# -----------------------------------------------------------------------------

def settings_text() -> str:
    """The tuned settings as JSON text, keys sorted so the same settings always
    give the same text (and the same hash)."""
    return json.dumps(settings.load(), sort_keys=True)


def model_version() -> str:
    """'<git commit>[-dirty]+settings.<hash>', e.g. 'a1b2c3d+settings.9f8e7d6c'.

    git commit  which code made the prediction. '-dirty' means there were
                uncommitted changes, so the commit alone doesn't fully
                describe the code (fine in a backtest; a live run should
                always come from a clean checkout).
    settings    the first 8 characters of a SHA-256 hash of the tuned
                settings, so two runs with the same commit but different
                settings files can be told apart.
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
    return f"{commit}+settings.{settings_hash}"


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
    # takes a few seconds, so clarity wins over saving it.)
    table = build_race_features(conn)
    history = table[table["race_date"] < race_date]
    if history.empty:
        raise ValueError(f"No earlier races to train on for event {event_id}")
    model = train_model(history, WEIGHT_2026, columns=STAGE_FEATURES[stage])

    # 3. Expected positions, and the model's order (ties: anchor slot, then
    # driver_id, so the order is always the same).
    anchor_column, anchor_weight, noise_sd = STAGE_SETTINGS[stage]
    race = add_baseline_ranks(race)          # champ_rank / practice_rank for this race
    if race[anchor_column].isna().any():
        # E.g. the practice stage with no practice laps at all (pre-2023):
        # there's no practice order to anchor on.
        raise ValueError(f"Event {event_id}, stage {stage}: no {anchor_column} for every driver")
    race = predict_positions(model, race)
    race = race.sort_values(["expected_position", anchor_column, "driver_id"]).reset_index(drop=True)
    race["predicted_rank"] = np.arange(1, len(race) + 1)

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
    # race_probabilities overwrites expected_position with the blend; keep the
    # model's raw number (that's what the log's expected_position means).
    drivers = (race[["driver_id", "constructor_id", "predicted_rank", "anchor_position",
                     "expected_position", "dnf_prob"]]
               .rename(columns={"dnf_prob": "p_dnf"})
               .merge(simulated[["driver_id", "p_win", "p_podium", "p_points", "mean_position"]],
                      on="driver_id"))

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
