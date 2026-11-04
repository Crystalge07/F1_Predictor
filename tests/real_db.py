"""Shared helpers for tests that look at the REAL database (read-only).

Some properties (does a better grid ever lower a win chance? how much does a
tiny nudge move the output?) depend on what the trained model actually
learned, so a hand-made 4-driver weekend can't show them. These tests:

    * open data/f1.sqlite (or $F1_LOG_DB) READ-ONLY: they can never change it
    * skip themselves if the file isn't there (e.g. a fresh clone or CI)
    * use one fixed race, 2026 round 19 (Mexico City), the same weekend the
      replay test uses, so results don't drift as new races are added
      (training uses only races before it)

Building the feature table takes a few seconds, so it's done once per test
session and shared (the _CACHE dict below).
"""

import os
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from f1predictor import config
from f1predictor.features.as_of import prediction_rows
from f1predictor.features.race_features import build_race_features
from f1predictor.features.stages import STAGE_FEATURES
from f1predictor.models.baselines import add_baseline_ranks
from f1predictor.models.position import WEIGHT_2026, predict_positions, train_model
from f1predictor.simulate.probabilities import STAGE_SETTINGS, race_probabilities

SEASON, ROUND = 2026, 19

# Filled on first use: the connection, the feature table, and per stage the
# trained model and the race's as-of rows.
_CACHE: dict = {}


def db_path() -> Path:
    return Path(os.environ.get("F1_LOG_DB", config.DB_PATH))


def connect_read_only() -> sqlite3.Connection:
    """The real database, opened so that any write raises an error."""
    path = db_path()
    if not path.exists():
        pytest.skip(f"no database at {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row          # same as db/connection.get_connection
    return conn


def stage_setup(stage: str, season: int = SEASON, rnd: int = ROUND) -> tuple:
    """(model, race rows) for one race at `stage` (default 2026 R19), trained
    exactly as predict/stage.py trains it: the stage's features, every race
    before this one, the frozen 2026 weight."""
    if "conn" not in _CACHE:
        _CACHE["conn"] = connect_read_only()
        _CACHE["table"] = build_race_features(_CACHE["conn"])
    key = (stage, season, rnd)
    if key not in _CACHE:
        conn = _CACHE["conn"]
        row = conn.execute("SELECT event_id FROM events WHERE season = ? AND round = ?",
                           (season, rnd)).fetchone()
        if row is None:
            pytest.skip(f"{season} round {rnd} not in the database")
        race, _ = prediction_rows(conn, row["event_id"], stage)
        table = _CACHE["table"]
        history = table[table["race_date"] < race["race_date"].iloc[0]]
        model = train_model(history, WEIGHT_2026, columns=STAGE_FEATURES[stage])
        _CACHE[key] = (model, race)
    return _CACHE[key]


def probabilities(model, race: pd.DataFrame, stage: str, n_sims: int = 2_000) -> pd.DataFrame:
    """p_win / p_podium per driver (indexed by driver_id), through the same
    steps as predict/stage.make_prediction: baseline ranks (the anchor) ->
    model's expected positions -> anchor/model blend -> Monte Carlo.

    The DNF chance is a flat 10% for everyone: these tests are about the
    position side, and a flat rate keeps DNFs from mixing into the comparison.
    The simulator's seed is fixed (0), so every call uses the SAME random
    numbers; differences between two calls come only from the inputs.
    """
    anchor, weight, noise_sd = STAGE_SETTINGS[stage]
    race = predict_positions(model, add_baseline_ranks(race)).assign(dnf_prob=0.1)
    out = race_probabilities(race, noise_sd=noise_sd, grid_weight=weight,
                             anchor_column=anchor, n_sims=n_sims)
    return out.set_index("driver_id")[["p_win", "p_podium"]]
