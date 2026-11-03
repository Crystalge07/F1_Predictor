"""The scorer: after a race, grade every logged prediction for it.

For each run in the prediction log whose race has a result, this writes one
`scores` row per metric, with the model's value next to the stage's
baseline on the SAME race:

    metric          model                         baseline
    position_mae    places off, on average        ... of the baseline order
    spearman        rank correlation              ...
    winner_hit      predicted winner won (1/0)    ...
    podium_hits     share of our top 3 on podium  ...
    brier_win       Brier score of p_win          slot rates (see below)
    brier_podium    Brier score of p_podium       slot rates
    lineup_missing  starters the run didn't have  (none)
    lineup_extra    predicted drivers who didn't start  (none)

The baseline uses the same information as the model. Its order is the
run's own `anchor_position` (the as-of starting grid after qualifying,
championship order or practice order before it), so it never sees the
official grid that is only published with the result. Its probabilities are
the historical win/podium rate of each slot in that order, from races before
this one (models/baselines.grid_probabilities), exactly as in Phase 3.

The driver list may not match the result: the run was made before the race
(features/as_of.py). Rule:
    * predicted but didn't start (e.g. ill on Sunday): dropped. Scoring is
      over the drivers who actually raced, as everywhere else.
    * started but wasn't predicted (a late replacement): added at the BACK of
      the predicted order, with p_win = p_podium = 0. The prediction really
      did miss them, so it pays for that, and lineup_missing records it.
Both the model and the baseline are treated this way, so the comparison is fair.

Scores can be rewritten: re-scoring a run (e.g. after the FIA changes a
result) replaces its rows. The prediction being scored never changes.
"""

import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

from f1predictor.evaluate.metrics import score_race
from f1predictor.features.race_features import build_race_features
from f1predictor.models.baselines import actual_result, add_baseline_ranks, grid_probabilities
from f1predictor.models.dnf import brier

# The baseline each stage is compared with, and the feature-table column that
# holds its slot (for the slot-rate probabilities).
BASELINES = {
    "pre_weekend": ("championship_order", "champ_rank"),
    "practice": ("practice_order", "practice_rank"),
    "post_quali": ("grid_order", "grid"),
}


# -----------------------------------------------------------------------------
# Reading what's needed for one run
# -----------------------------------------------------------------------------

def run_info(conn: sqlite3.Connection, run_id: int) -> dict:
    """The run's stage, its race session, and the race date."""
    row = conn.execute(
        """
        SELECT r.run_id, r.stage, r.event_id, e.race_date, s.session_id AS race_session_id
        FROM prediction_runs r
        JOIN events e   ON e.event_id = r.event_id
        JOIN sessions s ON s.event_id = r.event_id AND s.session_type = 'R'
        WHERE r.run_id = ?
        """, (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"No prediction run {run_id} (or its event has no race session)")
    return dict(row)


def logged_predictions(conn: sqlite3.Connection, run_id: int) -> pd.DataFrame:
    """driver_id, predicted_rank, anchor_position, p_win, p_podium for one run."""
    return pd.read_sql_query(
        "SELECT driver_id, predicted_rank, anchor_position, p_win, p_podium "
        "FROM predictions WHERE run_id = ? ORDER BY predicted_rank",
        conn, params=(run_id,))


# -----------------------------------------------------------------------------
# Lining up a prediction with the real field
# -----------------------------------------------------------------------------

def reconcile(order: list[str], actual: dict[str, int]) -> tuple[list[str], list[str], list[str]]:
    """Fit a predicted order to the drivers who actually raced.

    Returns (order, missing, extra):
        order    the predicted order, without non-starters, with unpredicted
                 starters added at the back (alphabetically: any order that
                 doesn't look at the result would do; this one is fixed).
        missing  starters the prediction didn't have
        extra    predicted drivers who didn't start
    """
    extra = [d for d in order if d not in actual]
    missing = sorted(d for d in actual if d not in set(order))
    return [d for d in order if d in actual] + missing, missing, extra


def _brier_scores(probs: pd.DataFrame, actual: dict[str, int]) -> dict[str, float]:
    """Brier win/podium over the drivers who raced. probs: driver_id, p_win, p_podium.
    A starter with no probability row (not predicted) counts as 0 for both."""
    field = pd.DataFrame({"driver_id": list(actual), "actual_position": list(actual.values())})
    field = field.merge(probs[["driver_id", "p_win", "p_podium"]], on="driver_id", how="left").fillna(
        {"p_win": 0.0, "p_podium": 0.0})
    return {
        "brier_win": brier(field["p_win"], (field["actual_position"] == 1).astype(int)),
        "brier_podium": brier(field["p_podium"], (field["actual_position"] <= 3).astype(int)),
    }


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------

def score_run(conn: sqlite3.Connection, run_id: int, history_table: pd.DataFrame | None = None) -> dict | None:
    """Score one run and write its scores. Returns {metric: (model, baseline)},
    or None if the race has no result yet (nothing to score).

    history_table: the result-based feature table with add_baseline_ranks
    applied (for the slot-rate baseline). Build it once and pass it in when
    scoring many runs; built here if not given.
    """
    info = run_info(conn, run_id)
    actual = actual_result(conn, info["race_session_id"])
    if not actual:
        return None
    logged = logged_predictions(conn, run_id)
    baseline_name, slot_column = BASELINES[info["stage"]]

    # --- Finishing order: the model's ranks vs the anchor's order ------------
    model_order, missing, extra = reconcile(logged["driver_id"].tolist(), actual)
    # Baseline order: sort by the anchor slot (ties by driver_id, fixed).
    anchor_sorted = logged.sort_values(["anchor_position", "driver_id"])["driver_id"].tolist()
    baseline_order, _, _ = reconcile(anchor_sorted, actual)
    model_scores = asdict(score_race(model_order, actual))
    baseline_scores = asdict(score_race(baseline_order, actual))

    # --- Probabilities: the run's p_win/p_podium vs historical slot rates ----
    if history_table is None:
        history_table = add_baseline_ranks(build_race_features(conn))
    history = history_table[history_table["race_date"] < info["race_date"]]
    # grid_probabilities looks up each driver's slot in `slot_column`; the
    # run's anchor_position IS that slot.
    slots = logged[["driver_id", "anchor_position"]].rename(columns={"anchor_position": slot_column})
    baseline_probs = grid_probabilities(history, slots, slot_column)
    model_scores.update(_brier_scores(logged, actual))
    baseline_scores.update(_brier_scores(baseline_probs, actual))

    # --- Write (replacing any earlier scores of this run) --------------------
    results = {metric: (model_scores[metric], baseline_scores[metric]) for metric in model_scores}
    results["lineup_missing"] = (len(missing), None)
    results["lineup_extra"] = (len(extra), None)
    scored_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with conn:   # one transaction: all of a run's scores, or none
        conn.executemany(
            # ON CONFLICT ... DO UPDATE ("upsert"): insert, or overwrite the
            # existing row for this (run, metric). Works the same in Postgres.
            """
            INSERT INTO scores (run_id, metric, model_value, baseline_value, baseline_name, scored_at_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id, metric) DO UPDATE SET
                model_value = excluded.model_value,
                baseline_value = excluded.baseline_value,
                baseline_name = excluded.baseline_name,
                scored_at_utc = excluded.scored_at_utc
            """,
            [(run_id, metric, float(model), None if base is None else float(base),
              None if base is None else baseline_name, scored_at)
             for metric, (model, base) in results.items()],
        )
    return results


def score_runs(conn: sqlite3.Connection, run_ids: list[int] | None = None) -> dict[int, dict]:
    """Score many runs (default: every logged run). Runs whose race has no
    result yet are skipped. Returns {run_id: scores} for the runs scored."""
    if run_ids is None:
        run_ids = [r[0] for r in conn.execute("SELECT run_id FROM prediction_runs ORDER BY run_id")]
    if not run_ids:
        return {}
    history_table = add_baseline_ranks(build_race_features(conn))   # built once for all runs
    scored = {}
    for run_id in run_ids:
        result = score_run(conn, run_id, history_table)
        if result is not None:
            scored[run_id] = result
    return scored
