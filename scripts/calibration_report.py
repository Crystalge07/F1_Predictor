"""Calibration report: predicted vs actual win and podium rates, per stage.

Usage (from the repo root, with the virtual environment active):
    python scripts/calibration_report.py [path/to/f1.sqlite]     (a few minutes)

Reads only. Two sets of predictions, never mixed:

    HELD-OUT   walk-forward predictions for the races that played no part in
               choosing the frozen settings (2025 + 2026 rounds 10+, the same
               34 races as docs/evaluation-protocol.md), made with the frozen
               settings. The honest one.
    LOGGED     the runs in the prediction log (2026 backtest, in-sample; plus
               any live runs, reported separately by mode), scored against
               the real result.

See evaluate/calibration.py for how to read the tables.
"""

import sys

import pandas as pd

from f1predictor.db.connection import get_connection
from f1predictor.evaluate.calibration import PODIUM_EDGES, WIN_EDGES, calibration_table
from f1predictor.evaluate.walk_forward import walk_forward
from f1predictor.features.race_features import build_race_features
from f1predictor.features.stages import STAGE_FEATURES
from f1predictor.models.baselines import add_baseline_ranks
from f1predictor.models.dnf import add_dnf_probability
from f1predictor.models.position import WEIGHT_2026, predict_positions, train_model
from f1predictor.simulate.probabilities import STAGE_SETTINGS, race_probabilities

LAST_CALIBRATION_ROUND_2026 = 9      # as in scripts/evaluate_stage_probabilities.py


def print_tables(title: str, rows: pd.DataFrame) -> None:
    """rows: p_win, p_podium, actual_position (one row per driver per race)."""
    races = rows["session_id"].nunique() if "session_id" in rows else rows["run_id"].nunique()
    print(f"\n=== {title}: {races} races, {len(rows)} driver rows ===")
    for name, column, edges, hit in [
        ("win", "p_win", WIN_EDGES, rows["actual_position"] == 1),
        ("podium", "p_podium", PODIUM_EDGES, rows["actual_position"] <= 3),
    ]:
        table = calibration_table(rows[column], hit, edges)
        print(f"  {name}: predicted total {rows[column].sum():.1f}, actual {int(hit.sum())}")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}",
                              formatters={"within_range": lambda b: "yes" if b else "NO"}))


def held_out(conn) -> None:
    features = add_baseline_ranks(build_race_features(conn))
    dnf = add_dnf_probability(features)[["session_id", "driver_id", "dnf_prob"]]
    sessions = features[["session_id", "season", "round"]].drop_duplicates()
    late_2026 = (sessions.season == 2026) & (sessions["round"] > LAST_CALIBRATION_ROUND_2026)
    test_ids = sessions[(sessions.season == 2025) | late_2026].session_id.tolist()

    for stage, columns in STAGE_FEATURES.items():
        anchor, weight, noise_sd = STAGE_SETTINGS[stage]

        def trainer(history, columns=columns):
            model = train_model(history, WEIGHT_2026, columns=columns)
            return lambda race: predict_positions(model, race)

        predicted = walk_forward(features, test_ids, trainer)
        rows = pd.concat([
            race_probabilities(race.merge(dnf, on=["session_id", "driver_id"], how="left"),
                               noise_sd=noise_sd, grid_weight=weight, anchor_column=anchor)
            for race in predicted.values()], ignore_index=True)
        print_tables(f"HELD-OUT {stage}", rows)


def logged(conn) -> None:
    rows = pd.read_sql_query(
        """
        SELECT r.run_id, r.stage, r.mode, p.driver_id, p.p_win, p.p_podium, res.position AS actual_position
        FROM prediction_runs r
        JOIN predictions p USING (run_id)
        JOIN sessions s ON s.event_id = r.event_id AND s.session_type = 'R'
        -- Only drivers who appear in the result (a predicted non-starter has
        -- no outcome; temporary debutant ids drop out here too).
        JOIN results res ON res.session_id = s.session_id AND res.driver_id = p.driver_id
        """, conn)
    for (mode, stage), group in rows.groupby(["mode", "stage"]):
        label = "in-sample" if mode == "backtest" else "live"
        print_tables(f"LOGGED {mode} {stage} ({label})", group)


def main() -> None:
    conn = get_connection(sys.argv[1]) if len(sys.argv) > 1 else get_connection()
    try:
        held_out(conn)
        logged(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
