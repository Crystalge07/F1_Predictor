"""Score the practice-stage model against its baselines.

Usage (from the repo root, with the virtual environment active):
    python scripts/evaluate_practice.py        (about a minute)

Practice stage = predicting after practice, before qualifying (see
features/stages.py). Every race from 2024 on is predicted walk-forward and
scored with the same metrics as everything else. Nothing is tuned, so every
season is reported.

Compared:
    championship order   the pre-weekend baseline
    practice order       best practice lap first: the obvious no-ML guess here
    pre-weekend model    what we knew before practice (does practice add anything?)
    practice model       the new one
    (grid, post-quali)   for scale: what qualifying adds on top
"""

import pandas as pd

from f1predictor.db.connection import get_connection
from f1predictor.evaluate.metrics import average_scores, score_race
from f1predictor.evaluate.walk_forward import walk_forward
from f1predictor.features.race_features import build_race_features
from f1predictor.features.stages import PRACTICE_FEATURES, PRE_WEEKEND_FEATURES
from f1predictor.models.baselines import actual_result, championship_order, grid_order, practice_order
from f1predictor.models.position import WEIGHT_2026, predict_order, train_model


def stage_trainer(columns: list[str]):
    """A walk-forward trainer for a model on the given feature list."""
    def trainer(history: pd.DataFrame):
        model = train_model(history, WEIGHT_2026, columns=columns)
        return lambda race: predict_order(model, race)
    return trainer


def main() -> None:
    conn = get_connection()
    try:
        features = build_race_features(conn)
        sessions = features[["session_id", "season"]].drop_duplicates()
        test_ids = sessions[sessions.season >= 2024].session_id.tolist()
        actuals = {s: actual_result(conn, s) for s in test_ids}
        grid = {s: grid_order(conn, s) for s in test_ids}
    finally:
        conn.close()

    race_rows = {s: features[features.session_id == s] for s in test_ids}
    champ = {s: championship_order(r) for s, r in race_rows.items()}
    practice = {s: practice_order(r) for s, r in race_rows.items()}
    print(f"Training {2 * len(test_ids)} walk-forward models (pre-weekend + practice)...")
    pre_weekend = walk_forward(features, test_ids, stage_trainer(PRE_WEEKEND_FEATURES))
    practice_model = walk_forward(features, test_ids, stage_trainer(PRACTICE_FEATURES))

    season_of = dict(zip(sessions.session_id, sessions.season))
    groups = {str(y): [s for s in test_ids if season_of[s] == y] for y in (2024, 2025, 2026)}
    groups["all"] = test_ids

    print(f"\n{'':<6} {'':<20} {'pos MAE':>8} {'spearman':>9} {'winner':>7} {'podium':>7}")
    for label, ids in groups.items():
        for name, preds in (("championship order", champ), ("practice order", practice),
                            ("pre-weekend model", pre_weekend), ("practice model", practice_model),
                            ("(grid, post-quali)", grid)):
            s = average_scores([score_race(preds[i], actuals[i]) for i in ids])
            print(f"{label:<6} {name:<20} {s['position_mae']:>8.2f} {s['spearman']:>9.3f} "
                  f"{s['winner_hit']:>7.0%} {s['podium_hits']:>7.0%}")
        print()


if __name__ == "__main__":
    main()
