"""Score the pre-weekend model against the championship-order baseline.

Usage (from the repo root, with the virtual environment active):
    python scripts/evaluate_pre_weekend.py        (about half a minute)

Pre-weekend = predicting before FP1, so no practice, qualifying or grid.
Every race from 2024 on is predicted walk-forward (model trained only on
earlier races) and scored with the same metrics as everything else.

Nothing is tuned here (same XGBoost settings and 2026 weight as Phase 2), so
every season can be reported without a held-out split.

The post-quali grid baseline is printed too, for scale: it shows how much
qualifying tells you that the pre-weekend stage has to do without.
"""

import pandas as pd

from f1predictor.db.connection import get_connection
from f1predictor.evaluate.metrics import average_scores, score_race
from f1predictor.evaluate.walk_forward import walk_forward
from f1predictor.features.race_features import build_race_features
from f1predictor.features.stages import PRE_WEEKEND_FEATURES
from f1predictor.models.baselines import actual_result, championship_order, grid_order
from f1predictor.models.position import predict_order, train_model


def pre_weekend_trainer(history: pd.DataFrame):
    model = train_model(history, columns=PRE_WEEKEND_FEATURES)
    return lambda race: predict_order(model, race)


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

    champ = {s: championship_order(features[features.session_id == s]) for s in test_ids}
    print(f"Training {len(test_ids)} walk-forward models...")
    model = walk_forward(features, test_ids, pre_weekend_trainer)

    season_of = dict(zip(sessions.session_id, sessions.season))
    groups = {str(y): [s for s in test_ids if season_of[s] == y] for y in (2024, 2025, 2026)}
    groups["all"] = test_ids

    print(f"\n{'':<6} {'':<20} {'pos MAE':>8} {'spearman':>9} {'winner':>7} {'podium':>7}")
    for label, ids in groups.items():
        for name, preds in (("championship order", champ), ("pre-weekend model", model),
                            ("(grid, post-quali)", grid)):
            s = average_scores([score_race(preds[i], actuals[i]) for i in ids])
            print(f"{label:<6} {name:<20} {s['position_mae']:>8.2f} {s['spearman']:>9.3f} "
                  f"{s['winner_hit']:>7.0%} {s['podium_hits']:>7.0%}")
        print()


if __name__ == "__main__":
    main()
