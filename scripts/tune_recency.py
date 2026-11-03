"""Choose the season decay (how much older seasons count) for every stage's model.

Usage (from the repo root, with the virtual environment active):
    python scripts/tune_recency.py        (about two minutes)

Each training row is weighted decay ** (seasons before the newest season), see
models/position.py. One decay is shared by all three stage models, so it's
chosen by their AVERAGE position error on the calibration races:

    CALIBRATE  2024 + 2026 rounds 1-9   (the same races used to calibrate the
                                         simulator; the held-out races, 2025 and
                                         2026 rounds 10+, are not touched here)

The winner is saved to tuned_settings.json as "season_decay".
"""

from f1predictor import settings
from f1predictor.db.connection import get_connection
from f1predictor.evaluate.metrics import average_scores, score_race
from f1predictor.evaluate.walk_forward import walk_forward
from f1predictor.features.race_features import build_race_features
from f1predictor.features.stages import STAGE_FEATURES
from f1predictor.models.baselines import actual_result
from f1predictor.models.position import predict_order, train_model

# 1.0 = every season counts the same (no recency weighting).
DECAY_CANDIDATES = [1.0, 0.8, 0.6, 0.4]
LAST_CALIBRATION_ROUND_2026 = 9


def main() -> None:
    conn = get_connection()
    try:
        features = build_race_features(conn)
        sessions = features[["session_id", "season", "round"]].drop_duplicates()
        early_2026 = (sessions.season == 2026) & (sessions["round"] <= LAST_CALIBRATION_ROUND_2026)
        calibrate = sessions[(sessions.season == 2024) | early_2026].session_id.tolist()
        actuals = {s: actual_result(conn, s) for s in calibrate}
    finally:
        conn.close()

    print(f"Average position error on {len(calibrate)} calibration races (lower = better)\n")
    print(f"  {'decay':<6} " + " ".join(f"{stage:>12}" for stage in STAGE_FEATURES) + f" {'mean':>8}")
    means = {}
    for decay in DECAY_CANDIDATES:
        maes = []
        for columns in STAGE_FEATURES.values():
            # Default arguments freeze this loop's values inside the lambda.
            def trainer(history, columns=columns, decay=decay):
                model = train_model(history, season_decay=decay, columns=columns)
                return lambda race: predict_order(model, race)
            predictions = walk_forward(features, calibrate, trainer)
            maes.append(average_scores([score_race(predictions[s], actuals[s]) for s in calibrate])["position_mae"])
        means[decay] = sum(maes) / len(maes)
        print(f"  {decay:<6} " + " ".join(f"{m:>12.3f}" for m in maes) + f" {means[decay]:>8.3f}")

    best = min(DECAY_CANDIDATES, key=means.get)
    settings.save(["season_decay"], best)
    print(f"\n-> chosen season_decay {best}, saved to tuned_settings.json")


if __name__ == "__main__":
    main()
