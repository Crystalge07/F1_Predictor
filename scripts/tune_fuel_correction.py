"""Choose the long-run fuel correction (ms gained per lap of fuel burned).

Usage (from the repo root, with the virtual environment active):
    python scripts/tune_fuel_correction.py        (about two minutes)

Long-run pace only feeds the PRACTICE stage model, so that's the model scored.
For each candidate effect we rebuild the feature table, run the practice model
walk-forward, and measure average position error:

    CALIBRATE  2024 + 2026 rounds 1-9   pick the effect (saved to tuned_settings.json)
    TEST       2025 + 2026 rounds 10+   chosen effect vs no correction, reported once
"""

import f1predictor.features.practice_pace as practice_pace
from f1predictor import settings
from f1predictor.db.connection import get_connection
from f1predictor.evaluate.metrics import average_scores, score_race
from f1predictor.evaluate.walk_forward import walk_forward
from f1predictor.features.race_features import build_race_features
from f1predictor.features.stages import PRACTICE_FEATURES
from f1predictor.models.baselines import actual_result
from f1predictor.models.position import predict_order, train_model

EFFECT_CANDIDATES = [0.0, 30.0, 60.0, 90.0]   # ms per lap; 0 = no correction
LAST_CALIBRATION_ROUND_2026 = 9


def trainer(history):
    model = train_model(history, columns=PRACTICE_FEATURES)
    return lambda race: predict_order(model, race)


def practice_mae(conn, effect: float, session_ids: list[int], actuals: dict) -> float:
    # The feature code reads this module-level constant, so setting it here
    # changes how long runs are corrected for the rebuild below.
    practice_pace.FUEL_EFFECT_MS_PER_LAP = effect
    features = build_race_features(conn)
    predictions = walk_forward(features, session_ids, trainer)
    return average_scores([score_race(predictions[s], actuals[s]) for s in session_ids])["position_mae"]


def main() -> None:
    conn = get_connection()
    try:
        sessions = build_race_features(conn)[["session_id", "season", "round"]].drop_duplicates()
        early_2026 = (sessions.season == 2026) & (sessions["round"] <= LAST_CALIBRATION_ROUND_2026)
        calibrate = sessions[(sessions.season == 2024) | early_2026].session_id.tolist()
        test = sessions[(sessions.season == 2025) | ((sessions.season == 2026) & ~early_2026)].session_id.tolist()
        actuals = {s: actual_result(conn, s) for s in calibrate + test}

        print(f"Practice-model position error on {len(calibrate)} calibration races:")
        maes = {e: practice_mae(conn, e, calibrate, actuals) for e in EFFECT_CANDIDATES}
        for e, m in maes.items():
            print(f"  {e:>4.0f} ms/lap   {m:.3f}")
        best = min(EFFECT_CANDIDATES, key=maes.get)
        settings.save(["fuel_effect_ms_per_lap"], best)
        print(f"  -> chosen {best:.0f} ms/lap (saved to tuned_settings.json)\n")

        none, chosen = practice_mae(conn, 0.0, test, actuals), practice_mae(conn, best, test, actuals)
        print(f"TEST ({len(test)} races): no correction {none:.3f}, {best:.0f} ms/lap {chosen:.3f}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
