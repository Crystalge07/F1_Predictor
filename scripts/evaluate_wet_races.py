"""How much would knowing it'll rain help? (An upper bound, using actual rain.)

Usage (from the repo root, with the virtual environment active):
    python scripts/evaluate_wet_races.py        (about half a minute)

The simulator can widen its noise for wet races (simulate/probabilities.wet_noise).
At prediction time that needs a rain FORECAST, which we don't have yet (Phase 5,
OpenWeatherMap). So here we pretend the forecast is perfect: rain_prob = 1 for
races that turned out wet, 0 otherwise (features/weather.py).

That is deliberately optimistic, so read the result as "the most a forecast
could add". If even a perfect forecast barely helps, wiring one up isn't worth
much; if it helps a lot, it is.

Post-quali stage only. Same split as everywhere:
    CALIBRATE  2024 + 2026 rounds 1-9   pick the wet multiplier
    TEST       2025 + 2026 rounds 10+   report

Caveat: weather data starts in 2023, and only 3 calibration races and 5 test
races were wet. Small samples: treat the numbers as a hint, not a verdict.
"""

import pandas as pd

from f1predictor import settings
from f1predictor.db.connection import get_connection
from f1predictor.evaluate.walk_forward import walk_forward
from f1predictor.features.race_features import build_race_features
from f1predictor.features.weather import race_rain
from f1predictor.models.dnf import add_dnf_probability
from f1predictor.models.position import predict_positions, train_model
from f1predictor.simulate.probabilities import race_probabilities, score_probabilities

MULTIPLIER_CANDIDATES = [1.0, 1.25, 1.5, 2.0, 3.0]   # 1.0 = ignore the rain
LAST_CALIBRATION_ROUND_2026 = 9


def trainer(history: pd.DataFrame):
    model = train_model(history)
    return lambda race: predict_positions(model, race)


def simulate_all(races: dict[int, pd.DataFrame], wet: dict[int, int], multiplier: float) -> pd.DataFrame:
    return pd.concat([race_probabilities(r, rain_prob=wet.get(s, 0), wet_multiplier=multiplier)
                      for s, r in races.items()], ignore_index=True)


def total(rows: pd.DataFrame) -> float:
    s = score_probabilities(rows)
    return s["brier_win"] + s["brier_podium"]


def main() -> None:
    conn = get_connection()
    try:
        features = build_race_features(conn)
        wet = dict(zip(*race_rain(conn)[["session_id", "wet"]].T.values))
    finally:
        conn.close()
    dnf = add_dnf_probability(features)[["session_id", "driver_id", "dnf_prob"]]

    sessions = features[["session_id", "season", "round"]].drop_duplicates()
    early_2026 = (sessions.season == 2026) & (sessions["round"] <= LAST_CALIBRATION_ROUND_2026)
    calibrate_ids = sessions[(sessions.season == 2024) | early_2026].session_id.tolist()
    test_ids = sessions[(sessions.season == 2025) | ((sessions.season == 2026) & ~early_2026)].session_id.tolist()

    print(f"Training {len(calibrate_ids) + len(test_ids)} walk-forward models...")
    predicted = walk_forward(features, calibrate_ids + test_ids, trainer)
    predicted = {s: r.merge(dnf, on=["session_id", "driver_id"], how="left") for s, r in predicted.items()}
    calibrate = {s: predicted[s] for s in calibrate_ids}
    test = {s: predicted[s] for s in test_ids}
    wet_test = {s: r for s, r in test.items() if wet.get(s, 0)}
    print(f"Wet races: {sum(wet.get(s, 0) for s in calibrate_ids)} calibration, {len(wet_test)} test\n")

    print("Calibration (win + podium Brier, lower = better):")
    scores = {m: total(simulate_all(calibrate, wet, m)) for m in MULTIPLIER_CANDIDATES}
    for m, s in scores.items():
        print(f"  wet multiplier {m:<5} {s:.5f}")
    best = min(MULTIPLIER_CANDIDATES, key=scores.get)
    settings.save(["wet_noise_multiplier"], best)
    print(f"  -> chosen {best} (saved to tuned_settings.json)\n")

    print(f"TEST, with a perfect rain forecast ({len(test)} races, {len(wet_test)} wet):")
    for label, races in (("all test races", test), ("wet test races only", wet_test)):
        ignore, use = total(simulate_all(races, wet, 1.0)), total(simulate_all(races, wet, best))
        print(f"  {label:<22} ignore rain {ignore:.5f}   use rain {use:.5f}   "
              f"({(ignore - use) / ignore:+.1%}, positive = rain helps)")


if __name__ == "__main__":
    main()
