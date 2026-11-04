"""Win/podium probabilities for the pre-weekend and practice stages.

Usage (from the repo root, with the virtual environment active):
    python scripts/evaluate_stage_probabilities.py        (a few minutes)

The same recipe as post-quali (scripts/evaluate_probabilities.py), with one
change: these stages have no grid, so the blend anchors on the stage's own
baseline order instead:

    stage         anchor (the "grid")      model features
    pre_weekend   championship order       history + standings
    practice      practice order           + practice pace

For each stage:
    1. walk-forward expected positions for every race from 2024
    2. CALIBRATE (2024 + 2026 rounds 1-9): pick anchor weight + noise together
    3. TEST (2025 + 2026 rounds 10+), scored once, against:
         slot baseline     historical win/podium rate for that place in the
                           baseline order (like the grid-slot baseline)
         anchor + sim      the simulator fed the baseline order only
         model             the full thing

The chosen settings are saved to tuned_settings.json (see settings.py).
"""

import pandas as pd

from f1predictor import settings
from f1predictor.db.connection import get_connection
from f1predictor.evaluate.splits import split_races
from f1predictor.evaluate.walk_forward import walk_forward
from f1predictor.features.race_features import build_race_features
from f1predictor.features.stages import STAGE_FEATURES
from f1predictor.models.baselines import add_baseline_ranks, grid_probabilities
from f1predictor.models.dnf import add_dnf_probability
from f1predictor.models.position import WEIGHT_2026, predict_positions, train_model
from f1predictor.simulate.probabilities import race_probabilities, score_probabilities

STAGES = {"pre_weekend": "champ_rank", "practice": "practice_rank"}
NOISE_CANDIDATES = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
ANCHOR_WEIGHT_CANDIDATES = [0.0, 0.25, 0.5, 0.75, 1.0]


def simulate_all(races, anchor, weight, noise_sd):
    return pd.concat([race_probabilities(r, noise_sd=noise_sd, grid_weight=weight, anchor_column=anchor)
                      for r in races.values()], ignore_index=True)


def total_brier(rows) -> float:
    s = score_probabilities(rows)
    return s["brier_win"] + s["brier_podium"]


def main() -> None:
    conn = get_connection()
    try:
        features = build_race_features(conn)
    finally:
        conn.close()
    features = add_baseline_ranks(features)
    dnf = add_dnf_probability(features)[["session_id", "driver_id", "dnf_prob"]]

    # Calibration / held-out races: the protocol's split (evaluate/splits.py).
    calibrate_ids, test_ids = split_races(features)

    for stage, anchor in STAGES.items():
        columns = STAGE_FEATURES[stage]

        def trainer(history, columns=columns):
            model = train_model(history, WEIGHT_2026, columns=columns)
            return lambda race: predict_positions(model, race)

        print(f"=== {stage} (anchor: {anchor}) ===")
        predicted = walk_forward(features, calibrate_ids + test_ids, trainer)
        predicted = {s: r.merge(dnf, on=["session_id", "driver_id"], how="left") for s, r in predicted.items()}
        calibrate = {s: predicted[s] for s in calibrate_ids}
        test = {s: predicted[s] for s in test_ids}

        scores = {(w, sd): total_brier(simulate_all(calibrate, anchor, w, sd))
                  for w in ANCHOR_WEIGHT_CANDIDATES for sd in NOISE_CANDIDATES}
        best_w, best_sd = min(scores, key=scores.get)
        settings.save(["probabilities", stage],
                      {"anchor_column": anchor, "anchor_weight": best_w, "noise_sd": best_sd})
        print(f"  calibrated: anchor weight {best_w}, noise_sd {best_sd} "
              f"(win + podium Brier {scores[(best_w, best_sd)]:.5f})")

        slot = pd.concat([grid_probabilities(features[features.race_date < r.race_date.iloc[0]], r, anchor)
                          for r in test.values()], ignore_index=True)
        rows = {
            "slot baseline": slot,
            "anchor + sim": simulate_all(test, anchor, 1.0, best_sd),
            "model": simulate_all(test, anchor, best_w, best_sd),
        }
        print(f"  TEST ({len(test_ids)} races)   {'win Brier':>10} {'podium Brier':>13}")
        for name, r in rows.items():
            s = score_probabilities(r)
            print(f"    {name:<16} {s['brier_win']:>10.5f} {s['brier_podium']:>13.5f}")
        base, model = score_probabilities(slot), score_probabilities(rows["model"])
        print(f"    model vs slot baseline: win {(base['brier_win'] - model['brier_win']) / base['brier_win']:+.1%}, "
              f"podium {(base['brier_podium'] - model['brier_podium']) / base['brier_podium']:+.1%}\n")


if __name__ == "__main__":
    main()
