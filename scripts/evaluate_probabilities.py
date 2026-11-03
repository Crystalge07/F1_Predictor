"""Calibrate the simulator's noise, then score win/podium probabilities.

Usage (from the repo root, with the virtual environment active):
    python scripts/evaluate_probabilities.py        (takes about a minute)

Pipeline for every race from 2024 on, each using only earlier races:
    1. position model trained on every earlier race -> expected_position
    2. DNF model (already history-only)              -> dnf_prob
    3. Monte Carlo, 10,000 races                     -> p_win, p_podium, ...

Same split philosophy as Phase 2: choose on some races, report on others.
    CALIBRATE  2024 + 2026 rounds 1-9   pick GRID_WEIGHT and NOISE_SD here
    TEST       2025 + 2026 rounds 10+   scored at the end

History (honest record): the first version simulated straight from the model
(no grid blend) and was scored on TEST once: 1.4% WORSE than the grid baseline
on both win and podium. The fix (the grid blend; see simulate/probabilities.py)
was then found using CALIBRATE races only, and TEST was scored a second time.

Compared on TEST, by Brier score (lower = better):
    grid baseline       P(win)/P(podium) = the historical rate for that grid slot
    grid + simulator    the simulator fed the grid only (grid_weight 1.0):
                        shows what the model adds on top of the grid
    model, flat DNF     our simulation, but everyone gets the same DNF chance
                        (shows what the DNF model adds on top)
    model               the full thing
"""

from f1predictor.db.connection import get_connection
from f1predictor.evaluate.walk_forward import walk_forward
from f1predictor.features.post_quali import build_post_quali_features
from f1predictor.models.baselines import grid_probabilities
from f1predictor.models.dnf import add_dnf_probability
from f1predictor.models.post_quali import WEIGHT_2026, predict_positions, train_model
from f1predictor.simulate.probabilities import race_probabilities, score_probabilities

import pandas as pd

NOISE_CANDIDATES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
GRID_WEIGHT_CANDIDATES = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
LAST_CALIBRATION_ROUND_2026 = 9


def position_trainer(history: pd.DataFrame):
    """Walk-forward trainer that returns expected positions (not just an order)."""
    model = train_model(history, WEIGHT_2026)
    return lambda race: predict_positions(model, race)


def simulate_all(races: dict[int, pd.DataFrame], noise_sd: float, grid_weight: float,
                 flat_dnf: float | None = None) -> pd.DataFrame:
    """Probabilities for every race, stacked into one table."""
    out = []
    for race in races.values():
        if flat_dnf is not None:
            race = race.assign(dnf_prob=flat_dnf)
        out.append(race_probabilities(race, noise_sd=noise_sd, grid_weight=grid_weight))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    conn = get_connection()
    try:
        features = build_post_quali_features(conn)
    finally:
        conn.close()

    # DNF probabilities are history-only by construction, so add them for all
    # rows at once. Keep only the columns we need, to merge onto predictions.
    dnf = add_dnf_probability(features)[["session_id", "driver_id", "dnf_prob"]]

    sessions = features[["session_id", "season", "round"]].drop_duplicates()
    early_2026 = (sessions.season == 2026) & (sessions["round"] <= LAST_CALIBRATION_ROUND_2026)
    calibrate_ids = sessions[(sessions.season == 2024) | early_2026].session_id.tolist()
    test_ids = sessions[(sessions.season == 2025) | ((sessions.season == 2026) & ~early_2026)].session_id.tolist()

    # Step 1 (the slow part): one walk-forward model per race.
    print(f"Training {len(calibrate_ids) + len(test_ids)} walk-forward models...")
    predicted = walk_forward(features, calibrate_ids + test_ids, position_trainer)
    predicted = {s: race.merge(dnf, on=["session_id", "driver_id"], how="left")
                 for s, race in predicted.items()}
    calibrate = {s: predicted[s] for s in calibrate_ids}
    test = {s: predicted[s] for s in test_ids}

    # Step 2: pick grid weight + noise together on CALIBRATE. Score = win +
    # podium Brier, so neither outcome is ignored.
    print(f"\nCalibrating on 2024 + 2026 rounds 1-{LAST_CALIBRATION_ROUND_2026} "
          f"({len(calibrate_ids)} races). Best 5 of {len(GRID_WEIGHT_CANDIDATES) * len(NOISE_CANDIDATES)}:")
    totals = {}
    for gw in GRID_WEIGHT_CANDIDATES:
        for sd in NOISE_CANDIDATES:
            s = score_probabilities(simulate_all(calibrate, sd, gw))
            totals[(gw, sd)] = s["brier_win"] + s["brier_podium"]
    for (gw, sd), total in sorted(totals.items(), key=lambda kv: kv[1])[:5]:
        print(f"  grid_weight {gw:<4}  noise_sd {sd:<4}  ->  win + podium Brier {total:.5f}")
    best_gw, best_sd = min(totals, key=totals.get)
    print(f"  -> chosen: grid_weight {best_gw}, noise_sd {best_sd}")

    # Step 3: everything on TEST, once.
    grid = pd.concat([grid_probabilities(features[features.race_date < race.race_date.iloc[0]], race)
                      for race in test.values()], ignore_index=True)
    model = simulate_all(test, best_sd, best_gw)
    grid_sim = simulate_all(test, best_sd, grid_weight=1.0)
    # Flat DNF = the average DNF chance the model handed out over the test races.
    flat = simulate_all(test, best_sd, best_gw, flat_dnf=float(pd.concat(test.values()).dnf_prob.mean()))

    print(f"\nTEST: 2025 + 2026 rounds {LAST_CALIBRATION_ROUND_2026 + 1}+ ({len(test_ids)} races)")
    print(f"  {'':<18} {'win Brier':>10} {'podium Brier':>13}")
    base = score_probabilities(grid)
    for name, rows in (("grid baseline", grid), ("grid + simulator", grid_sim),
                       ("model, flat DNF", flat), ("model", model)):
        s = score_probabilities(rows)
        print(f"  {name:<18} {s['brier_win']:>10.5f} {s['brier_podium']:>13.5f}")
    s = score_probabilities(model)
    print(f"\n  model vs grid: win {(base['brier_win'] - s['brier_win']) / base['brier_win']:+.1%}, "
          f"podium {(base['brier_podium'] - s['brier_podium']) / base['brier_podium']:+.1%} "
          f"(positive = better than the grid)")

    # Calibration check: when the model says ~X%, does it happen ~X% of the time?
    model["won"] = (model["actual_position"] == 1).astype(int)
    model["bucket"] = pd.cut(model["p_win"], [0, 0.05, 0.15, 0.3, 0.5, 1.0], include_lowest=True)
    table = model.groupby("bucket", observed=True).agg(drivers=("won", "size"),
                                                       predicted=("p_win", "mean"),
                                                       actual=("won", "mean"))
    print("\nWin calibration on TEST (predicted chance vs how often it happened):")
    for bucket, row in table.iterrows():
        print(f"  {str(bucket):<14} {int(row.drivers):>4} drivers   "
              f"predicted {row.predicted:>5.1%}   actual {row.actual:>5.1%}")


if __name__ == "__main__":
    main()
