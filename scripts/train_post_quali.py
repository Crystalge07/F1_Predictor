"""Train the post-qualifying XGBoost model, tune the 2026 weight, check the Phase 2 gate.

Usage (from the repo root, with the virtual environment active):
    python scripts/train_post_quali.py

The races are split into three groups, so that no number we report was used
to make a choice (that would flatter the model):

    TUNE   2026 rounds 1-9    pick the 2026 weight multiplier here
    GATE   2026 rounds 10+    untouched until the very end: the held-out test
    CHECK  2025 (all rounds)  a second, independent check on a whole season of
                              the old regulations (the 2026 weight can't
                              affect it: there are no 2026 rows before 2025 races)

Every race is predicted walk-forward (trained only on the races before it),
and scored against the grid-order baseline on exactly the same races.

Finally, the chosen model is retrained on ALL races and saved to
data/models/post_quali.json for the next phases.
"""

from f1predictor import config, settings
from f1predictor.db.connection import get_connection
from f1predictor.evaluate.metrics import average_scores, score_race
from f1predictor.evaluate.walk_forward import walk_forward
from f1predictor.features.race_features import POST_QUALI_FEATURES, build_race_features
from f1predictor.models.baselines import actual_result, grid_order
from f1predictor.models.position import predict_order, train_model

# Candidate 2026 multipliers. 1 = no extra weight (2026 counts like any season).
WEIGHT_CANDIDATES = [1, 2, 3, 5, 8, 12]
LAST_TUNING_ROUND = 9


def xgb_trainer(weight_2026: float):
    """A walk-forward trainer: fit on the history, return a race -> order function."""
    def trainer(history):
        model = train_model(history, weight_2026)
        return lambda race: predict_order(model, race)
    return trainer


def score(predictions: dict[int, list[str]], actuals: dict[int, dict[str, int]]) -> dict[str, float]:
    """Average the four metrics over a set of races."""
    return average_scores([score_race(predictions[s], actuals[s]) for s in predictions])


def print_comparison(label: str, model: dict, baseline: dict, n_races: int) -> None:
    print(f"\n{label}  ({n_races} races)")
    print(f"  {'':<10} {'pos MAE':>8} {'spearman':>9} {'winner':>7} {'podium':>7}")
    for name, s in (("grid", baseline), ("xgboost", model)):
        print(f"  {name:<10} {s['position_mae']:>8.2f} {s['spearman']:>9.3f} "
              f"{s['winner_hit']:>7.0%} {s['podium_hits']:>7.0%}")
    # % improvement in MAE: positive = the model is off by fewer places.
    gain = (baseline["position_mae"] - model["position_mae"]) / baseline["position_mae"]
    print(f"  MAE improvement over grid: {gain:+.1%}")


def main() -> None:
    conn = get_connection()
    try:
        features = build_race_features(conn)
        # Race lists for each group, in date order.
        races = (features[["session_id", "season", "round", "race_date"]]
                 .drop_duplicates().sort_values("race_date"))
        tune = races[(races.season == 2026) & (races["round"] <= LAST_TUNING_ROUND)].session_id.tolist()
        gate = races[(races.season == 2026) & (races["round"] > LAST_TUNING_ROUND)].session_id.tolist()
        check = races[races.season == 2025].session_id.tolist()

        # Real results and the baseline's predictions, from the same functions
        # that produced the Phase 2 baseline numbers.
        all_test = tune + gate + check
        actuals = {s: actual_result(conn, s) for s in all_test}
        baseline = {s: grid_order(conn, s) for s in all_test}
    finally:
        conn.close()

    def baseline_on(sessions):
        return score({s: baseline[s] for s in sessions}, actuals)

    # --- 1. Tune the 2026 weight on TUNE races only -----------------------------
    print(f"Tuning the 2026 weight on 2026 rounds 1-{LAST_TUNING_ROUND} "
          f"(grid baseline MAE {baseline_on(tune)['position_mae']:.2f})")
    results = {}
    for weight in WEIGHT_CANDIDATES:
        results[weight] = score(walk_forward(features, tune, xgb_trainer(weight)), actuals)
        print(f"  weight {weight:>2}:  MAE {results[weight]['position_mae']:.3f}  "
              f"spearman {results[weight]['spearman']:.3f}")
    # Lowest MAE wins; on an exact tie, min() keeps the first (smallest) weight.
    best_weight = min(WEIGHT_CANDIDATES, key=lambda w: results[w]["position_mae"])
    print(f"  -> chosen 2026 weight: {best_weight}")
    settings.save(["weight_2026"], float(best_weight))
    print(f"     saved to tuned_settings.json (review the change with `git diff src/f1predictor/tuned_settings.json`)")

    # --- 2. The gate: held-out races, scored once -------------------------------
    trainer = xgb_trainer(best_weight)
    gate_model = score(walk_forward(features, gate, trainer), actuals)
    check_model = score(walk_forward(features, check, trainer), actuals)
    print_comparison(f"GATE: 2026 rounds {LAST_TUNING_ROUND + 1}+", gate_model, baseline_on(gate), len(gate))
    print_comparison("CHECK: 2025 season", check_model, baseline_on(check), len(check))

    # The original gate ("beat the grid on MAE on both held-out sets") was not
    # met, and was redefined on 2026-11-02 (see docs/PRD.md, Decisions made):
    # the post-quali model only has to roughly match the grid, and the real
    # bar moves to Phase 3's probabilities. So we report, rather than pass/fail.
    for label, model, sessions in (("2026 held-out", gate_model, gate), ("2025", check_model, check)):
        diff = baseline_on(sessions)["position_mae"] - model["position_mae"]
        verdict = "ahead of" if diff > 0 else "behind"
        print(f"{label}: model is {abs(diff):.2f} places per driver {verdict} the grid")

    # --- 3. Final model on everything, saved for later phases -------------------
    final = train_model(features, best_weight)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / "post_quali.json"
    final.save_model(path)
    print(f"\nSaved final model (2026 weight {best_weight}) to {path}")

    # Which features the model leans on most ("gain" = how much a feature's
    # splits improved the predictions, as a share of the total).
    importance = final.get_booster().get_score(importance_type="gain")
    total = sum(importance.values())
    print("Feature importance (share of total gain):")
    for feature in sorted(POST_QUALI_FEATURES, key=lambda f: -importance.get(f, 0)):
        print(f"  {feature:<22} {importance.get(feature, 0) / total:>6.1%}")


if __name__ == "__main__":
    main()
