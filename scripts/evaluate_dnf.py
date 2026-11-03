"""Choose the DNF model's settings, then check them on races it never saw.

Usage (from the repo root, with the virtual environment active):
    python scripts/evaluate_dnf.py

Every probability only uses races before the one it's for, so no retraining
loop is needed: we compute probabilities for every race once per setting, then
score them with the Brier score (lower = better).

    CHOOSE  2019-2023   pick prior_strength and driver_weight here
    TEST    2024-2026   untouched until the end
(2018 is skipped in both: it has no history to learn from yet.)

Baselines to beat:
    field rate only   everyone gets the recent field-wide DNF rate
    constant          everyone gets the overall DNF rate of the CHOOSE seasons
"""

from itertools import product

from f1predictor.db.connection import get_connection
from f1predictor.features.post_quali import build_post_quali_features
from f1predictor.models.dnf import add_dnf_probability, brier

PRIOR_STRENGTHS = [5, 10, 20, 40, 80, 160]
DRIVER_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]   # 0 = team only, 1 = driver only


def main() -> None:
    conn = get_connection()
    try:
        features = build_post_quali_features(conn)
    finally:
        conn.close()

    # Score every combination on CHOOSE.
    results = []
    for k, w in product(PRIOR_STRENGTHS, DRIVER_WEIGHTS):
        df = add_dnf_probability(features, prior_strength=k, driver_weight=w)
        on = df[df["season"].between(2019, 2023)]
        results.append((brier(on["dnf_prob"], on["dnf"]), k, w))
    results.sort()

    print("Best settings on 2019-2023 (Brier, lower = better):")
    for score, k, w in results[:5]:
        print(f"  prior_strength {k:>3}  driver_weight {w:.2f}  ->  {score:.5f}")
    _, best_k, best_w = results[0]

    # Baselines and the chosen model, on CHOOSE and TEST.
    df = add_dnf_probability(features, prior_strength=best_k, driver_weight=best_w)
    choose = df["season"].between(2019, 2023)
    test = df["season"] >= 2024
    constant = df.loc[choose, "dnf"].mean()

    print(f"\n{'':<26} {'2019-2023':>10} {'2024-2026':>10}")
    for name, column in (("constant rate", None), ("field rate only", "field_dnf_rate"),
                         ("team only (shrunk)", "team_dnf_rate"),
                         ("driver only (shrunk)", "driver_dnf_rate"),
                         (f"chosen (k={best_k}, w={best_w})", "dnf_prob")):
        row = []
        for mask in (choose, test):
            part = df[mask]
            prob = constant if column is None else part[column]
            row.append(brier(prob, part["dnf"]))
        print(f"{name:<26} {row[0]:>10.5f} {row[1]:>10.5f}")

    print(f"\nActual DNF share: 2019-2023 {df.loc[choose, 'dnf'].mean():.1%}, "
          f"2024-2026 {df.loc[test, 'dnf'].mean():.1%}")


if __name__ == "__main__":
    main()
