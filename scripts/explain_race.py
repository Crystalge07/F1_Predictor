"""Print why the model predicted what it did for one race.

Usage (from the repo root, with the virtual environment active):
    python scripts/explain_race.py 2026 19                    (post-quali stage)
    python scripts/explain_race.py 2026 19 --stage practice
    python scripts/explain_race.py 2026 19 --stage pre_weekend --top 8

Trains the stage's model on every race BEFORE the chosen one (exactly what it
would have known that weekend), predicts the race, and prints each predicted
top-N driver's biggest reasons, with where they really finished.
"""

import argparse

from f1predictor.db.connection import get_connection
from f1predictor.explain.contributions import explain_race, top_reasons
from f1predictor.features.race_features import build_race_features
from f1predictor.features.stages import STAGE_FEATURES
from f1predictor.models.position import WEIGHT_2026, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("season", type=int)
    parser.add_argument("round", type=int)
    parser.add_argument("--stage", choices=sorted(STAGE_FEATURES), default="post_quali")
    parser.add_argument("--top", type=int, default=5, help="how many predicted front-runners to explain")
    args = parser.parse_args()

    conn = get_connection()
    try:
        features = build_race_features(conn)
    finally:
        conn.close()

    race = features[(features.season == args.season) & (features["round"] == args.round)]
    if race.empty:
        raise SystemExit(f"No race with results for {args.season} round {args.round}")
    history = features[features.race_date < race.race_date.iloc[0]]

    model = train_model(history, WEIGHT_2026, columns=STAGE_FEATURES[args.stage])
    explanation = explain_race(model, race)

    # One row per driver, best predicted first.
    drivers = (explanation.drop_duplicates("driver_id")
                          .merge(race[["driver_id", "actual_position"]], on="driver_id")
                          .sort_values("predicted"))
    print(f"{args.season} round {args.round}, stage: {args.stage}. "
          f"Model baseline (a typical driver): {drivers.baseline.iloc[0]:.1f}\n")
    for rank, row in enumerate(drivers.head(args.top).itertuples(), start=1):
        print(f"P{rank} predicted: {row.driver_id}  (model {row.predicted:.1f}, "
              f"actually finished P{row.actual_position})")
        for line in top_reasons(explanation, row.driver_id):
            print(f"    {line}")


if __name__ == "__main__":
    main()
