"""Score the grid-order baseline on every race in the database.

Usage (from the repo root, with the virtual environment active):
    python scripts/evaluate_baseline.py

Prints one line per season plus an overall line. These are the numbers the
Phase 2 model must beat.
"""

from f1predictor.db.connection import get_connection
from f1predictor.evaluate.metrics import average_scores, score_race
from f1predictor.models.baselines import actual_result, grid_order, race_sessions


def print_row(label: str, n_races: int, averages: dict[str, float]) -> None:
    print(f"{label:<8} {n_races:>5}  {averages['position_mae']:>8.2f}  "
          f"{averages['spearman']:>8.3f}  {averages['winner_hit']:>9.0%}  "
          f"{averages['podium_hits']:>9.0%}")


def main() -> None:
    conn = get_connection()
    try:
        races = race_sessions(conn)
        # Score every race once; keep (season, score) so we can group by season.
        scored = [(race["season"], score_race(grid_order(conn, race["session_id"]),
                                              actual_result(conn, race["session_id"])))
                  for race in races]
    finally:
        conn.close()

    print("Grid-order baseline (predicted finish = starting grid)\n")
    print(f"{'season':<8} {'races':>5}  {'pos MAE':>8}  {'spearman':>8}  "
          f"{'winner hit':>9}  {'podium hit':>9}")

    for season in sorted({season for season, _ in scored}):
        season_scores = [score for s, score in scored if s == season]
        print_row(str(season), len(season_scores), average_scores(season_scores))

    all_scores = [score for _, score in scored]
    print_row("all", len(all_scores), average_scores(all_scores))


if __name__ == "__main__":
    main()
