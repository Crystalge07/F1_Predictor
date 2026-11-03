"""Scoring a predicted finishing order against the real result.

Every model in the project (baselines, XGBoost, later stages) is scored with
these same functions, so their numbers are directly comparable. They're pure
functions: plain Python values in, numbers out. No database, no pandas.

The shapes used everywhere:
    predicted: list of driver_ids in predicted finishing order.
               predicted[0] is the predicted winner.
    actual:    dict {driver_id: actual finishing position}, positions 1..N.
               Retirements are included: they have a position too (ordered by
               laps completed), and leaving them out would flatter every model.

Metrics (per race), matching the PRD's scorecard:
    position_mae   on average, how many places off each driver was (lower = better)
    spearman       rank correlation: 1 = perfect order, 0 = random, -1 = reversed
    winner_hit     1 if our predicted winner won, else 0
    podium_hits    how many of our top 3 finished in the real top 3, divided by 3
Probability metrics (Brier score) come in Phase 3, when models output probabilities.
"""

from dataclasses import dataclass, asdict


@dataclass
class RaceScore:
    """The four metrics for one race."""
    position_mae: float
    spearman: float
    winner_hit: int
    podium_hits: float


def score_race(predicted: list[str], actual: dict[str, int]) -> RaceScore:
    """Score one race.

    Example:
        predicted = ['ver', 'nor', 'lec']
        actual    = {'nor': 1, 'ver': 2, 'lec': 3}
        -> position_mae 0.67 (two drivers 1 place off), winner_hit 0, podium_hits 1.0
    """
    # Both sides must be about the same drivers, or the comparison is
    # meaningless. Fail loudly rather than score half a grid.
    if set(predicted) != set(actual):
        missing = set(actual) - set(predicted)
        extra = set(predicted) - set(actual)
        raise ValueError(f"Predicted and actual drivers differ. Missing: {missing}, extra: {extra}")
    if len(predicted) != len(set(predicted)):
        raise ValueError("A driver appears twice in the predicted order")

    n = len(predicted)
    # Turn the predicted ORDER into predicted POSITIONS: index 0 -> position 1.
    # enumerate(..., start=1) yields (1, first_driver), (2, second_driver), ...
    predicted_position = {driver: pos for pos, driver in enumerate(predicted, start=1)}

    # Per-driver error: |predicted position - actual position|.
    errors = [abs(predicted_position[d] - actual[d]) for d in actual]
    position_mae = sum(errors) / n

    # Spearman's rank correlation. Because both sides are rankings 1..N with
    # no ties, it has a simple closed form:
    #     rho = 1 - 6 * sum(d^2) / (n * (n^2 - 1))
    # where d is each driver's position difference. Perfect order -> sum(d^2)
    # = 0 -> rho = 1. Fully reversed -> rho = -1.
    sum_d_squared = sum(e * e for e in errors)
    spearman = 1 - 6 * sum_d_squared / (n * (n * n - 1))

    # Who actually finished 1st / in the top 3.
    actual_winner = min(actual, key=actual.get)        # driver with the lowest position
    actual_podium = {d for d, pos in actual.items() if pos <= 3}

    winner_hit = int(predicted[0] == actual_winner)
    # Set intersection (&): drivers in both our top 3 and the real top 3.
    podium_hits = len(set(predicted[:3]) & actual_podium) / 3

    return RaceScore(position_mae, spearman, winner_hit, podium_hits)


def average_scores(scores: list[RaceScore]) -> dict[str, float]:
    """Average each metric over many races (each race counts equally).

    winner_hit averaged over races = winner hit RATE (e.g. 0.45 = 45% of races).
    """
    if not scores:
        raise ValueError("No races to average")
    # asdict turns a RaceScore into {'position_mae': ..., 'spearman': ..., ...}
    totals: dict[str, float] = {}
    for score in scores:
        for metric, value in asdict(score).items():
            totals[metric] = totals.get(metric, 0.0) + value
    return {metric: total / len(scores) for metric, total in totals.items()}
