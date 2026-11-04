"""Calibration: when the model says 20%, does it happen about 20% of the time?

Brier score (evaluate/scorer.py) mixes two things: how well the model
separates likely from unlikely drivers, and whether its numbers mean what
they say. Calibration looks at the second one alone:

    1. put every (driver, race) prediction in a bucket by its probability,
       e.g. "p_win between 5% and 10%"
    2. per bucket, compare the AVERAGE predicted probability with the share
       of those drivers who actually won

Well calibrated: the two columns match. Predicted > actual in the top buckets
= overconfident favourites; predicted < actual = underconfident.

With ~20 drivers per race, a season has only ~24 winners, so buckets are
small and the "actual" column is noisy. Each row carries its count, and a
rough 95% range for the actual rate (Wilson interval), so a gap can be
judged against luck.
"""

import math

import pandas as pd

WIN_EDGES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0]
PODIUM_EDGES = [0.0, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0]


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% range for a rate seen as hits / n (Wilson score interval).

    Better than "rate +/- 2 standard errors" for small n and rates near 0 or 1,
    which is exactly what the top and bottom buckets are.
    """
    if n == 0:
        return (math.nan, math.nan)
    rate = hits / n
    centre = (rate + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def calibration_table(prob: pd.Series, happened: pd.Series, edges: list[float]) -> pd.DataFrame:
    """One row per probability bucket: n, mean predicted, actual rate, 95% range.

    prob:      predicted probabilities, 0..1
    happened:  1/0 (or True/False), same index as prob
    edges:     bucket boundaries; a probability p falls in [edge_i, edge_i+1),
               except the last bucket, which includes 1.0
    """
    frame = pd.DataFrame({"prob": prob.astype(float), "hit": happened.astype(int)})
    # right=False: [a, b) buckets; include_lowest + the last edge nudged up a
    # hair puts exactly-1.0 predictions in the top bucket.
    bins = list(edges[:-1]) + [edges[-1] + 1e-9]
    frame["bucket"] = pd.cut(frame["prob"], bins=bins, right=False, include_lowest=True)
    rows = []
    for bucket, group in frame.groupby("bucket", observed=True):
        n, hits = len(group), int(group["hit"].sum())
        low, high = wilson_interval(hits, n)
        rows.append({
            "bucket": f"{bucket.left:.0%}-{min(bucket.right, 1.0):.0%}",
            "n": n,
            "predicted": group["prob"].mean(),
            "actual": hits / n,
            "actual_low": low,
            "actual_high": high,
            # Is the average prediction inside the range luck allows?
            "within_range": low <= group["prob"].mean() <= high,
        })
    return pd.DataFrame(rows)
