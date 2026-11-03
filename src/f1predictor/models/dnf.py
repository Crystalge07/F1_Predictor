"""DNF probabilities: each driver's chance of not finishing, before the race.

The position model (models/position.py) only predicts where a driver
finishes IF they finish. This module supplies the other half: how likely they
are not to finish at all. The Monte Carlo simulator combines the two.

A DNF here = not classified (position_text isn't a number: retired,
disqualified, withdrawn, ...). Same definition as the position model, which
treats exactly these rows as "no target". In the feature table that's simply
`finish_position` being NaN.

The estimate blends three rates, all from races BEFORE this one:

    field rate   share of all cars that failed to finish, over the last
                 FIELD_WINDOW races. The background level (it changes over the
                 years: reliability improves, then a new rulebook resets it).
    team rate    the team's cars, over their last TEAM_WINDOW races.
                 Mostly mechanical reliability.
    driver rate  the driver, over their last DRIVER_WINDOW races.
                 Mostly crashes and mistakes.

Shrinkage (the important idea). A rookie who crashed in 1 of their first 2
races does not have a 50% DNF chance; we've just seen too little. So each rate
is pulled towards the field rate as if the driver/team had `prior_strength`
extra races at exactly the field rate:

    shrunk = (their DNFs + prior_strength * field_rate) / (their races + prior_strength)

    2 races, 1 DNF, field 10%, prior_strength 10 -> (1 + 1) / (2 + 10) = 16.7%
    40 races, 20 DNFs (truly unreliable)         -> (20 + 1) / (40 + 10) = 42%

Few races -> close to the field rate. Many races -> close to their own record.
No history at all -> exactly the field rate.

Final probability = driver_weight * driver_rate + (1 - driver_weight) * team_rate.

Result (scripts/evaluate_dnf.py): on 2024-2026, which played no part in
choosing the settings, this beats a flat "everyone gets the average rate" by
about 1% in Brier score. Small, because DNFs are mostly random; the value for
the simulation is that a fragile car or a crash-prone driver gets a somewhat
higher chance instead of everyone getting the same 13%.
prior_strength and driver_weight are chosen in scripts/evaluate_dnf.py.
"""

import pandas as pd

from f1predictor import settings

FIELD_WINDOW = 40      # races (all cars) for the background rate (~2 seasons)
TEAM_WINDOW = 10       # races per team (two cars each -> about 20 starts)
DRIVER_WINDOW = 20     # races per driver
# Fallback before any history exists (the very first race in the database).
# About the long-run share of non-finishers in modern F1.
DEFAULT_DNF_RATE = 0.10

# Chosen in scripts/evaluate_dnf.py on 2019-2023 (field window chosen the same
# way, in a one-off search over 20/40/80). The big prior strength is itself a
# finding: DNFs are mostly luck. A driver's or team's own record only nudges
# them away from the field rate (someone with 20 races of history keeps 20/100
# of their own rate and 80/100 of the field's). The best few settings scored
# within 0.0001 Brier of each other, so these are not finely tuned values.
# The values live in tuned_settings.json (written by scripts/evaluate_dnf.py).
PRIOR_STRENGTH: float = settings.load()["dnf"]["prior_strength"]
DRIVER_WEIGHT: float = settings.load()["dnf"]["driver_weight"]


def _past_rolling_sum(series: pd.Series, window: int) -> pd.Series:
    """Sum over the previous `window` rows, NOT including the current row.

    rolling(window).sum() includes the current row; shift(1) moves the result
    down one, so row i sees rows i-window .. i-1 only. min_periods=1 means
    "sum whatever exists", so early rows aren't NaN just for having < window rows.
    The very first row of each group has no past at all and becomes 0.
    """
    return series.rolling(window, min_periods=1).sum().shift(1).fillna(0)


def add_dnf_probability(features: pd.DataFrame,
                        prior_strength: float = PRIOR_STRENGTH,
                        driver_weight: float = DRIVER_WEIGHT) -> pd.DataFrame:
    """Return a copy of the feature table with dnf_prob (and its ingredients) added.

    Needs columns: session_id, race_date, driver_id, constructor_id,
    finish_position (NaN = DNF). Rows must cover races in date order.
    Adds: field_dnf_rate, team_dnf_rate, driver_dnf_rate, dnf_prob, and the
    outcome column `dnf` (1/0), handy for scoring.
    """
    df = features.sort_values(["race_date", "session_id", "driver_id"]).copy()
    df["dnf"] = df["finish_position"].isna().astype(int)

    # --- Field rate: one row per race first, then roll over races ------------
    races = (df.groupby(["race_date", "session_id"], as_index=False)
               .agg(dnfs=("dnf", "sum"), starts=("dnf", "count")))
    past_dnfs = _past_rolling_sum(races["dnfs"], FIELD_WINDOW)
    past_starts = _past_rolling_sum(races["starts"], FIELD_WINDOW)
    # where(past_starts > 0): before any history, use the default instead of 0/0.
    races["field_dnf_rate"] = (past_dnfs / past_starts).where(past_starts > 0, DEFAULT_DNF_RATE)
    df = df.merge(races[["session_id", "field_dnf_rate"]], on="session_id", how="left")

    def shrink(dnfs: pd.Series, starts: pd.Series) -> pd.Series:
        return (dnfs + prior_strength * df["field_dnf_rate"]) / (starts + prior_strength)

    # --- Team rate: one row per (race, team), roll over that team's races -----
    team = (df.groupby(["race_date", "session_id", "constructor_id"], as_index=False)
              .agg(dnfs=("dnf", "sum"), starts=("dnf", "count")))
    by_team = team.groupby("constructor_id")
    team["team_past_dnfs"] = by_team["dnfs"].transform(lambda s: _past_rolling_sum(s, TEAM_WINDOW))
    team["team_past_starts"] = by_team["starts"].transform(lambda s: _past_rolling_sum(s, TEAM_WINDOW))
    df = df.merge(team[["session_id", "constructor_id", "team_past_dnfs", "team_past_starts"]],
                  on=["session_id", "constructor_id"], how="left")
    df["team_dnf_rate"] = shrink(df["team_past_dnfs"], df["team_past_starts"])

    # --- Driver rate: each row is already one race for one driver -------------
    by_driver = df.groupby("driver_id")
    driver_dnfs = by_driver["dnf"].transform(lambda s: _past_rolling_sum(s, DRIVER_WINDOW))
    driver_starts = by_driver["dnf"].transform(
        lambda s: _past_rolling_sum(pd.Series(1, index=s.index), DRIVER_WINDOW))
    df["driver_dnf_rate"] = shrink(driver_dnfs, driver_starts)

    # --- Blend ----------------------------------------------------------------
    df["dnf_prob"] = driver_weight * df["driver_dnf_rate"] + (1 - driver_weight) * df["team_dnf_rate"]
    return df.drop(columns=["team_past_dnfs", "team_past_starts"])


def brier(prob: pd.Series, outcome: pd.Series) -> float:
    """Brier score: mean of (probability - what happened)^2. Lower is better.

    outcome is 1 if it happened, 0 if not. Saying 10% for something that
    didn't happen costs 0.01; saying 90% costs 0.81. Always saying 50% scores
    0.25, so anything useful must beat that (and beat the simple field rate).
    """
    return float(((prob - outcome) ** 2).mean())
