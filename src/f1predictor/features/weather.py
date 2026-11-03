"""Race weather: how wet each race was.

Source: OpenF1's weather samples (about one a minute, 2023+), stored in the
`weather` table. `rainfall` says whether it was raining at that moment.

rain_fraction = share of the race's weather samples where it was raining.
0 = dry all race, 0.35 = raining about a third of the time.

IMPORTANT: this is what ACTUALLY happened, so it's only known after the race.
Using it to predict that same race would be leakage. It has two honest uses:
    1. Measuring how much a perfect rain forecast could help (an upper bound),
       which tells us whether wiring up a real forecast is worth it.
    2. Later, as the training label for "was this race wet" when a forecast
       source (OpenWeatherMap, Phase 5) provides the prediction-time input.
"""

import sqlite3

import pandas as pd

# A race counts as wet if it rained in at least this share of samples
# (about 3 laps' worth of a 60-lap race). Below it: a passing shower.
WET_THRESHOLD = 0.05


def race_rain(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per race session with weather data: session_id, rain_fraction, wet (0/1)."""
    query = """
        SELECT w.session_id, AVG(w.rainfall) AS rain_fraction
        FROM weather w
        JOIN sessions s ON s.session_id = w.session_id
        WHERE s.session_type = 'R'
        GROUP BY w.session_id
    """
    df = pd.read_sql_query(query, conn)
    df["wet"] = (df["rain_fraction"] >= WET_THRESHOLD).astype(int)
    return df
