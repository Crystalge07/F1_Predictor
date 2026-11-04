"""Which races chose the settings and which test them: the split, in ONE place.

docs/evaluation-protocol.md, "Which races were used for what":

    2018-2023                  training history only
    2024 + 2026 rounds 1-9     CALIBRATION: every tuned setting was chosen here
    2025 + 2026 rounds 10+     HELD OUT: reported, never used to choose

Before this module (architecture review, 2026-11-03) the same split was
typed into four scripts (LAST_CALIBRATION_ROUND_2026 = 9 three times,
LAST_TUNING_ROUND = 9 once, and the season numbers inline), so changing it
meant finding every copy. Every evaluation script now asks here.

The split is FROZEN with the settings: it describes what was done before the
2027 live test, it isn't a knob.
"""

import pandas as pd

CALIBRATION_SEASONS = (2024,)    # whole seasons used to choose settings
HELD_OUT_SEASONS = (2025,)       # whole seasons kept for reporting
SPLIT_SEASON = 2026              # the season split part-way through...
LAST_CALIBRATION_ROUND = 9       # ...rounds up to here calibrate, later ones are held out


def is_calibration(season: int, rnd: int) -> bool:
    """Did this race help choose the frozen settings?"""
    return season in CALIBRATION_SEASONS or (season == SPLIT_SEASON and rnd <= LAST_CALIBRATION_ROUND)


def is_held_out(season: int, rnd: int) -> bool:
    """Is this race one of the held-out test races?"""
    return season in HELD_OUT_SEASONS or (season == SPLIT_SEASON and rnd > LAST_CALIBRATION_ROUND)


def split_races(races: pd.DataFrame) -> tuple[list[int], list[int]]:
    """(calibration session_ids, held-out session_ids) from rows with
    session_id, season and round (one row per race, or many: duplicates are
    dropped). Races in neither group (2018-2023, 2027+) are in neither list."""
    races = races[["session_id", "season", "round"]].drop_duplicates()
    calibration = [s for s, season, rnd in races.itertuples(index=False) if is_calibration(season, rnd)]
    held_out = [s for s, season, rnd in races.itertuples(index=False) if is_held_out(season, rnd)]
    return calibration, held_out


def describe() -> tuple[str, str]:
    """Readable names of the two groups, for printed reports:
    ('2024 + 2026 rounds 1-9', '2025 + 2026 rounds 10+')."""
    calibration = " + ".join([str(s) for s in CALIBRATION_SEASONS]
                             + [f"{SPLIT_SEASON} rounds 1-{LAST_CALIBRATION_ROUND}"])
    held_out = " + ".join([str(s) for s in HELD_OUT_SEASONS]
                          + [f"{SPLIT_SEASON} rounds {LAST_CALIBRATION_ROUND + 1}+"])
    return calibration, held_out
