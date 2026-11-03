"""Backfill: load many seasons from Jolpica and OpenF1, resumably.

The order matters, season by season:
    1. Jolpica: calendar, results, qualifying. Creates events, sessions,
       drivers and teams.
    2. OpenF1 (2023+): lap-level data. Matches its sessions to the events from
       step 1, and its cars to the drivers in step 1's race results.
Then, once at the end:
    3. Re-match any session that still has unmatched drivers, in case newer
       Jolpica results now identify them.

Resumability (the whole point of this module):
    * OpenF1 sessions already logged 'ok' are skipped.
    * A Jolpica season is skipped only when all its parts are 'ok' AND were
      fetched after the season ended. So finished seasons are fetched once,
      while the current season is refreshed every run.
    * Each item commits on its own, so an interrupted run loses at most the
      item in progress. Re-running continues from there.
"""

import logging
import sqlite3
from collections import Counter
from datetime import date, datetime, timedelta

from f1predictor import config
from f1predictor.ingest import jolpica, openf1
from f1predictor.ingest.http import JsonHttpClient

logger = logging.getLogger(__name__)

# Jolpica can lag a few hours (occasionally longer) behind a race, so a
# season only counts as fully fetched if the fetch happened at least this
# long after its last race.
JOLPICA_LAG_DAYS = 2

JOLPICA_PARTS = ("schedule", "results", "sprint", "qualifying")


def jolpica_season_complete(conn: sqlite3.Connection, season: int) -> bool:
    """True if this season's Jolpica data is final and already loaded.

    Requires both:
      1. every part logged 'ok', and
      2. every one of those fetches happened >= JOLPICA_LAG_DAYS after the
         season's last race, i.e. after all results could have been published.
    """
    last_race = conn.execute(
        "SELECT MAX(race_date) FROM events WHERE season = ?", (season,)
    ).fetchone()[0]
    if last_race is None:
        return False  # never loaded

    # The earliest moment a fetch counts as "after the season was final".
    final_after = date.fromisoformat(last_race) + timedelta(days=JOLPICA_LAG_DAYS)

    for part in JOLPICA_PARTS:
        row = conn.execute(
            "SELECT status, ingested_at FROM ingestion_log WHERE source = 'jolpica' AND item_key = ?",
            (f"{season}-{part}",),
        ).fetchone()
        if row is None or row["status"] != "ok":
            return False
        # ingested_at is '2026-11-02T19:30:00+00:00'; [:10] is the date part.
        if date.fromisoformat(row["ingested_at"][:10]) < final_after:
            return False
    return True


def rematch_unmatched_entries(conn: sqlite3.Connection) -> int:
    """Re-run driver/team matching for every session that has unmatched cars.

    Returns how many cars gained a driver_id. Cheap: no network, just SQL.
    """
    session_ids = [row["session_id"] for row in conn.execute(
        "SELECT DISTINCT session_id FROM session_entries WHERE driver_id IS NULL")]

    before = conn.execute(
        "SELECT COUNT(*) FROM session_entries WHERE driver_id IS NULL").fetchone()[0]
    with conn:
        for session_id in session_ids:
            openf1.match_entries(conn, session_id)
    after = conn.execute(
        "SELECT COUNT(*) FROM session_entries WHERE driver_id IS NULL").fetchone()[0]
    return before - after


def backfill(
    conn: sqlite3.Connection,
    seasons: list[int],
    jolpica_client: JsonHttpClient,
    openf1_client: JsonHttpClient,
    include_openf1: bool = True,
    force: bool = False,
    now: datetime | None = None,
) -> Counter:
    """Load every season in `seasons`. Returns a tally of outcomes.

    Args:
        seasons: e.g. [2018, ..., 2026].
        jolpica_client / openf1_client: separate clients, because each API
            has its own rate limit (see config.py).
        include_openf1: False = Jolpica only (fast; useful for testing).
        force: Re-fetch everything, ignoring what's already logged 'ok'.
        now: Current time (tests pass a fixed one).

    Returns:
        Counter of (source, outcome), e.g. {('openf1', 'ok'): 450, ...}.
    """
    tally: Counter = Counter()

    for season in seasons:
        logger.info("===== Season %d =====", season)

        # --- 1. Jolpica ---
        if not force and jolpica_season_complete(conn, season):
            logger.info("Jolpica %d already complete, skipping", season)
            tally[("jolpica", "already done")] += len(JOLPICA_PARTS)
        else:
            for status in jolpica.ingest_season(conn, jolpica_client, season).values():
                tally[("jolpica", status)] += 1

        # --- 2. OpenF1 (only seasons it covers) ---
        if include_openf1 and season >= config.OPENF1_FIRST_SEASON:
            outcomes = openf1.ingest_season(conn, openf1_client, season,
                                            skip_done=not force, now=now)
            for item_key, status in outcomes.items():
                if not item_key.endswith("-sessions"):   # count sessions, not the list fetch
                    tally[("openf1", status)] += 1

    # --- 3. Re-match drivers now that all results are loaded ---
    newly_matched = rematch_unmatched_entries(conn)
    if newly_matched:
        logger.info("Re-matching identified %d more car(s)", newly_matched)

    return tally
