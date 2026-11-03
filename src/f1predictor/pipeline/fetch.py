"""Fetching for the weekend pipeline: do the planner's fetch actions, nothing more.

The planner (pipeline/plan.py) decides WHAT is ready to fetch; this module
fetches exactly that, using the same loaders as the backfill:

    fetch_session FP1/FP2/.../R  -> ingest/openf1.ingest_season, only those
                                    sessions of that round
    fetch_session grid           -> the same, for 'Q': its laps are already
                                    in, so only the starting grid is (re)tried
    fetch_jolpica ...            -> ingest/jolpica.ingest_season, once per season
                                    (it refreshes schedule, results, sprint and
                                    qualifying together; ~15 requests). Done
                                    FIRST, so OpenF1's race laps can be
                                    validated against the official result.

What makes it safe to run unattended (the PRD's "late or broken data"):

    retry with backoff   inside each request (ingest/http.JsonHttpClient):
                         timeouts, 429s and 5xx errors are retried 5 times with
                         doubling waits before a request counts as failed.
    validate first       every session and every Jolpica part is checked
                         before it's committed (validate/checks.py); a bad
                         batch is rolled back, never half-saved.
    alert and skip       a failure is recorded in ingestion_log ('failed' +
                         the error) and the run carries on. The next tick's
                         planner sees the item still isn't 'ok' and tries
                         again, until its give-up time (plan.GIVE_UP), after
                         which it's reported as 'gave_up' instead.

Nothing here raises for a data problem: it returns {item_key: status} so the
tick can report what happened.
"""

import logging
import sqlite3
from datetime import datetime

from f1predictor import config
from f1predictor.ingest import jolpica, openf1
from f1predictor.ingest.backfill import rematch_unmatched_entries
from f1predictor.ingest.http import JsonHttpClient
from f1predictor.pipeline.plan import Action

logger = logging.getLogger(__name__)


def make_clients() -> tuple[JsonHttpClient, JsonHttpClient]:
    """(jolpica_client, openf1_client), each with its API's rate limit (config.py)."""
    return (JsonHttpClient(min_interval_s=config.JOLPICA_MIN_INTERVAL_S),
            JsonHttpClient(min_interval_s=config.OPENF1_MIN_INTERVAL_S))


def run_fetches(conn: sqlite3.Connection, actions: list[Action], now: datetime,
                jolpica_client: JsonHttpClient, openf1_client: JsonHttpClient) -> dict[str, str]:
    """Do every fetch action in `actions` (other kinds are ignored).

    Returns {item_key: status} from the loaders, e.g.
    {'2026-20-FP1': 'ok', '2026-20-grid': 'skipped', '2026-results': 'failed'}.
    """
    outcomes: dict[str, str] = {}

    # --- Jolpica first: one refresh per season ---------------------------------
    # Before OpenF1 on purpose: OpenF1's race and sprint laps are validated
    # against Jolpica's official result (laps per driver, a driver missing
    # entirely; validate/checks.py), and those checks only run if the result
    # is already in. On race night both become ready in the same tick, so
    # with OpenF1 first the race laps would be saved unchecked.
    seasons = sorted({a.season for a in actions if a.kind == "fetch_jolpica"})
    for season in seasons:
        logger.info("Jolpica %d: refreshing schedule, results, sprint, qualifying", season)
        outcomes.update(jolpica.ingest_season(conn, jolpica_client, season))

    # --- OpenF1: one call per round, with just the sessions that are ready ---
    by_round: dict[tuple[int, int], set[str]] = {}
    for action in actions:
        if action.kind == "fetch_session":
            # The grid hangs off the qualifying session in the loader.
            session_type = "Q" if action.target == "grid" else action.target
            by_round.setdefault((action.season, action.round), set()).add(session_type)
    for (season, rnd), session_types in sorted(by_round.items()):
        logger.info("OpenF1 %d round %d: %s", season, rnd, ", ".join(sorted(session_types)))
        result = openf1.ingest_season(conn, openf1_client, season, rounds={rnd}, now=now,
                                      session_types=session_types, settle_hours=0)
        outcomes.update(result)

    # New Jolpica results can identify cars OpenF1 listed only by number
    # (e.g. a debut driver): re-run the matching. Cheap, no network.
    if seasons or by_round:
        newly = rematch_unmatched_entries(conn)
        if newly:
            logger.info("Re-matching identified %d more car(s)", newly)

    return outcomes
