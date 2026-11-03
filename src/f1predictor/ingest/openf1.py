"""Load lap-level data from the OpenF1 API (2023 onwards).

What OpenF1 gives us, per session (identified by OpenF1's `session_key`):
    /drivers       -> who drove: car number, acronym, name, team
    /laps          -> lap and sector times, speed traps, pit-out flag
    /stints        -> tyre compound and age for each stint
    /pit           -> pit stops: lap, time in pit lane, stationary time
    /weather       -> one sample per minute
    /race_control  -> flags, safety cars, penalties, deleted laps...
Plus /sessions?year=YYYY, the season's list of sessions, which we match to the
sessions Jolpica already created.

The module is laid out top to bottom as:
    1. FETCH:  calling the API (including OpenF1's "404 means empty" quirk)
    2. PARSE:  pure functions, OpenF1 values -> our column values
    3. MATCH:  OpenF1 sessions -> our sessions, and car numbers -> drivers/teams
    4. LOAD:   writing a session's data to SQLite
    5. ingest_season(): runs everything for one season, logging each session
"""

import logging
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Any, NamedTuple

import requests

from f1predictor import config
from f1predictor.db import ingestion_log
from f1predictor.ingest.http import JsonHttpClient

logger = logging.getLogger(__name__)

SOURCE = "openf1"  # the `source` value we write into ingestion_log

# OpenF1 session name -> our session code. Names not listed here (pre-season
# testing's "Day 1", "Day 2"...) are ignored.
# Note OpenF1 calls the 2023 Sprint Shootout "Sprint Qualifying"; both are 'SQ'.
SESSION_NAME_TO_TYPE: dict[str, str] = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Sprint Qualifying": "SQ",
    "Sprint Shootout": "SQ",
    "Sprint": "S",
    "Qualifying": "Q",
    "Race": "R",
}

# Only load a session once it started at least this long ago. A race lasts
# ~2 hours (longer with red flags), and OpenF1's free tier only serves data
# once a session is over. Exactly when to fetch is a Phase 5 (automation)
# question; this is a safe default for backfilling.
SESSION_SETTLE_HOURS = 3


class NoDataYet(Exception):
    """A session that should have happened has no laps (data not published yet)."""


# =============================================================================
# 1. FETCH
# =============================================================================

def fetch(client: JsonHttpClient, endpoint: str, **params: Any) -> list[dict]:
    """GET one OpenF1 endpoint and return its list of rows.

    Example: fetch(client, "laps", session_key=9662) -> [{...}, {...}, ...]

    OpenF1 quirk: a query with no matching rows returns HTTP 404 with
    {"detail": "No results found."} instead of an empty list. Our HTTP client
    treats 404 as a permanent error and raises, so we catch that one case
    and turn it into []. Every other error still propagates.
    """
    url = f"{config.OPENF1_BASE_URL}/{endpoint}"
    try:
        return client.get_json(url, params=params)
    except requests.HTTPError as exc:
        # exc.response is the HTTP response that caused the error.
        if exc.response is not None and exc.response.status_code == 404:
            return []
        raise


# =============================================================================
# 2. PARSE (pure functions: no network, no database)
# =============================================================================

def seconds_to_ms(seconds: float | None) -> int | None:
    """OpenF1 gives durations as float seconds (92.345); we store integer ms (92345)."""
    if seconds is None:
        return None
    # round(), not int(): 92.345 * 1000 may be 92344.99999 in floating point.
    return round(seconds * 1000)


def to_utc_iso(text: str | None, timespec: str = "milliseconds") -> str | None:
    """Normalize an OpenF1 timestamp to our format.

    '2024-12-08T13:09:54.663000+00:00' -> '2024-12-08T13:09:54.663Z'
    '2024-12-08T13:00:00+00:00' with timespec='seconds' -> '2024-12-08T13:00:00Z'

    Why bother: we compare timestamps as text (e.g. "weather samples during
    this lap"). Text comparison only works if every value has exactly the same
    format, so every timestamp in the database goes through one function.
    """
    if not text:
        return None
    # fromisoformat understands '+00:00' offsets and fractional seconds.
    parsed = datetime.fromisoformat(text)
    # Convert to UTC (a no-op for OpenF1, which already sends UTC, but it
    # makes the function correct for any offset).
    parsed_utc = parsed.astimezone(timezone.utc)
    # isoformat gives '...+00:00'; we use the shorter 'Z' (= UTC) like Jolpica.
    return parsed_utc.isoformat(timespec=timespec).replace("+00:00", "Z")


def entry_row(session_id: int, driver: dict) -> tuple:
    """One /drivers row -> one session_entries row (driver/team ids matched later)."""
    return (
        session_id,
        driver["driver_number"],
        driver.get("name_acronym"),
        driver.get("full_name"),
        driver.get("team_name"),
    )


def lap_row(session_id: int, lap: dict) -> tuple:
    return (
        session_id,
        lap["driver_number"],
        lap["lap_number"],
        to_utc_iso(lap.get("date_start")),
        seconds_to_ms(lap.get("lap_duration")),
        seconds_to_ms(lap.get("duration_sector_1")),
        seconds_to_ms(lap.get("duration_sector_2")),
        seconds_to_ms(lap.get("duration_sector_3")),
        lap.get("is_pit_out_lap"),
        lap.get("i1_speed"),
        lap.get("i2_speed"),
        lap.get("st_speed"),
    )


def stint_row(session_id: int, stint: dict) -> tuple:
    return (
        session_id,
        stint["driver_number"],
        stint["stint_number"],
        stint.get("compound"),
        stint.get("lap_start"),
        stint.get("lap_end"),
        stint.get("tyre_age_at_start"),
    )


def pit_row(session_id: int, pit: dict) -> tuple:
    # Newer data has `lane_duration` (entry to exit) and `stop_duration`
    # (stationary). Older data only has `pit_duration`, which is the lane time.
    lane_seconds = pit.get("lane_duration", pit.get("pit_duration"))
    return (
        session_id,
        pit["driver_number"],
        pit["lap_number"],
        to_utc_iso(pit.get("date")),
        seconds_to_ms(lane_seconds),
        seconds_to_ms(pit.get("stop_duration")),
    )


def weather_row(session_id: int, sample: dict) -> tuple:
    rainfall = sample.get("rainfall")
    return (
        session_id,
        to_utc_iso(sample["date"]),
        sample.get("air_temperature"),
        sample.get("track_temperature"),
        sample.get("humidity"),
        sample.get("pressure"),
        # OpenF1 sends 0/1; store a proper boolean (None stays None).
        None if rainfall is None else bool(rainfall),
        sample.get("wind_speed"),
        sample.get("wind_direction"),
    )


def race_control_row(session_id: int, msg: dict) -> tuple:
    return (
        session_id,
        to_utc_iso(msg["date"]),
        msg.get("category"),
        msg.get("flag"),
        msg.get("scope"),
        msg.get("sector"),
        msg.get("driver_number"),
        msg.get("lap_number"),
        msg["message"],
    )


# =============================================================================
# 3. MATCH
# =============================================================================

class MatchedSession(NamedTuple):
    """One OpenF1 session that we matched to one of our sessions."""
    session_id: int        # ours
    session_key: int       # OpenF1's
    round: int
    session_type: str
    start_time_utc: str
    item_key: str          # for ingestion_log, e.g. '2024-24-FP1'


def find_event_by_race_date(conn: sqlite3.Connection, season: int, race_day: date) -> sqlite3.Row | None:
    """Find our event whose race is within 1 day of `race_day`.

    Why "within 1 day" instead of equal: Jolpica's date and OpenF1's UTC
    timestamp can land on different calendar days for night races (Las Vegas
    starts Saturday evening local = Sunday morning UTC). Races are always at
    least a week apart, so a 1-day window can never pick the wrong event.
    """
    for event in conn.execute(
        "SELECT event_id, round, race_date FROM events WHERE season = ?", (season,)
    ):
        if abs(date.fromisoformat(event["race_date"]) - race_day) <= timedelta(days=1):
            return event
    return None


def match_sessions(conn: sqlite3.Connection, season: int, openf1_sessions: list[dict]) -> list[MatchedSession]:
    """Link a season's OpenF1 sessions to our sessions (creating any that are missing).

    OpenF1 groups sessions into "meetings" (a race weekend). For each meeting:
      1. Find its race. No race (pre-season testing) or a cancelled race
         (Imola 2023) means there's nothing to match, so skip the meeting.
      2. Find our event by race date.
      3. Upsert each session with its OpenF1 key and exact start time.
    """
    # Group sessions by meeting: {meeting_key: [session, session, ...]}
    meetings: dict[int, list[dict]] = {}
    for session in openf1_sessions:
        meetings.setdefault(session["meeting_key"], []).append(session)

    matched: list[MatchedSession] = []

    for meeting_sessions in meetings.values():
        # next(..., None) returns the first match, or None if there isn't one.
        race = next(
            (s for s in meeting_sessions
             if s["session_name"] == "Race" and not s.get("is_cancelled")),
            None,
        )
        if race is None:
            continue  # testing, or a cancelled weekend

        race_day = datetime.fromisoformat(race["date_start"]).astimezone(timezone.utc).date()
        event = find_event_by_race_date(conn, season, race_day)
        if event is None:
            # A real, non-cancelled race that Jolpica doesn't have. Worth a
            # warning (calendar mismatch), but not worth stopping for.
            logger.warning("No Jolpica event for OpenF1 race on %s (%s)",
                           race_day, race.get("country_name"))
            continue

        for session in meeting_sessions:
            session_type = SESSION_NAME_TO_TYPE.get(session["session_name"])
            if session_type is None or session.get("is_cancelled"):
                continue

            start_time_utc = to_utc_iso(session["date_start"], timespec="seconds")
            conn.execute(
                """
                INSERT INTO sessions (event_id, session_type, session_name,
                                      start_time_utc, openf1_session_key)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (event_id, session_type) DO UPDATE SET
                    -- Keep the name Jolpica gave it if there is one.
                    session_name = COALESCE(sessions.session_name, excluded.session_name),
                    -- OpenF1's exact start time beats Jolpica's (or fills it in
                    -- where Jolpica only had a date).
                    start_time_utc = excluded.start_time_utc,
                    openf1_session_key = excluded.openf1_session_key
                """,
                (event["event_id"], session_type, session["session_name"],
                 start_time_utc, session["session_key"]),
            )
            session_id = conn.execute(
                "SELECT session_id FROM sessions WHERE event_id = ? AND session_type = ?",
                (event["event_id"], session_type),
            ).fetchone()["session_id"]

            matched.append(MatchedSession(
                session_id=session_id,
                session_key=session["session_key"],
                round=event["round"],
                session_type=session_type,
                start_time_utc=start_time_utc,
                # :02d pads the round to 2 digits ('2024-06-FP1'), so keys sort nicely.
                item_key=f"{season}-{event['round']:02d}-{session_type}",
            ))

    return matched


def match_entries(conn: sqlite3.Connection, session_id: int) -> tuple[int, int]:
    """Fill in driver_id / constructor_id for every car in a session.

    Recomputes every entry each time (not just NULL ones), so re-running it
    after Jolpica publishes the race result upgrades earlier best guesses.

    Rules, most reliable first:
      Driver:
        1. Car number matches a race/sprint result at the SAME event
           (car numbers are unique within a weekend, so this is exact).
        2. Otherwise, acronym matches a driver's code, using their most recent
           result up to this season (covers practice before the race exists).
        3. Otherwise NULL (e.g. an FP1 rookie who has never raced).
      Team:
        1. Same as driver rule 1 (the team in that result).
        2. Otherwise, a teammate in this session with the same OpenF1
           team_name who WAS matched by rule 1 (this gives rookies a team).
        3. Otherwise, the team from the driver's most recent result.
        4. Otherwise NULL.

    Returns:
        (number of entries with a driver_id, number without).
    """
    session = conn.execute(
        "SELECT s.event_id, e.season FROM sessions s JOIN events e USING (event_id) "
        "WHERE s.session_id = ?",
        (session_id,),
    ).fetchone()
    entries = conn.execute(
        "SELECT car_number, acronym, team_name FROM session_entries WHERE session_id = ?",
        (session_id,),
    ).fetchall()

    # Pass 1: work out each car's driver (and team, when rule 1 applies).
    # Stored in a dict first, then written, so pass 2 can look at teammates.
    # {car_number: {"driver_id": ..., "constructor_id": ..., "exact": bool}}
    resolved: dict[int, dict[str, Any]] = {}
    for entry in entries:
        same_event = conn.execute(
            """
            SELECT r.driver_id, r.constructor_id
            FROM results r JOIN sessions s ON s.session_id = r.session_id
            WHERE s.event_id = ? AND r.car_number = ?
            -- Prefer the race result over the sprint result (they agree anyway).
            ORDER BY CASE s.session_type WHEN 'R' THEN 0 ELSE 1 END
            LIMIT 1
            """,
            (session["event_id"], entry["car_number"]),
        ).fetchone()

        if same_event is not None:
            resolved[entry["car_number"]] = {
                "driver_id": same_event["driver_id"],
                "constructor_id": same_event["constructor_id"],
                "exact": True,  # matched by rule 1: safe to copy to teammates
            }
            continue

        by_acronym = None
        if entry["acronym"]:
            # Acronyms are unique among a season's race drivers, and we take
            # the most recent use, so an old driver with the same code can't win.
            by_acronym = conn.execute(
                """
                SELECT d.driver_id
                FROM drivers d
                JOIN results r ON r.driver_id = d.driver_id
                JOIN sessions s ON s.session_id = r.session_id
                JOIN events e ON e.event_id = s.event_id
                WHERE d.code = ? AND e.season <= ?
                ORDER BY e.season DESC, e.round DESC
                LIMIT 1
                """,
                (entry["acronym"], session["season"]),
            ).fetchone()

        resolved[entry["car_number"]] = {
            "driver_id": by_acronym["driver_id"] if by_acronym else None,
            "constructor_id": None,  # decided in pass 2
            "exact": False,
        }

    # Pass 2: teams for cars not matched exactly.
    # Map OpenF1 team_name -> constructor_id, learned from exact matches.
    team_by_name: dict[str, str] = {}
    for entry in entries:
        match = resolved[entry["car_number"]]
        if match["exact"] and entry["team_name"]:
            team_by_name[entry["team_name"]] = match["constructor_id"]

    for entry in entries:
        match = resolved[entry["car_number"]]
        if match["exact"]:
            continue
        constructor_id = team_by_name.get(entry["team_name"])   # teammate rule
        if constructor_id is None and match["driver_id"] is not None:
            latest = conn.execute(
                """
                SELECT r.constructor_id
                FROM results r
                JOIN sessions s ON s.session_id = r.session_id
                JOIN events e ON e.event_id = s.event_id
                WHERE r.driver_id = ? AND e.season <= ?
                ORDER BY e.season DESC, e.round DESC
                LIMIT 1
                """,
                (match["driver_id"], session["season"]),
            ).fetchone()
            constructor_id = latest["constructor_id"] if latest else None
        match["constructor_id"] = constructor_id

    # Write everything back.
    for car_number, match in resolved.items():
        conn.execute(
            "UPDATE session_entries SET driver_id = ?, constructor_id = ? "
            "WHERE session_id = ? AND car_number = ?",
            (match["driver_id"], match["constructor_id"], session_id, car_number),
        )

    matched = sum(1 for m in resolved.values() if m["driver_id"] is not None)
    return matched, len(resolved) - matched


# =============================================================================
# 4. LOAD
# =============================================================================

def ingest_session(conn: sqlite3.Connection, client: JsonHttpClient, session_id: int, session_key: int) -> int:
    """Fetch and store everything OpenF1 has for one session. Returns rows written.

    All six endpoints are fetched BEFORE writing anything. That way a network
    failure halfway through never leaves a half-written session, and the
    database transaction (below) stays short.
    """
    drivers = fetch(client, "drivers", session_key=session_key)
    laps = fetch(client, "laps", session_key=session_key)
    if not laps:
        # A finished session always has laps. None means OpenF1 hasn't
        # published it yet (or lost it), so fail now and retry on a later run
        # rather than marking it 'ok' with nothing in it.
        raise NoDataYet(f"OpenF1 has no laps for session_key {session_key}")
    stints = fetch(client, "stints", session_key=session_key)
    pits = fetch(client, "pit", session_key=session_key)
    weather = fetch(client, "weather", session_key=session_key)
    race_control = fetch(client, "race_control", session_key=session_key)

    # `with conn:` = one transaction: commit if the block finishes, roll
    # back everything if any statement fails.
    with conn:
        conn.executemany(
            """
            INSERT INTO session_entries (session_id, car_number, acronym, full_name, team_name)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (session_id, car_number) DO UPDATE SET
                acronym = excluded.acronym, full_name = excluded.full_name,
                team_name = excluded.team_name
            """,
            [entry_row(session_id, d) for d in drivers],
        )

        # Safety net: if laps/stints/pits mention a car that /drivers left
        # out (a data gap), create a bare entry for it. Otherwise the foreign
        # key would reject that car's rows and fail the whole session.
        # A set comprehension: every distinct car number across the three lists.
        cars_seen = {row["driver_number"] for row in laps + stints + pits}
        conn.executemany(
            "INSERT INTO session_entries (session_id, car_number) VALUES (?, ?) "
            "ON CONFLICT (session_id, car_number) DO NOTHING",
            [(session_id, car) for car in sorted(cars_seen)],
        )

        conn.executemany(
            """
            INSERT INTO laps (session_id, car_number, lap_number, lap_start_utc,
                              lap_time_ms, sector1_ms, sector2_ms, sector3_ms,
                              is_pit_out_lap, i1_speed_kph, i2_speed_kph, st_speed_kph)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (session_id, car_number, lap_number) DO UPDATE SET
                lap_start_utc = excluded.lap_start_utc, lap_time_ms = excluded.lap_time_ms,
                sector1_ms = excluded.sector1_ms, sector2_ms = excluded.sector2_ms,
                sector3_ms = excluded.sector3_ms, is_pit_out_lap = excluded.is_pit_out_lap,
                i1_speed_kph = excluded.i1_speed_kph, i2_speed_kph = excluded.i2_speed_kph,
                st_speed_kph = excluded.st_speed_kph
            """,
            [lap_row(session_id, lap) for lap in laps],
        )
        conn.executemany(
            """
            INSERT INTO stints (session_id, car_number, stint_number, compound,
                                lap_start, lap_end, tyre_age_at_start)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (session_id, car_number, stint_number) DO UPDATE SET
                compound = excluded.compound, lap_start = excluded.lap_start,
                lap_end = excluded.lap_end, tyre_age_at_start = excluded.tyre_age_at_start
            """,
            [stint_row(session_id, s) for s in stints],
        )
        conn.executemany(
            """
            INSERT INTO pit_stops (session_id, car_number, lap_number, stop_time_utc,
                                   pit_lane_ms, stationary_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (session_id, car_number, lap_number) DO UPDATE SET
                stop_time_utc = excluded.stop_time_utc, pit_lane_ms = excluded.pit_lane_ms,
                stationary_ms = excluded.stationary_ms
            """,
            [pit_row(session_id, p) for p in pits],
        )
        conn.executemany(
            """
            INSERT INTO weather (session_id, sample_time_utc, air_temp_c, track_temp_c,
                                 humidity_pct, pressure_mbar, rainfall,
                                 wind_speed_ms, wind_direction_deg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (session_id, sample_time_utc) DO NOTHING
            """,
            [weather_row(session_id, w) for w in weather],
        )
        conn.executemany(
            """
            INSERT INTO race_control (session_id, message_time_utc, category, flag, scope,
                                      sector, car_number, lap_number, message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            -- DO NOTHING: an identical message at the same moment is a duplicate.
            ON CONFLICT (session_id, message_time_utc, message) DO NOTHING
            """,
            [race_control_row(session_id, m) for m in race_control],
        )

        matched, unmatched = match_entries(conn, session_id)

    if unmatched:
        logger.info("Session %d: %d car(s) not matched to a Jolpica driver", session_id, unmatched)

    return len(drivers) + len(laps) + len(stints) + len(pits) + len(weather) + len(race_control)


# =============================================================================
# 5. Putting it together
# =============================================================================

def ingest_season(
    conn: sqlite3.Connection,
    client: JsonHttpClient,
    season: int,
    rounds: set[int] | None = None,
    skip_done: bool = True,
    now: datetime | None = None,
) -> dict[str, str]:
    """Load all finished sessions of a season from OpenF1, logging each one.

    Jolpica's schedule for the season must be loaded first: we match OpenF1's
    sessions to the events it created.

    Args:
        rounds: Only load these rounds (e.g. {24}). None = the whole season.
        skip_done: Skip sessions already logged as 'ok'. This is what lets an
            interrupted backfill resume instead of starting over.
        now: The current time. Tests pass a fixed value; normally it's the real clock.

    Returns:
        {item_key: status}, e.g. {'2024-24-FP1': 'ok', ...}
    """
    now = now or datetime.now(timezone.utc)
    # Only sessions that started before this cutoff are considered finished.
    # isoformat + 'Z' gives text in the same format as start_time_utc, so the
    # two can be compared directly as strings.
    cutoff = (now - timedelta(hours=SESSION_SETTLE_HOURS)).isoformat(timespec="seconds")
    cutoff = cutoff.replace("+00:00", "Z")
    outcomes: dict[str, str] = {}

    # Step 1: match the season's sessions (one request for the whole season).
    try:
        openf1_sessions = fetch(client, "sessions", year=season)
        with conn:
            matched = match_sessions(conn, season, openf1_sessions)
    except Exception as exc:
        logger.error("OpenF1 %s-sessions failed: %s", season, exc)
        ingestion_log.record(conn, SOURCE, f"{season}-sessions", "failed", message=str(exc))
        outcomes[f"{season}-sessions"] = "failed"
        return outcomes  # nothing else can work without the session list
    ingestion_log.record(conn, SOURCE, f"{season}-sessions", "ok", row_count=len(matched))
    outcomes[f"{season}-sessions"] = "ok"

    # Step 2: load each finished session.
    for session in matched:
        if rounds is not None and session.round not in rounds:
            continue
        if session.start_time_utc > cutoff:
            continue  # hasn't happened (or finished) yet
        if skip_done and ingestion_log.is_done(conn, SOURCE, session.item_key):
            outcomes[session.item_key] = "already done"
            continue

        try:
            row_count = ingest_session(conn, client, session.session_id, session.session_key)
        except Exception as exc:
            # Alert and skip: record the failure and carry on with the next session.
            logger.error("OpenF1 %s failed: %s", session.item_key, exc)
            ingestion_log.record(conn, SOURCE, session.item_key, "failed", message=str(exc))
            outcomes[session.item_key] = "failed"
        else:
            logger.info("OpenF1 %s: %d rows", session.item_key, row_count)
            ingestion_log.record(conn, SOURCE, session.item_key, "ok", row_count=row_count)
            outcomes[session.item_key] = "ok"

    return outcomes
