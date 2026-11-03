"""Load the calendar, race/sprint results and qualifying from the Jolpica API.

What Jolpica gives us (per season):
    /{season}.json             -> schedule: events, circuits, session dates/times
    /{season}/results.json     -> race results   (session 'R')
    /{season}/sprint.json      -> sprint results (session 'S'), 2021+
    /{season}/qualifying.json  -> qualifying     (session 'Q')
Drivers and constructors arrive embedded inside every result row, so we
upsert them as we go instead of fetching them separately.

The module has three layers, top to bottom:
    1. FETCH: talk to the API (including merging paged responses)
    2. PARSE: pure functions, JSON values -> Python values (no I/O, easy to test)
    3. LOAD:  write rows into SQLite, using upserts so re-runs never duplicate
Finally, ingest_season() runs all of it for one season and logs each outcome.
"""

import logging
import sqlite3

from f1predictor import config
from f1predictor.db import ingestion_log
from f1predictor.ingest.http import JsonHttpClient

logger = logging.getLogger(__name__)

SOURCE = "jolpica"  # the `source` value we write into ingestion_log

# Jolpica's maximum page size. Asking for more still returns 100.
PAGE_LIMIT = 100

# Schedule JSON key -> (our session code, readable session name).
# The race itself isn't in this map: its date/time sit at the top level of
# each race object, not under a key.
# Note both SprintQualifying (2024+) and SprintShootout (2023) map to 'SQ':
# same role in the weekend (sets the sprint grid), different name per year.
SCHEDULE_SESSION_KEYS: dict[str, tuple[str, str]] = {
    "FirstPractice":    ("FP1", "Practice 1"),
    "SecondPractice":   ("FP2", "Practice 2"),
    "ThirdPractice":    ("FP3", "Practice 3"),
    "SprintQualifying": ("SQ",  "Sprint Qualifying"),
    "SprintShootout":   ("SQ",  "Sprint Shootout"),
    "Sprint":           ("S",   "Sprint"),
    "Qualifying":       ("Q",   "Qualifying"),
}


# =============================================================================
# 1. FETCH
# =============================================================================

def fetch_races(client: JsonHttpClient, path: str, result_key: str | None) -> list[dict]:
    """Fetch every page of a Jolpica endpoint and return the merged list of races.

    Jolpica pages by RESULT ROW, not by race: page 1 holds rows 1-100, which
    can end halfway through a race. So a race can appear on two consecutive
    pages, each holding part of its results. We merge those pieces back
    together by round number.

    Args:
        client: The HTTP client to fetch with.
        path: Endpoint path after the base URL, e.g. '2024/results'.
        result_key: Name of the per-race list to merge across pages
            ('Results', 'SprintResults', 'QualifyingResults'), or None for the
            schedule, which has no per-race list.

    Returns:
        One dict per race, in round order, each with its complete result list.
    """
    url = f"{config.JOLPICA_BASE_URL}/{path}.json"

    # Keyed by round number. Python dicts keep insertion order, so the output
    # stays in round order.
    races_by_round: dict[int, dict] = {}
    offset = 0

    while True:
        page = client.get_json(url, params={"limit": PAGE_LIMIT, "offset": offset})
        mr_data = page["MRData"]               # Ergast wraps everything in MRData
        total = int(mr_data["total"])          # total rows across ALL pages
        races = mr_data["RaceTable"]["Races"]

        for race in races:
            round_number = int(race["round"])
            if round_number in races_by_round and result_key is not None:
                # Second half of a race that was split across pages:
                # append its rows to the part we already have.
                races_by_round[round_number][result_key].extend(race[result_key])
            else:
                races_by_round[round_number] = race

        offset += PAGE_LIMIT
        # Stop once we've requested past the last row.
        if offset >= total:
            break

    return list(races_by_round.values())


# =============================================================================
# 2. PARSE (pure functions: no network, no database)
# =============================================================================

def to_int(value: str | None) -> int | None:
    """'12' -> 12. None or '' -> None. (Jolpica sends every number as a string.)"""
    if value is None or value == "":
        return None
    return int(value)


def to_float(value: str | None) -> float | None:
    """'25.9581' -> 25.9581. None or '' -> None."""
    if value is None or value == "":
        return None
    return float(value)


def parse_lap_time_ms(text: str | None) -> int | None:
    """Convert a lap time string to integer milliseconds.

    '1:29.708' -> 89708     (minutes:seconds.millis)
    '59.123'   -> 59123     (no minutes part)
    '' / None  -> None      (no time set, e.g. knocked out before Q2)
    """
    if not text:
        return None

    # Split off minutes if there's a colon. "1:29.708" -> ("1", "29.708")
    if ":" in text:
        minutes_text, seconds_text = text.split(":")
        minutes = int(minutes_text)
    else:
        minutes, seconds_text = 0, text

    # round() rather than int(): 29.708 * 1000 is 29707.999... in floating
    # point, and int() would truncate it to 29707.
    seconds_ms = round(float(seconds_text) * 1000)
    return minutes * 60_000 + seconds_ms


def to_utc_timestamp(date: str, time: str | None) -> str | None:
    """Combine Jolpica's date + time into one ISO-8601 UTC timestamp.

    ('2024-05-03', '16:30:00Z') -> '2024-05-03T16:30:00Z'
    ('2021-07-16', None)        -> None

    Older seasons only have dates for some sessions. We return None rather
    than guessing a time; the FastF1 loader fills in exact times later.
    """
    if not time:
        return None
    return f"{date}T{time}"


def event_format(race: dict) -> str:
    """Work out the weekend format from which session keys the race has.

    Names match FastF1's EventFormat values so both sources agree.
    """
    if "SprintQualifying" in race:
        return "sprint_qualifying"   # 2024+: FP1, SQ, Sprint, Q, Race
    if "SprintShootout" in race:
        return "sprint_shootout"     # 2023: FP1, Q, Shootout, Sprint, Race
    if "Sprint" in race:
        return "sprint"              # 2021-22: FP1, Q, FP2, Sprint, Race
    return "conventional"            # FP1, FP2, FP3, Q, Race


# =============================================================================
# 3. LOAD (write to the database)
# =============================================================================
# Each upsert function uses INSERT ... ON CONFLICT DO UPDATE: insert the row,
# or if its primary/unique key already exists, update it in place. That makes
# every load idempotent: running it twice gives the same database as once.

def upsert_circuit(conn: sqlite3.Connection, circuit: dict) -> None:
    location = circuit.get("Location", {})
    conn.execute(
        """
        INSERT INTO circuits (circuit_id, name, locality, country, latitude, longitude)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (circuit_id) DO UPDATE SET
            name = excluded.name, locality = excluded.locality,
            country = excluded.country, latitude = excluded.latitude,
            longitude = excluded.longitude
        """,
        (
            circuit["circuitId"],
            circuit["circuitName"],
            location.get("locality"),
            location.get("country"),
            to_float(location.get("lat")),
            to_float(location.get("long")),
        ),
    )


def upsert_driver(conn: sqlite3.Connection, driver: dict) -> None:
    conn.execute(
        """
        INSERT INTO drivers (driver_id, code, permanent_number, given_name,
                             family_name, date_of_birth, nationality)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (driver_id) DO UPDATE SET
            code = excluded.code, permanent_number = excluded.permanent_number,
            given_name = excluded.given_name, family_name = excluded.family_name,
            date_of_birth = excluded.date_of_birth, nationality = excluded.nationality
        """,
        (
            driver["driverId"],
            driver.get("code"),                       # .get(): missing for some old drivers
            to_int(driver.get("permanentNumber")),
            driver["givenName"],
            driver["familyName"],
            driver.get("dateOfBirth"),
            driver.get("nationality"),
        ),
    )


def upsert_constructor(conn: sqlite3.Connection, constructor: dict) -> None:
    conn.execute(
        """
        INSERT INTO constructors (constructor_id, name, nationality)
        VALUES (?, ?, ?)
        ON CONFLICT (constructor_id) DO UPDATE SET
            name = excluded.name, nationality = excluded.nationality
        """,
        (constructor["constructorId"], constructor["name"], constructor.get("nationality")),
    )


def upsert_event(conn: sqlite3.Connection, race: dict) -> int:
    """Insert/update one event and return its event_id."""
    season, round_number = int(race["season"]), int(race["round"])
    conn.execute(
        """
        INSERT INTO events (season, round, name, circuit_id, race_date, event_format)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (season, round) DO UPDATE SET
            name = excluded.name, circuit_id = excluded.circuit_id,
            race_date = excluded.race_date, event_format = excluded.event_format
        """,
        (season, round_number, race["raceName"], race["Circuit"]["circuitId"],
         race["date"], event_format(race)),
    )
    # The upsert doesn't hand back the id (in the update case there's no new
    # row), so look it up by the natural key.
    row = conn.execute(
        "SELECT event_id FROM events WHERE season = ? AND round = ?",
        (season, round_number),
    ).fetchone()
    return row["event_id"]


def upsert_session(
    conn: sqlite3.Connection,
    event_id: int,
    session_type: str,
    session_name: str,
    start_time_utc: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO sessions (event_id, session_type, session_name, start_time_utc)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (event_id, session_type) DO UPDATE SET
            session_name = excluded.session_name,
            -- COALESCE returns its first non-NULL argument. So if this load
            -- has no time (older seasons), keep the time already stored
            -- (e.g. one FastF1 filled in) instead of wiping it with NULL.
            start_time_utc = COALESCE(excluded.start_time_utc, sessions.start_time_utc)
        """,
        (event_id, session_type, session_name, start_time_utc),
    )


def get_session_id(
    conn: sqlite3.Connection, season: int, round_number: int, session_type: str
) -> int:
    """Look up the session_id for e.g. (2024, 6, 'R'). Raises if it doesn't exist."""
    row = conn.execute(
        """
        SELECT s.session_id
        FROM sessions AS s
        JOIN events AS e ON e.event_id = s.event_id
        WHERE e.season = ? AND e.round = ? AND s.session_type = ?
        """,
        (season, round_number, session_type),
    ).fetchone()
    if row is None:
        # This means results arrived for a session the schedule doesn't have.
        # Fail loudly: silently dropping results would corrupt the training data.
        raise ValueError(
            f"No {session_type} session for {season} round {round_number}; "
            "load the schedule first"
        )
    return row["session_id"]


def load_schedule(conn: sqlite3.Connection, client: JsonHttpClient, season: int) -> int:
    """Load one season's events, circuits and sessions. Returns events loaded."""
    races = fetch_races(client, str(season), result_key=None)

    for race in races:
        upsert_circuit(conn, race["Circuit"])
        event_id = upsert_event(conn, race)

        # One session row per schedule key the race actually has. Reading the
        # keys (instead of assuming FP1/FP2/FP3/Q/R) is what handles sprint
        # weekends without hard-coding which rounds are sprints.
        for key, (session_type, session_name) in SCHEDULE_SESSION_KEYS.items():
            if key in race:
                session = race[key]
                upsert_session(conn, event_id, session_type, session_name,
                               to_utc_timestamp(session["date"], session.get("time")))

        # The race session: its date/time are at the top level.
        upsert_session(conn, event_id, "R", "Race",
                       to_utc_timestamp(race["date"], race.get("time")))

    return len(races)


def load_results(
    conn: sqlite3.Connection, client: JsonHttpClient, season: int, sprint: bool = False
) -> int:
    """Load one season's race results (or sprint results). Returns rows loaded."""
    # Race and sprint results have identical row shapes; only the endpoint,
    # the JSON key and our session code differ.
    if sprint:
        path, result_key, session_type = f"{season}/sprint", "SprintResults", "S"
    else:
        path, result_key, session_type = f"{season}/results", "Results", "R"

    races = fetch_races(client, path, result_key)
    row_count = 0

    for race in races:
        session_id = get_session_id(conn, season, int(race["round"]), session_type)

        for result in race[result_key]:
            # Make sure the referenced driver/constructor rows exist first,
            # otherwise the foreign keys on `results` would reject this row.
            upsert_driver(conn, result["Driver"])
            upsert_constructor(conn, result["Constructor"])

            conn.execute(
                """
                INSERT INTO results (session_id, driver_id, constructor_id, car_number,
                                     grid, position, position_text, points,
                                     laps_completed, status, race_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (session_id, driver_id) DO UPDATE SET
                    constructor_id = excluded.constructor_id,
                    car_number = excluded.car_number, grid = excluded.grid,
                    position = excluded.position, position_text = excluded.position_text,
                    points = excluded.points, laps_completed = excluded.laps_completed,
                    status = excluded.status, race_time_ms = excluded.race_time_ms
                """,
                (
                    session_id,
                    result["Driver"]["driverId"],
                    result["Constructor"]["constructorId"],
                    to_int(result.get("number")),
                    to_int(result.get("grid")),        # official grid, after penalties
                    int(result["position"]),
                    result["positionText"],
                    float(result.get("points", 0)),
                    to_int(result.get("laps")),
                    result.get("status"),
                    # "Time" is missing for drivers who didn't finish. The
                    # nested .get() calls return None instead of crashing.
                    to_int(result.get("Time", {}).get("millis")),
                ),
            )
            row_count += 1

    return row_count


def load_qualifying(conn: sqlite3.Connection, client: JsonHttpClient, season: int) -> int:
    """Load one season's (Grand Prix) qualifying results. Returns rows loaded."""
    races = fetch_races(client, f"{season}/qualifying", "QualifyingResults")
    row_count = 0

    for race in races:
        session_id = get_session_id(conn, season, int(race["round"]), "Q")

        for result in race["QualifyingResults"]:
            upsert_driver(conn, result["Driver"])
            upsert_constructor(conn, result["Constructor"])
            conn.execute(
                """
                INSERT INTO qualifying (session_id, driver_id, constructor_id,
                                        position, q1_ms, q2_ms, q3_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (session_id, driver_id) DO UPDATE SET
                    constructor_id = excluded.constructor_id,
                    position = excluded.position, q1_ms = excluded.q1_ms,
                    q2_ms = excluded.q2_ms, q3_ms = excluded.q3_ms
                """,
                (
                    session_id,
                    result["Driver"]["driverId"],
                    result["Constructor"]["constructorId"],
                    int(result["position"]),
                    # A driver knocked out in Q1 has no Q2/Q3 keys at all.
                    parse_lap_time_ms(result.get("Q1")),
                    parse_lap_time_ms(result.get("Q2")),
                    parse_lap_time_ms(result.get("Q3")),
                ),
            )
            row_count += 1

    return row_count


# =============================================================================
# Putting it together
# =============================================================================

def ingest_season(conn: sqlite3.Connection, client: JsonHttpClient, season: int) -> dict[str, str]:
    """Load everything Jolpica has for one season, logging each part's outcome.

    Each part runs in its own transaction: if it fails halfway, its partial
    writes are rolled back, the failure is recorded in ingestion_log, and we
    move on to the next part instead of crashing ("alert and skip").

    The schedule must load first, since results attach to its sessions. If the
    schedule fails, the remaining parts would fail too, so we stop early.

    Returns:
        {item_key: status} for each part, e.g. {'2024-schedule': 'ok', ...}.
    """
    # (item_key, function to call). Lambdas delay the call until we're ready.
    parts = [
        (f"{season}-schedule",   lambda: load_schedule(conn, client, season)),
        (f"{season}-results",    lambda: load_results(conn, client, season)),
        (f"{season}-sprint",     lambda: load_results(conn, client, season, sprint=True)),
        (f"{season}-qualifying", lambda: load_qualifying(conn, client, season)),
    ]
    outcomes: dict[str, str] = {}

    for item_key, load in parts:
        try:
            # `with conn:` opens a transaction: it COMMITs if the block
            # finishes, and ROLLs BACK if an exception escapes it.
            with conn:
                row_count = load()
        except Exception as exc:
            # Deliberately broad: whatever went wrong (network, unexpected
            # JSON, a constraint violation), we record it and keep going.
            # The full message is stored for debugging.
            logger.error("Jolpica %s failed: %s", item_key, exc)
            ingestion_log.record(conn, SOURCE, item_key, "failed", message=str(exc))
            outcomes[item_key] = "failed"
            if item_key.endswith("schedule"):
                break
        else:
            logger.info("Jolpica %s: %d rows", item_key, row_count)
            ingestion_log.record(conn, SOURCE, item_key, "ok", row_count=row_count)
            outcomes[item_key] = "ok"

    return outcomes
