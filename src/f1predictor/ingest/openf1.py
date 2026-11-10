"""Load lap-level data from the OpenF1 API (2023 onwards).

What OpenF1 gives us, per session (identified by OpenF1's `session_key`):
    /drivers       -> who drove: car number, acronym, name, team
    /laps          -> lap and sector times, speed traps, pit-out flag
    /stints        -> tyre compound and age for each stint
    /pit           -> pit stops: lap, time in pit lane, stationary time
    /weather       -> one sample per minute
    /race_control  -> flags, safety cars, penalties, deleted laps...
    /starting_grid -> the race's grid after penalties (qualifying sessions only)
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
import unicodedata
from datetime import date, datetime, timedelta, timezone
from typing import Any, NamedTuple

import requests

from f1predictor import config
from f1predictor.db import ingestion_log
from f1predictor.ingest.http import JsonHttpClient
from f1predictor.validate import checks

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


def _match_driver(conn: sqlite3.Connection, session: sqlite3.Row, session_id: int,
                  entry: sqlite3.Row) -> dict[str, Any]:
    """Pass 1 of match_entries: driver rules 1-5 (and 3b) for ONE car (see its docstring).

    Returns {"driver_id", "constructor_id", "exact"}. constructor_id is only
    set by rules 1, 3 and 3b (all name the team); otherwise None, for pass 2.
    exact = matched by rule 1 (a result at this event), so its team can be
    taught to teammates in pass 2.
    """
    # Rule 1: the car number in a race/sprint result at this event.
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
        return {"driver_id": same_event["driver_id"], "constructor_id": same_event["constructor_id"],
                "exact": True}   # safe to copy to teammates

    if not entry["acronym"]:
        # Rule 2: the same car in another session of this weekend.
        same_car = conn.execute(
            """
            SELECT se.driver_id FROM session_entries se
            JOIN sessions s ON s.session_id = se.session_id
            WHERE s.event_id = ? AND se.car_number = ? AND se.session_id <> ?
              AND se.driver_id IS NOT NULL
            LIMIT 1
            """,
            (session["event_id"], entry["car_number"], session_id),
        ).fetchone()
        return {"driver_id": same_car["driver_id"] if same_car else None,
                "constructor_id": None, "exact": False}   # team: pass 2

    # Rule 3: a driver who qualified at this event under this acronym.
    qualified = conn.execute(
        """
        SELECT q.driver_id, q.constructor_id FROM qualifying q
        JOIN drivers d ON d.driver_id = q.driver_id
        JOIN sessions s ON s.session_id = q.session_id
        WHERE s.event_id = ? AND d.code = ?
        LIMIT 1
        """,
        (session["event_id"], entry["acronym"]),
    ).fetchone()
    if qualified is not None:
        return {"driver_id": qualified["driver_id"], "constructor_id": qualified["constructor_id"],
                "exact": False}   # by acronym, so don't teach teammates from it

    # Rule 3b: a debutant whose Jolpica row has NO code, so rule 3 can't see
    # him. Real case, Brazil 2026: Jolpica listed Rafael Camara in qualifying
    # with an empty code, his car 51 stayed unmatched until the race result,
    # and the post-quali prediction put him at the back as a "pit-lane
    # starter" although he started 11th. Before rule 4 on purpose: a driver
    # who qualified at THIS event is better evidence than the last driver
    # who happened to use the same acronym (who may have retired years ago).
    by_name = _code_less_qualifier_by_name(conn, session["event_id"], entry["full_name"])
    if by_name is not None:
        return {"driver_id": by_name["driver_id"], "constructor_id": by_name["constructor_id"],
                "exact": False}   # by name, so don't teach teammates from it

    # Rule 4: the acronym's most recent race driver up to this season.
    # Acronyms are unique among a season's race drivers, and we take the most
    # recent use, so an old driver with the same code can't win.
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
    # Rule 5: nobody (driver_id None).
    return {"driver_id": by_acronym["driver_id"] if by_acronym else None,
            "constructor_id": None, "exact": False}   # team: pass 2


def _fold_name(name: str | None) -> str:
    """A name in a form two sources can be compared in.

    'Nico HULKENBERG' (OpenF1) and 'Hülkenberg' (Jolpica) differ in case and
    accents, so: split accented letters into letter + accent mark (NFKD),
    drop the marks, lower-case (casefold), and squash runs of spaces.
        'Sergio PEREZ' -> 'sergio perez';  'Pérez' -> 'perez'
    """
    if not name:
        return ""
    decomposed = unicodedata.normalize("NFKD", name)
    # combining() is non-zero for accent marks (the parts NFKD split off).
    no_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(no_accents.casefold().split())


def _code_less_qualifier_by_name(conn: sqlite3.Connection, event_id: int,
                                 full_name: str | None) -> sqlite3.Row | None:
    """Rule 3b of match_entries: the one qualifier at this event with no
    Jolpica code whose family name ends OpenF1's full name, or None.

    Deliberately narrow, because OpenF1's names can be wrong (2023 Hungary
    FP1 listed junior-series drivers' names on the race drivers' cars):
      * only drivers who QUALIFIED at this same event (Jolpica qualifying);
      * only drivers with NO code: anyone with a code is rule 3's job, so a
        wrong name can't override a code match or steal a regular's id;
      * the family name must be the END of OpenF1's name, as whole words
        ('Rafael CAMARA' ends with 'camara'; 'Andrea Kimi ANTONELLI' with
        'antonelli'), compared without case or accents (_fold_name);
      * exactly one candidate, or nothing: two matches would be a guess.
    """
    name = _fold_name(full_name)
    if not name:
        return None
    candidates = conn.execute(
        """
        SELECT q.driver_id, q.constructor_id, d.family_name
        FROM qualifying q
        JOIN drivers d  ON d.driver_id = q.driver_id
        JOIN sessions s ON s.session_id = q.session_id
        WHERE s.event_id = ? AND (d.code IS NULL OR d.code = '')
        """,
        (event_id,),
    ).fetchall()
    matches = []
    for row in candidates:
        family = _fold_name(row["family_name"])
        # Whole words only: 'camara' must not match '... decamara'.
        if family and (name == family or name.endswith(" " + family)):
            matches.append(row)
    return matches[0] if len(matches) == 1 else None


def _fill_in_teams(conn: sqlite3.Connection, session: sqlite3.Row, entries: list[sqlite3.Row],
                   resolved: dict[int, dict[str, Any]]) -> None:
    """Pass 2 of match_entries: team rules 2-4 for cars whose team is still
    unknown (changes `resolved` in place; see match_entries' docstring)."""
    # Map OpenF1 team_name -> constructor_id, learned from exact matches.
    team_by_name: dict[str, str] = {}
    for entry in entries:
        match = resolved[entry["car_number"]]
        if match["exact"] and entry["team_name"]:
            team_by_name[entry["team_name"]] = match["constructor_id"]

    for entry in entries:
        match = resolved[entry["car_number"]]
        if match["exact"] or match["constructor_id"] is not None:   # team already known
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


def match_entries(conn: sqlite3.Connection, session_id: int) -> tuple[int, int]:
    """Fill in driver_id / constructor_id for every car in a session.

    Recomputes every entry each time (not just NULL ones), so re-running it
    after Jolpica publishes the race result upgrades earlier best guesses.

    Rules, most reliable first:
      Driver:
        1. Car number matches a race/sprint result at the SAME event
           (car numbers are unique within a weekend, so this is exact).
        2. No acronym (OpenF1's /drivers left the car out, but it set laps):
           the same car number's match in another session of this event.
           Car numbers are unique within a weekend. (Real case: Lawson's
           car 30 was missing from Mexico City 2026 FP3's driver list.)
        3. Acronym matches the code of a driver in this event's Jolpica
           QUALIFYING. Catches a debutant before their first race result
           exists (real case: Hirakawa, Mexico City 2026), so the post-quali
           prediction has them. Their team is the one they qualified for.
        3b. No code match, but exactly one driver who qualified at this
           event has NO Jolpica code and his family name ends OpenF1's
           full name (accents and case ignored). Real case: Camara, Brazil
           2026, whose Jolpica row had no code. Team: the qualifying one.
        4. Acronym matches a driver's code, using their most recent result
           up to this season (covers practice before the race exists).
        5. Otherwise NULL (e.g. an FP1 rookie who has never raced, or a
           debutant before qualifying: nobody knows their driver id yet).
      Team:
        1. Same as driver rule 1 (the team in that result), or rule 3 (the
           team in that qualifying result).
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
        # full_name is for rule 3b (matching a code-less debutant by name).
        "SELECT car_number, acronym, team_name, full_name FROM session_entries WHERE session_id = ?",
        (session_id,),
    ).fetchall()

    # Pass 1: work out each car's driver (and team, when rule 1 or 3 applies).
    # Stored in a dict first, then written, so pass 2 can look at teammates.
    # {car_number: {"driver_id": ..., "constructor_id": ..., "exact": bool}}
    resolved = {entry["car_number"]: _match_driver(conn, session, session_id, entry) for entry in entries}
    # Pass 2: teams for the cars not matched exactly.
    _fill_in_teams(conn, session, entries, resolved)

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

def ingest_session(
    conn: sqlite3.Connection, client: JsonHttpClient, session_id: int, session_key: int
) -> tuple[int, list[checks.Issue]]:
    """Fetch, store and validate everything OpenF1 has for one session.

    Returns:
        (rows written, warning-level issues found). Error-level issues raise
        ValidationError instead, and nothing is saved.

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

    # Which cars really took part: listed in /drivers, or set at least one lap.
    # OpenF1 occasionally sends stint/pit rows for car numbers that weren't in
    # the session at all (found in the 2023-2026 backfill: 46 "phantom" cars
    # with one stint, no laps and no name). Those rows are dropped here, so
    # they never create fake session entries.
    known_cars = {d["driver_number"] for d in drivers}
    cars_with_laps = {lap["driver_number"] for lap in laps}
    real_cars = known_cars | cars_with_laps          # | is set union
    dropped = sum(1 for row in stints + pits if row["driver_number"] not in real_cars)
    if dropped:
        logger.info("Session %d: dropped %d stint/pit rows for cars not in the session",
                    session_id, dropped)
    stints = [s for s in stints if s["driver_number"] in real_cars]
    pits = [p for p in pits if p["driver_number"] in real_cars]

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

        # Safety net: a car that set laps but that /drivers left out (a data
        # gap) gets a bare entry. Otherwise the foreign key would reject that
        # car's laps and fail the whole session. (Phantom cars with no laps
        # were already filtered out above.)
        conn.executemany(
            "INSERT INTO session_entries (session_id, car_number) VALUES (?, ?) "
            "ON CONFLICT (session_id, car_number) DO NOTHING",
            [(session_id, car) for car in sorted(cars_with_laps - known_cars)],
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

        # Validate inside the transaction: an error-level issue raises
        # ValidationError, which rolls back everything this session wrote.
        issues = checks.check_openf1_session(conn, session_id)
        checks.raise_if_errors(issues)

    if unmatched:
        logger.info("Session %d: %d car(s) not matched to a Jolpica driver", session_id, unmatched)

    row_count = len(drivers) + len(laps) + len(stints) + len(pits) + len(weather) + len(race_control)
    # Warnings (if any) go back to the caller, which records them.
    return row_count, issues


def load_starting_grid(
    conn: sqlite3.Connection, client: JsonHttpClient, session_id: int, session_key: int
) -> tuple[int, list[checks.Issue]]:
    """Fetch and store the starting grid for one QUALIFYING session.

    Kept separate from ingest_session because the grid can appear later than
    the qualifying laps (penalties are applied after qualifying), so it may
    need fetching again on its own. Safe to re-run: the session's old grid is
    replaced in one transaction.

    Returns (number of grid slots stored, warnings). 0 slots = OpenF1 has no
    grid yet. The warnings (if any) go back to the caller, which records
    them; today the only one is an accepted pit-lane shrink, which the tick
    alerts on (validate/checks.grid_shrink_problem).
    """
    grid = fetch(client, "starting_grid", session_key=session_key)
    # Only cars that took part in qualifying (the foreign key would reject
    # any other car anyway, and fail the whole batch).
    cars = {row["car_number"] for row in conn.execute(
        "SELECT car_number FROM session_entries WHERE session_id = ?", (session_id,))}
    rows = [(session_id, slot["driver_number"], slot["position"])
            for slot in grid
            if slot.get("driver_number") in cars and slot.get("position") is not None]
    # The grid we already hold for this session ({car: slot}; empty on the
    # first fetch). Read BEFORE the replace below deletes it.
    previous = {row[0]: row[1] for row in conn.execute(
        "SELECT car_number, position FROM starting_grid WHERE session_id = ?", (session_id,))}
    stored = len(previous)
    if not rows:
        # Nothing usable came back. Before any grid is stored that just means
        # "not published yet" (0 -> the caller logs 'skipped' and retries).
        # On a REFRESH it's an API hiccup, not OpenF1 withdrawing the grid:
        # keep the stored grid untouched and report its size, so the caller
        # logs 'ok' again and the planner still sees the grid as available.
        # (Before 2026-11-04 this path deleted the stored grid.)
        if stored:
            logger.warning("Session %d: empty starting-grid refresh; keeping the %d stored slots",
                           session_id, stored)
        return stored, []
    with conn:
        conn.execute("DELETE FROM starting_grid WHERE session_id = ?", (session_id,))
        conn.executemany(
            "INSERT INTO starting_grid (session_id, car_number, position) VALUES (?, ?, ?)", rows)
        # Validate inside the transaction: an error rolls back the DELETE
        # too, so the previous grid survives. previous_grid lets the check
        # tell a clean pit-lane move from a refresh that lost cars.
        issues = checks.check_starting_grid(conn, session_id, previous_grid=previous)
        checks.raise_if_errors(issues)
    return len(rows), issues


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
    session_types: set[str] | None = None,
    settle_hours: float = SESSION_SETTLE_HOURS,
    refresh_grid: bool = False,
) -> dict[str, str]:
    """Load all finished sessions of a season from OpenF1, logging each one.

    Jolpica's schedule for the season must be loaded first: we match OpenF1's
    sessions to the events it created.

    Args:
        rounds: Only load these rounds (e.g. {24}). None = the whole season.
        skip_done: Skip sessions already logged as 'ok'. This is what lets an
            interrupted backfill resume instead of starting over.
        now: The current time. Tests pass a fixed value; normally it's the real clock.
        session_types: Only load these kinds of session (e.g. {'FP2'}). None =
            all. The weekend pipeline passes exactly the sessions its planner
            found ready (pipeline/plan.py); 'Q' also (re)tries the starting grid.
        settle_hours: How long after a session STARTED it counts as finished.
            The backfill uses the safe default; the pipeline passes 0, since its
            planner already waited for each session's end (by session type).
        refresh_grid: Re-fetch a qualifying session's starting grid even if a
            previous attempt was successful. The live pipeline uses this
            immediately before its immutable post-quali prediction, so a
            penalty published since the first fetch is not missed.

    Returns:
        {item_key: status}, e.g. {'2024-24-FP1': 'ok', ...}
    """
    now = now or datetime.now(timezone.utc)
    # Only sessions that started before this cutoff are considered finished.
    # isoformat + 'Z' gives text in the same format as start_time_utc, so the
    # two can be compared directly as strings.
    cutoff = (now - timedelta(hours=settle_hours)).isoformat(timespec="seconds")
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
        if session_types is not None and session.session_type not in session_types:
            continue
        if session.start_time_utc > cutoff:
            continue  # hasn't happened (or finished) yet
        if skip_done and ingestion_log.is_done(conn, SOURCE, session.item_key):
            outcomes[session.item_key] = "already done"
        else:
            _ingest_logged(conn, client, session, outcomes)

        # After qualifying: also the official starting grid. Its own log key
        # ('2024-24-grid'), so a grid that isn't published yet is retried on
        # the next run without re-fetching the qualifying laps. Only once the
        # session itself loaded, since the grid hangs off its entries.
        if session.session_type == "Q" and outcomes[session.item_key] in ("ok", "already done"):
            grid_key = f"{session.item_key[:-2]}-grid"     # '2024-24-Q' -> '2024-24-grid'
            if skip_done and not refresh_grid and ingestion_log.is_done(conn, SOURCE, grid_key):
                outcomes[grid_key] = "already done"
                continue
            try:
                slots, grid_issues = load_starting_grid(conn, client, session.session_id,
                                                        session.session_key)
            except Exception as exc:
                logger.error("OpenF1 %s failed: %s", grid_key, exc)
                ingestion_log.record(conn, SOURCE, grid_key, "failed", message=str(exc))
                checks.record_issues(conn, SOURCE, grid_key, getattr(exc, "issues", []))
                outcomes[grid_key] = "failed"
                continue
            # 0 slots = not published yet: 'skipped', so it's tried again later.
            status = "ok" if slots else "skipped"
            ingestion_log.record(conn, SOURCE, grid_key, status, row_count=slots)
            # Warnings only (an error would have raised above), e.g. an
            # accepted pit-lane shrink; empty clears the item's old issues.
            checks.record_issues(conn, SOURCE, grid_key, grid_issues)
            outcomes[grid_key] = status

    return outcomes


def _ingest_logged(conn: sqlite3.Connection, client: JsonHttpClient,
                   session: MatchedSession, outcomes: dict[str, str]) -> None:
    """ingest_session for one session, recording the outcome in ingestion_log
    and in `outcomes`. Failures are logged and skipped, never raised (the
    PRD's "alert and skip instead of crashing")."""
    try:
        row_count, issues = ingest_session(conn, client, session.session_id, session.session_key)
    except Exception as exc:
        logger.error("OpenF1 %s failed: %s", session.item_key, exc)
        ingestion_log.record(conn, SOURCE, session.item_key, "failed", message=str(exc))
        # The error-level issues, if it was a ValidationError; else none.
        checks.record_issues(conn, SOURCE, session.item_key, getattr(exc, "issues", []))
        outcomes[session.item_key] = "failed"
    else:
        logger.info("OpenF1 %s: %d rows%s", session.item_key, row_count,
                    f" ({checks.summarize(issues)})" if issues else "")
        ingestion_log.record(conn, SOURCE, session.item_key, "ok", row_count=row_count,
                             message=checks.summarize(issues))
        checks.record_issues(conn, SOURCE, session.item_key, issues)
        outcomes[session.item_key] = "ok"
