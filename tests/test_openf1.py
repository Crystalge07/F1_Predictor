"""Tests for the OpenF1 loader: parsing, the 404 quirk, session matching,
driver/team matching, and loading a session into SQLite.

The fake payloads use OpenF1's real field names (trimmed to a few rows).
"""

from datetime import datetime, timezone

import pytest
import requests

from f1predictor.db.connection import get_connection, init_db
from f1predictor.ingest import openf1


# -----------------------------------------------------------------------------
# Fakes and setup helpers
# -----------------------------------------------------------------------------

def http_404() -> requests.HTTPError:
    """Build the error our HTTP client raises for a 404, as requests does."""
    response = requests.Response()
    response.status_code = 404
    return requests.HTTPError("404 Not Found", response=response)


class FakeClient:
    """Answers get_json() from {(endpoint, session_key or year): rows}.

    Anything not in the dict raises a 404, which is what real OpenF1 does
    for "no results".
    """

    def __init__(self, responses: dict):
        self.responses = responses

    def get_json(self, url, params=None):
        endpoint = url.rsplit("/", 1)[1]           # '.../v1/laps' -> 'laps'
        key = (endpoint, params.get("session_key", params.get("year")))
        if key not in self.responses:
            raise http_404()
        return self.responses[key]


@pytest.fixture
def conn():
    connection = get_connection(":memory:")
    init_db(connection)
    yield connection
    connection.close()


def add_event(conn, season: int, round_number: int, race_date: str) -> int:
    conn.execute("INSERT INTO circuits (circuit_id, name) VALUES ('c', 'C') "
                 "ON CONFLICT DO NOTHING")
    cursor = conn.execute(
        "INSERT INTO events (season, round, name, circuit_id, race_date) "
        "VALUES (?, ?, 'Test GP', 'c', ?)",
        (season, round_number, race_date),
    )
    return cursor.lastrowid


def add_session(conn, event_id: int, session_type: str) -> int:
    cursor = conn.execute(
        "INSERT INTO sessions (event_id, session_type) VALUES (?, ?)", (event_id, session_type)
    )
    return cursor.lastrowid


def add_result(conn, session_id: int, driver_id: str, code: str, constructor_id: str, car: int):
    """A race result, plus the driver and constructor rows it needs."""
    conn.execute(
        "INSERT INTO drivers (driver_id, code, given_name, family_name) VALUES (?, ?, 'X', 'Y') "
        "ON CONFLICT DO NOTHING", (driver_id, code))
    conn.execute(
        "INSERT INTO constructors (constructor_id, name) VALUES (?, ?) ON CONFLICT DO NOTHING",
        (constructor_id, constructor_id))
    conn.execute(
        "INSERT INTO results (session_id, driver_id, constructor_id, car_number, position, position_text) "
        "VALUES (?, ?, ?, ?, 1, '1')", (session_id, driver_id, constructor_id, car))


# -----------------------------------------------------------------------------
# Parse helpers and fetch
# -----------------------------------------------------------------------------

def test_seconds_to_ms():
    assert openf1.seconds_to_ms(92.345) == 92_345
    assert openf1.seconds_to_ms(24.941) == 24_941
    assert openf1.seconds_to_ms(None) is None


def test_to_utc_iso_normalizes_format():
    assert openf1.to_utc_iso("2024-12-08T13:09:54.663000+00:00") == "2024-12-08T13:09:54.663Z"
    assert openf1.to_utc_iso("2024-12-08T13:00:00+00:00", timespec="seconds") == "2024-12-08T13:00:00Z"
    # A non-UTC offset is converted: 17:00 at +04:00 is 13:00 UTC.
    assert openf1.to_utc_iso("2024-12-08T17:00:00+04:00", timespec="seconds") == "2024-12-08T13:00:00Z"
    assert openf1.to_utc_iso(None) is None


def test_fetch_turns_404_into_empty_list():
    assert openf1.fetch(FakeClient({}), "pit", session_key=1) == []


def test_fetch_still_raises_other_errors():
    class BrokenClient:
        def get_json(self, url, params=None):
            response = requests.Response()
            response.status_code = 500
            raise requests.HTTPError("500", response=response)

    with pytest.raises(requests.HTTPError):
        openf1.fetch(BrokenClient(), "laps", session_key=1)


# -----------------------------------------------------------------------------
# Matching sessions
# -----------------------------------------------------------------------------

def openf1_session(key, meeting, name, start, cancelled=False):
    return {"session_key": key, "meeting_key": meeting, "session_name": name,
            "date_start": start, "is_cancelled": cancelled, "country_name": "X"}


def test_match_sessions(conn):
    # Our event: a night race, which Jolpica dates on Saturday...
    event_id = add_event(conn, 2024, 22, "2024-11-23")
    add_session(conn, event_id, "R")

    sessions = [
        # ...while OpenF1's UTC race start is Sunday. Must still match.
        openf1_session(100, 1, "Practice 1", "2024-11-21T02:30:00+00:00"),
        openf1_session(101, 1, "Race", "2024-11-24T06:00:00+00:00"),
        # Pre-season testing: no Race session, so the meeting is skipped.
        openf1_session(200, 2, "Day 1", "2024-02-21T07:00:00+00:00"),
        # A cancelled weekend: skipped.
        openf1_session(300, 3, "Practice 1", "2024-05-17T11:30:00+00:00", cancelled=True),
        openf1_session(301, 3, "Race", "2024-05-19T13:00:00+00:00", cancelled=True),
    ]

    matched = openf1.match_sessions(conn, 2024, sessions)

    assert sorted((m.session_type, m.session_key, m.item_key) for m in matched) == [
        ("FP1", 100, "2024-22-FP1"),
        ("R", 101, "2024-22-R"),
    ]
    rows = conn.execute(
        "SELECT session_type, openf1_session_key, start_time_utc FROM sessions ORDER BY session_type"
    ).fetchall()
    # FP1 didn't exist yet and was created; R got its key and exact start time.
    assert [tuple(r) for r in rows] == [
        ("FP1", 100, "2024-11-21T02:30:00Z"),
        ("R", 101, "2024-11-24T06:00:00Z"),
    ]


# -----------------------------------------------------------------------------
# Matching drivers and teams (option 3)
# -----------------------------------------------------------------------------

def test_match_entries_handles_regulars_rookies_and_fallbacks(conn):
    # Round 1 (already raced): Lawson drove for RB.
    event1 = add_event(conn, 2025, 1, "2025-03-16")
    add_result(conn, add_session(conn, event1, "R"), "lawson", "LAW", "rb", 30)

    # Round 2: race done for Norris and Piastri; Lawson's round-2 result isn't in yet.
    event2 = add_event(conn, 2025, 2, "2025-03-23")
    race2 = add_session(conn, event2, "R")
    add_result(conn, race2, "norris", "NOR", "mclaren", 4)
    add_result(conn, race2, "piastri", "PIA", "mclaren", 81)
    fp1 = add_session(conn, event2, "FP1")

    conn.executemany(
        "INSERT INTO session_entries (session_id, car_number, acronym, team_name) VALUES (?, ?, ?, ?)",
        [
            (fp1, 4, "NOR", "McLaren"),      # regular: exact car-number match
            (fp1, 29, "HIR", "McLaren"),     # FP1 rookie, never raced
            (fp1, 30, "LAW", "Racing Bulls"),  # no result at this event yet -> acronym fallback
        ],
    )

    matched, unmatched = openf1.match_entries(conn, fp1)

    rows = conn.execute(
        "SELECT car_number, driver_id, constructor_id FROM session_entries ORDER BY car_number"
    ).fetchall()
    assert [tuple(r) for r in rows] == [
        (4, "norris", "mclaren"),
        (29, None, "mclaren"),    # unknown driver, but team copied from teammate Norris
        (30, "lawson", "rb"),     # team from his most recent result
    ]
    assert (matched, unmatched) == (2, 1)


# -----------------------------------------------------------------------------
# Loading a session
# -----------------------------------------------------------------------------

SESSION_KEY = 9662


def session_payload() -> dict:
    """A tiny but realistic session. Note car 99 appears in laps but not drivers."""
    return {
        ("drivers", SESSION_KEY): [
            {"driver_number": 4, "name_acronym": "NOR", "full_name": "Lando NORRIS",
             "team_name": "McLaren"},
        ],
        ("laps", SESSION_KEY): [
            {"driver_number": 4, "lap_number": 1, "date_start": "2024-12-08T13:03:00.100000+00:00",
             "lap_duration": 90.5, "duration_sector_1": 30.1, "duration_sector_2": 30.2,
             "duration_sector_3": 30.2, "is_pit_out_lap": False, "st_speed": 320},
            {"driver_number": 99, "lap_number": 1, "lap_duration": None},
        ],
        ("stints", SESSION_KEY): [
            {"driver_number": 4, "stint_number": 1, "compound": "MEDIUM",
             "lap_start": 1, "lap_end": 1, "tyre_age_at_start": 0},
        ],
        ("weather", SESSION_KEY): [
            {"date": "2024-12-08T13:00:00.000000+00:00", "air_temperature": 26.0, "rainfall": 0},
        ],
        # No 'pit' or 'race_control' entries -> the fake returns 404 -> [].
    }


def test_ingest_session_loads_rows_and_creates_missing_entries(conn):
    event_id = add_event(conn, 2024, 24, "2024-12-08")
    race = add_session(conn, event_id, "R")
    add_result(conn, race, "norris", "NOR", "mclaren", 4)

    rows = openf1.ingest_session(conn, FakeClient(session_payload()), race, SESSION_KEY)

    assert rows == 5   # 1 driver + 2 laps + 1 stint + 1 weather
    entries = conn.execute(
        "SELECT car_number, driver_id FROM session_entries ORDER BY car_number"
    ).fetchall()
    # Car 99 got a bare entry so its lap wasn't rejected by the foreign key.
    assert [tuple(r) for r in entries] == [(4, "norris"), (99, None)]

    lap = conn.execute(
        "SELECT lap_time_ms, compound, tyre_age, lap_start_utc FROM laps_detailed WHERE car_number = 4"
    ).fetchone()
    assert tuple(lap) == (90_500, "MEDIUM", 0, "2024-12-08T13:03:00.100Z")
    assert conn.execute("SELECT rainfall FROM weather").fetchone()[0] == 0   # False


def test_ingest_session_is_idempotent(conn):
    event_id = add_event(conn, 2024, 24, "2024-12-08")
    race = add_session(conn, event_id, "R")
    for _ in range(2):
        openf1.ingest_session(conn, FakeClient(session_payload()), race, SESSION_KEY)
    assert conn.execute("SELECT COUNT(*) FROM laps").fetchone()[0] == 2


def test_session_without_laps_raises_no_data_yet(conn):
    event_id = add_event(conn, 2024, 24, "2024-12-08")
    race = add_session(conn, event_id, "R")
    with pytest.raises(openf1.NoDataYet):
        openf1.ingest_session(conn, FakeClient({}), race, SESSION_KEY)


# -----------------------------------------------------------------------------
# A whole season
# -----------------------------------------------------------------------------

def test_ingest_season_skips_future_and_done_sessions_and_logs_failures(conn):
    add_event(conn, 2024, 24, "2024-12-08")
    responses = session_payload()
    responses[("sessions", 2024)] = [
        openf1_session(SESSION_KEY, 1, "Race", "2024-12-08T13:00:00+00:00"),
        openf1_session(9999, 1, "Qualifying", "2024-12-07T14:00:00+00:00"),  # no laps -> fails
    ]
    client = FakeClient(responses)

    # Pretend it's Saturday evening: qualifying has finished, the race hasn't started.
    saturday = datetime(2024, 12, 7, 20, 0, tzinfo=timezone.utc)
    outcomes = openf1.ingest_season(conn, client, 2024, now=saturday)
    assert outcomes == {"2024-sessions": "ok", "2024-24-Q": "failed"}

    # Monday: the race is loaded; qualifying is retried (and fails again).
    monday = datetime(2024, 12, 9, 12, 0, tzinfo=timezone.utc)
    outcomes = openf1.ingest_season(conn, client, 2024, now=monday)
    assert outcomes["2024-24-R"] == "ok"
    assert outcomes["2024-24-Q"] == "failed"

    # Running again skips the race, since it's already logged as ok.
    outcomes = openf1.ingest_season(conn, client, 2024, now=monday)
    assert outcomes["2024-24-R"] == "already done"

    log = conn.execute(
        "SELECT status, message FROM ingestion_log WHERE item_key = '2024-24-Q'"
    ).fetchone()
    assert log["status"] == "failed" and "no laps" in log["message"]
