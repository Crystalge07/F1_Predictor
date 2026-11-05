"""Tests for the OpenF1 loader: parsing, the 404 quirk, session matching,
driver/team matching, and loading a session into SQLite.

The fake payloads use OpenF1's real field names (trimmed to a few rows).
"""

from datetime import datetime, timezone

import pytest
import requests

from f1predictor.db.connection import get_connection, init_db
from f1predictor.db import ingestion_log
from f1predictor.ingest import openf1
from f1predictor.validate import checks


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


def add_qualifier(conn, quali_session_id: int, driver_id: str, code: str, constructor_id: str):
    """A Jolpica qualifying row (which, unlike results, has no car number)."""
    conn.execute("INSERT INTO drivers (driver_id, code, given_name, family_name) VALUES (?, ?, 'X', 'Y') "
                 "ON CONFLICT DO NOTHING", (driver_id, code))
    conn.execute("INSERT INTO constructors (constructor_id, name) VALUES (?, ?) ON CONFLICT DO NOTHING",
                 (constructor_id, constructor_id))
    conn.execute("INSERT INTO qualifying (session_id, driver_id, constructor_id, position) "
                 "VALUES (?, ?, ?, 1)", (quali_session_id, driver_id, constructor_id))


def test_a_debutant_is_matched_from_this_events_qualifying(conn):
    # Real case, Mexico City 2026: Hirakawa's first race. He has no race
    # result anywhere yet, so the acronym rule (most recent RESULT) can't
    # find him, and the post-quali prediction lost him. Jolpica's qualifying
    # for this event already lists him: that's enough to match car 50.
    event = add_event(conn, 2026, 19, "2026-11-01")
    add_session(conn, event, "R")
    quali = add_session(conn, event, "Q")
    add_qualifier(conn, quali, "ryo_hirakawa", "HIR", "haas")
    conn.execute("INSERT INTO session_entries (session_id, car_number, acronym, team_name) "
                 "VALUES (?, 50, 'HIR', 'Haas F1 Team')", (quali,))

    assert openf1.match_entries(conn, quali) == (1, 0)
    row = conn.execute("SELECT driver_id, constructor_id FROM session_entries").fetchone()
    # The team comes from his qualifying row (no result to take it from).
    assert tuple(row) == ("ryo_hirakawa", "haas")


def test_a_car_missing_from_the_driver_list_is_matched_from_the_same_weekend(conn):
    # Real case, Mexico City 2026 FP3: OpenF1's /drivers left out car 30, but
    # it set laps, so the loader made an entry with no acronym. Car 30 was
    # matched in FP2 of the same weekend, and car numbers are unique within
    # a weekend, so it's the same driver.
    event = add_event(conn, 2026, 19, "2026-11-01")
    add_session(conn, event, "R")
    add_result(conn, add_session(conn, add_event(conn, 2026, 18, "2026-10-25"), "R"),
               "lawson", "LAW", "rb", 30)                     # his latest team, from Austin
    fp2, fp3 = add_session(conn, event, "FP2"), add_session(conn, event, "FP3")
    conn.execute("INSERT INTO session_entries (session_id, car_number, acronym, team_name) "
                 "VALUES (?, 30, 'LAW', 'Racing Bulls')", (fp2,))
    conn.execute("INSERT INTO session_entries (session_id, car_number) VALUES (?, 30)", (fp3,))
    openf1.match_entries(conn, fp2)

    assert openf1.match_entries(conn, fp3) == (1, 0)
    row = conn.execute("SELECT driver_id, constructor_id FROM session_entries WHERE session_id = ?",
                       (fp3,)).fetchone()
    assert tuple(row) == ("lawson", "rb")


def test_a_nameless_car_is_not_matched_from_another_weekend(conn):
    # The same car number at a DIFFERENT event says nothing: numbers can be
    # reused (e.g. a reserve driver), so the rule stays within one weekend.
    other = add_event(conn, 2026, 18, "2026-10-25")
    add_session(conn, other, "R")
    other_fp1 = add_session(conn, other, "FP1")
    add_result(conn, add_session(conn, add_event(conn, 2026, 17, "2026-10-11"), "R"), "lawson", "LAW", "rb", 30)
    conn.execute("INSERT INTO session_entries (session_id, car_number, acronym) VALUES (?, 30, 'LAW')",
                 (other_fp1,))
    openf1.match_entries(conn, other_fp1)
    event = add_event(conn, 2026, 19, "2026-11-01")
    add_session(conn, event, "R")
    fp3 = add_session(conn, event, "FP3")
    conn.execute("INSERT INTO session_entries (session_id, car_number) VALUES (?, 30)", (fp3,))
    assert openf1.match_entries(conn, fp3) == (0, 1)


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

    rows, _issues = openf1.ingest_session(conn, FakeClient(session_payload()), race, SESSION_KEY)

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


def test_phantom_cars_in_stints_are_dropped(conn):
    # Real case (Austria 2025 qualifying): OpenF1 sent one stint each for car
    # numbers that weren't in the session, with no laps and no driver info.
    event_id = add_event(conn, 2024, 24, "2024-12-08")
    race = add_session(conn, event_id, "R")
    payload = session_payload()
    payload[("stints", SESSION_KEY)].append(
        {"driver_number": 7, "stint_number": 1, "compound": "SOFT",
         "lap_start": 1, "lap_end": 1, "tyre_age_at_start": 0})

    openf1.ingest_session(conn, FakeClient(payload), race, SESSION_KEY)

    cars = [r[0] for r in conn.execute("SELECT car_number FROM session_entries ORDER BY 1")]
    assert cars == [4, 99]      # no entry for phantom car 7
    assert conn.execute("SELECT COUNT(*) FROM stints WHERE car_number = 7").fetchone()[0] == 0


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


# -----------------------------------------------------------------------------
# Starting grid (Phase 4: the grid known BEFORE the race)
# -----------------------------------------------------------------------------

def test_load_starting_grid_keeps_known_cars_and_replaces_old_grid(conn):
    event_id = add_event(conn, 2024, 24, "2024-12-08")
    quali = add_session(conn, event_id, "Q")
    conn.executemany("INSERT INTO session_entries (session_id, car_number) VALUES (?, ?)",
                     [(quali, 4), (quali, 16)])
    grid = [
        {"position": 1, "driver_number": 16, "session_key": 77},
        {"position": 2, "driver_number": 4, "session_key": 77},
        {"position": 3, "driver_number": 99, "session_key": 77},   # not in qualifying: dropped
    ]
    client = FakeClient({("starting_grid", 77): grid})

    assert openf1.load_starting_grid(conn, client, quali, 77)[0] == 2
    # Running again (e.g. after a late penalty) replaces, never duplicates.
    grid[0]["position"], grid[1]["position"] = 2, 1
    assert openf1.load_starting_grid(conn, client, quali, 77)[0] == 2

    rows = conn.execute("SELECT car_number, position FROM starting_grid ORDER BY position").fetchall()
    assert [tuple(r) for r in rows] == [(4, 1), (16, 2)]


def test_missing_starting_grid_stores_nothing(conn):
    # OpenF1 answers 404 until the grid is published: that's 0 slots, not an error.
    event_id = add_event(conn, 2024, 24, "2024-12-08")
    quali = add_session(conn, event_id, "Q")
    assert openf1.load_starting_grid(conn, FakeClient({}), quali, 77)[0] == 0


def test_partial_starting_grid_is_rejected_and_keeps_the_last_good_grid(conn):
    event_id = add_event(conn, 2024, 24, "2024-12-08")
    quali = add_session(conn, event_id, "Q")
    conn.executemany("INSERT INTO session_entries (session_id, car_number) VALUES (?, ?)",
                     [(quali, car) for car in range(1, 7)])
    complete = [{"position": car, "driver_number": car, "session_key": 77}
                for car in range(1, 7)]
    client = FakeClient({("starting_grid", 77): complete})
    assert openf1.load_starting_grid(conn, client, quali, 77)[0] == 6

    # A one-row refresh is an incomplete API response, not five pit-lane
    # starters. Validation runs in the replacement transaction, so the last
    # complete grid remains available for a later retry.
    client.responses[("starting_grid", 77)] = complete[:1]
    with pytest.raises(checks.ValidationError, match="starting_grid_incomplete"):
        openf1.load_starting_grid(conn, client, quali, 77)
    rows = conn.execute(
        "SELECT car_number, position FROM starting_grid WHERE session_id = ? ORDER BY position",
        (quali,)).fetchall()
    assert [tuple(row) for row in rows] == [(car, car) for car in range(1, 7)]


def six_car_grid(conn):
    """A qualifying session with cars 1-6 and a complete grid stored (1..6)."""
    event_id = add_event(conn, 2024, 24, "2024-12-08")
    quali = add_session(conn, event_id, "Q")
    conn.executemany("INSERT INTO session_entries (session_id, car_number) VALUES (?, ?)",
                     [(quali, car) for car in range(1, 7)])
    complete = [{"position": car, "driver_number": car, "session_key": 77} for car in range(1, 7)]
    client = FakeClient({("starting_grid", 77): complete})
    assert openf1.load_starting_grid(conn, client, quali, 77)[0] == 6
    return quali, client, complete


def stored_grid(conn, quali):
    return [tuple(row) for row in conn.execute(
        "SELECT car_number, position FROM starting_grid WHERE session_id = ? ORDER BY position",
        (quali,))]


def test_empty_refresh_keeps_the_stored_grid(conn):
    # A refresh that comes back empty (404 / blip) must not wipe a good grid.
    # It reports the stored size, so ingest_season logs 'ok' again and the
    # planner keeps treating the grid as available.
    quali, client, _ = six_car_grid(conn)
    del client.responses[("starting_grid", 77)]
    assert openf1.load_starting_grid(conn, client, quali, 77)[0] == 6
    assert stored_grid(conn, quali) == [(car, car) for car in range(1, 7)]


def refresh_with(conn, client, quali, car_slots: dict[int, int]):
    """Refresh the stored six-car grid with OpenF1 now answering {car: slot}."""
    client.responses[("starting_grid", 77)] = [
        {"position": slot, "driver_number": car, "session_key": 77} for car, slot in car_slots.items()]
    return openf1.load_starting_grid(conn, client, quali, 77)


def test_mid_grid_pit_lane_move_is_accepted_and_flagged(conn):
    # Car 3 moves to the pit lane after the first fetch: OpenF1 drops it and
    # cars 4, 5, 6 each move up exactly one place. Accepted, with a warning
    # naming the car (the tick turns it into an alert).
    quali, client, _ = six_car_grid(conn)
    slots, issues = refresh_with(conn, client, quali, {1: 1, 2: 2, 4: 3, 5: 4, 6: 5})
    assert slots == 5
    assert stored_grid(conn, quali) == [(1, 1), (2, 2), (4, 3), (5, 4), (6, 5)]
    assert [(i.check, i.severity) for i in issues] == [("starting_grid_pit_lane_shrink", "warning")]
    assert "car 3 (was 3)" in issues[0].detail


def test_truncated_tail_is_rejected(conn):
    # Cars 5 and 6 missing, nothing else changed: a response cut short.
    quali, client, complete = six_car_grid(conn)
    with pytest.raises(checks.ValidationError, match="starting_grid_shrank.*at the back"):
        refresh_with(conn, client, quali, {1: 1, 2: 2, 3: 3, 4: 4})
    assert stored_grid(conn, quali) == [(car, car) for car in range(1, 7)]


def test_back_of_grid_car_makes_a_shrink_ambiguous(conn):
    # Car 2 (mid-grid) left cleanly, but car 6 was LAST: nobody behind it to
    # move up, so "pit lane" and "cut short" look the same. One ambiguous car
    # is enough to reject the whole refresh.
    quali, client, _ = six_car_grid(conn)
    with pytest.raises(checks.ValidationError, match=r"car\(s\) \[6\] were at the back"):
        refresh_with(conn, client, quali, {1: 1, 3: 2, 4: 3, 5: 4})
    assert stored_grid(conn, quali) == [(car, car) for car in range(1, 7)]


def test_shrink_with_any_other_reordering_is_rejected(conn):
    # Car 3 left and the rest moved up, but 4 and 5 also swapped: more than
    # a pit-lane move, so the stored grid stays.
    quali, client, _ = six_car_grid(conn)
    with pytest.raises(checks.ValidationError, match=r"other cars changed places as well: \[4, 5\]"):
        refresh_with(conn, client, quali, {1: 1, 2: 2, 5: 3, 4: 4, 6: 5})
    assert stored_grid(conn, quali) == [(car, car) for car in range(1, 7)]


def test_accepted_shrink_is_recorded_for_the_tick_to_alert(conn):
    # Through ingest_season: an accepted pit-lane shrink logs 'ok' AND leaves
    # its warning in validation_issues, which is where tick._fetch finds it.
    add_session(conn, add_event(conn, 2024, 24, "2024-12-08"), "Q")
    responses = session_payload()
    responses[("sessions", 2024)] = [
        openf1_session(1234, 1, "Race", "2024-12-08T13:00:00+00:00"),
        openf1_session(SESSION_KEY, 1, "Qualifying", "2024-12-07T14:00:00+00:00")]
    quali = conn.execute("SELECT session_id FROM sessions WHERE session_type = 'Q'").fetchone()[0]
    conn.executemany("INSERT INTO session_entries (session_id, car_number) VALUES (?, ?)",
                     [(quali, car) for car in (1, 2, 3)])          # car 4 comes from the payload
    responses[("starting_grid", SESSION_KEY)] = [
        {"position": slot, "driver_number": car, "session_key": SESSION_KEY}
        for car, slot in {1: 1, 4: 2, 2: 3, 3: 4}.items()]
    client = FakeClient(responses)
    monday = datetime(2024, 12, 9, 12, 0, tzinfo=timezone.utc)
    assert openf1.ingest_season(conn, client, 2024, now=monday, session_types={"Q"},
                                settle_hours=0)["2024-24-grid"] == "ok"

    # Car 4 (2nd) to the pit lane: 2 and 3 move up one place each.
    responses[("starting_grid", SESSION_KEY)] = [
        {"position": slot, "driver_number": car, "session_key": SESSION_KEY}
        for car, slot in {1: 1, 2: 2, 3: 3}.items()]
    again = openf1.ingest_season(conn, client, 2024, now=monday, session_types={"Q"},
                                 settle_hours=0, refresh_grid=True)
    assert again["2024-24-grid"] == "ok"
    recorded = [tuple(row) for row in conn.execute(
        "SELECT check_name, severity FROM validation_issues WHERE item_key = '2024-24-grid'")]
    assert recorded == [("starting_grid_pit_lane_shrink", "warning")]


def test_grid_shrink_problem_rules():
    old = {44: 1, 1: 2, 16: 3, 4: 4}
    assert checks.grid_shrink_problem(old, {44: 1, 16: 2, 4: 3}) is None          # 1 to the pit lane
    assert checks.grid_shrink_problem(old, {16: 1, 4: 2}) is None                  # 44 and 1 both
    assert "at the back" in checks.grid_shrink_problem(old, {44: 1, 1: 2, 16: 3})  # 4 was last
    assert "changed places" in checks.grid_shrink_problem(old, {44: 1, 4: 2, 16: 3})
    assert "not in the stored grid" in checks.grid_shrink_problem(old, {44: 1, 16: 2, 99: 3})


@pytest.mark.parametrize("positions, check", [
    ([1, 2, 4, 5, 6, 7], "starting_grid_position_too_large"),   # 7 with only 6 cars
    ([1, 2, 4, 5, 6, 7], "starting_grid_gaps"),                  # ...and place 3 empty
    ([1, 2, 3, 5, 6], "starting_grid_gaps"),                     # 5 cars, place 4 empty
])
def test_grid_with_a_gap_or_a_slot_past_the_field_is_rejected(conn, positions, check):
    event_id = add_event(conn, 2024, 24, "2024-12-08")
    quali = add_session(conn, event_id, "Q")
    conn.executemany("INSERT INTO session_entries (session_id, car_number) VALUES (?, ?)",
                     [(quali, car) for car in range(1, 7)])
    grid = [{"position": pos, "driver_number": car, "session_key": 77}
            for car, pos in zip(range(1, 7), positions)]
    with pytest.raises(checks.ValidationError, match=check):
        openf1.load_starting_grid(conn, FakeClient({("starting_grid", 77): grid}), quali, 77)
    assert stored_grid(conn, quali) == []


def test_ingest_season_can_load_only_chosen_sessions(conn):
    # The weekend pipeline asks for exactly the sessions its planner found ready.
    add_event(conn, 2024, 24, "2024-12-08")
    responses = session_payload()
    responses[("sessions", 2024)] = [
        openf1_session(SESSION_KEY, 1, "Race", "2024-12-08T13:00:00+00:00"),
        openf1_session(9999, 1, "Qualifying", "2024-12-07T14:00:00+00:00"),
    ]
    monday = datetime(2024, 12, 9, 12, 0, tzinfo=timezone.utc)
    outcomes = openf1.ingest_season(conn, FakeClient(responses), 2024, now=monday,
                                    session_types={"R"}, settle_hours=0)
    assert outcomes == {"2024-sessions": "ok", "2024-24-R": "ok"}     # Q not touched
