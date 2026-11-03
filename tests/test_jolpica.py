"""Tests for the Jolpica loader: parsing, page merging, and loading into SQLite.

The fake payloads below are trimmed copies of real Jolpica responses (same
keys, fewer rows), so the tests exercise the real JSON shapes.
"""

import pytest

from f1predictor.db.connection import get_connection, init_db
from f1predictor.ingest import jolpica
from f1predictor.validate import checks


# -----------------------------------------------------------------------------
# Parse helpers
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    # parametrize runs the same test once per (text, expected) pair.
    "text, expected",
    [
        ("1:29.708", 89_708),
        ("1:30.000", 90_000),
        ("59.123", 59_123),
        ("2:01.001", 121_001),
        ("", None),
        (None, None),
    ],
)
def test_parse_lap_time_ms(text, expected):
    assert jolpica.parse_lap_time_ms(text) == expected


def test_event_format_from_schedule_keys():
    assert jolpica.event_format({"SprintQualifying": {}, "Sprint": {}}) == "sprint_qualifying"
    assert jolpica.event_format({"SprintShootout": {}, "Sprint": {}}) == "sprint_shootout"
    assert jolpica.event_format({"Sprint": {}}) == "sprint"
    assert jolpica.event_format({"FirstPractice": {}}) == "conventional"


def test_to_utc_timestamp_without_time_is_none():
    assert jolpica.to_utc_timestamp("2024-05-03", "16:30:00Z") == "2024-05-03T16:30:00Z"
    assert jolpica.to_utc_timestamp("2021-07-16", None) is None


# -----------------------------------------------------------------------------
# Fake API payloads
# -----------------------------------------------------------------------------

CIRCUIT = {
    "circuitId": "miami", "circuitName": "Miami International Autodrome",
    "Location": {"lat": "25.9581", "long": "-80.2389", "locality": "Miami", "country": "USA"},
}

SCHEDULE_RACE = {
    "season": "2024", "round": "6", "raceName": "Miami Grand Prix", "Circuit": CIRCUIT,
    "date": "2024-05-05", "time": "20:00:00Z",
    "FirstPractice": {"date": "2024-05-03", "time": "16:30:00Z"},
    "SprintQualifying": {"date": "2024-05-03", "time": "20:30:00Z"},
    "Sprint": {"date": "2024-05-04", "time": "16:00:00Z"},
    "Qualifying": {"date": "2024-05-04", "time": "20:00:00Z"},
}


def result_row(driver_id: str, position: int, grid: int, millis: str | None) -> dict:
    """One race result row in Jolpica's shape."""
    row = {
        "number": str(position), "position": str(position), "positionText": str(position),
        "points": "0", "grid": str(grid), "laps": "57", "status": "Finished",
        "Driver": {"driverId": driver_id, "code": driver_id[:3].upper(),
                   "givenName": driver_id.title(), "familyName": driver_id.title()},
        "Constructor": {"constructorId": "mclaren", "name": "McLaren"},
    }
    if millis:
        row["Time"] = {"millis": millis}
    return row


def page(races: list[dict], total: int) -> dict:
    """Wrap races in Jolpica's MRData envelope."""
    return {"MRData": {"total": str(total), "RaceTable": {"Races": races}}}


class FakeClient:
    """Answers get_json() from a dict of {(path, offset): response}."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.requested: list[tuple[str, int]] = []

    def get_json(self, url, params=None):
        # Turn the full URL back into just the path, e.g. '2024/results'.
        path = url.removeprefix(jolpica.config.JOLPICA_BASE_URL + "/").removesuffix(".json")
        key = (path, params["offset"])
        self.requested.append(key)
        return self.responses[key]


# -----------------------------------------------------------------------------
# Fetch: page merging
# -----------------------------------------------------------------------------

def test_fetch_races_merges_a_race_split_across_pages(monkeypatch):
    # Pretend pages hold 2 rows, so 3 rows of round 6 span two pages.
    monkeypatch.setattr(jolpica, "PAGE_LIMIT", 2)
    meta = {"season": "2024", "round": "6"}
    client = FakeClient({
        ("2024/results", 0): page([{**meta, "Results": ["a", "b"]}], total=3),
        ("2024/results", 2): page([{**meta, "Results": ["c"]}], total=3),
    })

    races = jolpica.fetch_races(client, "2024/results", "Results")

    assert len(races) == 1
    assert races[0]["Results"] == ["a", "b", "c"]
    assert client.requested == [("2024/results", 0), ("2024/results", 2)]


# -----------------------------------------------------------------------------
# Load: end to end into an in-memory database
# -----------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def allow_tiny_fake_races(monkeypatch):
    """The fake races below have 2 drivers, far fewer than any real race.

    The validation check `too_few_results` would (correctly) reject them, so
    for these loader tests we lower its threshold. autouse=True applies this
    fixture to every test in this file without each test asking for it.
    The checks themselves are tested in test_checks.py.
    """
    monkeypatch.setattr(checks, "MIN_DRIVERS_PER_RESULT", 1)


@pytest.fixture
def conn():
    connection = get_connection(":memory:")
    init_db(connection)
    yield connection
    connection.close()


def fake_season_client() -> FakeClient:
    return FakeClient({
        ("2024", 0): page([SCHEDULE_RACE], total=1),
        ("2024/results", 0): page([{**SCHEDULE_RACE, "Results": [
            result_row("norris", 1, 5, "5719000"),
            result_row("piastri", 2, 3, None),     # no Time key, like a non-finisher
        ]}], total=2),
        ("2024/sprint", 0): page([], total=0),
        ("2024/qualifying", 0): page([{**SCHEDULE_RACE, "QualifyingResults": [
            {"position": "1", "Q1": "1:28.000", "Q2": "1:27.500",
             "Driver": {"driverId": "norris", "givenName": "Lando", "familyName": "Norris"},
             "Constructor": {"constructorId": "mclaren", "name": "McLaren"}},
        ]}], total=1),
    })


def test_ingest_season_loads_everything(conn):
    outcomes = jolpica.ingest_season(conn, fake_season_client(), 2024)
    assert set(outcomes.values()) == {"ok"}

    # Sprint weekend: FP1, SQ, S, Q, R (no FP2/FP3), read from the schedule keys.
    session_types = {r["session_type"] for r in conn.execute("SELECT session_type FROM sessions")}
    assert session_types == {"FP1", "SQ", "S", "Q", "R"}

    event = conn.execute("SELECT * FROM events").fetchone()
    assert event["event_format"] == "sprint_qualifying"

    results = conn.execute(
        "SELECT driver_id, grid, race_time_ms FROM results ORDER BY position"
    ).fetchall()
    assert [tuple(r) for r in results] == [("norris", 5, 5719000), ("piastri", 3, None)]

    quali = conn.execute("SELECT q1_ms, q2_ms, q3_ms FROM qualifying").fetchone()
    assert tuple(quali) == (88_000, 87_500, None)


def test_ingest_season_is_idempotent(conn):
    jolpica.ingest_season(conn, fake_season_client(), 2024)
    jolpica.ingest_season(conn, fake_season_client(), 2024)   # run again
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 2


def test_validation_error_rolls_back_and_is_recorded(conn):
    client = fake_season_client()
    # Both drivers classified 1st: positions aren't 1..N -> error-level issue.
    client.responses[("2024/results", 0)] = page([{**SCHEDULE_RACE, "Results": [
        result_row("norris", 1, 5, None),
        result_row("piastri", 1, 3, None),
    ]}], total=2)

    outcomes = jolpica.ingest_season(conn, client, 2024)

    assert outcomes["2024-results"] == "failed"
    assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 0   # nothing saved
    issue = conn.execute(
        "SELECT check_name, severity FROM validation_issues WHERE item_key = '2024-results'"
    ).fetchone()
    assert tuple(issue) == ("positions_not_sequential", "error")


def test_failed_part_is_logged_and_rolled_back(conn):
    client = fake_season_client()
    # Results for round 99, which isn't in the schedule -> get_session_id raises.
    client.responses[("2024/results", 0)] = page(
        [{**SCHEDULE_RACE, "round": "99", "Results": [result_row("norris", 1, 1, None)]}],
        total=1,
    )

    outcomes = jolpica.ingest_season(conn, client, 2024)

    assert outcomes["2024-results"] == "failed"
    assert outcomes["2024-qualifying"] == "ok"        # later parts still ran
    assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 0
    log = conn.execute(
        "SELECT status, message FROM ingestion_log WHERE item_key = '2024-results'"
    ).fetchone()
    assert log["status"] == "failed" and "round 99" in log["message"]
