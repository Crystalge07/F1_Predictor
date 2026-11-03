"""Tests for the backfill's resume logic.

The real loaders are replaced with fakes (via monkeypatch) that just record
which seasons they were asked for. That isolates what we're testing here:
the decisions about what to skip and what to re-fetch.
"""

import pytest

from f1predictor.db import ingestion_log
from f1predictor.db.connection import get_connection, init_db
from f1predictor.ingest import backfill as backfill_module


@pytest.fixture
def conn():
    connection = get_connection(":memory:")
    init_db(connection)
    yield connection
    connection.close()


def add_season(conn, season: int, last_race: str) -> None:
    conn.execute("INSERT INTO circuits (circuit_id, name) VALUES ('c', 'C') ON CONFLICT DO NOTHING")
    conn.execute("INSERT INTO events (season, round, name, circuit_id, race_date) "
                 "VALUES (?, 1, 'GP', 'c', ?)", (season, last_race))


def log_jolpica(conn, season: int, fetched_at: str, status: str = "ok") -> None:
    for part in backfill_module.JOLPICA_PARTS:
        ingestion_log.record(conn, "jolpica", f"{season}-{part}", status)
    # record() stamps the real current time; overwrite it with the test's time.
    conn.execute("UPDATE ingestion_log SET ingested_at = ?", (fetched_at,))


# -----------------------------------------------------------------------------
# When is a Jolpica season complete?
# -----------------------------------------------------------------------------

def test_season_fetched_after_it_ended_is_complete(conn):
    add_season(conn, 2024, "2024-12-08")
    log_jolpica(conn, 2024, "2025-01-15T10:00:00+00:00")
    assert backfill_module.jolpica_season_complete(conn, 2024)


def test_season_fetched_during_the_season_is_not_complete(conn):
    # Fetched mid-season: later races' results weren't published yet.
    add_season(conn, 2026, "2026-12-06")
    log_jolpica(conn, 2026, "2026-11-02T10:00:00+00:00")
    assert not backfill_module.jolpica_season_complete(conn, 2026)


def test_season_fetched_the_day_after_the_finale_is_not_complete(conn):
    # Inside the lag window: the final results might not be in Jolpica yet.
    add_season(conn, 2024, "2024-12-08")
    log_jolpica(conn, 2024, "2024-12-09T10:00:00+00:00")
    assert not backfill_module.jolpica_season_complete(conn, 2024)


def test_season_with_a_failed_part_is_not_complete(conn):
    add_season(conn, 2024, "2024-12-08")
    log_jolpica(conn, 2024, "2025-01-15T10:00:00+00:00")
    ingestion_log.record(conn, "jolpica", "2024-qualifying", "failed")
    assert not backfill_module.jolpica_season_complete(conn, 2024)


# -----------------------------------------------------------------------------
# The backfill loop
# -----------------------------------------------------------------------------

@pytest.fixture
def fake_loaders(monkeypatch):
    """Replace both loaders with fakes. Returns the list of calls they received."""
    calls = []

    def fake_jolpica(conn, client, season):
        calls.append(("jolpica", season))
        return {f"{season}-results": "ok"}

    def fake_openf1(conn, client, season, skip_done, now):
        calls.append(("openf1", season, skip_done))
        return {f"{season}-sessions": "ok", f"{season}-01-R": "ok"}

    monkeypatch.setattr(backfill_module.jolpica, "ingest_season", fake_jolpica)
    monkeypatch.setattr(backfill_module.openf1, "ingest_season", fake_openf1)
    return calls


def test_openf1_only_runs_for_seasons_it_covers(conn, fake_loaders):
    backfill_module.backfill(conn, [2022, 2023], None, None)
    assert fake_loaders == [("jolpica", 2022), ("jolpica", 2023), ("openf1", 2023, True)]


def test_complete_jolpica_seasons_are_skipped_unless_forced(conn, fake_loaders):
    add_season(conn, 2024, "2024-12-08")
    log_jolpica(conn, 2024, "2025-01-15T10:00:00+00:00")

    tally = backfill_module.backfill(conn, [2024], None, None, include_openf1=False)
    assert fake_loaders == []
    assert tally[("jolpica", "already done")] == 4

    backfill_module.backfill(conn, [2024], None, None, include_openf1=False, force=True)
    assert fake_loaders == [("jolpica", 2024)]


def test_tally_counts_openf1_sessions_not_the_session_list(conn, fake_loaders):
    tally = backfill_module.backfill(conn, [2024], None, None)
    assert tally[("openf1", "ok")] == 1    # just '2024-01-R', not '2024-sessions'


def test_rematch_fills_in_drivers_once_results_exist(conn):
    add_season(conn, 2026, "2026-03-08")
    event_id = conn.execute("SELECT event_id FROM events").fetchone()[0]
    fp1 = conn.execute("INSERT INTO sessions (event_id, session_type) VALUES (?, 'FP1')",
                       (event_id,)).lastrowid
    race = conn.execute("INSERT INTO sessions (event_id, session_type) VALUES (?, 'R')",
                        (event_id,)).lastrowid
    # A debut driver's FP1 entry, loaded before Jolpica had any result for them.
    conn.execute("INSERT INTO session_entries (session_id, car_number, acronym) "
                 "VALUES (?, 41, 'LIN')", (fp1,))

    # Later: the race result arrives.
    conn.execute("INSERT INTO drivers (driver_id, code, given_name, family_name) "
                 "VALUES ('lindblad', 'LIN', 'Arvid', 'Lindblad')")
    conn.execute("INSERT INTO constructors (constructor_id, name) VALUES ('rb', 'RB')")
    conn.execute("INSERT INTO results (session_id, driver_id, constructor_id, car_number, "
                 "position, position_text) VALUES (?, 'lindblad', 'rb', 41, 1, '1')", (race,))

    assert backfill_module.rematch_unmatched_entries(conn) == 1
    assert conn.execute("SELECT driver_id FROM session_entries").fetchone()[0] == "lindblad"
