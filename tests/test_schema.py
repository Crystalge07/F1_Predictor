"""Tests for the database schema.

These don't test F1 data. They test that the schema protects us from bad data:
foreign keys reject orphans, UNIQUE stops duplicates, CHECK rejects invalid
values. If someone later edits schema.sql and breaks one of these guarantees,
a test fails.
"""

import sqlite3

import pytest

from f1predictor.db.connection import get_connection, init_db, list_tables


@pytest.fixture
def conn():
    """A fresh, empty, in-memory database with the schema applied.

    A pytest "fixture" is setup code shared by tests: any test that takes a
    parameter named `conn` gets this. ":memory:" means nothing touches disk,
    and every test starts from a clean database.
    """
    connection = get_connection(":memory:")
    init_db(connection)
    yield connection          # the test runs at this point
    connection.close()        # cleanup after the test finishes


def insert_minimal_session(conn) -> int:
    """Insert the smallest valid circuit -> event -> session chain.

    Returns the new session_id. Several tests need a session to attach data to.
    """
    conn.execute("INSERT INTO circuits (circuit_id, name) VALUES ('monza', 'Monza')")
    cursor = conn.execute(
        "INSERT INTO events (season, round, name, circuit_id, race_date) "
        "VALUES (2024, 16, 'Italian Grand Prix', 'monza', '2024-09-01')"
    )
    # lastrowid is the auto-assigned id of the row we just inserted.
    event_id = cursor.lastrowid
    cursor = conn.execute(
        "INSERT INTO sessions (event_id, session_type) VALUES (?, 'R')", (event_id,)
    )
    return cursor.lastrowid


def test_all_tables_created(conn):
    expected = {
        "circuits", "drivers", "constructors", "events", "sessions",
        "results", "qualifying", "laps", "stints", "weather",
        "track_status", "ingestion_log",
    }
    # <= means "is a subset of": every expected name must be present.
    assert expected <= set(list_tables(conn))


def test_init_db_is_safe_to_rerun(conn):
    # Running the schema a second time must not fail or wipe data.
    insert_minimal_session(conn)
    init_db(conn)
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_foreign_key_rejects_unknown_driver(conn):
    session_id = insert_minimal_session(conn)
    # 'nobody' isn't in the drivers table, so this lap must be rejected.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO laps (session_id, driver_id, lap_number) VALUES (?, 'nobody', 1)",
            (session_id,),
        )


def test_duplicate_event_rejected(conn):
    insert_minimal_session(conn)
    # Same season + round as the event already inserted.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO events (season, round, name, circuit_id, race_date) "
            "VALUES (2024, 16, 'Duplicate', 'monza', '2024-09-01')"
        )


def test_invalid_session_type_rejected(conn):
    insert_minimal_session(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO sessions (event_id, session_type) VALUES (1, 'Warmup')")


def test_stints_view_groups_laps(conn):
    session_id = insert_minimal_session(conn)
    conn.execute(
        "INSERT INTO drivers (driver_id, given_name, family_name) "
        "VALUES ('leclerc', 'Charles', 'Leclerc')"
    )
    # 3 laps on mediums (stint 1), then 2 laps on hards (stint 2).
    laps = [
        (1, 1, "MEDIUM", 1), (2, 1, "MEDIUM", 2), (3, 1, "MEDIUM", 3),
        (4, 2, "HARD", 1), (5, 2, "HARD", 2),
    ]
    # executemany runs the same INSERT once per tuple in the list.
    conn.executemany(
        "INSERT INTO laps (session_id, driver_id, lap_number, stint, compound, tyre_life) "
        f"VALUES ({session_id}, 'leclerc', ?, ?, ?, ?)",
        laps,
    )

    stints = conn.execute(
        "SELECT stint, compound, start_lap, end_lap, lap_count FROM stints ORDER BY stint"
    ).fetchall()

    # Convert sqlite3.Row objects to tuples so we can compare them directly.
    assert [tuple(row) for row in stints] == [
        (1, "MEDIUM", 1, 3, 3),
        (2, "HARD", 4, 5, 2),
    ]
