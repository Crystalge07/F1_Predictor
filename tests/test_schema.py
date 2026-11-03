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


def insert_driver(conn, driver_id: str = "leclerc") -> None:
    conn.execute(
        "INSERT INTO drivers (driver_id, given_name, family_name) VALUES (?, 'X', 'Y')",
        (driver_id,),
    )


def test_all_tables_created(conn):
    expected = {
        "circuits", "drivers", "constructors", "events", "sessions",
        "results", "qualifying", "session_entries", "laps", "stints",
        "pit_stops", "weather", "race_control", "ingestion_log", "laps_detailed",
    }
    # <= means "is a subset of": every expected name must be present.
    assert expected <= set(list_tables(conn))


def test_init_db_is_safe_to_rerun(conn):
    # Running the schema a second time must not fail or wipe data.
    insert_minimal_session(conn)
    init_db(conn)
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


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


def test_result_for_unknown_driver_rejected(conn):
    session_id = insert_minimal_session(conn)
    # 'nobody' isn't in drivers, so the foreign key must reject this row.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO results (session_id, driver_id, constructor_id, position, position_text) "
            "VALUES (?, 'nobody', 'nobody', 1, '1')",
            (session_id,),
        )


# --- Session entries: the "drivers Jolpica doesn't know" design (option 3) ---

def test_entry_without_driver_id_is_allowed(conn):
    # An FP1 rookie: we know the car number and name, but not a Jolpica id.
    session_id = insert_minimal_session(conn)
    conn.execute(
        "INSERT INTO session_entries (session_id, car_number, acronym, full_name) "
        "VALUES (?, 22, 'HIR', 'Ryo HIRAKAWA')",
        (session_id,),
    )
    row = conn.execute("SELECT driver_id FROM session_entries").fetchone()
    assert row["driver_id"] is None


def test_entry_with_unknown_driver_id_rejected(conn):
    # NULL is fine, but a non-NULL driver_id must point at a real driver.
    session_id = insert_minimal_session(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO session_entries (session_id, car_number, driver_id) "
            "VALUES (?, 22, 'made_up_id')",
            (session_id,),
        )


def test_lap_for_car_not_in_session_rejected(conn):
    session_id = insert_minimal_session(conn)
    # No session_entries row for car 99 -> the composite foreign key rejects it.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO laps (session_id, car_number, lap_number) VALUES (?, 99, 1)",
            (session_id,),
        )


def test_laps_detailed_view_attaches_driver_and_tyres(conn):
    session_id = insert_minimal_session(conn)
    insert_driver(conn, "leclerc")
    conn.execute(
        "INSERT INTO session_entries (session_id, car_number, acronym, driver_id) "
        "VALUES (?, 16, 'LEC', 'leclerc')",
        (session_id,),
    )
    # Stint 1: laps 1-3 on new mediums. Stint 2: laps 4-5 on hards that
    # already had 2 laps on them.
    conn.executemany(
        "INSERT INTO stints (session_id, car_number, stint_number, compound, "
        "lap_start, lap_end, tyre_age_at_start) VALUES (?, 16, ?, ?, ?, ?, ?)",
        [(session_id, 1, "MEDIUM", 1, 3, 0), (session_id, 2, "HARD", 4, 5, 2)],
    )
    # executemany runs the same INSERT once per tuple in the list.
    conn.executemany(
        "INSERT INTO laps (session_id, car_number, lap_number) VALUES (?, 16, ?)",
        [(session_id, n) for n in range(1, 6)],
    )

    rows = conn.execute(
        "SELECT lap_number, driver_id, compound, tyre_age FROM laps_detailed ORDER BY lap_number"
    ).fetchall()

    # Convert sqlite3.Row objects to tuples so we can compare them directly.
    assert [tuple(r) for r in rows] == [
        (1, "leclerc", "MEDIUM", 0),
        (2, "leclerc", "MEDIUM", 1),
        (3, "leclerc", "MEDIUM", 2),
        (4, "leclerc", "HARD", 2),     # used set: starts at age 2
        (5, "leclerc", "HARD", 3),
    ]
