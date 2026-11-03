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


# --- Prediction log (Phase 4): predictions are frozen once written ---

def insert_run(conn, mode: str = "backtest") -> int:
    """One prediction run, with one driver's prediction and one feature row.

    Returns the run_id. The event/session chain comes from insert_minimal_session.
    """
    insert_minimal_session(conn)
    insert_driver(conn, "leclerc")
    cursor = conn.execute(
        "INSERT INTO prediction_runs (event_id, stage, mode, as_of_utc, created_at_utc, "
        "model_version, settings_json) "
        "VALUES (1, 'post_quali', ?, '2024-08-31T15:00:00Z', '2024-08-31T15:05:00Z', 'abc123', '{}')",
        (mode,),
    )
    run_id = cursor.lastrowid
    conn.execute(
        "INSERT INTO predictions (run_id, driver_id, predicted_rank, p_win, p_podium, p_points, p_dnf) "
        "VALUES (?, 'leclerc', 1, 0.4, 0.7, 0.9, 0.1)",
        (run_id,),
    )
    conn.execute(
        "INSERT INTO prediction_features (run_id, driver_id, feature, value, contribution) "
        "VALUES (?, 'leclerc', 'grid', 1, -2.5)",
        (run_id,),
    )
    return run_id


def count(conn, table: str) -> int:
    # The table name comes from the test itself, never from outside input,
    # so formatting it into the SQL is safe here.
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_prediction_log_tables_created(conn):
    assert {"prediction_runs", "predictions", "prediction_features", "scores"} <= set(list_tables(conn))


@pytest.mark.parametrize("mode", ["backtest", "live"])
@pytest.mark.parametrize("sql", [
    "UPDATE prediction_runs SET model_version = 'changed'",
    "UPDATE predictions SET p_win = 0.99",
    "UPDATE prediction_features SET value = 5",
])
def test_logged_predictions_can_never_be_edited(conn, mode, sql):
    # Whatever the mode, a logged prediction is frozen: no quietly fixing it
    # after the race.
    insert_run(conn, mode)
    with pytest.raises(sqlite3.IntegrityError, match="never edited"):
        conn.execute(sql)


@pytest.mark.parametrize("table", ["prediction_runs", "predictions", "prediction_features"])
def test_live_predictions_can_never_be_deleted(conn, table):
    insert_run(conn, "live")
    with pytest.raises(sqlite3.IntegrityError, match="never deleted"):
        conn.execute(f"DELETE FROM {table}")
    assert count(conn, table) == 1


def test_backtest_run_delete_removes_its_rows(conn):
    # Backtests can be re-run: deleting the run cascades to its predictions
    # and feature rows, so nothing is left dangling.
    run_id = insert_run(conn, "backtest")
    conn.execute("DELETE FROM prediction_runs WHERE run_id = ?", (run_id,))
    for table in ["prediction_runs", "predictions", "prediction_features"]:
        assert count(conn, table) == 0


def test_invalid_stage_and_mode_rejected(conn):
    insert_minimal_session(conn)
    for stage, mode in [("after_fp1", "live"), ("post_quali", "replay")]:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO prediction_runs (event_id, stage, mode, as_of_utc, created_at_utc, "
                "model_version, settings_json) VALUES (1, ?, ?, 'x', 'x', 'v', '{}')",
                (stage, mode),
            )


def test_probability_outside_0_to_1_rejected(conn):
    run_id = insert_run(conn)
    insert_driver(conn, "sainz")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO predictions (run_id, driver_id, predicted_rank, p_win, p_podium, p_points, p_dnf) "
            "VALUES (?, 'sainz', 2, 1.5, 0.7, 0.9, 0.1)",
            (run_id,),
        )


def test_two_drivers_cannot_share_a_predicted_place(conn):
    run_id = insert_run(conn)
    insert_driver(conn, "sainz")
    # leclerc already holds predicted_rank 1 in this run.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO predictions (run_id, driver_id, predicted_rank, p_win, p_podium, p_points, p_dnf) "
            "VALUES (?, 'sainz', 1, 0.3, 0.6, 0.9, 0.1)",
            (run_id,),
        )


def test_feature_row_needs_a_prediction(conn):
    # A feature row for a driver with no prediction in that run is an orphan.
    run_id = insert_run(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO prediction_features (run_id, driver_id, feature, value) "
            "VALUES (?, 'nobody', 'grid', 3)",
            (run_id,),
        )


def test_scores_can_be_rewritten(conn):
    # Unlike predictions, scores may be redone if the FIA changes a result.
    run_id = insert_run(conn, "live")
    conn.execute(
        "INSERT INTO scores (run_id, metric, model_value, baseline_value, baseline_name, scored_at_utc) "
        "VALUES (?, 'position_mae', 3.1, 3.4, 'grid_order', 'x')",
        (run_id,),
    )
    conn.execute("UPDATE scores SET model_value = 2.9 WHERE run_id = ?", (run_id,))
    assert conn.execute("SELECT model_value FROM scores").fetchone()[0] == 2.9
