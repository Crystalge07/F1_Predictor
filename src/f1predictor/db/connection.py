"""Opening the SQLite database and creating its tables.

Every other module gets its database connection from `get_connection()`, so
connection settings (like enforcing foreign keys) are set in exactly one place.
"""

import sqlite3
from pathlib import Path

from f1predictor import config


def get_connection(db_path: Path | str = config.DB_PATH) -> sqlite3.Connection:
    """Open a connection to the database, with our standard settings applied.

    Args:
        db_path: Which database file to open. Defaults to the main database.
            Tests pass ":memory:" to get a throwaway in-memory database that
            disappears when the connection closes.

    Returns:
        An open sqlite3 connection. The caller should close it when done.
    """
    # sqlite3.connect creates the file if it doesn't exist yet.
    conn = sqlite3.connect(db_path)

    # IMPORTANT: SQLite does NOT enforce foreign keys unless you turn them on,
    # and the setting only lasts for this one connection. Without this line,
    # inserting a lap for a driver_id that doesn't exist would silently succeed.
    # With it, SQLite raises an error, which catches loader bugs early.
    conn.execute("PRAGMA foreign_keys = ON")

    # By default, rows come back as plain tuples: row[0], row[1]...
    # sqlite3.Row lets us also access columns by name: row["driver_id"],
    # which is much more readable and survives column reordering.
    conn.row_factory = sqlite3.Row

    return conn


def init_db(conn: sqlite3.Connection, schema_path: Path = config.SCHEMA_PATH) -> None:
    """Create all tables, views and indexes defined in schema.sql.

    Safe to call on an existing database: every statement in the schema uses
    IF NOT EXISTS, so existing tables (and their data) are left alone.

    Note: the schema only ADDS missing tables. Columns added to an existing
    table later are handled by the small `added_columns` list below, which
    runs ALTER TABLE for any that are missing.
    """
    # Read the whole .sql file as one string...
    schema_sql = schema_path.read_text(encoding="utf-8")

    # ...and run it. executescript() runs multiple ;-separated statements,
    # unlike execute(), which only accepts one.
    conn.executescript(schema_sql)

    # Small migrations: columns added to an existing table after it was first
    # created. CREATE TABLE IF NOT EXISTS skips tables that already exist, so
    # a database made before the column was added needs an ALTER TABLE.
    # Each entry is (table, column, column definition).
    added_columns = [
        ("prediction_runs", "lineup_source", "TEXT"),   # Phase 4 step 3
    ]
    for table, column, definition in added_columns:
        # PRAGMA table_info lists a table's columns, one row each ("name" = column name).
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    # executescript commits automatically, but committing explicitly makes the
    # intent obvious to anyone reading this.
    conn.commit()


def list_tables(conn: sqlite3.Connection) -> list[str]:
    """Return the names of all tables and views, alphabetically.

    sqlite_master is SQLite's built-in catalog of everything in the database.
    Handy for checking that init_db worked.
    """
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type IN ('table', 'view') "
        "ORDER BY name"
    ).fetchall()
    return [row["name"] for row in rows]
