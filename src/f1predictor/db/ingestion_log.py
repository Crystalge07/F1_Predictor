"""Read and write the ingestion_log table: a record of what we've fetched.

Every fetch (a season's schedule, a session's laps...) records one row here
with its outcome. Two uses:
  * Resuming: the backfill can skip items already marked 'ok'.
  * Debugging: failed items keep their error message, so you can see what
    broke without re-running anything.
"""

import sqlite3
from datetime import datetime, timezone


def record(
    conn: sqlite3.Connection,
    source: str,
    item_key: str,
    status: str,
    row_count: int | None = None,
    message: str | None = None,
) -> None:
    """Record the outcome of fetching one item, replacing any earlier attempt.

    Args:
        source: 'jolpica' or 'openf1'.
        item_key: A readable id for what was fetched, e.g. '2024-results'.
        status: 'ok', 'failed' or 'skipped' (the schema rejects anything else).
        row_count: Rows written, as a sanity number.
        message: Error text when status is 'failed'.
    """
    # Current time in UTC, as ISO-8601 text: '2026-11-02T19:30:00+00:00'.
    # Always UTC, so timestamps from a laptop and GitHub Actions compare correctly.
    now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # "Upsert": insert a new row, or if (source, item_key) already exists,
    # overwrite it. `excluded` refers to the row we tried to insert.
    # This syntax works in both SQLite (3.24+) and Postgres.
    conn.execute(
        """
        INSERT INTO ingestion_log (source, item_key, status, row_count, message, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (source, item_key) DO UPDATE SET
            status      = excluded.status,
            row_count   = excluded.row_count,
            message     = excluded.message,
            ingested_at = excluded.ingested_at
        """,
        # The ? placeholders are filled from this tuple. NEVER build SQL with
        # f-strings from data: placeholders handle quoting safely.
        (source, item_key, status, row_count, message, now_utc),
    )
    conn.commit()


def is_done(conn: sqlite3.Connection, source: str, item_key: str) -> bool:
    """True if this item was already fetched successfully."""
    row = conn.execute(
        "SELECT status FROM ingestion_log WHERE source = ? AND item_key = ?",
        (source, item_key),
    ).fetchone()
    # fetchone() returns None when no row matches (never attempted).
    return row is not None and row["status"] == "ok"
