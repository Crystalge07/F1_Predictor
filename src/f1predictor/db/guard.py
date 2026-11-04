"""The upload guard: never replace the shared database with one that lost live predictions.

The scheduled pipeline (.github/workflows/tick.yml) downloads the database,
runs a tick, and uploads it back. If anything went wrong in between (a bug,
a crash mid-write, a truncated file), uploading would overwrite the only copy
of the permanent prediction log. So before every upload, the new file is
compared with the one that was downloaded:

    1. it's a healthy SQLite file (PRAGMA integrity_check says 'ok');
    2. every LIVE run in the old file is in the new file, with exactly the
       same run row, predictions and features (live runs are never edited);
    3. the new file has at least as many live runs.

Only if all three hold does the upload go ahead. Backtest runs aren't
checked: they can be deleted and re-made on purpose (scripts/replay_season.py).

Which columns are compared: the ones the OLD file has, by name. A column added
since (a schema change; db/connection.init_db adds it with ALTER TABLE) isn't
part of the old record, so it can't make a live run look "changed" and block
every upload from then on. A column the new file has LOST is a problem: that
part of the live log is gone. (Before 2026-11-04 the fingerprint used SELECT *,
so any new column on these tables would have blocked all uploads.)
"""

import hashlib
import sqlite3


# The tables that make up one live run, and the order their rows are read in.
LIVE_TABLES = {
    "prediction_runs": "",
    "predictions": "ORDER BY driver_id",
    "prediction_features": "ORDER BY driver_id, feature",
}


def table_columns(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """{table: its column names, in table order} for LIVE_TABLES."""
    return {table: [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
            for table in LIVE_TABLES}


def live_fingerprints(conn: sqlite3.Connection,
                      columns: dict[str, list[str]] | None = None) -> dict[int, str]:
    """{run_id: SHA-256 of everything logged for that live run}.

    `columns` says which columns of each table to include (default: all the
    connection's own columns). The guard passes the OLD file's columns for
    both files, so the two fingerprints cover the same fields.
    """
    columns = columns or table_columns(conn)
    fingerprints = {}
    for (run_id,) in conn.execute("SELECT run_id FROM prediction_runs WHERE mode = 'live' ORDER BY run_id"):
        digest = hashlib.sha256()
        # The run row, then its predictions and features in a fixed order.
        # repr() of each row tuple is an exact, stable text form. Columns are
        # named (quoted) in the old file's order, which is what SELECT * gave,
        # so fingerprints match those of the earlier SELECT * version.
        for table, order in LIVE_TABLES.items():
            names = ", ".join(f'"{name}"' for name in columns[table])
            for row in conn.execute(f"SELECT {names} FROM {table} WHERE run_id = ? {order}", (run_id,)):
                digest.update(repr(tuple(row)).encode("utf-8"))
        fingerprints[run_id] = digest.hexdigest()
    return fingerprints


def upload_problems(old_path: str, new_path: str) -> list[str]:
    """Reasons NOT to upload new_path in place of old_path (empty = safe)."""
    try:
        return _compare(old_path, new_path)
    except sqlite3.DatabaseError as error:
        # A truncated or corrupt file may not even open as a database.
        return [f"the new database can't be read: {error}"]


def _compare(old_path: str, new_path: str) -> list[str]:
    problems = []
    new = sqlite3.connect(new_path)
    old = sqlite3.connect(old_path)
    try:
        check = new.execute("PRAGMA integrity_check").fetchone()[0]
        if check != "ok":
            return [f"the new database is damaged: {check}"]
        old_columns, new_columns = table_columns(old), table_columns(new)
        dropped = {table: [c for c in cols if c not in new_columns[table]]
                   for table, cols in old_columns.items()}
        dropped = {table: cols for table, cols in dropped.items() if cols}
        if dropped:
            # Can't compare (and live data is gone): say so and stop here.
            return [f"columns missing from the new database: {dropped}"]
        before = live_fingerprints(old, old_columns)
        after = live_fingerprints(new, old_columns)
        lost = sorted(set(before) - set(after))
        changed = sorted(r for r in before if r in after and before[r] != after[r])
        if lost:
            problems.append(f"live runs missing from the new database: {lost}")
        if changed:
            problems.append(f"live runs changed in the new database: {changed}")
        if len(after) < len(before):
            problems.append(f"fewer live runs ({len(after)}) than before ({len(before)})")
    finally:
        new.close()
        old.close()
    return problems
