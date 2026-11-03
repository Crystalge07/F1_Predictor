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
"""

import hashlib
import sqlite3


def live_fingerprints(conn: sqlite3.Connection) -> dict[int, str]:
    """{run_id: SHA-256 of everything logged for that live run}."""
    fingerprints = {}
    for (run_id,) in conn.execute("SELECT run_id FROM prediction_runs WHERE mode = 'live' ORDER BY run_id"):
        digest = hashlib.sha256()
        # Every column of the run row, then its predictions and features in a
        # fixed order. repr() of each row tuple is an exact, stable text form.
        for query in ("SELECT * FROM prediction_runs WHERE run_id = ?",
                      "SELECT * FROM predictions WHERE run_id = ? ORDER BY driver_id",
                      "SELECT * FROM prediction_features WHERE run_id = ? ORDER BY driver_id, feature"):
            for row in conn.execute(query, (run_id,)):
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
        before, after = live_fingerprints(old), live_fingerprints(new)
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
