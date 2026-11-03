"""Audit the whole database: run every validation check and print a report.

Usage (from the repo root, with the virtual environment active):
    python scripts/validate.py                  # the main database
    python scripts/validate.py path/to/other.sqlite

The loaders already validate each item as they save it. This script is for
checking everything at once: after a backfill, after changing a check's
threshold, or whenever you want to know the state of the data. It only reads;
it never changes the database.
"""

import sys
from collections import Counter

from f1predictor import config
from f1predictor.db.connection import get_connection
from f1predictor.validate import checks


def audit(conn) -> list[checks.Issue]:
    """Run every check over every season and session that has data."""
    issues: list[checks.Issue] = []
    seasons = [row["season"] for row in conn.execute(
        "SELECT DISTINCT season FROM events ORDER BY season")]

    for season in seasons:
        for part in ("schedule", "results", "sprint", "qualifying"):
            issues.extend(checks.check_jolpica_part(conn, season, part))

    # Every session that has OpenF1 lap data.
    for row in conn.execute("SELECT DISTINCT session_id FROM laps ORDER BY session_id"):
        issues.extend(checks.check_openf1_session(conn, row["session_id"]))

    return issues


def describe_session(conn, session_id: int | None) -> str:
    """'2024 R24 FP1' for a session id; 'season' for season-level issues."""
    if session_id is None:
        return "season"
    row = conn.execute(
        "SELECT e.season, e.round, s.session_type FROM sessions s "
        "JOIN events e USING (event_id) WHERE s.session_id = ?", (session_id,)).fetchone()
    return f"{row['season']} R{row['round']:02d} {row['session_type']}"


def main() -> None:
    # sys.argv is the command line as a list: [script name, arg1, ...].
    db_path = sys.argv[1] if len(sys.argv) > 1 else config.DB_PATH
    conn = get_connection(db_path)
    try:
        issues = audit(conn)

        # Errors first, then warnings, each sorted by session.
        for issue in sorted(issues, key=lambda i: (i.severity != "error", i.session_id or 0)):
            print(f"[{issue.severity.upper():7}] {describe_session(conn, issue.session_id):14} "
                  f"{issue.check}: {issue.detail}")

        counts = Counter((i.severity, i.check) for i in issues)
        print("\nSummary:" if counts else "No issues found.")
        for (severity, check), n in sorted(counts.items()):
            print(f"  {severity:7} {check:28} {n}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
