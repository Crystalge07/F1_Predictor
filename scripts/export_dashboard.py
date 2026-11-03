"""Export the dashboard's JSON files from the database (Phase 6, step 1).

    python scripts/export_dashboard.py data/f1.sqlite site

writes site/data/*.json (src/f1predictor/export/dashboard.py) and prints
the export's hash. The GitHub Actions workflow runs it after every tick and
deploys the site to GitHub Pages only if the hash differs from the live one.

The database is opened READ-ONLY (SQLite's mode=ro): the export can't change
anything, least of all the frozen prediction log.
"""

import argparse
import sqlite3
from pathlib import Path

from f1predictor.export.dashboard import export


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path, help="the SQLite database to read")
    parser.add_argument("out", type=Path, help="the site folder; files go in <out>/data/")
    args = parser.parse_args()
    if not args.db.exists():                       # mode=ro won't create it, but say why clearly
        parser.error(f"no database at {args.db}")
    conn = sqlite3.connect(f"file:{args.db.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    version = export(conn, args.out)
    print(f"exported {version['files']} files, hash {version['hash']}")


if __name__ == "__main__":
    main()
