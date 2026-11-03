"""Check a database is safe to upload in place of the one downloaded (db/guard.py).

Usage:
    python scripts/check_db_upload.py OLD.sqlite NEW.sqlite

Exit code 0 = safe to upload; 1 = don't (the reasons are printed). The
scheduled workflow runs this between the tick and the upload.
"""

import sys

from f1predictor.db.guard import live_fingerprints, upload_problems


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    old_path, new_path = sys.argv[1], sys.argv[2]
    problems = upload_problems(old_path, new_path)
    if problems:
        print("NOT safe to upload:\n  - " + "\n  - ".join(problems))
        return 1
    import sqlite3
    n_live = len(live_fingerprints(sqlite3.connect(new_path)))
    print(f"Safe to upload: integrity ok, all {n_live} live runs intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
