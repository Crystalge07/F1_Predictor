"""Create the database and all its tables.

Usage (from the repo root, with the virtual environment active):
    python scripts/init_db.py

Safe to run repeatedly: existing tables and data are left untouched.
"""

from f1predictor import config
from f1predictor.db.connection import get_connection, init_db, list_tables


def main() -> None:
    # Make sure data/ exists before SQLite tries to create a file inside it.
    config.ensure_data_dirs()

    conn = get_connection()
    try:
        init_db(conn)
        print(f"Database ready at {config.DB_PATH}")
        print("Tables and views:", ", ".join(list_tables(conn)))
    finally:
        # try/finally makes sure the connection closes even if init_db fails.
        conn.close()


# This block only runs when the file is executed directly
# (python scripts/init_db.py), not when another file imports it.
if __name__ == "__main__":
    main()
