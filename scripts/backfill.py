"""Fill the database with historical F1 data. Safe to stop and re-run.

Usage (from the repo root, with the virtual environment active):
    python scripts/backfill.py                      # all seasons in config (2018-2026)
    python scripts/backfill.py --seasons 2024 2025  # just these
    python scripts/backfill.py --no-openf1          # Jolpica only (~1 minute)
    python scripts/backfill.py --force              # re-fetch everything

A full run takes ~1.5 hours, almost all of it waiting on OpenF1's rate limit
(30 requests/minute). Progress is printed and also written to data/backfill.log.
If it's interrupted, just run it again: finished work is skipped.
"""

import argparse
import logging
import sys
import time

from f1predictor import config
from f1predictor.db.connection import get_connection, init_db
from f1predictor.ingest.backfill import backfill
from f1predictor.ingest.http import JsonHttpClient


def parse_args() -> argparse.Namespace:
    """Read command-line options. argparse also generates --help automatically."""
    parser = argparse.ArgumentParser(description="Backfill F1 data into the database.")
    parser.add_argument("--seasons", type=int, nargs="+", default=config.SEASONS,
                        help="Seasons to load (default: %(default)s)")
    parser.add_argument("--no-openf1", action="store_true",
                        help="Skip OpenF1 lap data (Jolpica only)")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch items already loaded successfully")
    parser.add_argument("--db", default=config.DB_PATH,
                        help="Database file (default: %(default)s)")
    return parser.parse_args()


def setup_logging() -> None:
    """Send log messages to both the terminal and data/backfill.log."""
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                                  datefmt="%H:%M:%S")
    terminal = logging.StreamHandler(sys.stdout)
    logfile = logging.FileHandler(config.DATA_DIR / "backfill.log", encoding="utf-8")
    for handler in (terminal, logfile):
        handler.setFormatter(formatter)

    root = logging.getLogger()          # the parent of every logger in the program
    root.setLevel(logging.INFO)
    root.addHandler(terminal)
    root.addHandler(logfile)


def main() -> None:
    args = parse_args()
    config.ensure_data_dirs()           # data/ must exist before the log file is opened
    setup_logging()
    log = logging.getLogger("backfill")

    conn = get_connection(args.db)
    init_db(conn)                       # creates any missing tables; harmless otherwise

    # One client per API, each spaced to that API's rate limit.
    jolpica_client = JsonHttpClient(min_interval_s=config.JOLPICA_MIN_INTERVAL_S)
    openf1_client = JsonHttpClient(min_interval_s=config.OPENF1_MIN_INTERVAL_S)

    log.info("Backfilling seasons %s into %s", args.seasons, args.db)
    started = time.monotonic()
    try:
        tally = backfill(conn, args.seasons, jolpica_client, openf1_client,
                         include_openf1=not args.no_openf1, force=args.force)
    except KeyboardInterrupt:
        # Ctrl+C: the item in progress was rolled back; everything before it
        # is committed and logged, so a re-run resumes from here.
        log.warning("Interrupted. Run the same command again to resume.")
        sys.exit(1)
    finally:
        conn.close()

    minutes = (time.monotonic() - started) / 60
    log.info("Done in %.1f minutes.", minutes)
    for (source, status), count in sorted(tally.items()):
        log.info("  %-8s %-13s %d", source, status, count)
    if any(status == "failed" for _, status in tally):
        log.info("Some items failed; see ingestion_log / validation_issues, "
                 "or just re-run to retry them.")
    log.info("Audit the data with: python scripts/validate.py")


if __name__ == "__main__":
    main()
