"""Run one tick of the weekend pipeline: fetch, predict, score, report.

Usage (from the repo root, with the virtual environment active):
    python scripts/tick.py                    (the real thing: may log LIVE runs)
    python scripts/tick.py --dry-run          (fetches data, logs no predictions)
    python scripts/tick.py --dry-run --now 2026-11-07T17:20:00Z   (pretend it's then)

The scheduler (GitHub Actions) runs the first form every hour. See
pipeline/tick.py for what a tick does.

Exit code: 0 if the tick raised no NEW alert, 1 if it did (a failed fetch, a
blocked prediction, a missed stage, a session given up on). A non-zero exit
is what makes GitHub Actions email about the run. Alerts already raised by
an earlier tick don't count again.

--now is only allowed with --dry-run: a live prediction must be made at the
real time, never a pretend one.
"""

import argparse
import logging
import sys

from f1predictor.db.connection import get_connection, init_db
from f1predictor.pipeline.plan import parse_utc
from f1predictor.pipeline.tick import run_tick


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="fetch data, but log no predictions or scores")
    parser.add_argument("--now", help="pretend the time is this (UTC ISO); dry runs only")
    args = parser.parse_args()
    if args.now and not args.dry_run:
        parser.error("--now needs --dry-run: live predictions are only made at the real time")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    conn = get_connection()
    try:
        init_db(conn)
        report = run_tick(conn, now=parse_utc(args.now) if args.now else None, dry_run=args.dry_run)
    finally:
        conn.close()
    print("\n".join(report.lines()))
    return 1 if report.new_alerts else 0


if __name__ == "__main__":
    sys.exit(main())
