"""Replay a past season race by race, as if live: predict, log, score.

Usage (from the repo root, with the virtual environment active):
    python scripts/replay_season.py                 (2026, every stage; ~5 minutes)
    python scripts/replay_season.py --season 2026 --rounds 10 19
    python scripts/replay_season.py --summary-only  (just print the totals again)

For every race of the season that has a result, in calendar order, and every
stage (pre-weekend, practice, post-quali):
    1. predict it with predict/stage.predict_stage, which only uses what was
       known at that stage's cutoff, trains on earlier races only, and logs the
       run as mode 'backtest';
    2. score it with evaluate/scorer.py against the stage's baseline.
Then print the season totals per stage.

Re-runnable: the season's earlier BACKTEST runs are deleted first (with their
predictions, features and scores). Live runs are never touched, and the
database refuses to delete them anyway (db/schema.sql triggers).

The numbers are IN-SAMPLE for 2026: rounds 1-9 chose the frozen settings, and
rounds 10+ were looked at many times during development
(docs/evaluation-protocol.md). This replay checks that the whole pipeline
works end to end; the 2027 live season is the real test.
"""

import argparse
import time

from f1predictor.db.connection import get_connection, init_db
from f1predictor.evaluate.scorer import score_runs, summary
from f1predictor.features.as_of import STAGES
from f1predictor.predict.stage import model_version, predict_stage

# Metrics in the order they're printed, with how to show each value.
PRINTED = [
    ("position_mae", "places off (avg)", "{:.2f}"),
    ("spearman", "rank correlation", "{:.3f}"),
    ("winner_hit", "winner picked", "{:.0%}"),
    ("podium_hits", "podium drivers picked", "{:.0%}"),
    ("brier_win", "win Brier", "{:.4f}"),
    ("brier_podium", "podium Brier", "{:.4f}"),
]


def season_events(conn, season: int, first: int, last: int) -> list:
    """event_id, round, name of every race in the season that has a result."""
    return conn.execute(
        """
        SELECT e.event_id, e.round, e.name
        FROM events e
        JOIN sessions s ON s.event_id = e.event_id AND s.session_type = 'R'
        WHERE e.season = ? AND e.round BETWEEN ? AND ?
          AND EXISTS (SELECT 1 FROM results r WHERE r.session_id = s.session_id)
        ORDER BY e.round
        """, (season, first, last)).fetchall()


def delete_backtest_runs(conn, event_ids: list[int]) -> int:
    """Delete earlier backtest runs for these events (cascades to their rows)."""
    if not event_ids:
        return 0
    placeholders = ", ".join("?" * len(event_ids))
    with conn:
        cursor = conn.execute(
            f"DELETE FROM prediction_runs WHERE mode = 'backtest' AND event_id IN ({placeholders})",
            event_ids)
    return cursor.rowcount


def print_summary(conn, season: int, label: str, rounds=None) -> None:
    table = summary(conn, season, rounds=rounds)
    if table.empty:
        print(f"\n{label}: nothing scored")
        return
    print(f"\n{label}")
    for stage in STAGES:
        rows = table[table["stage"] == stage].set_index("metric")
        if rows.empty:
            continue
        races = int(rows["races"].max())
        baseline_name = rows["baseline_name"].dropna().iloc[0]
        print(f"\n  {stage}  ({races} races, baseline: {baseline_name})")
        print(f"    {'':<24}{'model':>9}{'baseline':>10}{'better by':>11}")
        for metric, name, fmt in PRINTED:
            if metric not in rows.index:
                continue
            r = rows.loc[metric]
            print(f"    {name:<24}{fmt.format(r['model']):>9}{fmt.format(r['baseline']):>10}"
                  f"{r['improvement']:>+11.1%}")
        missing = rows.loc["lineup_missing", "model"] * races if "lineup_missing" in rows.index else 0
        extra = rows.loc["lineup_extra", "model"] * races if "lineup_extra" in rows.index else 0
        print(f"    driver list: {missing:.0f} starters not predicted, "
              f"{extra:.0f} predicted drivers who didn't start")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--rounds", type=int, nargs=2, metavar=("FIRST", "LAST"), default=(1, 99))
    parser.add_argument("--summary-only", action="store_true",
                        help="don't predict anything, just print the totals of the logged runs")
    args = parser.parse_args()

    conn = get_connection()
    try:
        init_db(conn)   # make sure the log tables (and any added columns) exist
        if not args.summary_only:
            events = season_events(conn, args.season, *args.rounds)
            deleted = delete_backtest_runs(conn, [e["event_id"] for e in events])
            print(f"Replaying {args.season}: {len(events)} races, code {model_version()}"
                  f" ({deleted} earlier backtest runs deleted)\n")

            run_ids = []
            for event in events:
                started = time.time()
                done, skipped = [], []
                for stage in STAGES:
                    try:
                        run_ids.append(predict_stage(conn, event["event_id"], stage, mode="backtest"))
                        done.append(stage)
                    except ValueError as error:
                        # E.g. no practice laps -> no practice order to anchor on.
                        # Skipped and reported, never filled in with something else.
                        skipped.append(f"{stage} ({error})")
                line = f"  R{event['round']:>2} {event['name']:<32} {len(done)} stages, {time.time() - started:.0f}s"
                print(line + (f"  SKIPPED: {'; '.join(skipped)}" if skipped else ""))

            print("\nScoring...")
            scored = score_runs(conn, run_ids)
            print(f"  {len(scored)} of {len(run_ids)} runs scored")

        label = f"{args.season} backtest (IN-SAMPLE: see docs/evaluation-protocol.md)"
        print_summary(conn, args.season, label, rounds=tuple(args.rounds))
        if args.season == 2026 and tuple(args.rounds) == (1, 99):
            # The same split as the evaluation protocol, for comparison with
            # the frozen numbers (both parts are in-sample here).
            print_summary(conn, 2026, "2026 rounds 1-9 (the races that chose the settings)", rounds=(1, 9))
            print_summary(conn, 2026, "2026 rounds 10+ (held out, but looked at often)", rounds=(10, 99))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
