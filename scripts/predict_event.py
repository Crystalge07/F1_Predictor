"""Predict one race at one stage by hand: a dry run first, then (optionally) live.

Usage (from the repo root, with the virtual environment active):
    python scripts/predict_event.py 2026 20 pre_weekend           (dry run: prints, writes nothing)
    python scripts/predict_event.py 2026 20 pre_weekend --live    (writes the live run)

Live runs can never be edited or deleted (db/schema.sql), so a bad input
would stay in the log forever. That's why --live only goes ahead if ALL of
these hold:

    1. a dry run of the same race and stage was made earlier, with the same
       code and settings (it's saved in data/dry_runs/), and the prediction
       made now is identical to it: what gets logged is exactly what was checked;
    2. the input checks pass (predict/stage.check_prediction: field size,
       probabilities add up, no more than a few driver changes vs last race);
    3. the stage's cutoff hasn't passed (a "live" prediction made after the
       session started wouldn't be honest);
    4. there's no live run for this race and stage yet;
    5. the git working tree is clean, so model_version names a real commit.

The dry run prints the inputs worth checking before going live: where the
driver list came from and how it differs from the last race, the cutoff,
how many features are unknown per driver, and whether the probabilities add up.
"""

import argparse
import json
from datetime import datetime, timezone


from f1predictor import config
from f1predictor.db.connection import get_connection, init_db
from f1predictor.features.as_of import event_info, previous_race_lineup, STAGES
from f1predictor.predict.stage import check_prediction, live_problems, model_version, make_prediction, write_run

DRY_RUN_DIR = config.DATA_DIR / "dry_runs"


def dry_run_path(season: int, rnd: int, stage: str):
    return DRY_RUN_DIR / f"{season}-{rnd:02d}-{stage}.json"


def as_record(prediction, version: str) -> dict:
    """Everything that would be logged, as plain JSON (rounded, so it compares cleanly)."""
    return {
        "model_version": version,
        "as_of_utc": prediction.as_of_utc,
        "lineup_source": prediction.lineup_source,
        "n_training_races": prediction.n_training_races,
        "drivers": json.loads(prediction.drivers.round(6).to_json(orient="records")),
        "features": json.loads(prediction.features.round(6).to_json(orient="records")),
    }


def print_checks(conn, info, prediction) -> None:
    drivers = prediction.drivers
    print(f"cutoff (as_of): {prediction.as_of_utc}   driver list from: {prediction.lineup_source}   "
          f"trained on {prediction.n_training_races} races\n")
    show = drivers.assign(
        **{c: drivers[c].map("{:.1%}".format) for c in ["p_win", "p_podium", "p_points", "p_dnf"]},
        expected_position=drivers["expected_position"].round(2),
        mean_position=drivers["mean_position"].round(2))
    print(show.to_string(index=False))

    # Checks worth reading before going live.
    print(f"\n{len(drivers)} drivers; win chances add up to {drivers['p_win'].sum():.3f}, "
          f"podium to {drivers['p_podium'].sum():.3f} (should be ~1 and ~3)")
    last = set(previous_race_lineup(conn, info["race_date"])["driver_id"])
    now = set(drivers["driver_id"])
    print(f"vs last race's starters: new {sorted(now - last) or 'none'}, gone {sorted(last - now) or 'none'}")
    unknown = prediction.features.assign(unknown=prediction.features["value"].isna())
    per_driver = unknown.groupby("driver_id")["unknown"].sum()
    n_features = unknown["feature"].nunique()
    flagged = per_driver[per_driver > 0].sort_values(ascending=False)
    print(f"unknown feature values (of {n_features} per driver): "
          + (", ".join(f"{d} {int(n)}" for d, n in flagged.items()) if not flagged.empty else "none"))
    problems = check_prediction(conn, prediction)
    print("input checks: " + ("all passed" if not problems else "FAILED: " + "; ".join(problems)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("season", type=int)
    parser.add_argument("round", type=int)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--live", action="store_true", help="write the run to the log (after a matching dry run)")
    args = parser.parse_args()

    conn = get_connection()
    try:
        init_db(conn)
        row = conn.execute("SELECT event_id FROM events WHERE season = ? AND round = ?",
                           (args.season, args.round)).fetchone()
        if row is None:
            raise SystemExit(f"No event {args.season} round {args.round}")
        event_id = row["event_id"]
        info = event_info(conn, event_id)
        version = model_version()
        print(f"{args.season} round {args.round}, stage {args.stage}, code {version}"
              f"  [{'LIVE' if args.live else 'DRY RUN'}]\n")

        prediction = make_prediction(conn, event_id, args.stage)
        print_checks(conn, info, prediction)
        record = as_record(prediction, version)
        path = dry_run_path(args.season, args.round, args.stage)

        if not args.live:
            DRY_RUN_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1), encoding="utf-8")
            print(f"\nDry run saved to {path}. Nothing was logged. Re-run with --live to log it.")
            return

        # --- The live checks (see the module docstring) ------------------------
        problems = []
        if not path.exists():
            problems.append("no dry run for this race and stage: run without --live first")
        elif json.loads(path.read_text(encoding="utf-8")) != record:
            problems.append(f"the prediction differs from the dry run in {path} "
                            "(data, code or settings changed): dry-run again and re-check")
        # The same checks the pipeline tick uses (predict/stage.live_problems):
        # input sanity, cutoff not passed, no live run yet, clean code.
        problems += live_problems(conn, prediction, datetime.now(timezone.utc))
        if problems:
            raise SystemExit("NOT logged:\n  - " + "\n  - ".join(problems))

        run_id = write_run(conn, prediction, "live")
        print(f"\nLogged live run {run_id}. It can never be edited or deleted.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
