"""Export what the dashboard shows, as small static JSON files.

The dashboard (Phase 6) is a React app on GitHub Pages: plain files, no
server. It never reads the database. After each hourly tick, the workflow
runs this export and, if anything changed, deploys the app together with
these files (decision 2026-11-03, docs/PRD.md). Why not read the database:
a browser can't fetch the release asset (no cross-site permission), it's
35 MB where a page needs ~30-100 KB, and this way every number (baselines,
"% better", rehearsal vs live) is worked out once, here, in tested Python.
The export only READS the database (scripts/export_dashboard.py opens it
read-only).

Files (under <out>/data/):

    index.json        the calendar of every season with predictions: each
                      race's status and file, plus which race is next
    races/2026-20.json  one race: its sessions, every logged run (each
                      stage, live and backtest), every driver's prediction
                      and SHAP factors, the result and the scores
    scorecard.json    per season and mode: average of each metric per
                      stage, vs the baseline, with the label that must be
                      shown next to it (scorer.scorecard_label)
    version.json      a hash of all the files above, and when they were
                      exported. The workflow compares the hash with the live
                      site's to decide whether to deploy.

Every file is deterministic: the same database gives byte-identical files
(sorted keys, rounded numbers, fixed orders, no clock times except in
version.json's exported_at_utc, which isn't part of the hash). So an hourly
tick that changed nothing deploys nothing.
"""

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from f1predictor.evaluate.scorer import (HIGHER_IS_BETTER, UnresolvedDriverError, logged_predictions,
                                         real_driver_ids, run_info, scorecard_label, scorecards, summary)
from f1predictor.explain.contributions import FEATURE_LABELS
from f1predictor.features.provisional import is_provisional, resolve
from f1predictor.models.baselines import actual_result
from f1predictor.pipeline.plan import parse_utc, to_utc_text

SCHEMA = 1                 # bumped if a file's shape changes, so the app can tell
STAGE_ORDER = ["pre_weekend", "practice", "post_quali"]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _num(value, places: int):
    """A float rounded for the files (smaller, and stable across runs), or
    None for missing / NaN values (JSON has no NaN)."""
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, places)


def race_key(season: int, rnd: int) -> str:
    return f"{season}-{rnd:02d}"


def _dump(data) -> bytes:
    # sort_keys + fixed separators: the same data always gives the same bytes.
    return (json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


# -----------------------------------------------------------------------------
# One race
# -----------------------------------------------------------------------------

def _race_start(conn, event_id) -> str | None:
    row = conn.execute("SELECT start_time_utc FROM sessions WHERE event_id = ? AND session_type = 'R'",
                       (event_id,)).fetchone()
    return row[0] if row else None


def build_race(conn: sqlite3.Connection, event_id: int) -> dict:
    """Everything the dashboard shows about one race (see the module docstring)."""
    event = conn.execute(
        "SELECT e.event_id, e.season, e.round, e.name, e.race_date, c.name AS circuit, c.locality, c.country "
        "FROM events e JOIN circuits c USING (circuit_id) WHERE e.event_id = ?", (event_id,)).fetchone()
    sessions = {r["session_type"]: r["start_time_utc"] for r in conn.execute(
        "SELECT session_type, start_time_utc FROM sessions WHERE event_id = ? ORDER BY start_time_utc",
        (event_id,))}

    # --- The result (None until the race has one) ---------------------------
    result_rows = conn.execute(
        "SELECT r.driver_id, r.constructor_id, r.car_number, r.position, r.position_text, r.grid, "
        "r.points, r.status, r.laps_completed FROM results r JOIN sessions s USING (session_id) "
        "WHERE s.event_id = ? AND s.session_type = 'R' ORDER BY r.position", (event_id,)).fetchall()
    result = [{"driver_id": r["driver_id"], "team_id": r["constructor_id"], "car_number": r["car_number"],
               "position": r["position"], "position_text": r["position_text"], "grid": r["grid"],
               "points": _num(r["points"], 1), "status": r["status"], "laps": r["laps_completed"]}
              for r in result_rows] or None

    # --- Every logged run, stage by stage ------------------------------------
    runs = []
    run_rows = conn.execute(
        "SELECT run_id, stage, mode, as_of_utc, created_at_utc, lineup_source, model_version, n_training_races "
        "FROM prediction_runs WHERE event_id = ?", (event_id,)).fetchall()
    # Order: live before backtest, then by stage through the weekend.
    run_rows = sorted(run_rows, key=lambda r: (r["mode"] != "live", STAGE_ORDER.index(r["stage"]), r["run_id"]))
    for run in run_rows:
        factors = {}
        for f in conn.execute("SELECT driver_id, feature, value, contribution FROM prediction_features "
                              "WHERE run_id = ?", (run["run_id"],)):
            factors.setdefault(f["driver_id"], []).append(
                {"feature": f["feature"], "value": _num(f["value"], 4), "contribution": _num(f["contribution"], 4)})
        drivers = []
        for p in conn.execute("SELECT * FROM predictions WHERE run_id = ? ORDER BY predicted_rank", (run["run_id"],)):
            # Biggest effect first; ties by name, so the order is fixed.
            ordered = sorted(factors.get(p["driver_id"], []),
                             key=lambda f: (-abs(f["contribution"] or 0), f["feature"]))
            drivers.append({
                "driver_id": p["driver_id"], "team_id": p["constructor_id"],
                "predicted_rank": p["predicted_rank"],
                "anchor_position": _num(p["anchor_position"], 3),
                "expected_position": _num(p["expected_position"], 3),
                "mean_position": _num(p["mean_position"], 3),
                "p_win": _num(p["p_win"], 4), "p_podium": _num(p["p_podium"], 4),
                "p_points": _num(p["p_points"], 4), "p_dnf": _num(p["p_dnf"], 4),
                "factors": ordered,
            })
        scores = {s["metric"]: {"model": _num(s["model_value"], 4), "baseline": _num(s["baseline_value"], 4),
                                "baseline_name": s["baseline_name"]}
                  for s in conn.execute("SELECT metric, model_value, baseline_value, baseline_name FROM scores "
                                        "WHERE run_id = ? ORDER BY metric", (run["run_id"],))}
        runs.append({
            "run_id": run["run_id"], "stage": run["stage"], "mode": run["mode"],
            "as_of_utc": run["as_of_utc"], "logged_at_utc": run["created_at_utc"],
            "lineup_source": run["lineup_source"], "model_version": run["model_version"],
            "n_training_races": run["n_training_races"],
            "drivers": drivers, "scores": scores or None,
        })

    # --- Names: one table per file, so each run lists ids only ---------------
    driver_ids = {d["driver_id"] for run in runs for d in run["drivers"]} | {r["driver_id"] for r in result or []}
    # A debutant predicted under a temporary id: who he turned out to be,
    # once the race has a result (features/provisional.py). None before that,
    # or if he can't be told (the tick alerts about that).
    resolved = resolve(conn, event_id, sorted(d for d in driver_ids if is_provisional(d))) if result else {}
    driver_ids |= {r for r in resolved.values() if r}
    drivers = {}
    for d in sorted(driver_ids):
        row = conn.execute("SELECT given_name, family_name, code FROM drivers WHERE driver_id = ?", (d,)).fetchone()
        drivers[d] = {"name": f"{row['given_name']} {row['family_name']}" if row else d,
                      "code": row["code"] if row else None,
                      "provisional": is_provisional(d),
                      "resolved_as": resolved.get(d) if is_provisional(d) else None}
    team_ids = sorted({d["team_id"] for run in runs for d in run["drivers"] if d["team_id"]}
                      | {r["team_id"] for r in result or []})
    teams = {t: conn.execute("SELECT name FROM constructors WHERE constructor_id = ?", (t,)).fetchone()[0]
             for t in team_ids}

    # Plain-English names of the factors in this file (explain/contributions.py),
    # so the dashboard never keeps a second copy that could drift.
    features = sorted({f["feature"] for run in runs for d in run["drivers"] for f in d["factors"]})
    feature_labels = {f: FEATURE_LABELS.get(f, f.replace("_", " ")) for f in features}

    return {
        "schema": SCHEMA, "season": event["season"], "round": event["round"], "name": event["name"],
        "feature_labels": feature_labels,
        "race_date": event["race_date"], "circuit": event["circuit"], "locality": event["locality"],
        "country": event["country"], "sessions": sessions,
        "status": "finished" if result else ("predicted" if runs else "upcoming"),
        "runs": runs, "result": result, "drivers": drivers, "teams": teams,
    }


# -----------------------------------------------------------------------------
# The calendar, the scorecard
# -----------------------------------------------------------------------------

def exported_events(conn: sqlite3.Connection, now: datetime) -> list[sqlite3.Row]:
    """Every event of every season that has a logged prediction, plus the next
    race's season (so a new season's calendar shows before its first prediction)."""
    seasons = {r[0] for r in conn.execute(
        "SELECT DISTINCT e.season FROM prediction_runs r JOIN events e USING (event_id)")}
    upcoming = next_race(conn, now)
    if upcoming:
        seasons.add(upcoming["season"])
    if not seasons:
        return []
    marks = ",".join("?" * len(seasons))
    return conn.execute(f"SELECT event_id, season, round, name, race_date FROM events WHERE season IN ({marks}) "
                        "ORDER BY season, round", sorted(seasons)).fetchall()


def next_race(conn: sqlite3.Connection, now: datetime):
    """The first race that hasn't started at `now` (None after the last one)."""
    return conn.execute(
        "SELECT e.event_id, e.season, e.round FROM events e JOIN sessions s ON s.event_id = e.event_id "
        "AND s.session_type = 'R' WHERE s.start_time_utc > ? ORDER BY s.start_time_utc LIMIT 1",
        (to_utc_text(now),)).fetchone()


def build_index(conn: sqlite3.Connection, races: dict[int, dict], now: datetime) -> dict:
    upcoming = next_race(conn, now)
    seasons = {}
    for event in exported_events(conn, now):
        race = races.get(event["event_id"])
        seasons.setdefault(event["season"], []).append({
            "round": event["round"], "name": event["name"], "race_date": event["race_date"],
            "race_start_utc": _race_start(conn, event["event_id"]),
            "status": race["status"] if race else "upcoming",
            # Which stages were logged live / backtest (the app's badges).
            "live_stages": [r["stage"] for r in race["runs"] if r["mode"] == "live"] if race else [],
            "backtest_stages": [r["stage"] for r in race["runs"] if r["mode"] == "backtest"] if race else [],
            "file": f"races/{race_key(event['season'], event['round'])}.json" if race else None,
        })
    return {
        "schema": SCHEMA,
        "seasons": [{"season": s, "races": rs} for s, rs in sorted(seasons.items())],
        "next_race": {"season": upcoming["season"], "round": upcoming["round"]} if upcoming else None,
    }


# Calibration groups: predicted chance from..to (the last includes 1.0).
CALIBRATION_EDGES = [0.0, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0]


def per_race_scores(conn: sqlite3.Connection, season: int, mode: str) -> dict[str, list[dict]]:
    """{stage: [one entry per scored race, in calendar order]}: each race's
    scores, model vs baseline (the same `scores` rows summary() averages)."""
    rows = conn.execute(
        """
        SELECT r.stage, e.season, e.round, e.name, s.metric, s.model_value, s.baseline_value
        FROM scores s
        JOIN prediction_runs r ON r.run_id = s.run_id
        JOIN events e          ON e.event_id = r.event_id
        WHERE e.season = ? AND r.mode = ?
        ORDER BY e.round, s.metric
        """, (season, mode)).fetchall()
    races: dict[str, dict[int, dict]] = {}
    for row in rows:
        race = races.setdefault(row["stage"], {}).setdefault(
            row["round"], {"season": row["season"], "round": row["round"], "name": row["name"], "metrics": {}})
        race["metrics"][row["metric"]] = {"model": _num(row["model_value"], 4), "baseline": _num(row["baseline_value"], 4)}
    return {stage: list(races[stage].values()) for stage in STAGE_ORDER if stage in races}


def calibration(conn: sqlite3.Connection, season: int, mode: str, stage: str) -> dict[str, list[dict]]:
    """How often things predicted at X% happened, for win and podium.

    Over every scored run of this group and stage: each driver who STARTED
    the race, with the run's predicted chance (0 if the run didn't have
    them: the scorer's rule, evaluate/scorer.py, so this matches the Brier
    scores), and whether it happened (won / finished top 3). Grouped by
    predicted chance (CALIBRATION_EDGES); empty groups are left out.
    """
    runs = conn.execute(
        "SELECT r.run_id FROM prediction_runs r JOIN events e ON e.event_id = r.event_id "
        "WHERE e.season = ? AND r.mode = ? AND r.stage = ? "
        "AND EXISTS (SELECT 1 FROM scores s WHERE s.run_id = r.run_id) ORDER BY r.run_id",
        (season, mode, stage)).fetchall()
    pairs = {"win": [], "podium": []}          # (predicted chance, happened 0/1)
    for (run_id,) in runs:
        info = run_info(conn, run_id)
        actual = actual_result(conn, info["race_session_id"])
        try:
            logged = real_driver_ids(conn, run_id, info["event_id"], logged_predictions(conn, run_id))
        except UnresolvedDriverError:
            continue                            # (not scored either; the tick alerts)
        chances = {row.driver_id: (row.p_win, row.p_podium) for row in logged.itertuples()}
        for driver, position in actual.items():
            p_win, p_podium = chances.get(driver, (0.0, 0.0))
            pairs["win"].append((p_win, int(position == 1)))
            pairs["podium"].append((p_podium, int(position <= 3)))

    groups = {}
    for outcome, values in pairs.items():
        bins = []
        for low, high in zip(CALIBRATION_EDGES, CALIBRATION_EDGES[1:]):
            inside = [(p, hit) for p, hit in values if low <= p < high or (high == 1.0 and p == 1.0)]
            if inside:
                bins.append({"from": low, "to": high, "n": len(inside),
                             "predicted": _num(sum(p for p, _ in inside) / len(inside), 4),
                             "observed": _num(sum(hit for _, hit in inside) / len(inside), 4)})
        groups[outcome] = bins
    return groups


def build_scorecard(conn: sqlite3.Connection) -> dict:
    """Per (season, mode): each stage's average of each metric vs its baseline,
    each race's scores, and calibration. The groups are NEVER added together
    (docs/evaluation-protocol.md): each carries the label that must be shown
    with it."""
    groups = []
    for season, mode in scorecards(conn):
        table = summary(conn, season, mode)
        stages = {}
        for row in table.itertuples():
            stages.setdefault(row.stage, {})[row.metric] = {
                "races": int(row.races), "model": _num(row.model, 4), "baseline": _num(row.baseline, 4),
                "baseline_name": row.baseline_name, "improvement": _num(row.improvement, 4),
                "higher_is_better": HIGHER_IS_BETTER.get(row.metric),
            }
        stage_names = [s for s in STAGE_ORDER if s in stages]
        groups.append({"season": season, "mode": mode, "label": scorecard_label(season, mode),
                       "stages": {s: stages[s] for s in stage_names},
                       "per_race": per_race_scores(conn, season, mode),
                       "calibration": {s: calibration(conn, season, mode, s) for s in stage_names}})
    return {"schema": SCHEMA, "groups": groups}


# -----------------------------------------------------------------------------
# Writing it all
# -----------------------------------------------------------------------------

def export(conn: sqlite3.Connection, out_dir: Path | str, now: datetime | None = None) -> dict:
    """Write every file under <out_dir>/data/. Returns version.json's content.

    Files from an earlier export that are no longer produced are removed, so
    the folder is always exactly this export.
    """
    now = now or datetime.now(timezone.utc)
    data_dir = Path(out_dir) / "data"
    files: dict[str, bytes] = {}

    races = {}
    for event in exported_events(conn, now):
        has_runs = conn.execute("SELECT 1 FROM prediction_runs WHERE event_id = ?", (event["event_id"],)).fetchone()
        upcoming = next_race(conn, now)
        if has_runs or (upcoming and upcoming["event_id"] == event["event_id"]):
            races[event["event_id"]] = build_race(conn, event["event_id"])
            files[f"races/{race_key(event['season'], event['round'])}.json"] = _dump(races[event["event_id"]])
    files["index.json"] = _dump(build_index(conn, races, now))
    files["scorecard.json"] = _dump(build_scorecard(conn))

    # The hash covers every file's path and bytes, in a fixed order.
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.encode() + b"\0" + files[path] + b"\0")
    version = {"schema": SCHEMA, "hash": digest.hexdigest(), "exported_at_utc": to_utc_text(now),
               "files": len(files)}
    files["version.json"] = _dump(version)

    # Write, and remove anything left over from an earlier export.
    data_dir.mkdir(parents=True, exist_ok=True)
    for old in data_dir.rglob("*.json"):
        if old.relative_to(data_dir).as_posix() not in files:
            old.unlink()
    for path, content in files.items():
        target = data_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return version
