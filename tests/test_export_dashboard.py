"""The dashboard export (src/f1predictor/export/dashboard.py).

The React dashboard shows ONLY what these files hold, so the tests check
that the files say exactly what the prediction log says, nothing more:

    * every logged number is in the race file, equal to the log (rounded);
    * a race's status, result and scores appear when they exist, not before;
    * a debutant's temporary id comes with his name and, after the race, who he was;
    * the scorecard equals the scorer's own summary, and keeps live and
      backtest apart, each with its label;
    * the same database always gives the same bytes and hash (so an hourly
      tick that changed nothing deploys nothing), and any change changes the hash;
    * the export never writes to the database.

Data: the fake world's 2027 rounds 1-3 run hour by hour by the real
pipeline (test_simulated_weekend.simulate), with two debutants in round 2
(as in test_debutants.py).
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from f1predictor.db.connection import get_connection
from f1predictor.evaluate.scorer import score_runs, summary
from f1predictor.export.dashboard import export
from f1predictor.predict import stage

from test_simulated_weekend import END, simulate

ROOT = Path(__file__).resolve().parent.parent
DEBUTANTS = {(2027, 2): {"d05": {"id": "n50", "number": 50}, "d12": {"id": "n51", "number": 51}}}
N50 = "provisional_2027_r02_car50"


@pytest.fixture(scope="module")
def world():
    mp = pytest.MonkeyPatch()
    mp.setattr(stage, "model_version", lambda: "abc1234+settings.00000000")
    try:
        conn, _ = simulate(debutants=DEBUTANTS)
        yield conn
    finally:
        mp.undo()


@pytest.fixture
def exported(world, tmp_path):
    """Export at the end of the simulation. Returns a reader: read('index.json')."""
    export(world, tmp_path, now=END)

    def read(name):
        return json.loads((tmp_path / "data" / name).read_text())
    return read


def copy_of(conn):
    """An in-memory copy, for tests that change the database."""
    copy = get_connection(":memory:")
    conn.backup(copy)
    return copy


def files(folder: Path) -> dict[str, bytes]:
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*.json"))}


# -----------------------------------------------------------------------------
# What's in the files
# -----------------------------------------------------------------------------

def test_one_file_per_race_with_predictions_plus_the_index_and_scorecard(world, tmp_path):
    export(world, tmp_path, now=END)
    # 2027 rounds 1-3 have logged runs; 2026 (history only, no runs) has
    # no files at all. Plain files, no NaN anywhere (JSON can't hold it,
    # and a browser would fail to read the file).
    assert set(files(tmp_path / "data")) == {"index.json", "scorecard.json", "version.json",
                                            "races/2027-01.json", "races/2027-02.json", "races/2027-03.json"}
    for content in files(tmp_path / "data").values():
        json.loads(content, parse_constant=lambda c: pytest.fail(f"{c} in a file"))


def test_every_logged_number_is_in_the_race_file_exactly(world, exported):
    race = exported("races/2027-01.json")
    assert [r["stage"] for r in race["runs"]] == ["pre_weekend", "practice", "post_quali"]
    for run in race["runs"]:
        logged = world.execute("SELECT * FROM predictions WHERE run_id = ? ORDER BY predicted_rank",
                               (run["run_id"],)).fetchall()
        # Same drivers, in the same order, with the same numbers (rounded to 4 places).
        assert [d["driver_id"] for d in run["drivers"]] == [p["driver_id"] for p in logged]
        for d, p in zip(run["drivers"], logged):
            for column in ("p_win", "p_podium", "p_points", "p_dnf"):
                assert d[column] == pytest.approx(p[column], abs=5e-5)
            assert d["predicted_rank"] == p["predicted_rank"]
            # Every SHAP factor of the driver, biggest effect first.
            n_features = world.execute("SELECT COUNT(*) FROM prediction_features WHERE run_id = ? "
                                       "AND driver_id = ?", (run["run_id"], p["driver_id"])).fetchone()[0]
            effects = [abs(f["contribution"]) for f in d["factors"]]
            assert len(effects) == n_features > 0 and effects == sorted(effects, reverse=True)
        # Run details the dashboard shows (when it was logged, its cutoff, the code).
        row = world.execute("SELECT as_of_utc, created_at_utc, model_version FROM prediction_runs "
                            "WHERE run_id = ?", (run["run_id"],)).fetchone()
        assert (run["as_of_utc"], run["logged_at_utc"], run["model_version"]) == tuple(row)
        assert run["mode"] == "live"


def test_results_and_scores_appear_only_once_the_race_has_them(exported):
    finished, upcoming = exported("races/2027-01.json"), exported("races/2027-03.json")
    # Round 1: raced. The result, in finishing order, and every run's scores.
    assert finished["status"] == "finished"
    assert [r["position"] for r in finished["result"]] == list(range(1, 21))
    assert all(run["scores"]["position_mae"]["baseline"] is not None for run in finished["runs"])
    # Round 3: only its pre-weekend prediction so far. No result, no scores.
    assert upcoming["status"] == "predicted"
    assert upcoming["result"] is None
    assert [(r["stage"], r["scores"]) for r in upcoming["runs"]] == [("pre_weekend", None)]


def test_names_and_teams_come_with_the_file_and_a_debutant_is_identified(exported):
    race = exported("races/2027-02.json")
    # Every id in the file has a name (one table per file, ids in the runs).
    ids = {d["driver_id"] for run in race["runs"] for d in run["drivers"]} | {r["driver_id"] for r in race["result"]}
    assert ids <= set(race["drivers"])
    assert race["drivers"]["d01"] == {"name": "Test D01", "code": "D01", "provisional": False, "resolved_as": None}
    assert race["teams"]["t03"] == "T03"
    # The debutant was predicted at practice under a temporary id; the file
    # says so, gives OpenF1's name, and (the race being over) who he was.
    assert race["drivers"][N50] == {"name": "Rookie N50", "code": None, "provisional": True, "resolved_as": "n50"}
    assert "n50" in race["drivers"]


def test_every_factor_has_its_plain_english_name(exported):
    # The Why page shows these names; they come from the one Python list
    # (explain/contributions.FEATURE_LABELS), not a copy in the app.
    from f1predictor.explain.contributions import FEATURE_LABELS
    race = exported("races/2027-01.json")
    used = {f["feature"] for run in race["runs"] for d in run["drivers"] for f in d["factors"]}
    assert set(race["feature_labels"]) == used
    assert race["feature_labels"] == {f: FEATURE_LABELS[f] for f in used}
    assert race["feature_labels"]["grid"] == "starting position"


def test_the_index_lists_the_calendar_and_the_next_race(world, exported, tmp_path):
    index = exported("index.json")
    [season] = index["seasons"]
    assert season["season"] == 2027
    assert [(r["round"], r["status"], r["file"]) for r in season["races"]] == [
        (1, "finished", "races/2027-01.json"), (2, "finished", "races/2027-02.json"),
        (3, "predicted", "races/2027-03.json")]
    assert season["races"][0]["live_stages"] == ["pre_weekend", "practice", "post_quali"]
    assert season["races"][2]["race_start_utc"] == "2027-03-21T15:00:00Z"
    assert index["next_race"] == {"season": 2027, "round": 3}
    # After the last race of the calendar has started: no next race.
    later = tmp_path / "later"
    export(world, later, now=datetime(2027, 3, 22, tzinfo=timezone.utc))
    assert json.loads((later / "data" / "index.json").read_text())["next_race"] is None


def test_the_scorecard_is_the_scorers_summary_with_live_and_backtest_kept_apart(world, tmp_path):
    # Add a BACKTEST run of round 1 (and score it), so both kinds exist.
    conn = copy_of(world)
    event_id = conn.execute("SELECT event_id FROM events WHERE season = 2027 AND round = 1").fetchone()[0]
    score_runs(conn, [stage.predict_stage(conn, event_id, "post_quali", mode="backtest", n_sims=500)])
    export(conn, tmp_path, now=END)
    card = json.loads((tmp_path / "data" / "scorecard.json").read_text())
    groups = {(g["season"], g["mode"]): g for g in card["groups"]}
    # Two groups, never added together, each with the label the scorer gives it.
    assert list(groups) == [(2027, "live"), (2027, "backtest")]
    assert groups[(2027, "live")]["label"].startswith("2027 live: the real test")
    assert groups[(2027, "backtest")]["label"].startswith("2027 backtest (in-sample")
    # The live numbers are exactly the scorer's summary of LIVE runs only:
    # the backtest run's scores aren't mixed in.
    live = summary(conn, 2027, "live")
    for row in live.itertuples():
        exported_row = groups[(2027, "live")]["stages"][row.stage][row.metric]
        assert exported_row["races"] == row.races == 2
        assert exported_row["model"] == pytest.approx(row.model, abs=5e-5)
    assert groups[(2027, "backtest")]["stages"]["post_quali"]["position_mae"]["races"] == 1
    assert list(groups[(2027, "backtest")]["stages"]) == ["post_quali"]
    # Race by race and calibration are kept apart the same way: the backtest
    # group has its one race (20 starters), the live group its two.
    backtest = groups[(2027, "backtest")]
    assert [r["round"] for r in backtest["per_race"]["post_quali"]] == [1]
    assert sum(b["n"] for b in backtest["calibration"]["post_quali"]["podium"]) == 20
    assert sum(b["n"] for b in groups[(2027, "live")]["calibration"]["post_quali"]["podium"]) == 40


def test_the_scorecard_has_each_races_scores_exactly_as_scored(world, exported):
    group = exported("scorecard.json")["groups"][0]
    for stage, races in group["per_race"].items():
        # One entry per scored race (rounds 1 and 2), each metric equal to the scores table.
        assert [r["round"] for r in races] == [1, 2]
        for race in races:
            for metric, value in race["metrics"].items():
                row = world.execute(
                    "SELECT s.model_value, s.baseline_value FROM scores s JOIN prediction_runs r USING (run_id) "
                    "JOIN events e USING (event_id) WHERE e.round = ? AND r.stage = ? AND r.mode = 'live' "
                    "AND s.metric = ?", (race["round"], stage, metric)).fetchone()
                assert value["model"] == pytest.approx(row[0], abs=5e-5)
                assert (value["baseline"] is None) == (row[1] is None)


def test_calibration_counts_every_starter_once_and_what_really_happened(world, exported):
    # Calibration: drivers grouped by predicted chance; for each group, the
    # average chance and how often it happened. Checked against a hand
    # count: two scored races, 20 starters each, so 40 drivers per stage;
    # exactly 2 winners and 6 podium finishers among them; and the average
    # predicted chance is the logged one.
    group = exported("scorecard.json")["groups"][0]
    for stage, outcomes in group["calibration"].items():
        for outcome, bins in outcomes.items():
            assert sum(b["n"] for b in bins) == 40
            happened = sum(b["n"] * b["observed"] for b in bins)
            # (rates are rounded to 4 places, like every number in the files)
            assert happened == pytest.approx({"win": 2, "podium": 6}[outcome], abs=0.002)
            for b in bins:
                assert b["from"] <= b["predicted"] <= b["to"]
            # Total predicted chance = the logged chances of the drivers who
            # STARTED (the scorer's rule: d05 and d12, replaced in round 2,
            # drop out of the pre-weekend run; a debutant's temporary id
            # counts as the driver he turned out to be, and both started).
            column = {"win": "p_win", "podium": "p_podium"}[outcome]
            total = world.execute(
                f"SELECT SUM(p.{column}) FROM predictions p JOIN prediction_runs r USING (run_id) "
                "JOIN events e USING (event_id) JOIN sessions rs ON rs.event_id = e.event_id AND rs.session_type = 'R' "
                "WHERE r.mode = 'live' AND r.stage = ? AND e.round <= 2 AND (p.driver_id LIKE 'provisional_%' "
                "OR p.driver_id IN (SELECT driver_id FROM results WHERE session_id = rs.session_id))",
                (stage,)).fetchone()[0]
            # (each group average is rounded to 4 places: up to 40 x 0.00005 in total)
            assert sum(b["n"] * b["predicted"] for b in bins) == pytest.approx(total, abs=0.002)


# -----------------------------------------------------------------------------
# Same data, same files; any change, a new hash
# -----------------------------------------------------------------------------

def test_the_same_database_gives_the_same_bytes_and_hash(world, tmp_path):
    # Two exports an hour apart with nothing changed in between: identical
    # files, except the export time in version.json, which isn't hashed.
    # This is what makes "deploy only if the hash changed" work.
    first = export(world, tmp_path / "a", now=END)
    second = export(world, tmp_path / "b", now=END.replace(hour=13))
    a, b = files(tmp_path / "a" / "data"), files(tmp_path / "b" / "data")
    assert {k: v for k, v in a.items() if k != "version.json"} == {k: v for k, v in b.items() if k != "version.json"}
    assert first["hash"] == second["hash"]
    assert first["exported_at_utc"] != second["exported_at_utc"]


def test_any_change_in_what_is_shown_changes_the_hash(world, tmp_path):
    before = export(world, tmp_path / "a", now=END)["hash"]
    # Scores may be rewritten (e.g. after a penalty changes a result).
    conn = copy_of(world)
    conn.execute("UPDATE scores SET model_value = model_value + 0.5 WHERE metric = 'position_mae' "
                 "AND run_id = (SELECT MIN(run_id) FROM prediction_runs)")
    assert export(conn, tmp_path / "b", now=END)["hash"] != before


def test_files_from_an_earlier_export_that_no_longer_exist_are_removed(world, tmp_path):
    # The folder is exactly one export: a race file that shouldn't be there
    # (e.g. a race renumbered on the calendar) doesn't linger on the site.
    stale = tmp_path / "data" / "races" / "1999-01.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    export(world, tmp_path, now=END)
    assert not stale.exists()


# -----------------------------------------------------------------------------
# The script: read-only
# -----------------------------------------------------------------------------

def test_the_script_never_writes_to_the_database(world, tmp_path):
    # The real script, on a database file, as the workflow runs it. The file
    # is byte-for-byte the same afterwards (SQLite opened with mode=ro).
    db = tmp_path / "f1.sqlite"
    disk = get_connection(db)
    world.backup(disk)
    disk.close()
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_dashboard.py"), str(db), str(tmp_path / "site")],
                         capture_output=True, text=True, check=True).stdout
    assert out.startswith("exported 6 files, hash ")
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    assert (tmp_path / "site" / "data" / "index.json").exists()


def test_the_script_refuses_a_missing_database(tmp_path):
    # (Rather than SQLite creating an empty one and exporting nothing.)
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_dashboard.py"),
                          str(tmp_path / "missing.sqlite"), str(tmp_path / "site")], capture_output=True, text=True)
    assert run.returncode != 0 and "no database at" in run.stderr
    assert not (tmp_path / "missing.sqlite").exists()


# -----------------------------------------------------------------------------
# The React app's test data is this export
# -----------------------------------------------------------------------------

def same_data(a, b, path="") -> list[str]:
    """Differences between two JSON values: same keys, lists and texts;
    numbers within 0.001 (the model's floating point may differ slightly
    between machines; the shape and the ids may not)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys {sorted(set(a) ^ set(b))}"]
        return [d for k in a for d in same_data(a[k], b[k], f"{path}.{k}")]
    if path.endswith(".logged_at_utc"):
        # The real clock time the simulated run was written (write_run
        # stamps the actual time, on purpose): different every simulation.
        # Only its form must match.
        return [] if all(isinstance(x, str) and x.endswith("Z") and len(x) == 20 for x in (a, b)) \
            else [f"{path}: {a!r} vs {b!r}"]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: {len(a)} vs {len(b)} items"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in same_data(x, y, f"{path}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return [] if abs(a - b) <= 1e-3 else [f"{path}: {a} vs {b}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def test_the_react_apps_test_data_is_what_the_export_writes(world, tmp_path):
    # web/src/test/data/ (scripts/export_test_data.py) is what the React
    # tests load. If the export changes and the files aren't regenerated,
    # the app would be tested against an old format: this fails instead.
    # Fix: run `python scripts/export_test_data.py` and commit the files.
    export(world, tmp_path, now=END)
    committed = ROOT / "web" / "src" / "test" / "data"
    fresh = tmp_path / "data"
    assert set(files(committed)) == set(files(fresh))
    for name in files(fresh):
        a, b = json.loads((committed / name).read_text()), json.loads((fresh / name).read_text())
        if name == "version.json":     # the hash covers exact bytes; the numbers may differ a hair
            a, b = {**a, "hash": None}, {**b, "hash": None}
        differences = same_data(a, b)
        assert differences == [], f"{name} is out of date (run scripts/export_test_data.py): {differences[:5]}"
