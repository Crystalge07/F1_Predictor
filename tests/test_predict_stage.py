"""Tests for predicting a stage and writing it to the log (predict/stage.py).

Reuses the two-race weekend from test_as_of.py: race 1 is history, race 2 is
predicted, and race 2's real result is full of surprises (penalties, a sick
driver, a pit-lane start) that must not leak into any prediction.
"""

import re
import sqlite3

import pytest

from f1predictor.db.connection import init_db
from f1predictor.features.stages import STAGE_FEATURES
from f1predictor.predict.stage import make_prediction, model_version, predict_stage, write_run

# Importing the fixtures makes pytest offer them to the tests in this file too.
from test_as_of import conn, weekend  # noqa: F401

N_SIMS = 2_000   # plenty for these checks, and fast


def rows(conn, sql, *params):
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def test_post_quali_run_is_logged(conn, weekend):
    run_id = predict_stage(conn, weekend["event"], "post_quali", n_sims=N_SIMS)

    [run] = rows(conn, "SELECT * FROM prediction_runs WHERE run_id = ?", run_id)
    assert run["stage"] == "post_quali" and run["mode"] == "backtest"
    assert run["as_of_utc"] == "2024-03-17T15:00:00Z"        # race start
    assert run["lineup_source"] == "starting_grid"
    assert run["n_training_races"] == 1
    assert run["model_version"] == model_version()

    preds = rows(conn, "SELECT * FROM predictions WHERE run_id = ? ORDER BY predicted_rank", run_id)
    # The penalty grid's four drivers (with 'd', who was ill on Sunday), not the result's.
    assert sorted(p["driver_id"] for p in preds) == ["a", "b", "c", "d"]
    assert [p["predicted_rank"] for p in preds] == [1, 2, 3, 4]
    # The anchor is the starting grid after penalties.
    assert {p["driver_id"]: p["anchor_position"] for p in preds} == {"b": 1, "c": 2, "a": 3, "d": 4}
    # Exactly one winner per simulated race (unless everyone retires, which is
    # vanishingly rare), so the win chances add up to ~1.
    assert sum(p["p_win"] for p in preds) == pytest.approx(1.0, abs=0.01)
    assert all(0 <= p["p_dnf"] <= 1 for p in preds)

    # One feature row per driver per feature the model used.
    n_features = conn.execute("SELECT COUNT(*) FROM prediction_features WHERE run_id = ?",
                              (run_id,)).fetchone()[0]
    assert n_features == 4 * len(STAGE_FEATURES["post_quali"])
    [grid] = rows(conn, "SELECT value FROM prediction_features "
                        "WHERE run_id = ? AND driver_id = 'a' AND feature = 'grid'", run_id)
    assert grid["value"] == 3.0


def test_pre_weekend_run_uses_only_pre_weekend_features(conn, weekend):
    run_id = predict_stage(conn, weekend["event"], "pre_weekend", n_sims=N_SIMS)
    [run] = rows(conn, "SELECT * FROM prediction_runs WHERE run_id = ?", run_id)
    assert run["lineup_source"] == "previous_race"
    features = {r["feature"] for r in rows(conn, "SELECT DISTINCT feature FROM prediction_features "
                                                 "WHERE run_id = ?", run_id)}
    assert features == set(STAGE_FEATURES["pre_weekend"])
    assert "grid" not in features


def test_practice_stage_without_practice_laps_is_refused(conn, weekend):
    # The fixture has practice ENTRIES but no laps, so there is no practice
    # order to anchor on: refuse rather than log a half-made prediction.
    with pytest.raises(ValueError, match="practice_rank"):
        predict_stage(conn, weekend["event"], "practice", n_sims=N_SIMS)
    assert conn.execute("SELECT COUNT(*) FROM prediction_runs").fetchone()[0] == 0


@pytest.mark.parametrize("stage", ["pre_weekend", "post_quali"])
def test_changing_the_result_changes_no_prediction(conn, weekend, stage):
    # The leakage test, end to end: rewrite race 2's result completely; the
    # prediction (order, probabilities, SHAP) must not move at all.
    before = make_prediction(conn, weekend["event"], stage, n_sims=N_SIMS)
    conn.execute("UPDATE results SET position = 5 - position, grid = 9 WHERE session_id = ?",
                 (weekend["R"],))
    after = make_prediction(conn, weekend["event"], stage, n_sims=N_SIMS)
    assert before.drivers.equals(after.drivers)
    assert before.features.equals(after.features)


def test_unknown_mode_rejected_before_anything_is_written(conn, weekend):
    with pytest.raises(ValueError, match="Unknown mode"):
        predict_stage(conn, weekend["event"], "post_quali", mode="dry_run")
    assert conn.execute("SELECT COUNT(*) FROM prediction_runs").fetchone()[0] == 0


def test_a_failed_write_leaves_no_half_run(conn, weekend):
    prediction = make_prediction(conn, weekend["event"], "post_quali", n_sims=N_SIMS)
    # A feature row for a driver with no prediction breaks a foreign key
    # halfway through the write...
    prediction.features.loc[0, "driver_id"] = "e"
    with pytest.raises(sqlite3.IntegrityError):
        write_run(conn, prediction, "backtest")
    # ...and the whole run is rolled back, not just the failing row.
    for table in ["prediction_runs", "predictions", "prediction_features"]:
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table


def test_live_run_is_logged_and_cannot_be_deleted(conn, weekend):
    run_id = predict_stage(conn, weekend["event"], "post_quali", mode="live", n_sims=N_SIMS)
    with pytest.raises(sqlite3.IntegrityError, match="never deleted"):
        conn.execute("DELETE FROM prediction_runs WHERE run_id = ?", (run_id,))


def test_a_second_live_run_is_refused_even_without_the_live_checks(conn, weekend):
    # The tick checks "no live run yet" (live_problems) before writing. This
    # skips that check and calls write_run directly, the way a buggy script
    # might: the database itself must refuse the second live run, and
    # write_run's transaction must leave no stray rows from the attempt.
    prediction = make_prediction(conn, weekend["event"], "post_quali", n_sims=N_SIMS)
    write_run(conn, prediction, "live")
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ["prediction_runs", "predictions", "prediction_features"]}
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        write_run(conn, prediction, "live")
    for table, n in counts.items():
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == n, table


def test_model_version_names_commit_and_settings():
    # e.g. 'a1b2c3d+settings.9f8e7d6c' or 'a1b2c3d-dirty+settings.9f8e7d6c'
    assert re.fullmatch(r"(nogit|[0-9a-f]{4,40}(-dirty)?)\+settings\.[0-9a-f]{8}", model_version())


def test_init_db_adds_lineup_source_to_an_old_log(tmp_path):
    # A database whose prediction_runs predates the lineup_source column.
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.execute("""CREATE TABLE prediction_runs (
        run_id INTEGER PRIMARY KEY, event_id INTEGER NOT NULL, stage TEXT NOT NULL,
        mode TEXT NOT NULL, as_of_utc TIMESTAMP NOT NULL, created_at_utc TIMESTAMP NOT NULL,
        model_version TEXT NOT NULL, settings_json TEXT NOT NULL, n_training_races INTEGER)""")
    old.close()
    conn = sqlite3.connect(path)
    init_db(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(prediction_runs)")}
    assert "lineup_source" in columns
    conn.close()
