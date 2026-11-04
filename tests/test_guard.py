"""Tests for the upload guard (db/guard.py): the live log must survive every upload."""

import shutil
import sqlite3

import pytest

from f1predictor.db.connection import get_connection, init_db
from f1predictor.db.guard import upload_problems


@pytest.fixture
def db(tmp_path):
    """A database file with one live run (and one backtest run)."""
    path = tmp_path / "old.sqlite"
    conn = get_connection(path)
    init_db(conn)
    conn.execute("INSERT INTO circuits (circuit_id, name) VALUES ('x', 'X')")
    conn.execute("INSERT INTO drivers (driver_id, given_name, family_name) VALUES ('a', 'A', 'A')")
    event = conn.execute("INSERT INTO events (season, round, name, circuit_id, race_date) "
                         "VALUES (2026, 20, 'GP', 'x', '2026-11-08')").lastrowid
    for mode in ("live", "backtest"):
        run = conn.execute("INSERT INTO prediction_runs (event_id, stage, mode, as_of_utc, created_at_utc, "
                           "model_version, settings_json) VALUES (?, 'pre_weekend', ?, 't', 't', 'v', '{}')",
                           (event, mode)).lastrowid
        conn.execute("INSERT INTO predictions (run_id, driver_id, predicted_rank, p_win, p_podium, p_points, p_dnf) "
                     "VALUES (?, 'a', 1, 0.3, 0.6, 0.9, 0.1)", (run,))
    conn.commit()
    conn.close()
    return path


def copy(db, tmp_path):
    new = tmp_path / "new.sqlite"
    shutil.copy(db, new)
    return new


def test_unchanged_or_grown_database_is_safe(db, tmp_path):
    new = copy(db, tmp_path)
    conn = get_connection(new)
    conn.execute("INSERT INTO pipeline_ticks (started_at_utc, dry_run) VALUES ('t', 0)")   # normal tick bookkeeping
    conn.execute("DELETE FROM prediction_runs WHERE mode = 'backtest'")                     # allowed
    conn.commit()
    conn.close()
    assert upload_problems(db, new) == []


def test_a_lost_live_run_blocks_the_upload(db, tmp_path):
    new = copy(db, tmp_path)
    conn = sqlite3.connect(new)          # a plain connection: bypasses nothing, triggers still apply...
    conn.execute("DROP TRIGGER prediction_runs_no_live_delete")   # ...so simulate a bug that removed them
    conn.execute("DROP TRIGGER predictions_no_live_delete")
    conn.execute("DELETE FROM predictions")
    conn.execute("DELETE FROM prediction_runs WHERE mode = 'live'")
    conn.commit()
    conn.close()
    problems = upload_problems(db, new)
    assert any("missing" in p for p in problems)


def test_a_changed_live_prediction_blocks_the_upload(db, tmp_path):
    new = copy(db, tmp_path)
    conn = sqlite3.connect(new)
    conn.execute("DROP TRIGGER predictions_no_update")
    conn.execute("UPDATE predictions SET p_win = 0.9 WHERE run_id = 1")
    conn.commit()
    conn.close()
    assert any("changed" in p for p in upload_problems(db, new))


def test_a_damaged_file_blocks_the_upload(db, tmp_path):
    new = tmp_path / "new.sqlite"
    new.write_bytes(db.read_bytes()[:4096])          # truncated mid-file
    problems = upload_problems(db, new)
    assert problems and ("damaged" in problems[0] or "can't be read" in problems[0])


def test_a_new_column_does_not_block_the_upload(db, tmp_path):
    """A schema change that adds a column (filled in for old rows with NULL or
    a default) must not make every live run look changed."""
    new = copy(db, tmp_path)
    conn = sqlite3.connect(new)
    conn.execute("ALTER TABLE predictions ADD COLUMN model_rank INTEGER")
    conn.execute("ALTER TABLE prediction_runs ADD COLUMN note TEXT DEFAULT 'x'")
    conn.commit()
    conn.close()
    assert upload_problems(db, new) == []


def test_a_changed_live_value_still_blocks_after_a_new_column(db, tmp_path):
    new = copy(db, tmp_path)
    conn = sqlite3.connect(new)
    conn.execute("ALTER TABLE predictions ADD COLUMN model_rank INTEGER")
    conn.execute("DROP TRIGGER predictions_no_update")
    conn.execute("UPDATE predictions SET p_win = 0.9 WHERE run_id = 1")
    conn.commit()
    conn.close()
    assert any("changed" in p for p in upload_problems(db, new))


def test_a_dropped_column_blocks_the_upload(db, tmp_path):
    new = copy(db, tmp_path)
    conn = sqlite3.connect(new)
    conn.execute("ALTER TABLE predictions DROP COLUMN p_points")
    conn.commit()
    conn.close()
    assert any("columns missing" in p and "p_points" in p for p in upload_problems(db, new))


def test_fingerprints_match_the_old_select_star_version(db):
    """Naming the columns gives the same fingerprint SELECT * did, so the
    first upload after this change compares like with like."""
    import hashlib
    from f1predictor.db.guard import live_fingerprints
    conn = sqlite3.connect(db)
    digest = hashlib.sha256()
    for query in ("SELECT * FROM prediction_runs WHERE run_id = ?",
                  "SELECT * FROM predictions WHERE run_id = ? ORDER BY driver_id",
                  "SELECT * FROM prediction_features WHERE run_id = ? ORDER BY driver_id, feature"):
        for row in conn.execute(query, (1,)):
            digest.update(repr(tuple(row)).encode("utf-8"))
    assert live_fingerprints(conn) == {1: digest.hexdigest()}
