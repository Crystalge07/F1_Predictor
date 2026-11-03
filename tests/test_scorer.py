"""Tests for the scorer (evaluate/scorer.py).

The weekend from test_as_of.py is a hard one to score: the prediction lists
a, b, c, d, but on race day 'd' was ill and the reserve 'e' started (and won):
    result: e 1st, c 2nd, b 3rd, a 4th
So every run has one predicted non-starter (d) and one unpredicted starter (e).
"""

import pandas as pd
import pytest

from f1predictor.evaluate.scorer import reconcile, score_run, score_runs
from f1predictor.models.dnf import brier
from f1predictor.predict.stage import predict_stage

from test_as_of import conn, weekend  # noqa: F401  (fixtures)

N_SIMS = 2_000


def scores_of(conn, run_id):
    return {r["metric"]: dict(r) for r in conn.execute("SELECT * FROM scores WHERE run_id = ?", (run_id,))}


def test_reconcile_drops_non_starters_and_appends_unpredicted():
    order, missing, extra = reconcile(["a", "b", "c", "d"], {"e": 1, "c": 2, "b": 3, "a": 4})
    assert order == ["a", "b", "c", "e"]
    assert missing == ["e"] and extra == ["d"]


def test_post_quali_run_is_scored_against_the_as_of_grid(conn, weekend):
    run_id = predict_stage(conn, weekend["event"], "post_quali", n_sims=N_SIMS)
    score_run(conn, run_id)
    s = scores_of(conn, run_id)

    assert s["lineup_missing"]["model_value"] == 1         # 'e'
    assert s["lineup_extra"]["model_value"] == 1           # 'd'
    assert s["lineup_missing"]["baseline_value"] is None

    # Baseline order = the logged starting grid (b, c, a, d), d dropped, e at
    # the back: [b, c, a, e] vs actual e1 c2 b3 a4 -> errors 2, 0, 1, 3.
    assert s["position_mae"]["baseline_value"] == pytest.approx(1.5)
    assert s["position_mae"]["baseline_name"] == "grid_order"
    # Nobody predicted 'e', so neither side can have picked the winner.
    assert s["winner_hit"]["model_value"] == 0 and s["winner_hit"]["baseline_value"] == 0

    # Baseline probabilities: slot rates from race 1 (pole won, slots 1-3 on
    # the podium). b (slot 1) gets p_win 1; e gets 0 and won.
    #   win:    b (1-0)^2 + e (0-1)^2       over 4 starters = 0.5
    #   podium: a (slot 3, 1-0)^2 + e (0-1)^2               = 0.5
    assert s["brier_win"]["baseline_value"] == pytest.approx(0.5)
    assert s["brier_podium"]["baseline_value"] == pytest.approx(0.5)

    # The model's Brier comes from its logged probabilities, with e at 0.
    logged = pd.read_sql_query("SELECT driver_id, p_win FROM predictions WHERE run_id = ?",
                               conn, params=(run_id,)).set_index("driver_id")["p_win"]
    field = {"e": 1, "c": 2, "b": 3, "a": 4}
    p = pd.Series([logged.get(d, 0.0) for d in field])
    won = pd.Series([int(pos == 1) for pos in field.values()])
    assert s["brier_win"]["model_value"] == pytest.approx(brier(p, won))


def test_pre_weekend_baseline_is_championship_order(conn, weekend):
    run_id = predict_stage(conn, weekend["event"], "pre_weekend", n_sims=N_SIMS)
    score_run(conn, run_id)
    assert scores_of(conn, run_id)["spearman"]["baseline_name"] == "championship_order"


def test_rescoring_replaces_scores(conn, weekend):
    run_id = predict_stage(conn, weekend["event"], "post_quali", n_sims=N_SIMS)
    score_run(conn, run_id)
    n_before = conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
    # The FIA changes the result after the race (say, a time penalty): a and c
    # swap places, so it's now e, a, b, c.
    conn.execute("UPDATE results SET position = CASE driver_id WHEN 'c' THEN 4 WHEN 'a' THEN 2 "
                 "ELSE position END WHERE session_id = ?", (weekend["R"],))
    before = scores_of(conn, run_id)["position_mae"]["baseline_value"]
    score_run(conn, run_id)
    after = scores_of(conn, run_id)["position_mae"]["baseline_value"]
    assert conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == n_before   # replaced, not added
    assert after != before


def test_run_without_a_result_is_skipped(conn, weekend):
    run_id = predict_stage(conn, weekend["event"], "post_quali", n_sims=N_SIMS)
    conn.execute("DELETE FROM results WHERE session_id = ?", (weekend["R"],))
    assert score_runs(conn) == {}
    assert conn.execute("SELECT COUNT(*) FROM scores WHERE run_id = ?", (run_id,)).fetchone()[0] == 0
