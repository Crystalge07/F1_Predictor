"""A real weekend, replayed: 2026 Mexico City Grand Prix (round 19), no network.

The simulated weekends (test_simulated_weekend.py, test_sprint_weekend.py)
use an invented world: 20 tidy drivers, round-number timings, perfect data.
This test runs the same hourly tick against what Jolpica and OpenF1 REALLY
returned for Mexico City 2026 (recorded by scripts/record_replay_fixture.py,
served on a fake clock by tests/replay_f1.py). It checks what an invented
world can't: real JSON, a 22-car field, real car numbers matched to Jolpica
drivers, real session times (Mexico City is UTC-6: FP1 at 12:30 local is
18:30Z), and real data passing every validation check.

The weekend, in UTC (from the recording):
    FP1  Fri Oct 30 18:30     FP2  Fri 22:00     FP3  Sat Oct 31 17:30
    Q    Sat 21:00            R    Sun Nov 1 20:00

Setup = the database as it would have been when the pipeline was switched
on, the Monday after Austin (R18): 2025 and 2026 results up to Austin from
Jolpica, and Austin's OpenF1 data (it's inside the planner's 10-day look-back,
so without it the planner would go and fetch it). Then: one tick per hour,
Monday Oct 26 to Tuesday Nov 3, and nothing else.

Not tested here: model quality. With only two seasons of results and one
weekend of practice data to learn from, these predictions aren't comparable
to the Phase 4 backtest (docs/evaluation-protocol.md). This is about the pipeline.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from f1predictor.db.connection import get_connection, init_db
from f1predictor.ingest import jolpica, openf1
from f1predictor.ingest.backfill import rematch_unmatched_entries
from f1predictor.pipeline.plan import parse_utc
from f1predictor.pipeline.tick import run_tick
from f1predictor.predict import stage

from replay_f1 import ReplayF1
from test_probability_invariants import invariant_problems, order_problems

START = datetime(2026, 10, 26, 0, 0, tzinfo=timezone.utc)    # Monday after Austin
END = datetime(2026, 11, 5, 0, 0, tzinfo=timezone.utc)       # Thursday after Mexico City: past
                                                             # its 72h re-check (R20 pre-weekend waits for it)
HOUR = timedelta(hours=1)
N_SIMS = 2_000


def utc(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def replay():
    """The replayed week, run once and shared. Returns (conn, {tick time: report})."""
    mp = pytest.MonkeyPatch()
    mp.setattr(stage, "model_version", lambda: "abc1234+settings.00000000")   # a clean commit
    try:
        conn = get_connection(":memory:")
        init_db(conn)
        world = ReplayF1(now=START)
        # The database at go-live: Jolpica history, and Austin's OpenF1 data.
        for season in (2025, 2026):
            jolpica.ingest_season(conn, world, season)
        openf1.ingest_season(conn, world, 2026, rounds={18}, now=START, settle_hours=0)
        rematch_unmatched_entries(conn)

        reports, now = {}, START
        while now <= END:
            world.now = now
            reports[now] = run_tick(conn, now=now, jolpica_client=world, openf1_client=world, n_sims=N_SIMS)
            now += HOUR
        yield conn, reports
    finally:
        mp.undo()


def r19_session_ids(conn):
    return {r["session_type"]: r["session_id"] for r in conn.execute(
        "SELECT s.session_type, s.session_id FROM sessions s JOIN events e USING (event_id) "
        "WHERE e.season = 2026 AND e.round = 19")}


def runs(conn):
    return {(r["round"], r["stage"]): r for r in conn.execute(
        "SELECT e.round, r.* FROM prediction_runs r JOIN events e USING (event_id) WHERE r.mode = 'live'")}


def logged_at(reports):
    """{'2026 R19 practice': tick time} for every live run logged."""
    out = {}
    for now, report in reports.items():
        for line in report.predicted:
            out[line.split(" -> ")[0]] = now
    return out


# -----------------------------------------------------------------------------
# The weekend runs itself
# -----------------------------------------------------------------------------

def test_every_stage_is_logged_live_when_expected(replay):
    conn, reports = replay
    assert set(runs(conn)) == {(19, "pre_weekend"), (19, "practice"), (19, "post_quali"),
                               (20, "pre_weekend")}
    # When each was logged, given the real session times and the modelled
    # publishing delays (tests/replay_f1.py):
    assert logged_at(reports) == {
        # Pre-weekend waits for the previous result's 72h re-check (race
        # start + 2h + 30 min + 72h, at the next tick): Austin's, Mexico's.
        "2026 R19 pre_weekend": utc("2026-10-28T23:00:00"),
        "2026 R19 practice": utc("2026-10-31T19:00:00"),      # FP3 fetched (ended 18:30, +30 min)
        "2026 R19 post_quali": utc("2026-11-01T00:00:00"),    # OpenF1 starting grid (Q end + 2h)
        "2026 R20 pre_weekend": utc("2026-11-04T23:00:00"),
    }
    for (rnd, stage_name), run in runs(conn).items():
        assert logged_at(reports)[f"2026 R{rnd} {stage_name}"] < parse_utc(run["as_of_utc"])


def test_the_race_is_scored_and_nothing_is_alerted(replay):
    conn, reports = replay
    scored = conn.execute("SELECT COUNT(DISTINCT s.run_id) FROM scores s JOIN prediction_runs r USING (run_id) "
                          "JOIN events e USING (event_id) WHERE e.round = 19").fetchone()[0]
    assert scored == 3
    assert reports[utc("2026-11-01T23:00:00")].scored == ["2026 R19: 3 run(s)"]
    # No alert of any kind: nothing invalid, late, given up on, blocked or missed.
    assert [m for report in reports.values() for m in report.new_alerts] == []
    assert conn.execute("SELECT COUNT(*) FROM pipeline_ticks WHERE finished_at_utc IS NULL").fetchone()[0] == 0


# -----------------------------------------------------------------------------
# Real data, handled correctly
# -----------------------------------------------------------------------------

def test_real_data_loads_and_passes_validation(replay):
    conn, _ = replay
    status = dict(conn.execute("SELECT item_key, status FROM ingestion_log WHERE item_key LIKE '2026-19-%'"))
    assert status == {**{f"2026-19-{t}": "ok" for t in ("FP1", "FP2", "FP3", "Q", "R", "grid")},
                      # The replay runs past race + 72h, so the result was re-checked
                      # twice (pipeline/plan.RESULT_RECHECKS); it hadn't changed.
                      "2026-19-results-recheck-24h": "ok", "2026-19-results-recheck-72h": "ok"}
    assert conn.execute("SELECT COUNT(*) FROM validation_issues WHERE severity = 'error'").fetchone()[0] == 0
    # A real quirk, handled: OpenF1's FP3 driver list has 21 cars, but 22
    # set laps. The missing one (Lawson, car 30) is still saved, and matched
    # from the same car in FP1/FP2 (ingest/openf1.match_entries, rule 2).
    ids = r19_session_ids(conn)
    cars_with_laps = conn.execute("SELECT COUNT(DISTINCT car_number) FROM laps WHERE session_id = ?",
                                  (ids["FP3"],)).fetchone()[0]
    assert cars_with_laps == 22
    car_30 = conn.execute("SELECT acronym, driver_id FROM session_entries WHERE session_id = ? "
                          "AND car_number = 30", (ids["FP3"],)).fetchone()
    assert tuple(car_30) == (None, "lawson")


def test_every_race_driver_is_matched_in_every_session(replay):
    # OpenF1 knows cars by number and acronym; Jolpica by driver id. Every
    # car of a race driver must be matched in every session it ran, or its
    # practice pace and its prediction row would be lost.
    conn, _ = replay
    unmatched = conn.execute(
        "SELECT s.session_type, se.car_number, se.acronym FROM session_entries se "
        "JOIN sessions s USING (session_id) JOIN events e USING (event_id) "
        "WHERE e.season = 2026 AND e.round = 19 AND se.driver_id IS NULL "
        "AND EXISTS (SELECT 1 FROM laps l WHERE l.session_id = se.session_id AND l.car_number = se.car_number)"
    ).fetchall()
    # The only cars left unmatched: FP1 outings by young drivers who have
    # never raced or qualified in F1, so Jolpica has no driver id for them.
    # They drive in FP1 only, and no lineup is taken from FP1.
    assert {r["session_type"] for r in unmatched} == {"FP1"}
    assert sorted(r["acronym"] for r in unmatched) == [
        "BRO", "CRA", "FOR", "HER", "IWA", "MIN", "SLA", "TSO", "VES"]


def test_session_times_are_stored_in_utc(replay):
    # Mexico City is UTC-6: FP1 at 12:30 local is stored as 18:30Z, and every
    # cutoff in the log is the same UTC instant.
    conn, _ = replay
    starts = dict(conn.execute(
        "SELECT s.session_type, s.start_time_utc FROM sessions s JOIN events e USING (event_id) "
        "WHERE e.season = 2026 AND e.round = 19"))
    assert starts == {"FP1": "2026-10-30T18:30:00Z", "FP2": "2026-10-30T22:00:00Z",
                      "FP3": "2026-10-31T17:30:00Z", "Q": "2026-10-31T21:00:00Z",
                      "R": "2026-11-01T20:00:00Z"}
    cutoffs = {stage_name: run["as_of_utc"] for (rnd, stage_name), run in runs(conn).items() if rnd == 19}
    assert cutoffs == {"pre_weekend": starts["FP1"], "practice": starts["Q"], "post_quali": starts["R"]}


def test_what_each_stage_knew_about_who_was_racing(replay):
    # Ryo Hirakawa made his F1 debut here, for Haas (car 50), in place of
    # Ocon. How much of that each stage could know:
    conn, _ = replay
    starters = {r[0] for r in conn.execute("SELECT driver_id FROM results WHERE session_id = ?",
                                           (r19_session_ids(conn)["R"],))}

    def field(stage_name):
        return {r[0] for r in conn.execute("SELECT driver_id FROM predictions WHERE run_id = ?",
                                           (runs(conn)[(19, stage_name)]["run_id"],))}

    # pre_weekend: last race's lineup, by design (features/as_of.py), so
    # Ocon instead of Hirakawa. (The open question: a hand-edited entry list.)
    assert starters - field("pre_weekend") == {"ryo_hirakawa"}
    assert field("pre_weekend") - starters == {"ocon"}
    # practice: the full real field. Before qualifying Jolpica doesn't know
    # the debutant, so he has no driver id, but his car (50) was still
    # driving in FP3, so he's predicted under a temporary id
    # (features/provisional.py), with Haas as his team...
    debutant = "provisional_2026_r19_car50"
    assert field("practice") == starters - {"ryo_hirakawa"} | {debutant}
    practice_run = runs(conn)[(19, "practice")]["run_id"]
    assert conn.execute("SELECT constructor_id FROM predictions WHERE run_id = ? AND driver_id = ?",
                        (practice_run, debutant)).fetchone()[0] == "haas"
    # ...his FP1-FP3 laps count (he has a practice pace feature)...
    pace = conn.execute("SELECT value FROM prediction_features WHERE run_id = ? AND driver_id = ? "
                        "AND feature = 'short_run_gap_pct'", (practice_run, debutant)).fetchone()[0]
    assert pace is not None
    # ...and after the race he's scored as Hirakawa: nobody is missing.
    missing = conn.execute("SELECT model_value FROM scores WHERE run_id = ? AND metric = 'lineup_missing'",
                           (practice_run,)).fetchone()[0]
    assert missing == 0
    # post_quali: the full real field, debutant included: matched through
    # this weekend's qualifying (ingest/openf1.match_entries, rule 3).
    assert field("post_quali") == starters and len(starters) == 22


def test_post_quali_used_the_real_starting_grid_and_the_real_field(replay):
    conn, _ = replay
    run = runs(conn)[(19, "post_quali")]
    assert run["lineup_source"] == "starting_grid"
    predicted = dict(conn.execute("SELECT driver_id, anchor_position FROM predictions WHERE run_id = ?",
                                  (run["run_id"],)))
    # The anchor of every driver is their slot on OpenF1's real starting grid...
    grid = dict(conn.execute(
        "SELECT se.driver_id, g.position FROM starting_grid g JOIN session_entries se "
        "ON se.session_id = g.session_id AND se.car_number = g.car_number WHERE g.session_id = ?",
        (r19_session_ids(conn)["Q"],)))
    assert predicted == {d: float(p) for d, p in grid.items()}
    # ...and the predicted field is exactly the 22 drivers in the real result.
    starters = {r[0] for r in conn.execute("SELECT driver_id FROM results WHERE session_id = ?",
                                           (r19_session_ids(conn)["R"],))}
    assert set(predicted) == starters and len(starters) == 22
    lineup_missing = conn.execute("SELECT model_value FROM scores WHERE run_id = ? AND metric = 'lineup_missing'",
                                  (run["run_id"],)).fetchone()[0]
    assert lineup_missing == 0


def test_logged_runs_obey_probability_invariants(replay):
    # The probability rules from test_probability_invariants.py (win chances
    # add up to 1, podium to 3, points to 10, ...), checked on the three runs
    # the hourly tick actually wrote to the log: real laps, all 22 cars, and
    # the debutant predicted under a temporary id. This is the only place the
    # PRACTICE stage gets these checks (the small fake weekend has no laps).
    conn, _ = replay
    problems = {}
    for (rnd, stage_name), run in runs(conn).items():
        rows = pd.read_sql_query("SELECT * FROM predictions WHERE run_id = ?", conn,
                                 params=(run["run_id"],))
        assert len(rows) == 22, f"R{rnd} {stage_name}: {len(rows)} drivers logged"
        # These runs are made by the current code, so the order rule applies too.
        found = invariant_problems(rows) + order_problems(rows)
        if found:
            problems[f"R{rnd} {stage_name}"] = found
    assert problems == {}
