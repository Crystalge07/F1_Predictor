"""Debutants: a driver with no Jolpica id yet, predicted under a temporary id.

The rule (features/provisional.py): a car that can't be matched to a driver
but is still driving in the last session before qualifying is a race driver
we can't name yet. The practice stage predicts him as
provisional_<season>_r<round>_car<number>; after the race the scorer turns
that back into his real id. If it can't, the run is NOT scored and the tick
alerts, rather than scoring it wrongly or silently leaving it out.

Most tests run the real pipeline, hour by hour, over the fake world's
rounds 1 and 2 (tests/test_simulated_weekend.simulate), with new drivers
put into round 2's seats (tests/fake_f1.py, `debutants`). Four worlds:

    TWO        two debutants at the same weekend, all sessions
    STAND_IN   a debutant who drives practice only; the regular driver is
               back for qualifying and the race. His temporary id never
               matches anyone: the alert case.
    NO_TEAM    a debutant whose team OpenF1 doesn't give
    FP1_ONLY   a young driver's FP1 outing: not in the race, must NOT be predicted

The last tests check the identification step directly, on hand-built data.
"""

import pytest

from f1predictor.db.connection import get_connection, init_db
from f1predictor.evaluate.scorer import UnresolvedDriverError, real_driver_ids
from f1predictor.features.provisional import provisional_id, resolve
from f1predictor.predict import stage

from test_simulated_weekend import simulate

import pandas as pd

N50, N51 = "provisional_2027_r02_car50", "provisional_2027_r02_car51"
WORLDS = {
    # d05 is team t03's second seat, d12 team t06's second seat.
    "TWO": {(2027, 2): {"d05": {"id": "n50", "number": 50}, "d12": {"id": "n51", "number": 51}}},
    "STAND_IN": {(2027, 2): {"d05": {"id": "n50", "number": 50, "sessions": {"FP1", "FP2", "FP3"}}}},
    "NO_TEAM": {(2027, 2): {"d05": {"id": "n50", "number": 50, "team_name": False}}},
    "FP1_ONLY": {(2027, 2): {"d05": {"id": "n50", "number": 50, "sessions": {"FP1"}}}},
}


@pytest.fixture(scope="module")
def world():
    """{name: (conn, reports)}: each world simulated once, when first asked for."""
    mp = pytest.MonkeyPatch()
    mp.setattr(stage, "model_version", lambda: "abc1234+settings.00000000")   # a clean commit
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = simulate(debutants=WORLDS[name])
        return cache[name]
    try:
        yield get
    finally:
        mp.undo()


def run_id(conn, stage_name, rnd=2):
    return conn.execute("SELECT r.run_id FROM prediction_runs r JOIN events e USING (event_id) "
                        "WHERE e.round = ? AND r.stage = ? AND r.mode = 'live'", (rnd, stage_name)).fetchone()[0]


def field(conn, stage_name, rnd=2):
    return dict(conn.execute("SELECT driver_id, constructor_id FROM predictions WHERE run_id = ?",
                             (run_id(conn, stage_name, rnd),)).fetchall())


def scores(conn, stage_name):
    return dict(conn.execute("SELECT metric, model_value FROM scores WHERE run_id = ?",
                             (run_id(conn, stage_name),)).fetchall())


def all_alerts(reports):
    return [m for report in reports for m in report.new_alerts]


REGULARS = {f"d{i:02d}" for i in range(1, 21)}


# -----------------------------------------------------------------------------
# TWO: two debutants at the same weekend
# -----------------------------------------------------------------------------

def test_two_debutants_are_both_in_the_practice_prediction_each_with_his_own_team(world):
    conn, _ = world("TWO")
    practice = field(conn, "practice")
    # The full real field: 18 regulars, plus the two new drivers in place
    # of d05 and d12, each under his own temporary id (car numbers differ,
    # so the ids do), each in his seat's team (from his teammate's OpenF1
    # team name: on practice day nothing is matched exactly yet).
    assert set(practice) == REGULARS - {"d05", "d12"} | {N50, N51}
    assert (practice[N50], practice[N51]) == ("t03", "t06")


def test_their_practice_laps_count(world):
    conn, _ = world("TWO")
    # The practice stage's point is practice pace: a debutant with no pace
    # would be predicted blind. Both have the feature.
    pace = dict(conn.execute(
        "SELECT driver_id, value FROM prediction_features WHERE run_id = ? AND feature = 'short_run_gap_pct'",
        (run_id(conn, "practice"),)).fetchall())
    assert pace[N50] is not None and pace[N51] is not None


def test_placeholder_driver_rows_are_named_and_never_matchable(world):
    conn, _ = world("TWO")
    rows = {r["driver_id"]: r for r in conn.execute(
        "SELECT driver_id, code, given_name, family_name FROM drivers WHERE driver_id LIKE 'provisional_%'")}
    assert set(rows) == {N50, N51}
    # Named from OpenF1, for the dashboard...
    assert (rows[N50]["given_name"], rows[N50]["family_name"]) == ("Rookie", "N50")
    # ...and no 3-letter code: the matcher finds drivers by code, so a
    # placeholder can never be mistaken for a real driver later.
    assert rows[N50]["code"] is None and rows[N51]["code"] is None
    # Nothing in OpenF1's data was matched to a placeholder.
    assert conn.execute("SELECT COUNT(*) FROM session_entries WHERE driver_id LIKE 'provisional_%'"
                        ).fetchone()[0] == 0


def test_after_qualifying_they_are_predicted_under_their_real_ids(world):
    conn, _ = world("TWO")
    # Their first qualifying gives them Jolpica ids, so from post-quali on
    # the temporary ids are gone.
    assert set(field(conn, "post_quali")) == REGULARS - {"d05", "d12"} | {"n50", "n51"}
    assert set(field(conn, "pre_weekend", rnd=3)) == REGULARS - {"d05", "d12"} | {"n50", "n51"}


def test_both_are_scored_as_themselves_and_nothing_is_alerted(world):
    conn, reports = world("TWO")
    practice = scores(conn, "practice")
    # Both temporary ids were turned into n50 / n51 (who started in cars
    # 50 / 51), so the prediction had the whole field: nobody missing, nobody extra.
    assert practice["lineup_missing"] == 0 and practice["lineup_extra"] == 0
    # (The pre-weekend stage, by design, still had d05 and d12: last race's lineup.)
    pre = scores(conn, "pre_weekend")
    assert pre["lineup_missing"] == 2 and pre["lineup_extra"] == 2
    assert all_alerts(reports) == []


# -----------------------------------------------------------------------------
# STAND_IN: a temporary id that never matches anyone
# -----------------------------------------------------------------------------

def test_a_practice_stand_in_is_predicted_at_practice(world):
    conn, _ = world("STAND_IN")
    # He drove FP3 in d05's seat, so as far as anyone could tell on Saturday
    # morning, he was racing. (d05 wasn't in FP3, so he's out of this stage.)
    assert set(field(conn, "practice")) == REGULARS - {"d05"} | {N50}


def test_an_unmatched_temporary_id_leaves_the_run_unscored_and_alerts_once(world):
    conn, reports = world("STAND_IN")
    # d05 was back for qualifying and the race, so nobody started in car 50:
    # the temporary id can't be turned into anyone. The practice run is
    # NOT scored (not scored as if he were someone, and not quietly left out)...
    assert scores(conn, "practice") == {}
    # ...the race's other runs are scored as usual...
    assert scores(conn, "pre_weekend") and scores(conn, "post_quali")
    # ...and it's alerted, once (the tick tries again every hour), naming
    # the race, the run and the car.
    alerts = all_alerts(reports)
    assert len(alerts) == 1
    assert alerts[0].startswith("unresolved_driver: 2027 R2: run ")
    assert N50 in alerts[0] and "NOT scored" in alerts[0]
    assert conn.execute("SELECT alert_key FROM pipeline_alerts").fetchall()[0][0] == \
        f"unresolved_driver:2027:2:{N50}"


def test_the_unscored_run_is_scored_once_the_car_is_matched(world):
    conn, _ = world("STAND_IN")
    # The fix the alert asks for: say who drove car 50. (Here: his real id
    # turns up, e.g. Jolpica lists him. It must be someone the run didn't
    # already predict, or he'd be counted twice: see the last tests.)
    # Then scoring finds him (provisional.resolve, rule 2) and the run is
    # scored, as made. On a copy, so the other tests keep the unfixed one.
    fixed = get_connection(":memory:")
    conn.backup(fixed)
    fixed.execute("INSERT INTO drivers (driver_id, given_name, family_name) VALUES ('n50', 'Test', 'N50')")
    fixed.execute("UPDATE session_entries SET driver_id = 'n50' WHERE car_number = 50 AND driver_id IS NULL")
    from f1predictor.evaluate.scorer import score_runs
    unresolved = {}
    assert list(score_runs(fixed, [run_id(fixed, "practice")], unresolved)) == [run_id(fixed, "practice")]
    assert unresolved == {}


# -----------------------------------------------------------------------------
# NO_TEAM and FP1_ONLY: when the rule must NOT add someone
# -----------------------------------------------------------------------------

def test_a_debutant_with_no_known_team_is_left_out_not_blocking_the_prediction(world):
    conn, reports = world("NO_TEAM")
    # Without a team his DNF chance can't be computed, and a prediction with
    # a missing probability is blocked. Rather than lose the whole practice
    # prediction, he's left out (as before this rule existed): 19 drivers.
    assert set(field(conn, "practice")) == REGULARS - {"d05"}
    assert scores(conn, "practice")["lineup_missing"] == 1
    # After qualifying he has his real id and team (from Jolpica).
    assert field(conn, "post_quali")["n50"] == "t03"
    assert all_alerts(reports) == []


def test_an_fp1_only_driver_is_not_predicted(world):
    conn, reports = world("FP1_ONLY")
    # A young driver's FP1 outing: he's gone by FP3, so he's not in the race.
    assert set(field(conn, "practice")) == REGULARS
    assert conn.execute("SELECT COUNT(*) FROM drivers WHERE driver_id LIKE 'provisional_%'").fetchone()[0] == 0
    assert scores(conn, "practice")["lineup_missing"] == 0
    assert all_alerts(reports) == []


# -----------------------------------------------------------------------------
# Identifying a temporary id after the race (hand-built data)
# -----------------------------------------------------------------------------

@pytest.fixture
def weekend():
    """One event (2027 R2) with FP3, Q and R sessions, and drivers x, y, z."""
    conn = get_connection(":memory:")
    init_db(conn)
    conn.execute("INSERT INTO circuits (circuit_id, name) VALUES ('c', 'C')")
    conn.execute("INSERT INTO constructors (constructor_id, name) VALUES ('t', 'T')")
    for d in "xyz":
        conn.execute("INSERT INTO drivers (driver_id, given_name, family_name) VALUES (?, 'A', 'B')", (d,))
    event = conn.execute("INSERT INTO events (season, round, name, circuit_id, race_date) "
                         "VALUES (2027, 2, 'GP', 'c', '2027-03-14')").lastrowid
    ids = {t: conn.execute("INSERT INTO sessions (event_id, session_type) VALUES (?, ?)", (event, t)).lastrowid
           for t in ("FP3", "Q", "R")}
    return conn, event, ids


def add_result(conn, ids, driver, car, position):
    conn.execute("INSERT INTO results (session_id, driver_id, constructor_id, car_number, position, "
                 "position_text) VALUES (?, ?, 't', ?, ?, ?)", (ids["R"], driver, car, position, str(position)))


def add_entry(conn, session_id, car, driver):
    conn.execute("INSERT INTO session_entries (session_id, car_number, driver_id) VALUES (?, ?, ?)",
                 (session_id, car, driver))


def test_resolve_uses_the_race_result_first_then_other_sessions(weekend):
    conn, event, ids = weekend
    add_result(conn, ids, "x", 50, 1)                 # car 50 started the race: x
    add_entry(conn, ids["Q"], 51, "y")                # car 51 only matched in qualifying: y
    found = resolve(conn, event, [provisional_id(2027, 2, 50), provisional_id(2027, 2, 51),
                                  provisional_id(2027, 2, 52), "z"])
    # Real ids are left alone; car 52 is nowhere: None (-> alert).
    assert found == {provisional_id(2027, 2, 50): "x", provisional_id(2027, 2, 51): "y",
                     provisional_id(2027, 2, 52): None}


def test_a_car_that_points_to_two_drivers_is_not_guessed(weekend):
    # A mid-weekend swap with no race result for the car: two candidates.
    conn, event, ids = weekend
    add_entry(conn, ids["FP3"], 50, "x")
    add_entry(conn, ids["Q"], 50, "y")
    assert resolve(conn, event, [provisional_id(2027, 2, 50)]) == {provisional_id(2027, 2, 50): None}


def logged(*driver_ids):
    return pd.DataFrame({"driver_id": list(driver_ids)})


def test_scoring_refuses_a_temporary_id_that_would_duplicate_a_driver(weekend):
    conn, event, ids = weekend
    add_result(conn, ids, "x", 50, 1)
    add_result(conn, ids, "y", 2, 2)
    # Fine: car 50 -> x, and x isn't otherwise in the prediction.
    out = real_driver_ids(conn, 1, event, logged(provisional_id(2027, 2, 50), "y"))
    assert list(out["driver_id"]) == ["x", "y"]
    # Car 50 -> x, but the run ALSO predicted x under his own id: one
    # person twice. Refused, not merged.
    with pytest.raises(UnresolvedDriverError) as error:
        real_driver_ids(conn, 1, event, logged(provisional_id(2027, 2, 50), "x"))
    assert error.value.driver_ids == [provisional_id(2027, 2, 50)]


def test_scoring_refuses_two_temporary_ids_that_turn_into_the_same_driver(weekend):
    conn, event, ids = weekend
    add_result(conn, ids, "x", 50, 1)
    add_entry(conn, ids["Q"], 51, "x")     # (a data mistake: car 51 matched to x too)
    with pytest.raises(UnresolvedDriverError) as error:
        real_driver_ids(conn, 1, event, logged(provisional_id(2027, 2, 50), provisional_id(2027, 2, 51)))
    assert error.value.driver_ids == sorted([provisional_id(2027, 2, 50), provisional_id(2027, 2, 51)])
