"""Practice pace (features/practice_pace.py) on awkward sessions.

The two features:
    long_run_gap_pct   per session: each driver's median long-run lap vs the
                       best driver's, in %; then averaged over the weekend's
                       sessions. Compared WITHIN a session.
    short_run_gap_pct  each driver's single best practice lap of the weekend vs
                       the fastest anyone set, in %. Compared ACROSS sessions.

Passing tests (behaviour that is right):
    * a wet session doesn't distort long runs (everyone in it is slowed alike)
    * wet laps don't matter for the best lap if the driver also ran dry
Documented weakness (passing, describes what happens today):
    * a session where only ONE driver did a long run makes that driver the
      reference, so they score 0% (the "fastest") for it. Real data: 5 of 225
      sessions, 18 with three or fewer long-runners.

FIXED (were BUGS 3-5), all found in the real data. features/practice_pace.py
now drops implausible laps before computing either feature:
    3. one glitched FAST lap used to become the weekend's reference and shift
       every other driver's short-run gap. 2026 R4 Miami FP1: Piastri 60.4 s
       when the next best was 89.3 s -> 21 drivers at ~51-54%. Now: a lap
       more than 7% faster than every other driver's best AND than the
       driver's own next-best lap is dropped (exactly 1 of 100,074 real laps:
       this one).
    4. a driver whose only timed lap wasn't a flying lap got an absurd gap.
       2023 R12 FP1 (sprint weekend, so FP1 is the only practice):
       Verstappen's one lap took 1,631 s (red flag) -> 1,224%. Now: a best
       lap outside 107% of the weekend's typical best gives NaN (unknown).
    5. a stint of mostly cool-down laps counted as a "long run": the filter
       was 105% of the stint's MEDIAN, and when most laps are slow the median
       is slow. 2026 R17 (Singapore) FP1: Perez's stint (median 147 s,
       fastest 95 s) -> 52.7%. Now: 107% of the stint's FASTEST lap.
Real data after the fix (2023-2026): worst short-run gap 10.7% (was 1,224%),
worst long-run gap 14.8% (was 52.7%).
"""

import pandas as pd
import pytest

from f1predictor.features.practice_pace import _long_run_pace, _short_run_pace

RACE = 100   # the race session the laps help predict


def stint(driver: str, session: int, lap_times_s: list[float], session_type: str = "FP2",
          stint_number: int = 1, first_lap: int = 1) -> list[dict]:
    """Rows for one tyre stint, in the shape load_pre_quali_laps() returns.

    The stint's first lap (out-lap) and last lap (in-lap) are added here at
    a slow 120 s: _long_run_pace drops both, so `lap_times_s` are exactly the
    laps that count.
    """
    times = [120.0] + list(lap_times_s) + [120.0]
    last_lap = first_lap + len(times) - 1
    return [{
        "race_session_id": RACE, "session_id": session, "session_type": session_type,
        "driver_id": driver, "lap_number": first_lap + i, "lap_time_ms": t * 1000,
        "is_pit_out_lap": int(i == 0), "stint_number": stint_number,
        "stint_first_lap": first_lap, "stint_last_lap": last_lap,
    } for i, t in enumerate(times)]


def laps(*stints) -> pd.DataFrame:
    return pd.DataFrame([row for s in stints for row in s])


def long_gap(table: pd.DataFrame) -> dict[str, float]:
    return _long_run_pace(table).set_index("driver_id")["long_run_gap_pct"].to_dict()


def short_gap(table: pd.DataFrame) -> dict[str, float]:
    return _short_run_pace(table).set_index("driver_id")["short_run_gap_pct"].to_dict()


# -----------------------------------------------------------------------------
# What works
# -----------------------------------------------------------------------------

def test_wet_session_does_not_distort_long_runs():
    # FP2 dry: y is 1% slower than x. FP3 wet: everyone 10% slower, y still 1% off.
    table = laps(
        stint("x", 2, [90.0] * 6), stint("y", 2, [90.9] * 6),
        stint("x", 3, [99.0] * 6, "FP3"), stint("y", 3, [99.99] * 6, "FP3"),
    )
    gaps = long_gap(table)
    assert gaps["x"] == pytest.approx(0.0)
    assert gaps["y"] == pytest.approx(1.0)     # not pulled towards the wet session's times


def test_wet_laps_do_not_matter_when_the_driver_also_ran_dry():
    # x and y both ran wet FP1 (slow) and dry FP2. Best laps come from FP2.
    table = laps(
        stint("x", 1, [105.0] * 3, "FP1"), stint("y", 1, [101.0] * 3, "FP1"),
        stint("x", 2, [90.0] * 3), stint("y", 2, [90.9] * 3),
    )
    gaps = short_gap(table)
    assert gaps["x"] == pytest.approx(0.0)
    assert gaps["y"] == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# Documented weakness
# -----------------------------------------------------------------------------

def test_lone_long_runner_is_scored_as_the_fastest_in_that_session():
    # FP2: x and y long-run, y is 2% slower. FP3: only y does a long run (x
    # does short stints), so y is the session's reference: 0% there.
    # The weekend average for y becomes (2% + 0%) / 2 = 1%, although nothing
    # suggests y got faster. This is what the code does TODAY; if the rule
    # changes (e.g. ignore sessions with fewer than ~3 long-runners), update it.
    table = laps(
        stint("x", 2, [90.0] * 6), stint("y", 2, [91.8] * 6),
        stint("x", 3, [90.0] * 2, "FP3"), stint("y", 3, [91.8] * 6, "FP3"),
    )
    gaps = long_gap(table)
    assert gaps["y"] == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# Were bugs (fixed)
# -----------------------------------------------------------------------------

def test_one_glitched_lap_does_not_shift_everyone():
    # Was BUG 3. Real-world shape of 2026 Miami FP1: normal laps around 89-91 s, and one
    # timing glitch of 60 s for z. x and y are 1% apart and should stay so.
    table = laps(
        stint("x", 1, [89.3, 89.5, 89.6], "FP1"),
        stint("y", 1, [90.2, 90.4, 90.5], "FP1"),
        stint("z", 1, [60.4, 89.8, 90.0], "FP1"),
    )
    gaps = short_gap(table)
    # Measured against x's 89.3 s, y is 1.0% off. Before the fix both were
    # measured against the 60.4 s glitch: x ~48%, y ~49%, and the 1% between
    # them was stretched to 1.5%.
    assert gaps["y"] - gaps["x"] == pytest.approx(1.0, abs=0.05)
    assert gaps["x"] < 1.0


def test_a_lap_that_is_not_a_flying_lap_gives_no_gap():
    # Was BUG 4. Like Verstappen in 2023 Belgium FP1: one "lap" lasting 27 minutes
    # (it spans a red flag). There's no pace information in it, so the gap
    # should be unknown (NaN), or at least something a real car could do.
    # v's lap is built by hand, NOT with stint(): stint() adds 120 s out/in
    # laps, which would hand v a normal-looking best lap. In the real data
    # this was Verstappen's ONLY timed lap of the weekend's practice.
    lone_lap = [{**stint("v", 1, [], "FP1")[0], "lap_time_ms": 1_631_100}]
    table = laps(stint("x", 1, [123.2, 123.5], "FP1"), lone_lap)
    gap = short_gap(table)["v"]
    assert pd.isna(gap)                    # before the fix: 1,224%


def test_mostly_cool_down_stint_is_not_a_long_run():
    # Was BUG 5. x does a proper long run at 92 s. p alternates two push laps (95 s) with
    # cool-down laps (~147 s), like Perez's 2026 Singapore FP1 stint. Six
    # cool-down laps survived the old 105%-of-median filter, which made a
    # "long run" at ~60% off the pace. Now only the 2 push laps pass, too
    # few for a long run: p has no long-run gap at all.
    table = laps(
        stint("x", 1, [92.0] * 6, "FP1"),
        stint("p", 1, [95.0, 147.0, 146.0, 148.0, 95.5, 147.5, 146.5, 147.0], "FP1"),
    )
    assert "p" not in long_gap(table)


def test_real_dry_laps_in_a_rain_hit_session_are_not_glitches():
    # 2024 Japan FP2: rain arrived after a few cars set dry laps. h's dry
    # 95.2 s is far faster than h's later wet laps and than the session's
    # typical (wet) best, but NOT faster than another car's dry lap (94.7 s),
    # so it's real and must be kept.
    table = laps(
        stint("p", 2, [94.7, 96.9, 99.1]),
        stint("h", 2, [95.2, 104.0, 105.0]),
        stint("w1", 2, [104.5, 105.5]), stint("w2", 2, [105.0, 106.0]),
        stint("w3", 2, [106.0, 107.0]),
    )
    assert short_gap(table)["h"] == pytest.approx((95.2 / 94.7 - 1) * 100)
