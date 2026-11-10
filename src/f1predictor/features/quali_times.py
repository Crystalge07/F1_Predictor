"""Qualifying lap times for a live post-qualifying prediction.

The model's quali_gap_pct feature ("% slower than pole") is trained on
Jolpica's official Q1/Q2/Q3 times. Live, Jolpica can publish qualifying in
two steps: first the ORDER with no times at all, and the times a day or two
later. That happened at 2026 R20 Brazil: the after-qualifying prediction was
logged with Jolpica's order but 22 of 22 times missing, so the gap was blank
for every driver (release backups f1-Sun / f1-Mon; times arrived ~49h later).

OpenF1's qualifying laps are in by then (they're fetched ~30 minutes after
the session), and each driver's best OpenF1 lap IS their official time once
two kinds of laps are removed:

    deleted laps  a lap time deleted by the stewards (track limits, yellow
                  flags) is still in OpenF1's lap list. Race control says
                  "CAR 23 (ALB) TIME 1:35.457 DELETED - ...", so a lap with
                  that car and exactly that time is dropped.
    glitches      a timing glitch far faster than the field (the same rule
                  as practice: practice_pace._drop_glitches).

Checked on every qualifying session in the database with both sources
(2023-2026, 89 sessions, 2026-11-10): the best OpenF1 lap equals Jolpica's
best time exactly for 1,772 of 1,778 drivers. Of the other 6, one is a 32 s
glitch (2023, dropped by the glitch rule) and the rest are a few laps
deleted with no time in the message.

When the fallback is used: ONLY when Jolpica has the session's qualifying
rows but no time for anyone (the "order first, times later" case). If any
driver has a Jolpica time, Jolpica is complete for this purpose: a driver
with no time there really set none (crashed, didn't run), and training saw
him the same way, so he must not be given an OpenF1 lap instead.
"""

import re
import sqlite3

import pandas as pd

from f1predictor.features.practice_pace import _drop_glitches

# "CAR 23 (ALB) TIME 1:35.457 DELETED - DOUBLE YELLOW AT TURN 12"
# -> car 23, 95457 ms. Messages that delete a lap without giving its time
# ("CAR 77 (BOT) LAP DELETED - ...") can't be matched to a lap, so they're
# not used (rare: see the check in the module docstring).
DELETED_TIME = re.compile(r"CAR (\d+)\b.*?\bTIME (?:(\d+):)?(\d+\.\d+) DELETED")

# Where the times came from, for the tick report and alerts.
SOURCE_JOLPICA = "jolpica"
SOURCE_OPENF1 = "openf1"     # Jolpica had the order but no times
SOURCE_NONE = "none"         # no times anywhere: quali_gap_pct will be blank


def deleted_lap_times(conn: sqlite3.Connection, session_id: int) -> set[tuple[int, int]]:
    """{(car number, lap time in ms)} of laps race control deleted in this session."""
    deleted = set()
    for (message,) in conn.execute(
            "SELECT message FROM race_control WHERE session_id = ? AND message LIKE '%DELETED%'",
            (session_id,)):
        match = DELETED_TIME.search(message)
        if match:
            car, minutes, seconds = match.groups()
            # round(): 35.457 * 1000 is 35456.999... in floating point.
            deleted.add((int(car), int(minutes or 0) * 60_000 + round(float(seconds) * 1000)))
    return deleted


def openf1_best_laps(conn: sqlite3.Connection, session_id: int) -> pd.DataFrame:
    """driver_id, openf1_best_ms: each matched driver's best valid OpenF1 lap in
    this (qualifying) session. Empty if OpenF1 has no laps for it."""
    laps = pd.read_sql_query(
        "SELECT l.session_id, l.car_number, se.driver_id, l.lap_time_ms "
        "FROM laps l JOIN session_entries se "
        "  ON se.session_id = l.session_id AND se.car_number = l.car_number "
        "WHERE l.session_id = ? AND l.lap_time_ms IS NOT NULL AND se.driver_id IS NOT NULL",
        conn, params=(session_id,))
    if laps.empty:
        return pd.DataFrame(columns=["driver_id", "openf1_best_ms"])
    deleted = deleted_lap_times(conn, session_id)
    if deleted:
        is_deleted = [(car, ms) in deleted for car, ms in zip(laps["car_number"], laps["lap_time_ms"])]
        laps = laps[~pd.Series(is_deleted, index=laps.index)]
    laps = _drop_glitches(laps)
    return (laps.groupby("driver_id", as_index=False)["lap_time_ms"].min()
                .rename(columns={"lap_time_ms": "openf1_best_ms"}))


def qualifying_times(conn: sqlite3.Connection, session_id: int) -> tuple[pd.DataFrame, str]:
    """The qualifying rows for a prediction, and where the times came from.

    Returns (rows, source): rows has driver_id, quali_position, q1_ms, q2_ms,
    q3_ms (Jolpica, as stored) and openf1_best_ms, which is filled only when
    the fallback is used (see the module docstring), else empty.
    race_features._add_same_weekend_features uses openf1_best_ms only where a
    driver has no Q1/Q2/Q3 time.
    """
    rows = pd.read_sql_query(
        "SELECT driver_id, position AS quali_position, q1_ms, q2_ms, q3_ms "
        "FROM qualifying WHERE session_id = ?", conn, params=(session_id,))
    rows["openf1_best_ms"] = float("nan")
    if rows.empty:
        return rows, SOURCE_NONE
    if rows[["q1_ms", "q2_ms", "q3_ms"]].notna().any().any():
        return rows, SOURCE_JOLPICA
    # Jolpica: the order, but no times for anyone. Use OpenF1's.
    best = openf1_best_laps(conn, session_id)
    if best.empty:
        return rows, SOURCE_NONE
    rows = rows.drop(columns="openf1_best_ms").merge(best, on="driver_id", how="left")
    return rows, SOURCE_OPENF1


def jolpica_times_in(conn: sqlite3.Connection, event_id: int) -> bool:
    """True if Jolpica's qualifying for this event has at least one lap time."""
    return conn.execute(
        "SELECT EXISTS (SELECT 1 FROM qualifying q JOIN sessions s ON s.session_id = q.session_id "
        "WHERE s.event_id = ? AND s.session_type = 'Q' "
        "AND COALESCE(q.q1_ms, q.q2_ms, q.q3_ms) IS NOT NULL)", (event_id,)).fetchone()[0] == 1
