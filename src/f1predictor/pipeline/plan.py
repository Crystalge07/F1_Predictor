"""The pipeline's planner: given the time now, what needs doing?

The scheduler (Phase 5) doesn't run "the FP2 job" or "the race job". It runs
ONE job, the tick, every hour or so, and the tick asks this module what is
due right now. That makes the pipeline:

    * idempotent  running a tick twice does the work once: anything already
                  done (data fetched, run logged, race scored) isn't planned again;
    * self-healing a tick that fails or never fires is simply caught up by the
                  next one, as long as the stage's cutoff hasn't passed;
    * calendar-driven: nothing is hard-coded per weekend. Sprint weekends,
                  odd start times and cancelled sessions all come from the
                  sessions table (PRD edge case: "read it from the calendar").

plan(conn, now) only READS the database and returns a list of Actions. It
never fetches or writes anything, so it's easy to test with a fake clock.

The actions, in the order a weekend produces them:

    fetch_session   OpenF1 data for a session that has ended (+ FETCH_DELAY),
                    or (target 'grid') the starting grid once qualifying is in
    fetch_jolpica   Jolpica results/qualifying, once qualifying or the race
                    has ended and its rows aren't in yet; or the season's
                    schedule, if the current year has no events yet
    predict         a live prediction for a stage, once its inputs are in and
                    before its cutoff (features/as_of.stage_cutoff)
    score           a race's logged runs, once the result is in
    late            informational: Jolpica data still missing JOLPICA_LATE
                    after it was due (still retried).
    gave_up         informational: a session's data still isn't in GIVE_UP
                    after it was ready; it's no longer retried.
    missed          informational: a stage's cutoff passed with no live run,
                    after live predicting started (the first live run).
                    Nothing is done about it: a "live" prediction made after
                    its cutoff wouldn't be honest (docs/evaluation-protocol.md).

When is each stage due? (all times UTC, from the calendar)

    pre_weekend  the previous race's result is in (round 1: within
                 PRE_SEASON_WINDOW of FP1), and the first session hasn't started.
    practice     every session before qualifying has been fetched, OR it's
                 within LAST_CALL (2h) of qualifying (predict with whatever practice
                 data exists: a cancelled or broken session mustn't block it).
    post_quali   qualifying results are in, AND either OpenF1's starting grid
                 is in or it's within LAST_CALL of the race (then the predictor
                 falls back to qualifying order, see features/as_of.py).
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from f1predictor.features.as_of import PRE_QUALI_SESSION_TYPES

# How long each kind of session usually lasts. The data can't be fetched
# before the session ends; red flags can make a session run long, which the
# FETCH_DELAY margin (and simply retrying on the next tick) absorbs.
SESSION_LENGTH = {
    "FP1": timedelta(hours=1), "FP2": timedelta(hours=1), "FP3": timedelta(hours=1),
    "SQ": timedelta(minutes=45), "S": timedelta(hours=1),
    "Q": timedelta(hours=1), "R": timedelta(hours=2),
}
# Extra wait after a session's expected end before fetching: OpenF1 needs a
# little time to finish publishing a session.
FETCH_DELAY = timedelta(minutes=30)
# How close to a cutoff a stage is predicted with whatever data exists.
# Must be comfortably longer than the gap between ticks: the scheduler ticks
# hourly and GitHub Actions can start a scheduled job 10-30 minutes late, so
# 2 hours gives at least two ticks a chance. (45 minutes looked fine, but the
# simulated-weekend test showed an on-the-hour tick could never land in it.)
LAST_CALL = timedelta(hours=2)
# Jolpica results usually appear within hours, occasionally days. Past this,
# the planner keeps retrying but also reports the part as 'late' (an alert).
JOLPICA_LATE = timedelta(hours=48)
# How long to keep retrying a session's OpenF1 data after it should have
# been ready. Practice stages don't wait for it anyway (LAST_CALL); this just
# stops a session OpenF1 never publishes from being retried for days.
GIVE_UP = timedelta(hours=24)
# Round 1 has no previous race this season: predict it within this long of FP1.
PRE_SEASON_WINDOW = timedelta(days=14)
# Which events the planner looks at: those whose race is within this window
# of now. Wide enough to catch up after a few missed ticks, narrow enough that
# old seasons are never re-planned.
LOOK_BACK = timedelta(days=10)
LOOK_AHEAD = timedelta(days=14)


@dataclass(frozen=True)
class Action:
    """One thing the tick should do. frozen=True: actions can't be changed after creation."""
    kind: str                    # 'fetch_session', 'fetch_jolpica', 'predict', 'score', 'late', 'gave_up', 'missed'
    event_id: int
    season: int
    round: int
    target: str                  # session type ('FP2'), stage ('practice'), Jolpica part, or 'race'
    reason: str                  # a human-readable "why now", for the log and the alert

    def __str__(self) -> str:
        return f"{self.kind:<13} {self.season} R{self.round:>2} {self.target:<11} {self.reason}"


# -----------------------------------------------------------------------------
# Time helpers. The database stores UTC times as ISO text ending in 'Z'
# ('2026-11-06T15:30:00Z'); Python 3.10's fromisoformat doesn't accept the
# 'Z', so it's swapped for the equivalent '+00:00'.
# -----------------------------------------------------------------------------

def parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def to_utc_text(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -----------------------------------------------------------------------------
# Facts about one event, read from the database
# -----------------------------------------------------------------------------

def _events_near(conn: sqlite3.Connection, now: datetime) -> list[sqlite3.Row]:
    """Events whose race starts within [now - LOOK_BACK, now + LOOK_AHEAD]."""
    return conn.execute(
        """
        SELECT e.event_id, e.season, e.round, s.start_time_utc AS race_start
        FROM events e
        JOIN sessions s ON s.event_id = e.event_id AND s.session_type = 'R'
        WHERE s.start_time_utc BETWEEN ? AND ?
        ORDER BY s.start_time_utc
        """, (to_utc_text(now - LOOK_BACK), to_utc_text(now + LOOK_AHEAD))).fetchall()


def _sessions(conn: sqlite3.Connection, event_id: int) -> dict[str, datetime]:
    """{session_type: start time} for one event (sessions without a time are left out)."""
    return {r["session_type"]: parse_utc(r["start_time_utc"])
            for r in conn.execute("SELECT session_type, start_time_utc FROM sessions "
                                  "WHERE event_id = ? AND start_time_utc IS NOT NULL", (event_id,))}


def _openf1_done(conn: sqlite3.Connection, season: int, rnd: int, item: str) -> bool:
    """True if OpenF1 item '2026-20-FP2' (or '2026-20-grid') was fetched OK.
    Same keys as ingest/openf1.py writes."""
    row = conn.execute("SELECT status FROM ingestion_log WHERE source = 'openf1' AND item_key = ?",
                       (f"{season}-{rnd:02d}-{item}",)).fetchone()
    return row is not None and row["status"] == "ok"


def _has_rows(conn: sqlite3.Connection, table: str, event_id: int, session_type: str) -> bool:
    """True if `table` (results or qualifying) has rows for this event's session."""
    # The table name is one of two fixed strings chosen in this module, never input.
    return conn.execute(
        f"SELECT EXISTS (SELECT 1 FROM {table} t JOIN sessions s ON s.session_id = t.session_id "
        "WHERE s.event_id = ? AND s.session_type = ?)", (event_id, session_type)).fetchone()[0] == 1


def _live_run_exists(conn: sqlite3.Connection, event_id: int, stage: str) -> bool:
    return conn.execute("SELECT EXISTS (SELECT 1 FROM prediction_runs WHERE event_id = ? "
                        "AND stage = ? AND mode = 'live')", (event_id, stage)).fetchone()[0] == 1


def _previous_race_has_result(conn: sqlite3.Connection, season: int, rnd: int) -> bool | None:
    """Whether the previous round's race result is in. None if this is round 1."""
    if rnd == 1:
        return None
    return conn.execute(
        """
        SELECT EXISTS (SELECT 1 FROM results r
                       JOIN sessions s ON s.session_id = r.session_id AND s.session_type = 'R'
                       JOIN events e   ON e.event_id = s.event_id
                       WHERE e.season = ? AND e.round = ?)
        """, (season, rnd - 1)).fetchone()[0] == 1


def _unscored_runs(conn: sqlite3.Connection, event_id: int) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM prediction_runs r WHERE r.event_id = ? "
        "AND NOT EXISTS (SELECT 1 FROM scores s WHERE s.run_id = r.run_id)", (event_id,)).fetchone()[0]


# -----------------------------------------------------------------------------
# The plan
# -----------------------------------------------------------------------------

def plan(conn: sqlite3.Connection, now: datetime) -> list[Action]:
    """Everything due at `now` (a timezone-aware datetime), in a sensible order:
    fetches first (later steps need their data), then predictions, then scoring."""
    actions: list[Action] = []
    # When live predicting started: the first live run's creation time.
    first_live = conn.execute("SELECT MIN(created_at_utc) FROM prediction_runs WHERE mode = 'live'").fetchone()[0]
    live_since = parse_utc(first_live) if first_live else None

    # A new season's calendar: nothing else can be planned without it. Once
    # it's in, Jolpica refreshes it with every results fetch, and OpenF1
    # updates exact session times on every fetch, so this only fires when
    # the current year has no events at all (e.g. the first ticks of January).
    season_now = now.year
    if not conn.execute("SELECT 1 FROM events WHERE season = ?", (season_now,)).fetchone():
        actions.append(Action("fetch_jolpica", 0, season_now, 0, "schedule",
                              f"no {season_now} calendar in the database yet"))
    for event in _events_near(conn, now):
        event_id, season, rnd = event["event_id"], event["season"], event["round"]
        starts = _sessions(conn, event_id)

        def act(kind, target, reason):
            actions.append(Action(kind, event_id, season, rnd, target, reason))

        # --- Fetching ---------------------------------------------------------
        for session_type, start in sorted(starts.items(), key=lambda kv: kv[1]):
            ready_at = start + SESSION_LENGTH.get(session_type, timedelta(hours=2)) + FETCH_DELAY
            if now < ready_at or _openf1_done(conn, season, rnd, session_type):
                continue
            if now < ready_at + GIVE_UP:
                act("fetch_session", session_type, f"{session_type} ended; OpenF1 data not in yet")
            else:
                # Tried every tick for a day: stop, and say so once per tick
                # instead of hammering the API until LOOK_BACK runs out.
                act("gave_up", session_type,
                    f"no OpenF1 data {GIVE_UP.total_seconds() / 3600:.0f}h after it was ready; not retried")
        # The starting grid: published after qualifying (penalties can change
        # it), so it's fetched on its own once qualifying is in, and retried
        # every tick until it appears or the race starts.
        if ("Q" in starts and "R" in starts and now < starts["R"]
                and _openf1_done(conn, season, rnd, "Q") and not _openf1_done(conn, season, rnd, "grid")):
            act("fetch_session", "grid", "qualifying is in; starting grid not published yet")
        # Jolpica parts: (session, table its rows land in, part name). Sprint
        # results live in `results` under the sprint session (ingest/jolpica.py).
        for session_type, table, part in [("Q", "qualifying", "qualifying"), ("S", "results", "sprint"),
                                          ("R", "results", "results")]:
            if session_type in starts:
                ready_at = starts[session_type] + SESSION_LENGTH[session_type] + FETCH_DELAY
                if now >= ready_at and not _has_rows(conn, table, event_id, session_type):
                    act("fetch_jolpica", part, f"{session_type} ended; Jolpica {part} not in yet")
                    if now >= ready_at + JOLPICA_LATE:
                        act("late", part, f"Jolpica {part} still missing "
                                          f"{JOLPICA_LATE.total_seconds() / 3600:.0f}h after {session_type}; still retrying")

        # --- Predicting -------------------------------------------------------
        # (stage, cutoff, are its inputs in?, explanation)
        first_start = min(starts.values()) if starts else None
        previous = _previous_race_has_result(conn, season, rnd)
        pre_quali = [t for t in starts if t in PRE_QUALI_SESSION_TYPES
                     and ("Q" not in starts or starts[t] < starts["Q"])]
        stages = []
        if first_start is not None:
            ready = (now >= first_start - PRE_SEASON_WINDOW) if previous is None else previous
            stages.append(("pre_weekend", first_start, ready,
                           "previous race result is in" if previous else "season opener is near"))
        if "Q" in starts:
            fetched = all(_openf1_done(conn, season, rnd, t) for t in pre_quali)
            last_call = now >= starts["Q"] - LAST_CALL
            stages.append(("practice", starts["Q"], fetched or last_call,
                           "practice sessions fetched" if fetched else "last call before qualifying"))
        if "R" in starts:
            quali_in = _has_rows(conn, "qualifying", event_id, "Q")
            grid_in = _openf1_done(conn, season, rnd, "grid")
            last_call = now >= starts["R"] - LAST_CALL
            stages.append(("post_quali", starts["R"], quali_in and (grid_in or last_call),
                           "starting grid is in" if grid_in else "last call before the race (qualifying order)"))
        for stage, cutoff, ready, why in stages:
            if _live_run_exists(conn, event_id, stage):
                continue
            if now >= cutoff:
                # Only a miss if the pipeline was already live then: races
                # before the first live run ever (the backtested past) never
                # had a live prediction to make.
                if live_since is not None and cutoff > live_since:
                    act("missed", stage, f"cutoff {to_utc_text(cutoff)} passed with no live run")
            elif ready:
                act("predict", stage, why)

        # --- Scoring ----------------------------------------------------------
        if _has_rows(conn, "results", event_id, "R"):
            unscored = _unscored_runs(conn, event_id)
            if unscored:
                act("score", "race", f"result is in; {unscored} logged run(s) not scored")

    order = {"fetch_session": 0, "fetch_jolpica": 1, "predict": 2, "score": 3, "late": 4, "gave_up": 5, "missed": 6}
    # sorted() is stable: within a kind, the calendar order above is kept.
    return sorted(actions, key=lambda a: order[a.kind])
