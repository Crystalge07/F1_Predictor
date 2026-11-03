"""Validation checks: catch broken or incomplete data before (and after) it's saved.

Every check is a small function that reads the database and returns a list of
Issues (empty = all good). Each Issue has a severity:

    'error'   Data is clearly broken or incomplete (e.g. positions 1, 2, 2, 4).
              Loaders call these checks INSIDE their transaction, and raise
              ValidationError if any error is found. The transaction rolls
              back, nothing bad is saved, and the item is logged 'failed'
              so the next run retries it.
    'warning' Unusual, but possibly real (19 starters instead of 20). The data
              is saved and the warning recorded for a human to look at.

Thresholds were calibrated on real 2018 and 2024 data, so the checks don't
raise false alarms on things like pit-lane starts or retirements.

Grouped by what they check:
    check_schedule        a season's calendar           (after Jolpica schedule)
    check_classification  a race/sprint result          (after Jolpica results)
    check_qualifying      a qualifying result           (after Jolpica qualifying)
    check_openf1_session  a session's lap-level data    (after OpenF1 session)
"""

import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

# --- Thresholds (named, so they're easy to find and tune) ---

# Fewer classified drivers than this means a partial / broken result.
# Every race since 2018 has had at least 19.
MIN_DRIVERS_PER_RESULT = 15

# OpenF1 laps for a race should cover at least this share of the laps Jolpica
# says were completed. Less means OpenF1 only has part of the race.
MIN_LAP_COVERAGE = 0.9

# Allowed difference between OpenF1's lap count for a driver and Jolpica's
# laps_completed. 1, because OpenF1 also records the (untimed) lap on which a
# driver retired, while Jolpica doesn't count it.
LAP_COUNT_TOLERANCE = 1

# How many drivers may qualify and then have no race result before the result
# counts as broken. A driver can genuinely withdraw between qualifying and the
# race, and Jolpica then leaves them out of the result: in 2018-2026 that
# happened exactly once (Stroll, 2025 round 9). So one missing qualifier is a
# warning; two or more is far likelier a partial result than two withdrawals.
MAX_WITHDRAWALS = 1

# Warn if more than this share of a session's laps have no tyre stint.
MAX_LAPS_WITHOUT_STINT = 0.05

# No F1 lap has ever been anywhere near this fast (Austria, the shortest lap,
# is ~63s). Anything quicker is a timing glitch.
MIN_PLAUSIBLE_LAP_MS = 55_000


@dataclass
class Issue:
    """One problem found by a check.

    @dataclass auto-generates __init__, __repr__ and __eq__ from the fields
    below, so Issue("x", "error", "detail") just works and prints nicely.
    """
    check: str                   # short machine-friendly name
    severity: str                # 'error' or 'warning'
    detail: str                  # human-readable explanation
    session_id: int | None = None


class ValidationError(Exception):
    """Raised by loaders when error-level issues are found. Carries the issues."""

    def __init__(self, issues: list[Issue]):
        self.issues = issues
        # The exception's message lists every error, so it reads well in logs.
        super().__init__("; ".join(f"{i.check}: {i.detail}" for i in issues))


def raise_if_errors(issues: list[Issue]) -> None:
    """Raise ValidationError if any issue is an error; otherwise do nothing."""
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise ValidationError(errors)


# =============================================================================
# Jolpica checks
# =============================================================================

def check_schedule(conn: sqlite3.Connection, season: int) -> list[Issue]:
    """The season has events, numbered 1..N, each with a race session."""
    issues: list[Issue] = []
    rounds = [row["round"] for row in conn.execute(
        "SELECT round FROM events WHERE season = ? ORDER BY round", (season,))]

    if not rounds:
        return [Issue("no_events", "error", f"No events for {season}")]

    # list(range(1, n + 1)) is [1, 2, ..., n]. Gaps would suggest a missing race.
    if rounds != list(range(1, len(rounds) + 1)):
        issues.append(Issue("rounds_not_sequential", "warning",
                            f"{season} rounds are {rounds}, expected 1..{len(rounds)}"))

    missing_race = [row["round"] for row in conn.execute(
        """
        SELECT e.round FROM events e
        WHERE e.season = ?
          -- NOT EXISTS: true when the subquery finds no row at all.
          AND NOT EXISTS (SELECT 1 FROM sessions s
                          WHERE s.event_id = e.event_id AND s.session_type = 'R')
        """, (season,))]
    if missing_race:
        issues.append(Issue("event_missing_race_session", "error",
                            f"{season} rounds without a race session: {missing_race}"))
    return issues


def check_classification(conn: sqlite3.Connection, session_id: int) -> list[Issue]:
    """A race or sprint result is complete and self-consistent."""
    issues: list[Issue] = []
    rows = conn.execute(
        "SELECT position, car_number, grid FROM results WHERE session_id = ?", (session_id,)
    ).fetchall()
    n = len(rows)

    if n < MIN_DRIVERS_PER_RESULT:
        issues.append(Issue("too_few_results", "error",
                            f"Only {n} results (expected at least {MIN_DRIVERS_PER_RESULT})",
                            session_id))

    # Positions must be exactly 1, 2, ..., n: no gaps, no duplicates.
    positions = sorted(row["position"] for row in rows)
    if positions != list(range(1, n + 1)):
        issues.append(Issue("positions_not_sequential", "error",
                            f"Positions are {positions}", session_id))

    # Two drivers can't share a car number in one race.
    duplicate_cars = _duplicates(row["car_number"] for row in rows)
    if duplicate_cars:
        issues.append(Issue("duplicate_car_numbers", "error",
                            f"Car numbers used twice: {duplicate_cars}", session_id))

    # Two drivers can't share a grid slot. grid = 0 means pit-lane start, and
    # several drivers can start from the pit lane, so 0 is excluded.
    duplicate_grid = _duplicates(row["grid"] for row in rows if row["grid"])
    if duplicate_grid:
        issues.append(Issue("duplicate_grid_slots", "error",
                            f"Grid slots used twice: {duplicate_grid}", session_id))

    # Compare with the season's usual field size (20 until 2025, 22 in 2026).
    # Using the most common count adapts to new teams without hard-coding it.
    usual = _usual_result_count(conn, session_id)
    if usual is not None and n != usual:
        issues.append(Issue("unusual_entry_count", "warning",
                            f"{n} results; this season usually has {usual}", session_id))

    # Drivers who qualified at this event but are missing from this result.
    # The checks above can miss a dropped row: if the LAST finisher is the
    # one missing, positions are still 1..n with no gap. Every driver who
    # starts is in the result (retirements included), so a missing qualifier
    # is either a withdrawal (rare, see MAX_WITHDRAWALS) or a broken result.
    # Only for races: sprint qualifying isn't stored in the qualifying table.
    # Skipped when the event's qualifying isn't loaded (e.g. mid-backfill).
    missing = [row["driver_id"] for row in conn.execute(
        """
        SELECT q.driver_id FROM qualifying q
        JOIN sessions qs   ON qs.session_id = q.session_id AND qs.session_type = 'Q'
        JOIN sessions race ON race.event_id = qs.event_id AND race.session_type = 'R'
        WHERE race.session_id = ?
          AND NOT EXISTS (SELECT 1 FROM results r
                          WHERE r.session_id = race.session_id AND r.driver_id = q.driver_id)
        ORDER BY q.driver_id
        """, (session_id,))]
    if len(missing) > MAX_WITHDRAWALS:
        issues.append(Issue("qualifiers_missing_from_result", "error",
                            f"{len(missing)} drivers qualified but have no result: {missing}", session_id))
    elif missing:
        issues.append(Issue("qualifier_missing_from_result", "warning",
                            f"{missing[0]} qualified but has no result (withdrew?)", session_id))
    return issues


def check_qualifying(conn: sqlite3.Connection, session_id: int) -> list[Issue]:
    """A qualifying result is self-consistent and matches the race's field."""
    issues: list[Issue] = []
    positions = sorted(row["position"] for row in conn.execute(
        "SELECT position FROM qualifying WHERE session_id = ?", (session_id,)))
    n = len(positions)

    # Only a WARNING for qualifying (unlike races). Calibration on real data
    # found that when a driver is disqualified from qualifying (Monaco, Zandvoort
    # and Baku 2024), Jolpica keeps them at their original position while
    # moving everyone behind up, so two drivers share a position. That's
    # correct data we must not block. The race's `grid` column is what
    # reflects the disqualification, and that's what the model will use.
    if positions != list(range(1, n + 1)):
        issues.append(Issue("positions_not_sequential", "warning",
                            f"Qualifying positions are {positions} "
                            "(often a qualifying disqualification)", session_id))

    # How many drivers got a race result at the same event (if the race exists yet).
    race_count = conn.execute(
        """
        SELECT COUNT(*) FROM results r
        JOIN sessions race ON race.session_id = r.session_id AND race.session_type = 'R'
        JOIN sessions q ON q.event_id = race.event_id
        WHERE q.session_id = ?
        """, (session_id,)).fetchone()[0]
    if race_count and n != race_count:
        issues.append(Issue("quali_vs_race_count", "warning",
                            f"{n} qualifiers but {race_count} race results", session_id))
    return issues


def check_jolpica_part(conn: sqlite3.Connection, season: int, part: str) -> list[Issue]:
    """Run the right checks for one Jolpica ingest part ('schedule', 'results', ...)."""
    if part == "schedule":
        return check_schedule(conn, season)

    # Every other part checks each session of one type that has data.
    session_type, table, check = {
        "results":    ("R", "results", check_classification),
        "sprint":     ("S", "results", check_classification),
        "qualifying": ("Q", "qualifying", check_qualifying),
    }[part]
    # `table` comes from the fixed dict above, never from outside input, so
    # putting it into the SQL string with an f-string is safe here.
    session_ids = [row["session_id"] for row in conn.execute(
        f"""
        SELECT DISTINCT s.session_id FROM sessions s
        JOIN events e ON e.event_id = s.event_id
        JOIN {table} t ON t.session_id = s.session_id
        WHERE e.season = ? AND s.session_type = ?
        """, (season, session_type))]

    issues: list[Issue] = []
    for session_id in session_ids:
        issues.extend(check(conn, session_id))
    return issues


# =============================================================================
# OpenF1 checks
# =============================================================================

def check_openf1_session(conn: sqlite3.Connection, session_id: int) -> list[Issue]:
    """A session's lap-level data is complete and plausible."""
    issues: list[Issue] = []
    session_type = conn.execute(
        "SELECT session_type FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()["session_type"]

    # --- Races and sprints: compare lap counts with Jolpica's official result ---
    if session_type in ("R", "S"):
        per_driver = conn.execute(
            """
            SELECT se.acronym, r.laps_completed,
                   (SELECT COUNT(*) FROM laps l
                    WHERE l.session_id = se.session_id AND l.car_number = se.car_number) AS openf1_laps
            FROM session_entries se
            JOIN results r ON r.session_id = se.session_id AND r.driver_id = se.driver_id
            WHERE se.session_id = ?
              -- Disqualified drivers ('D') are recorded by Jolpica with 0 laps
              -- even though they drove the whole race (e.g. Hamilton, Leclerc
              -- and Gasly in China 2025), so their counts can't be compared.
              AND r.position_text <> 'D'
            """, (session_id,)).fetchall()

        if per_driver:  # only when Jolpica's result is already loaded
            expected = sum(row["laps_completed"] or 0 for row in per_driver)
            actual = sum(row["openf1_laps"] for row in per_driver)
            if expected and actual < MIN_LAP_COVERAGE * expected:
                issues.append(Issue("race_laps_incomplete", "error",
                                    f"OpenF1 has {actual} laps, official result implies {expected}",
                                    session_id))

            mismatched = [
                f"{row['acronym']} ({row['openf1_laps']} vs {row['laps_completed']})"
                for row in per_driver
                if abs(row["openf1_laps"] - (row["laps_completed"] or 0)) > LAP_COUNT_TOLERANCE
            ]
            if mismatched:
                issues.append(Issue("lap_count_mismatch", "warning",
                                    "OpenF1 vs official laps: " + ", ".join(mismatched), session_id))

            # A driver who drove (official laps completed > the retirement-lap
            # tolerance) but has NO OpenF1 laps at all. The coverage check above
            # can't see this: it only counts drivers OpenF1 has an entry for, so a
            # car missing from OpenF1 entirely drops out of both sides of the sum.
            # Matched by car number, which both sources have even when a car
            # isn't matched to a driver_id yet. Only when OpenF1 has data for
            # the session (per_driver non-empty). Calibrated on 2023-2026 (112 races
            # and sprints with OpenF1 laps): never happens in good data.
            no_laps = [row["driver_id"] for row in conn.execute(
                """
                SELECT r.driver_id FROM results r
                WHERE r.session_id = ? AND r.laps_completed > ? AND r.position_text <> 'D'
                  AND NOT EXISTS (SELECT 1 FROM laps l
                                  WHERE l.session_id = r.session_id AND l.car_number = r.car_number)
                ORDER BY r.driver_id
                """, (session_id, LAP_COUNT_TOLERANCE))]
            if no_laps:
                issues.append(Issue("driver_missing_from_laps", "error",
                                    f"Official result has laps for {no_laps}, OpenF1 has none", session_id))

    # --- Unmatched drivers: expected in practice (rookies), not in Q/S/R ---
    if session_type in ("Q", "SQ", "S", "R"):
        # Only cars that actually set laps: an unmatched car with no laps
        # can't affect any pace feature, so it isn't worth a warning.
        unmatched = [row["acronym"] or f"#{row['car_number']}" for row in conn.execute(
            """
            SELECT se.acronym, se.car_number FROM session_entries se
            WHERE se.session_id = ? AND se.driver_id IS NULL
              AND EXISTS (SELECT 1 FROM laps l
                          WHERE l.session_id = se.session_id AND l.car_number = se.car_number)
            """, (session_id,))]
        if unmatched:
            issues.append(Issue("unmatched_drivers", "warning",
                                f"Not matched to a Jolpica driver: {unmatched}", session_id))

    # --- Tyre coverage: laps with no stint have unknown compound/tyre age ---
    total_laps, laps_without_stint = conn.execute(
        "SELECT COUNT(*), SUM(compound IS NULL) FROM laps_detailed WHERE session_id = ?",
        (session_id,)).fetchone()
    if total_laps and (laps_without_stint or 0) > MAX_LAPS_WITHOUT_STINT * total_laps:
        issues.append(Issue("laps_without_stint", "warning",
                            f"{laps_without_stint} of {total_laps} laps have no tyre stint",
                            session_id))

    # --- Impossible lap times: timing glitches that would poison pace features ---
    too_fast = conn.execute(
        "SELECT COUNT(*) FROM laps WHERE session_id = ? AND lap_time_ms < ?",
        (session_id, MIN_PLAUSIBLE_LAP_MS)).fetchone()[0]
    if too_fast:
        issues.append(Issue("impossible_lap_times", "warning",
                            f"{too_fast} laps faster than {MIN_PLAUSIBLE_LAP_MS / 1000:.0f}s",
                            session_id))

    # --- Weather: every session should have some samples ---
    if not conn.execute("SELECT 1 FROM weather WHERE session_id = ? LIMIT 1",
                        (session_id,)).fetchone():
        issues.append(Issue("no_weather", "warning", "No weather samples", session_id))

    return issues


# =============================================================================
# Recording issues
# =============================================================================

def record_issues(conn: sqlite3.Connection, source: str, item_key: str, issues: list[Issue]) -> None:
    """Replace the stored issues for one item with `issues` (may be empty).

    Replacing (not appending) means validation_issues always shows the
    current state: once a problem is fixed and re-validated, it disappears.
    """
    found_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    with conn:
        conn.execute("DELETE FROM validation_issues WHERE source = ? AND item_key = ?",
                     (source, item_key))
        conn.executemany(
            "INSERT INTO validation_issues (source, item_key, check_name, severity, "
            "session_id, detail, found_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(source, item_key, i.check, i.severity, i.session_id, i.detail, found_at)
             for i in issues],
        )


def summarize(issues: list[Issue]) -> str | None:
    """'2 warnings' / '1 error, 3 warnings' / None if there are none. For log messages."""
    counts = Counter(i.severity for i in issues)
    parts = [f"{counts[s]} {s}{'s' if counts[s] != 1 else ''}"
             for s in ("error", "warning") if counts[s]]
    return ", ".join(parts) or None


# =============================================================================
# Helpers
# =============================================================================

def _duplicates(values) -> list:
    """Values that appear more than once, e.g. [1, 2, 2, 3, 3] -> [2, 3]."""
    counts = Counter(v for v in values if v is not None)
    return sorted(v for v, c in counts.items() if c > 1)


def _usual_result_count(conn: sqlite3.Connection, session_id: int) -> int | None:
    """Most common number of results per session, for this session's season and type."""
    counts = [row["n"] for row in conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM results r
        JOIN sessions s ON s.session_id = r.session_id
        JOIN events e ON e.event_id = s.event_id
        JOIN sessions this ON this.session_id = ?
        JOIN events this_e ON this_e.event_id = this.event_id
        WHERE e.season = this_e.season AND s.session_type = this.session_type
        GROUP BY s.session_id
        """, (session_id,))]
    if not counts:
        return None
    # most_common(1) -> [(value, how_often)]; [0][0] picks the value.
    return Counter(counts).most_common(1)[0][0]
