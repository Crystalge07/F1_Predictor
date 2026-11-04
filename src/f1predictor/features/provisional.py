"""Debutants: predicting a driver who has no Jolpica id yet.

The problem. OpenF1 knows cars by number; the model knows drivers by their
Jolpica id. A driver gets a Jolpica id with his first qualifying or race
result, so a DEBUTANT has none during practice. His car can't be matched
(ingest/openf1.match_entries), and the practice-stage prediction used to
leave him out: 21 drivers instead of 22 (Hirakawa, Mexico City 2026), and
every rookie at round 1.

The rule. A car that runs in the LAST session before qualifying (FP3, or
the sprint sessions) is in the race: FP1-only drivers (young drivers' FP1
outings) are gone by then and the race drivers are back (the same reasoning
as features/as_of.last_pre_quali_lineup). So an unmatched car with laps in
that session is a race driver we can't name yet: a debutant. Checked on
2023-2026 (89 weekends): no unmatched car ever ran in that session, so the
rule changes nothing in history; it only acts live, before a debutant's
first result.

The temporary id. Such a car is predicted as

    provisional_<season>_r<round>_car<number>     e.g. provisional_2026_r19_car4

* per weekend: car numbers are unique within a weekend, not across them,
  and by the next weekend the driver has his real id anyway;
* never confused with a real id (no Jolpica id starts with "provisional_");
* a placeholder row in `drivers` is written with the run (the log's
  foreign key needs one), named from OpenF1 ('Ryo HIRAKAWA'), with no
  3-letter code, so the matcher's acronym rules can never match a car to it;
* the model treats him like any driver with no history (a rookie); his
  practice laps DO count (features/practice_pace.py), and his team is his
  teammate's (the matched car with the same OpenF1 team name).

After the race, the scorer turns the temporary id back into the real one
(resolve, below) and scores the prediction as made. If it can't, the run is
NOT scored, and the tick alerts (pipeline/tick.py): a score is never quietly
dropped or quietly wrong.
"""

import sqlite3

import pandas as pd

PREFIX = "provisional_"


def provisional_id(season: int, rnd: int, car_number: int) -> str:
    return f"{PREFIX}{season}_r{rnd:02d}_car{int(car_number)}"


def is_provisional(driver_id) -> bool:
    return isinstance(driver_id, str) and driver_id.startswith(PREFIX)


def car_number_of(driver_id: str) -> int:
    """provisional_2026_r19_car4 -> 4"""
    return int(driver_id.rsplit("_car", 1)[1])


def provisional_cars(conn: sqlite3.Connection, event_id: int | None = None) -> pd.DataFrame:
    """Unmatched cars with laps in each weekend's last session before qualifying.

    One row per car: event_id, session_id (that last session), car_number,
    driver_id (the temporary id), constructor_id (from OpenF1's team name;
    may be NULL), full_name, acronym. event_id=None: every weekend.

    "Last session" = the latest FP1/FP2/FP3/SQ/S that starts before
    qualifying and has matched entries, exactly as as_of.last_pre_quali_lineup
    chooses it.
    """
    rows = pd.read_sql_query(
        f"""
        WITH pre AS (
            SELECT s.event_id, s.session_id, s.start_time_utc
            FROM sessions s
            -- No qualifying on the calendar (cancelled): up to the race instead,
            -- as features/as_of.stage_cutoff does.
            LEFT JOIN sessions q    ON q.event_id = s.event_id AND q.session_type = 'Q'
            JOIN sessions race      ON race.event_id = s.event_id AND race.session_type = 'R'
            WHERE s.session_type IN ('FP1', 'FP2', 'FP3', 'SQ', 'S')
              AND s.start_time_utc < COALESCE(q.start_time_utc, race.start_time_utc)
              AND EXISTS (SELECT 1 FROM session_entries x
                          WHERE x.session_id = s.session_id AND x.driver_id IS NOT NULL)
              {"AND s.event_id = :event_id" if event_id is not None else ""}
        ),
        last AS (
            SELECT p.event_id, p.session_id FROM pre p
            WHERE p.start_time_utc = (SELECT MAX(p2.start_time_utc) FROM pre p2
                                      WHERE p2.event_id = p.event_id)
        )
        SELECT last.event_id, last.session_id, e.season, e.round,
               se.car_number, se.full_name, se.acronym,
               -- His team: the matcher's, or else his teammate's: a matched car
               -- in the same session under the same OpenF1 team name. (On
               -- practice day nobody is matched exactly yet, so the matcher's
               -- team-name rule has nothing to learn from.)
               COALESCE(se.constructor_id,
                        (SELECT MIN(x.constructor_id) FROM session_entries x
                         WHERE x.session_id = se.session_id AND x.team_name = se.team_name
                           AND x.constructor_id IS NOT NULL)) AS constructor_id
        FROM last
        JOIN events e           ON e.event_id = last.event_id
        JOIN session_entries se ON se.session_id = last.session_id
        WHERE se.driver_id IS NULL
          -- It actually drove in that session (a name on a list isn't enough).
          AND EXISTS (SELECT 1 FROM laps l WHERE l.session_id = se.session_id
                                             AND l.car_number = se.car_number)
        ORDER BY last.event_id, se.car_number
        """, conn, params={"event_id": event_id})
    rows["driver_id"] = [provisional_id(s, r, c) for s, r, c in
                         zip(rows["season"], rows["round"], rows["car_number"])]
    return rows[["event_id", "session_id", "car_number", "driver_id", "constructor_id",
                 "full_name", "acronym"]]


def add_placeholder_drivers(conn: sqlite3.Connection, event_id: int, driver_ids) -> None:
    """A `drivers` row for each temporary id (the prediction log's foreign key
    needs one). Called inside write_run's transaction, so it's written with
    the run or not at all. Re-running is harmless (ON CONFLICT DO NOTHING)."""
    wanted = [d for d in driver_ids if is_provisional(d)]
    if not wanted:
        return
    names = {row["driver_id"]: row["full_name"] for _, row in provisional_cars(conn, event_id).iterrows()}
    for driver_id in wanted:
        # 'Ryo HIRAKAWA' -> given 'Ryo', family 'HIRAKAWA'. No name known: say which car.
        given, _, family = (names.get(driver_id) or "").partition(" ")
        conn.execute(
            "INSERT INTO drivers (driver_id, code, permanent_number, given_name, family_name) "
            "VALUES (?, NULL, NULL, ?, ?) ON CONFLICT (driver_id) DO NOTHING",
            (driver_id, given or "New driver", family or f"car {car_number_of(driver_id)}"))


def resolve(conn: sqlite3.Connection, event_id: int, driver_ids) -> dict[str, str | None]:
    """{temporary id: real driver id, or None if it can't be told} for one weekend.

    Most reliable first, the same weekend only (a car number means nothing
    across weekends):
      1. the official race result: the driver who started in that car;
      2. the car matched in any other session of the weekend (e.g. the
         qualifying session's entries, matched through Jolpica qualifying).
    Two different drivers for one car (a mid-weekend swap): None. Guessing
    would score the prediction against the wrong person.
    """
    resolved = {}
    for driver_id in [d for d in driver_ids if is_provisional(d)]:
        car = car_number_of(driver_id)
        in_result = conn.execute(
            "SELECT DISTINCT r.driver_id FROM results r JOIN sessions s ON s.session_id = r.session_id "
            "WHERE s.event_id = ? AND s.session_type = 'R' AND r.car_number = ?", (event_id, car)).fetchall()
        candidates = in_result or conn.execute(
            "SELECT DISTINCT se.driver_id FROM session_entries se JOIN sessions s ON s.session_id = se.session_id "
            "WHERE s.event_id = ? AND se.car_number = ? AND se.driver_id IS NOT NULL", (event_id, car)).fetchall()
        resolved[driver_id] = candidates[0][0] if len(candidates) == 1 else None
    return resolved
