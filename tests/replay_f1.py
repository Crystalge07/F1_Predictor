"""Replay recorded real API responses on a fake clock (tests/test_real_weekend_replay.py).

The recording (tests/fixtures/replay_2026_r19.json.gz, made by
scripts/record_replay_fixture.py) is what Jolpica and OpenF1 returned AFTER
the weekend: everything is in it. A live pipeline never sees it all at once,
so this server answers get_json(url, params) like the real APIs would have
at `replay.now`, releasing each piece only once it would have existed:

    OpenF1 session data   PUBLISH_DELAY after the session's real end (OpenF1's date_end)
    OpenF1 starting grid  GRID_DELAY after qualifying's real end
    Jolpica qualifying    PUBLISH_DELAY after qualifying ends
    Jolpica race/sprint   RESULTS_DELAY after the race/sprint ends
    schedule, sessions    always (published long before)

The delays are MODELLED (the same numbers as tests/fake_f1.py): the
recording can't show when each piece first appeared. Everything else (the
JSON itself, driver numbers, names, timings, session times) is real.
Anything not recorded answers 404, like OpenF1 does for "no data".
"""

import gzip
import json
from datetime import datetime, timedelta
from pathlib import Path

from fake_f1 import GRID_DELAY, PUBLISH_DELAY, RESULTS_DELAY, http_404

FIXTURE = Path(__file__).parent / "fixtures" / "replay_2026_r19.json.gz"

# How long Jolpica's sessions run (Jolpica's schedule has start times only).
JOLPICA_LENGTH = {"Qualifying": timedelta(hours=1), "Sprint": timedelta(hours=1),
                  "Race": timedelta(hours=2)}


def _utc(date: str, time: str | None) -> datetime:
    """Jolpica date + time ('2026-11-01', '20:00:00Z') -> aware UTC datetime."""
    return datetime.fromisoformat(f"{date}T{(time or '00:00:00Z').replace('Z', '+00:00')}")


class ReplayF1:
    def __init__(self, now: datetime, fixture: Path = FIXTURE):
        with gzip.open(fixture, "rt", encoding="utf-8") as f:
            self.recording = json.load(f)
        self.now = now
        self.requests = 0
        # OpenF1 sessions by key, for their real end times.
        self.sessions = {s["session_key"]: s
                         for key, rows in self.recording["openf1"].items() if key.startswith("sessions?")
                         for s in rows}

    def get_json(self, url, params=None):
        self.requests += 1
        params = params or {}
        if "jolpi" in url:
            return self._jolpica(url.split("/ergast/f1/", 1)[1].removesuffix(".json"), params)
        return self._openf1(url.rsplit("/", 1)[1], params)

    # --- Jolpica -----------------------------------------------------------------
    def _jolpica(self, path, params):
        races = self.recording["jolpica"].get(path)
        if races is None:
            raise http_404()
        part = path.split("/")[1] if "/" in path else "schedule"
        # (session that must have ended, its key in the schedule, delay, the per-race list)
        rule = {"results": ("Race", RESULTS_DELAY, "Results"),
                "sprint": ("Sprint", RESULTS_DELAY, "SprintResults"),
                "qualifying": ("Qualifying", PUBLISH_DELAY, "QualifyingResults")}.get(part)
        if rule:
            session, delay, _ = rule
            schedule = {r["round"]: r for r in self.recording["jolpica"][path.split("/")[0]]}
            races = [r for r in races if self._ended(schedule[r["round"]], session) + delay <= self.now]
        # One page with everything: fetch_races asks for offset 0, sees
        # offset + limit >= total, and stops (exactly as with a short season).
        if params.get("offset", 0) > 0:
            races = []
        list_key = rule[2] if rule else None
        rows = sum(len(r[list_key]) for r in races) if list_key else len(races)
        return {"MRData": {"total": str(rows), "RaceTable": {"Races": races}}}

    def _ended(self, race: dict, session: str) -> datetime:
        """When `session` of this race ended. `race` is the round's entry in the
        season SCHEDULE: real result responses don't repeat every session's
        time, only the race's own date."""
        if session == "Race":
            start = _utc(race["date"], race.get("time"))
        else:
            start = _utc(race[session]["date"], race[session].get("time"))
        return start + JOLPICA_LENGTH[session]

    # --- OpenF1 ------------------------------------------------------------------
    def _openf1(self, endpoint, params):
        if endpoint == "sessions":
            rows = self.recording["openf1"].get(f"sessions?year={params['year']}")
        else:
            key = params["session_key"]
            end = datetime.fromisoformat(self.sessions[key]["date_end"]) if key in self.sessions else None
            delay = GRID_DELAY if endpoint == "starting_grid" else PUBLISH_DELAY
            if end is None or self.now < end + delay:
                raise http_404()                      # not published yet
            rows = self.recording["openf1"].get(f"{endpoint}?session_key={key}")
        if not rows:
            raise http_404()                          # OpenF1's way of saying "no rows"
        return rows
