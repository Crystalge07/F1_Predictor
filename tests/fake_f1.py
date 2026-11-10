"""A fake F1 world for the simulated-weekend test: Jolpica + OpenF1 on a fake clock.

It answers get_json(url, params) like the real APIs do, in the same JSON
shapes, so the pipeline runs its REAL loaders, validation, matching,
predictions and scoring. Only the network is fake.

Data appears the way it does in real life, based on `world.now`:
    calendar, sessions list   always (published long before the season)
    OpenF1 session data       PUBLISH_DELAY after the session ends (404 before)
    OpenF1 starting grid      GRID_DELAY after qualifying ends
    Jolpica qualifying        PUBLISH_DELAY after qualifying ends
    Jolpica race results      RESULTS_DELAY after the race ends (Jolpica lags)

The field: 20 drivers ('d01'..'d20', codes 'D01'.., car numbers 1..20) in 10
teams. Lower driver number = faster, plus a small fixed shuffle per round so
results aren't identical every race. Every round's starting grid has one
penalty: the pole-sitter drops to 4th, so "grid from the starting grid" and
"grid from qualifying order" are easy to tell apart.

Knobs that make things go wrong on purpose (all keyed by (season, round, X)):

    broken       a set: OpenF1 data for session X is never published; X = 'grid'
                 means no starting grid for that round.
    cancelled    a set: session X isn't on the calendar at all (Jolpica's
                 schedule and OpenF1's session list both leave it out).
    empty_until  {key: datetime}: the data is LATE. Before that time, OpenF1
                 answers session X with an empty lap list, or (X = 'results')
                 Jolpica's results leave the round out. Tests can change it.
    sprint_rounds  a set of (season, round): those are sprint weekends
                 (SPRINT_OFFSETS), with sprint results on Jolpica's /sprint.
    tamper       {key: kind}: the data is BROKEN, until a test removes the key.
                 OpenF1 session X: 'drop_driver' (d20 missing from drivers,
                 laps and stints) or 'half_laps' (only the first half of the
                 laps). X = 'results': 'drop_two' (the last two finishers are
                 missing from Jolpica's race result). X = 'qualifying':
                 'no_times' (Jolpica's qualifying has the order but no Q1/Q2/Q3
                 times, as at 2026 R20 Brazil; the times "arrive" when a test
                 removes the key).

And one that isn't a fault, keyed by (season, round):

    debutants    {(season, round): {seat: spec}}: a NEW driver takes seat `seat`
                 (one of d01..d20: its team and its pace) for that weekend,
                 under his own id and car number. spec keys:
                     "id"        e.g. "n50" (code 'N50', a Jolpica id only
                                 from his first qualifying/race result)
                     "number"    his car number, e.g. 50
                     "sessions"  which sessions he drives (default: all); the
                                 seat's own driver drives the others. E.g.
                                 {"FP1"}: a young driver's FP1 outing;
                                 {"FP1", "FP2", "FP3"}: a practice stand-in.
                     "team_name" False: OpenF1 doesn't give his team.
"""

from datetime import datetime, timedelta, timezone

import requests

PUBLISH_DELAY = timedelta(minutes=15)
GRID_DELAY = timedelta(hours=2)
RESULTS_DELAY = timedelta(hours=1)
LENGTH = {"FP1": 1, "FP2": 1, "FP3": 1, "Q": 1, "R": 2, "SQ": 0.75, "S": 1}   # hours
RACE_LAPS = 20
SPRINT_LAPS = 10
POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
SPRINT_POINTS = [8, 7, 6, 5, 4, 3, 2, 1]

DRIVERS = [f"d{i:02d}" for i in range(1, 21)]
TEAM = {d: f"t{(i // 2) + 1:02d}" for i, d in enumerate(DRIVERS)}   # d01,d02 -> t01 ...
NUMBER = {d: i for i, d in enumerate(DRIVERS, start=1)}

# Each round: race day; sessions run Fri (FP1 12:00, FP2 16:00), Sat (FP3
# 11:00, Q 15:00), Sun (R 15:00), all UTC.
CALENDAR = {
    2026: ["2026-11-15", "2026-11-22"],                     # history: Jolpica only
    2027: ["2027-03-07", "2027-03-14", "2027-03-21"],
}
SESSION_OFFSETS = {"FP1": (-2, 12), "FP2": (-2, 16), "FP3": (-1, 11), "Q": (-1, 15), "R": (0, 15)}
# A sprint weekend (the 2024+ format, FakeF1(sprint_rounds=...)): FP2 and FP3
# give way to Sprint Qualifying (Fri 16:00) and the Sprint (Sat 11:00), both
# before qualifying.
SPRINT_OFFSETS = {"FP1": (-2, 12), "SQ": (-2, 16), "S": (-1, 11), "Q": (-1, 15), "R": (0, 15)}
ALL_OFFSETS = {**SESSION_OFFSETS, **SPRINT_OFFSETS}
# Fixed order for session keys. SQ and S go last, so the keys of normal
# sessions are the same as before sprints existed in this world.
ALL_SESSION_TYPES = ["FP1", "FP2", "FP3", "Q", "R", "SQ", "S"]
# The names each API uses for a session.
JOLPICA_KEY = {"FP1": "FirstPractice", "FP2": "SecondPractice", "FP3": "ThirdPractice",
               "SQ": "SprintQualifying", "S": "Sprint", "Q": "Qualifying"}
OPENF1_NAME = {"FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3",
               "SQ": "Sprint Qualifying", "S": "Sprint", "Q": "Qualifying", "R": "Race"}


def http_404() -> requests.HTTPError:
    response = requests.Response()
    response.status_code = 404
    return requests.HTTPError("404 Not Found", response=response)


def session_start(race_day: str, session_type: str) -> datetime:
    days, hour = ALL_OFFSETS[session_type]
    day = datetime.fromisoformat(race_day).replace(tzinfo=timezone.utc) + timedelta(days=days)
    return day.replace(hour=hour)


def session_key(season: int, rnd: int, session_type: str) -> int:
    """A unique OpenF1-style key, e.g. 2027, 2, 'Q' -> 20270204."""
    return season * 10_000 + rnd * 100 + ALL_SESSION_TYPES.index(session_type)


def order(season: int, rnd: int, salt: int) -> list[str]:
    """Drivers best first: by number, with a small fixed shuffle per round/session."""
    return sorted(DRIVERS, key=lambda d: NUMBER[d] + ((NUMBER[d] * 7 + rnd * 3 + salt) % 5) * 0.6)


def starting_grid(season: int, rnd: int) -> list[str]:
    """Qualifying order with one penalty: the pole-sitter drops to 4th."""
    quali = order(season, rnd, salt=1)
    return quali[1:4] + quali[:1] + quali[4:]


class FakeF1:
    def __init__(self, now: datetime, broken: set | None = None, cancelled: set | None = None,
                 empty_until: dict | None = None, tamper: dict | None = None,
                 sprint_rounds: set | None = None, debutants: dict | None = None):
        self.now = now
        self.broken = broken or set()
        self.cancelled = cancelled or set()
        self.empty_until = empty_until or {}
        self.tamper = tamper or {}
        self.sprint_rounds = sprint_rounds or set()
        self.debutants = debutants or {}
        self.requests = 0

    # --- helpers ---------------------------------------------------------------
    def _weekend(self, season, rnd) -> list[str]:
        """The round's sessions in calendar order, without cancelled ones."""
        offsets = SPRINT_OFFSETS if (season, rnd) in self.sprint_rounds else SESSION_OFFSETS
        return [t for t in offsets if (season, rnd, t) not in self.cancelled]

    def _late(self, season, rnd, what) -> bool:
        """True while `what` is still empty_until-delayed (see the module docstring)."""
        until = self.empty_until.get((season, rnd, what))
        return until is not None and self.now < until

    def _published(self, season, rnd, session_type, delay=PUBLISH_DELAY) -> bool:
        race_day = CALENDAR[season][rnd - 1]
        end = session_start(race_day, session_type) + timedelta(hours=LENGTH[session_type])
        return self.now >= end + delay

    def _find_key(self, key):
        for season, days in CALENDAR.items():
            for rnd in range(1, len(days) + 1):
                for session_type in ALL_SESSION_TYPES:
                    if session_key(season, rnd, session_type) == key:
                        return season, rnd, session_type
        raise http_404()

    def _debutant(self, season, rnd, session_type, seat):
        """The debutant spec if a debutant drives `seat` in this session, else None."""
        spec = self.debutants.get((season, rnd), {}).get(seat)
        if spec and session_type in spec.get("sessions", ALL_SESSION_TYPES):
            return spec
        return None

    def _who(self, season, rnd, session_type, seat):
        """(driver id, car number) in `seat` for this session."""
        spec = self._debutant(season, rnd, session_type, seat)
        return (spec["id"], spec["number"]) if spec else (seat, NUMBER[seat])

    def _driver(self, season, rnd, session_type, seat):
        driver_id, number = self._who(season, rnd, session_type, seat)
        return {"driverId": driver_id, "code": driver_id.upper(), "permanentNumber": str(number),
                "givenName": "Test", "familyName": driver_id.upper()}

    def _number(self, season, rnd, session_type, seat) -> int:
        return self._who(season, rnd, session_type, seat)[1]

    @staticmethod
    def _constructor(d):
        return {"constructorId": TEAM[d], "name": TEAM[d].upper()}

    # --- the API ---------------------------------------------------------------
    def get_json(self, url, params=None):
        self.requests += 1
        params = params or {}
        if "jolpi" in url:
            return self._jolpica(url.split("/ergast/f1/", 1)[1].removesuffix(".json"), params)
        return self._openf1(url.rsplit("/", 1)[1], params)

    def _jolpica(self, path, params):
        parts = path.split("/")
        season = int(parts[0])
        if season not in CALENDAR or params.get("offset", 0) > 0:
            races = []
        elif len(parts) == 1:
            races = [self._schedule_race(season, rnd) for rnd in range(1, len(CALENDAR[season]) + 1)]
        elif parts[1] == "results":
            races = [dict(self._schedule_race(season, rnd), Results=self._results(season, rnd))
                     for rnd in range(1, len(CALENDAR[season]) + 1)
                     if season < 2027 or (self._published(season, rnd, "R", RESULTS_DELAY)
                                          and not self._late(season, rnd, "results"))]
        elif parts[1] == "qualifying":
            races = [dict(self._schedule_race(season, rnd), QualifyingResults=self._qualifying(season, rnd))
                     for rnd in range(1, len(CALENDAR[season]) + 1)
                     # A cancelled qualifying has no results, like the real API.
                     if (season, rnd, "Q") not in self.cancelled
                     and (season < 2027 or self._published(season, rnd, "Q"))]
        elif parts[1] == "sprint":
            races = [dict(self._schedule_race(season, rnd), SprintResults=self._sprint_results(season, rnd))
                     for rnd in range(1, len(CALENDAR[season]) + 1)
                     if (season, rnd) in self.sprint_rounds
                     and self._published(season, rnd, "S", RESULTS_DELAY)]
        else:
            races = []
        rows = sum(len(r.get("Results") or r.get("QualifyingResults") or r.get("SprintResults") or [None])
                   for r in races)
        return {"MRData": {"total": str(rows), "RaceTable": {"Races": races}}}

    def _schedule_race(self, season, rnd):
        race_day = CALENDAR[season][rnd - 1]
        race = {"season": str(season), "round": str(rnd), "raceName": f"Test GP {rnd}",
                "Circuit": {"circuitId": f"c{rnd}", "circuitName": f"Circuit {rnd}", "Location": {}},
                "date": race_day, "time": "15:00:00Z"}
        # Every session but the race (whose time is the race's own date/time
        # above). Cancelled sessions are left out: not on the calendar.
        for session_type in self._weekend(season, rnd):
            if session_type == "R":
                continue
            start = session_start(race_day, session_type)
            race[JOLPICA_KEY[session_type]] = {"date": start.strftime("%Y-%m-%d"), "time": start.strftime("%H:%M:%SZ")}
        return race

    def _results(self, season, rnd):
        grid = {d: i for i, d in enumerate(starting_grid(season, rnd), start=1)}
        finishers = order(season, rnd, salt=2)
        if self.tamper.get((season, rnd, "results")) == "drop_two":
            finishers = finishers[:-2]           # a partial result: last two rows lost
        return [{"number": str(self._number(season, rnd, "R", d)), "position": str(p), "positionText": str(p),
                 "points": str(POINTS[p - 1] if p <= 10 else 0), "Driver": self._driver(season, rnd, "R", d),
                 "Constructor": self._constructor(d), "grid": str(grid[d]), "laps": str(RACE_LAPS),
                 "status": "Finished", "Time": {"millis": str(5_400_000 + p * 1000)}}
                for p, d in enumerate(finishers, start=1)]

    def _sprint_results(self, season, rnd):
        """A 10-lap sprint: grid = sprint qualifying order, its own finishing order."""
        grid = {d: i for i, d in enumerate(order(season, rnd, salt=len("SQ")), start=1)}
        return [{"number": str(self._number(season, rnd, "S", d)), "position": str(p), "positionText": str(p),
                 "points": str(SPRINT_POINTS[p - 1] if p <= 8 else 0), "Driver": self._driver(season, rnd, "S", d),
                 "Constructor": self._constructor(d), "grid": str(grid[d]), "laps": str(SPRINT_LAPS),
                 "status": "Finished", "Time": {"millis": str(1_800_000 + p * 1000)}}
                for p, d in enumerate(order(season, rnd, salt=3), start=1)]

    def _qualifying(self, season, rnd):
        no_times = self.tamper.get((season, rnd, "qualifying")) == "no_times"
        return [{"number": str(self._number(season, rnd, "Q", d)), "position": str(p),
                 "Driver": self._driver(season, rnd, "Q", d), "Constructor": self._constructor(d),
                 **({} if no_times else {"Q1": f"1:{20 + p * 0.1:06.3f}"})}
                for p, d in enumerate(order(season, rnd, salt=1), start=1)]

    def _openf1(self, endpoint, params):
        if endpoint == "sessions":
            season = params["year"]
            if season not in CALENDAR or season < 2027:
                raise http_404()
            return [{"session_key": session_key(season, rnd, t), "meeting_key": season * 100 + rnd,
                     "session_name": OPENF1_NAME[t],
                     "date_start": session_start(day, t).isoformat(), "country_name": "Testland"}
                    for rnd, day in enumerate(CALENDAR[season], start=1)
                    for t in self._weekend(season, rnd)]

        season, rnd, session_type = self._find_key(params["session_key"])
        if endpoint == "starting_grid":
            if (session_type != "Q" or (season, rnd, "grid") in self.broken
                    or not self._published(season, rnd, "Q", GRID_DELAY)):
                raise http_404()
            return [{"driver_number": self._number(season, rnd, "Q", d), "position": p}
                    for p, d in enumerate(starting_grid(season, rnd), start=1)]
        if (season, rnd, session_type) in self.broken or not self._published(season, rnd, session_type):
            raise http_404()                     # not published (yet, or ever)

        tamper = self.tamper.get((season, rnd, session_type))
        # 'drop_driver': d20 isn't in this session's data at all.
        drivers = DRIVERS[:-1] if tamper == "drop_driver" else DRIVERS
        car = {d: self._number(season, rnd, session_type, d) for d in drivers}
        if endpoint == "drivers":
            out = []
            for d in drivers:
                driver_id, number = self._who(season, rnd, session_type, d)
                spec = self._debutant(season, rnd, session_type, d) or {}
                out.append({"driver_number": number, "name_acronym": driver_id.upper(),
                            "full_name": f"Rookie {driver_id.upper()}" if spec else d,
                            "team_name": None if spec.get("team_name") is False else TEAM[d].upper()})
            return out
        n_laps = {"Q": 3, "SQ": 3, "R": RACE_LAPS, "S": SPRINT_LAPS}.get(session_type, 8)
        if endpoint == "laps" and self._late(season, rnd, session_type):
            return []                            # published, but no laps in it yet
        # 'half_laps': OpenF1 only has the first half of the session.
        laps_served = n_laps // 2 if tamper == "half_laps" else n_laps
        pace = {d: i for i, d in enumerate(order(season, rnd, salt=len(session_type)), start=1)}
        if endpoint == "laps":
            start = session_start(CALENDAR[season][rnd - 1], session_type)
            return [{"driver_number": car[d], "lap_number": lap,
                     "date_start": (start + timedelta(minutes=2 * lap)).isoformat(),
                     # Out-lap and in-lap slower; a clean run in between.
                     "lap_duration": 90 + pace[d] * 0.1 + (5 if lap in (1, n_laps) else lap * 0.02),
                     "is_pit_out_lap": lap == 1}
                    for d in drivers for lap in range(1, laps_served + 1)]
        if endpoint == "stints":
            return [{"driver_number": car[d], "stint_number": 1, "compound": "MEDIUM",
                     "lap_start": 1, "lap_end": n_laps, "tyre_age_at_start": 0} for d in drivers]
        if endpoint == "weather":
            return [{"date": session_start(CALENDAR[season][rnd - 1], session_type).isoformat(),
                     "air_temperature": 22.0, "rainfall": 0}]
        raise http_404()                          # pit, race_control: nothing
