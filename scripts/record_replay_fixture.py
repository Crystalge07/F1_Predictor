"""Record the real API responses for one past weekend, for the replay test.

Run once (it uses the network); the result is committed as a test fixture so
the test itself never touches the network:

    python scripts/record_replay_fixture.py

What it records (tests/test_real_weekend_replay.py explains why):

    Jolpica   2025 and 2026: schedule, race results, sprint results,
              qualifying. 2025 is training history (and last season's points).
    OpenF1    the 2026 session list, and for every session of 2026 R18
              (Austin, loaded as history) and R19 (Mexico City, the replayed
              weekend): drivers, laps, stints, pit, weather, race_control,
              plus the starting grid of each qualifying session.

Requests go through the project's own fetch functions (jolpica.fetch_races,
openf1.fetch), so the recording holds exactly what the loaders would receive.
Jolpica pages are merged into one list per endpoint; the replay server hands
them back as a single page, which fetch_races reads the same way.

Output: tests/fixtures/replay_2026_r19.json.gz, gzipped JSON:
    {"recorded_at": ..., "jolpica": {"2026/results": [race, ...], ...},
     "openf1": {"sessions?year=2026": [...], "laps?session_key=9999": [...], ...}}
"""

import gzip
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from f1predictor.ingest import jolpica, openf1
from f1predictor.pipeline.fetch import make_clients

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "replay_2026_r19.json.gz"

JOLPICA_SEASONS = [2025, 2026]
# (path suffix, the per-race list to merge across pages; None for the schedule)
JOLPICA_PARTS = [("", None), ("/results", "Results"), ("/sprint", "SprintResults"),
                 ("/qualifying", "QualifyingResults")]
SEASON = 2026
ROUNDS = [18, 19]                    # Austin (history), Mexico City (replayed)
# Every per-session endpoint ingest/openf1.ingest_session reads.
SESSION_ENDPOINTS = ["drivers", "laps", "stints", "pit", "weather", "race_control"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    jolpica_client, openf1_client = make_clients()     # real clients, real rate limits
    recording = {"recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "jolpica": {}, "openf1": {}}

    # --- Jolpica: whole seasons (small: a few hundred result rows each) -------
    for season in JOLPICA_SEASONS:
        for suffix, result_key in JOLPICA_PARTS:
            path = f"{season}{suffix}"
            recording["jolpica"][path] = jolpica.fetch_races(jolpica_client, path, result_key)
            logging.info("Jolpica %s: %d races", path, len(recording["jolpica"][path]))

    # --- OpenF1: the season's session list, then two weekends' sessions -------
    sessions = openf1.fetch(openf1_client, "sessions", year=SEASON)
    recording["openf1"][f"sessions?year={SEASON}"] = sessions

    # Which OpenF1 sessions belong to R18 and R19? OpenF1 has no round number,
    # so match by the race date in Jolpica's schedule (same rule as the loader,
    # ingest/openf1.match_sessions): the meeting whose Race is on that day.
    schedule = {int(r["round"]): r["date"] for r in recording["jolpica"][str(SEASON)]}
    for rnd in ROUNDS:
        race = next(s for s in sessions if s["session_name"] == "Race"
                    and s["date_start"][:10] == schedule[rnd])
        weekend = [s for s in sessions if s["meeting_key"] == race["meeting_key"]
                   and s["session_name"] in openf1.SESSION_NAME_TO_TYPE]
        for session in weekend:
            key = session["session_key"]
            endpoints = SESSION_ENDPOINTS + (["starting_grid"] if session["session_name"] == "Qualifying" else [])
            for endpoint in endpoints:
                rows = openf1.fetch(openf1_client, endpoint, session_key=key)
                recording["openf1"][f"{endpoint}?session_key={key}"] = rows
            logging.info("OpenF1 R%d %s (%d): recorded", rnd, session["session_name"], key)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        json.dump(recording, f, separators=(",", ":"))      # compact: no spaces
    logging.info("Wrote %s (%.0f KB)", OUT, OUT.stat().st_size / 1024)


if __name__ == "__main__":
    main()
