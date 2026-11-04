"""Central configuration: every path and constant the project uses lives here.

Why a single config module?
    If the database path were written out in five different files, changing it
    would mean finding all five. Here, every other module does
    `from f1predictor import config` and reads `config.DB_PATH`.

Why plain constants instead of a config file (YAML/TOML)?
    We only have a handful of settings so far. Plain Python is the simplest
    thing that works, and you can jump straight to the definition in an editor.
    If settings start varying between environments (laptop vs GitHub Actions),
    we can read overrides from environment variables then.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Path(__file__) is this file: <repo>/src/f1predictor/config.py
# .resolve() turns it into an absolute path (following any symlinks).
# .parents[2] climbs up three levels: f1predictor/ -> src/ -> <repo>/
# Building paths from this file's location (not the current working directory)
# means scripts work no matter which folder you run them from.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Everything generated locally goes under data/ (gitignored).
DATA_DIR: Path = PROJECT_ROOT / "data"

# The SQLite database file: a single file holding all of our tables.
DB_PATH: Path = DATA_DIR / "f1.sqlite"

# Where the SQL file that defines our tables will live (written in Step 2).
# Trained models are saved here (gitignored with the rest of data/; any model
# can be rebuilt by rerunning its training script).
MODELS_DIR: Path = DATA_DIR / "models"
SCHEMA_PATH: Path = Path(__file__).resolve().parent / "db" / "schema.sql"

# ---------------------------------------------------------------------------
# Seasons to backfill
# ---------------------------------------------------------------------------

# 2018: the first season of the training history. (It was chosen as FastF1's
# first season with lap timing; FastF1 was dropped for OpenF1, whose lap data
# starts in 2023, so 2018-2022 now have results and qualifying only. Jolpica
# has results back to 1950, but older seasons add little to today's model.)
FIRST_SEASON: int = 2018

# The latest season to backfill. Written explicitly rather than computed from
# today's date, so a run in January doesn't silently start fetching an empty
# new season. Bump this at the start of each new season.
LAST_SEASON: int = 2026

# A convenience list: [2018, 2019, ..., 2026]. range() excludes its end value,
# hence the + 1.
SEASONS: list[int] = list(range(FIRST_SEASON, LAST_SEASON + 1))

# ---------------------------------------------------------------------------
# External APIs
# ---------------------------------------------------------------------------

# Jolpica-F1 is the community-run replacement for the old Ergast API, and its
# URLs and JSON format are the same as Ergast's. Example request:
#   https://api.jolpi.ca/ergast/f1/2024/results.json
JOLPICA_BASE_URL: str = "https://api.jolpi.ca/ergast/f1"

# Jolpica's documented limits: about 4 requests/second in bursts, with a hard
# cap of 200 requests/hour. One request every 0.5s stays under the burst limit;
# a full 2018-2026 Jolpica backfill is ~120 requests, under the hourly cap.
JOLPICA_MIN_INTERVAL_S: float = 0.5

# OpenF1: lap-level data (laps, stints, pit stops, weather, race control).
# Example request: https://api.openf1.org/v1/laps?session_key=9662
OPENF1_BASE_URL: str = "https://api.openf1.org/v1"

# OpenF1's historical data starts with the 2023 season. Earlier seasons get
# Jolpica data (results, qualifying) only.
OPENF1_FIRST_SEASON: int = 2023

# OpenF1's free tier allows 3 requests/second and 30 requests/minute.
# The per-minute limit is the tighter one: 60s / 30 = one request every 2s.
OPENF1_MIN_INTERVAL_S: float = 2.0


def ensure_data_dirs() -> None:
    """Create the local data folders if they don't exist yet.

    This is a function, not something that runs at import time, so that
    importing config never touches the filesystem. Scripts call it
    explicitly when they need the folders.
    """
    # parents=True also creates any missing parent folders.
    # exist_ok=True means "don't raise an error if it's already there".
    DATA_DIR.mkdir(parents=True, exist_ok=True)
