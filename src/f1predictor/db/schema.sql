-- =============================================================================
-- schema.sql: every table in the F1 predictor database (Phase 1: raw data).
--
-- Conventions used throughout:
--   * Durations are INTEGER milliseconds, with column names ending in _ms.
--     Integers are exact and sort correctly; floats of seconds are not exact.
--   * Timestamps are UTC, stored as ISO-8601 text ('2024-03-02T15:00:00Z').
--     We declare them as TIMESTAMP / DATE (standard SQL); SQLite stores the
--     text as-is, and Postgres would store real date types.
--   * IDs from Jolpica (driver_id, constructor_id, circuit_id) are reused as our
--     primary keys, e.g. 'max_verstappen', 'red_bull', 'monza'. They're stable,
--     readable, and they're what FastF1 also exposes, so both sources join on them.
--   * Every statement uses IF NOT EXISTS, so running this file again on an
--     existing database is safe and changes nothing.
--
-- Phase 1 only stores raw facts. Features, predictions and scores get their
-- own tables in later phases.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- Reference tables: things that exist independently of any one race.
-- -----------------------------------------------------------------------------

-- One row per circuit ever used.
CREATE TABLE IF NOT EXISTS circuits (
    circuit_id   TEXT PRIMARY KEY,   -- Jolpica id, e.g. 'silverstone'
    name         TEXT NOT NULL,      -- 'Silverstone Circuit'
    locality     TEXT,               -- 'Silverstone'
    country      TEXT,               -- 'UK'
    latitude     REAL,               -- for weather lookups in a later phase
    longitude    REAL
);

-- One row per driver. Note: no team column here. A driver's team can change
-- mid-season, so the team is recorded on each result row instead.
CREATE TABLE IF NOT EXISTS drivers (
    driver_id        TEXT PRIMARY KEY,  -- Jolpica id, e.g. 'hamilton'
    code             TEXT,              -- 3-letter code, e.g. 'HAM' (FastF1 uses this)
    permanent_number INTEGER,           -- e.g. 44 (can be NULL for older/one-off drivers)
    given_name       TEXT NOT NULL,
    family_name      TEXT NOT NULL,
    date_of_birth    DATE,
    nationality      TEXT
);

-- One row per constructor (team). Rebrands get new ids in Jolpica
-- (e.g. 'alphatauri' -> 'rb'), which we keep as-is; linking them is a
-- feature-engineering decision for a later phase.
CREATE TABLE IF NOT EXISTS constructors (
    constructor_id TEXT PRIMARY KEY,    -- e.g. 'mclaren'
    name           TEXT NOT NULL,       -- 'McLaren'
    nationality    TEXT
);


-- -----------------------------------------------------------------------------
-- Calendar: events (Grands Prix) and the sessions within them.
-- -----------------------------------------------------------------------------

-- One row per Grand Prix weekend.
CREATE TABLE IF NOT EXISTS events (
    -- SQLite-specific: "INTEGER PRIMARY KEY" auto-assigns 1, 2, 3...
    -- (In Postgres this line would be: event_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY)
    -- We use a numeric id so other tables can reference an event with one
    -- column instead of two (season + round).
    event_id     INTEGER PRIMARY KEY,
    season       INTEGER NOT NULL,      -- 2024
    round        INTEGER NOT NULL,      -- 1, 2, 3... order within the season
    name         TEXT NOT NULL,         -- 'Bahrain Grand Prix'
    circuit_id   TEXT NOT NULL REFERENCES circuits (circuit_id),
    race_date    DATE NOT NULL,         -- date of the Sunday (or Saturday) race
    -- Weekend format, as named by FastF1: 'conventional', 'sprint',
    -- 'sprint_shootout', 'sprint_qualifying'. Stored rather than hard-coded
    -- because the PRD says to always read the schedule from the calendar.
    event_format TEXT,

    -- The real-world identity of an event is (season, round). UNIQUE stops us
    -- from ever inserting the same race twice.
    UNIQUE (season, round)
);

-- One row per session within an event.
CREATE TABLE IF NOT EXISTS sessions (
    session_id     INTEGER PRIMARY KEY,  -- auto-assigned (see note on events)
    event_id       INTEGER NOT NULL REFERENCES events (event_id),
    -- Our own normalized session codes. Different years used confusing names
    -- (in 2021 "Sprint Qualifying" was actually the sprint race), so the
    -- loader maps every name onto these 7 codes:
    --   FP1/FP2/FP3 = practice, SQ = sprint qualifying/shootout,
    --   S = sprint race, Q = qualifying, R = race.
    session_type   TEXT NOT NULL
                   CHECK (session_type IN ('FP1', 'FP2', 'FP3', 'SQ', 'S', 'Q', 'R')),
    session_name   TEXT,                 -- the original name, kept for reference
    start_time_utc TIMESTAMP,            -- scheduled start; the scheduler uses this later

    -- An event has at most one of each session type.
    UNIQUE (event_id, session_type)
);


-- -----------------------------------------------------------------------------
-- Session results.
-- -----------------------------------------------------------------------------

-- Final classification for race (R) and sprint (S) sessions. Source: Jolpica.
CREATE TABLE IF NOT EXISTS results (
    session_id     INTEGER NOT NULL REFERENCES sessions (session_id),
    driver_id      TEXT NOT NULL REFERENCES drivers (driver_id),
    -- The team for THIS race. This is what makes mid-season swaps work.
    constructor_id TEXT NOT NULL REFERENCES constructors (constructor_id),
    car_number     INTEGER,             -- can differ from permanent_number (e.g. #1 for the champion)
    -- Official STARTING grid, after penalties (not qualifying order!).
    -- 0 means the driver started from the pit lane.
    grid           INTEGER,
    -- Finishing order 1..N. Every driver gets one, including retirements
    -- (they're ordered by laps completed). Use position_text to tell them apart.
    position       INTEGER NOT NULL,
    -- '1', '2', ... for classified finishers; 'R' retired, 'D' disqualified,
    -- 'W' withdrawn, 'N' not classified, 'E' excluded, 'F' failed to qualify.
    position_text  TEXT NOT NULL,
    points         REAL NOT NULL DEFAULT 0,   -- REAL because half points exist (Spa 2021)
    laps_completed INTEGER,
    status         TEXT,                -- 'Finished', '+1 Lap', 'Engine', 'Accident', ...
    race_time_ms   INTEGER,             -- total race time; NULL for non-finishers / lapped cars

    -- One result per driver per session.
    PRIMARY KEY (session_id, driver_id)
);

-- Qualifying results for Q and SQ sessions.
-- Source: Jolpica for Q; FastF1 for SQ (Jolpica doesn't publish sprint quali).
CREATE TABLE IF NOT EXISTS qualifying (
    session_id     INTEGER NOT NULL REFERENCES sessions (session_id),
    driver_id      TEXT NOT NULL REFERENCES drivers (driver_id),
    constructor_id TEXT NOT NULL REFERENCES constructors (constructor_id),
    position       INTEGER NOT NULL,    -- qualifying position (NOT the starting grid)
    q1_ms          INTEGER,             -- best lap in each part; NULL if not set / knocked out
    q2_ms          INTEGER,
    q3_ms          INTEGER,

    PRIMARY KEY (session_id, driver_id)
);


-- -----------------------------------------------------------------------------
-- Lap-level data. Source: FastF1. Available for every session type.
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS laps (
    session_id      INTEGER NOT NULL REFERENCES sessions (session_id),
    driver_id       TEXT NOT NULL REFERENCES drivers (driver_id),
    lap_number      INTEGER NOT NULL,

    lap_time_ms     INTEGER,            -- NULL when timing failed or the lap wasn't completed
    sector1_ms      INTEGER,
    sector2_ms      INTEGER,
    sector3_ms      INTEGER,
    -- Session clock time (ms since the session started) when this lap began.
    -- Lets us match laps to weather samples and track status changes.
    lap_start_ms    INTEGER,

    -- Tyre information: the raw material for stints and long-run pace.
    stint           INTEGER,            -- 1, 2, 3... increments after each pit stop
    compound        TEXT,               -- 'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'
    tyre_life       INTEGER,            -- laps on this set, including laps from earlier sessions
    fresh_tyre      BOOLEAN,            -- was the set new when fitted?

    is_pit_in_lap   BOOLEAN,            -- driver entered the pits at the end of this lap
    is_pit_out_lap  BOOLEAN,            -- driver left the pits at the start of this lap
    -- Track status code(s) seen during the lap, e.g. '1' green, '4' safety car,
    -- '6' virtual SC. Multiple codes can appear ('124'). Used to drop
    -- non-representative laps when computing pace.
    track_status    TEXT,
    is_accurate     BOOLEAN,            -- FastF1's own flag: timing for this lap passed its checks
    is_deleted      BOOLEAN,            -- lap time deleted by stewards (e.g. track limits)
    position        INTEGER,            -- running position at the end of the lap (races only)

    PRIMARY KEY (session_id, driver_id, lap_number)
);

-- Stints: one row per driver per tyre stint. This is a VIEW, i.e. a saved
-- query that behaves like a read-only table. We compute it from laps instead
-- of storing it, so the two can never disagree.
CREATE VIEW IF NOT EXISTS stints AS
SELECT
    session_id,
    driver_id,
    stint,
    -- A stint uses one tyre compound; MAX just picks it (ignores NULLs).
    MAX(compound)        AS compound,
    MIN(lap_number)      AS start_lap,
    MAX(lap_number)      AS end_lap,
    COUNT(*)             AS lap_count,
    MIN(tyre_life)       AS tyre_life_at_start,   -- > 1 means a used set
    MAX(fresh_tyre)      AS fresh_tyre
FROM laps
WHERE stint IS NOT NULL
GROUP BY session_id, driver_id, stint;


-- -----------------------------------------------------------------------------
-- Conditions during a session. Source: FastF1.
-- -----------------------------------------------------------------------------

-- Weather, sampled about once a minute during each session.
CREATE TABLE IF NOT EXISTS weather (
    session_id         INTEGER NOT NULL REFERENCES sessions (session_id),
    sample_ms          INTEGER NOT NULL,   -- session clock time of this sample
    air_temp_c         REAL,
    track_temp_c       REAL,
    humidity_pct       REAL,
    pressure_mbar      REAL,
    rainfall           BOOLEAN,            -- was it raining at this moment?
    wind_speed_ms      REAL,               -- metres per second
    wind_direction_deg INTEGER,

    PRIMARY KEY (session_id, sample_ms)
);

-- Every change in track status: green, yellow, safety car, VSC, red flag.
-- Lets us count safety cars / red flags per race (the "chaos" edge cases).
CREATE TABLE IF NOT EXISTS track_status (
    session_id INTEGER NOT NULL REFERENCES sessions (session_id),
    change_ms  INTEGER NOT NULL,   -- session clock time when the status changed
    -- '1' green, '2' yellow, '4' safety car, '5' red flag,
    -- '6' VSC deployed, '7' VSC ending
    status     TEXT NOT NULL,
    message    TEXT,               -- e.g. 'SCDeployed', 'AllClear'

    PRIMARY KEY (session_id, change_ms, status)
);


-- -----------------------------------------------------------------------------
-- Pipeline bookkeeping.
-- -----------------------------------------------------------------------------

-- One row per thing we've tried to fetch, with how it went. The backfill reads
-- this to skip work already done (so it can be stopped and restarted), and
-- the validation step writes failures here instead of crashing.
CREATE TABLE IF NOT EXISTS ingestion_log (
    source      TEXT NOT NULL,       -- 'jolpica' or 'fastf1'
    -- What was fetched, as a readable key, e.g. '2024-05-R' (season-round-session)
    -- or '2024-schedule'. A text key keeps one table usable for every kind of fetch.
    item_key    TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('ok', 'failed', 'skipped')),
    row_count   INTEGER,             -- rows written, as a quick sanity number
    message     TEXT,                -- error details when status = 'failed'
    ingested_at TIMESTAMP NOT NULL,  -- when this attempt happened (UTC)

    -- Only the latest attempt per item is kept; retries overwrite it.
    PRIMARY KEY (source, item_key)
);


-- -----------------------------------------------------------------------------
-- Indexes: speed up the lookups we know we'll do often. (Primary keys and
-- UNIQUE constraints already get an index automatically.)
-- -----------------------------------------------------------------------------

-- "All sessions for this event" and "all results for this driver" come up
-- constantly in feature building (e.g. a driver's last 5 finishes).
CREATE INDEX IF NOT EXISTS idx_sessions_event   ON sessions (event_id);
CREATE INDEX IF NOT EXISTS idx_results_driver   ON results (driver_id);
CREATE INDEX IF NOT EXISTS idx_qualifying_driver ON qualifying (driver_id);
CREATE INDEX IF NOT EXISTS idx_events_circuit   ON events (circuit_id);
