-- =============================================================================
-- schema.sql: every table in the F1 predictor database (Phase 1: raw data).
--
-- Two sources feed it:
--   * Jolpica (2018+): calendar, race/sprint results, qualifying, drivers, teams
--   * OpenF1  (2023+): who drove in each session, laps, stints, pit stops,
--                      weather, race control messages
--
-- Conventions used throughout:
--   * Durations are INTEGER milliseconds, with column names ending in _ms.
--     Integers are exact and sort correctly; floats of seconds are not exact.
--   * Timestamps are UTC, stored as ISO-8601 text ('2024-03-02T15:00:00Z'),
--     with column names ending in _utc. We declare them as TIMESTAMP / DATE
--     (standard SQL); SQLite stores the text as-is, and Postgres would store
--     real date types.
--   * IDs from Jolpica (driver_id, constructor_id, circuit_id) are reused as our
--     primary keys, e.g. 'max_verstappen', 'red_bull', 'monza'. They're stable
--     and readable.
--   * Every statement uses IF NOT EXISTS, so running this file again on an
--     existing database is safe and changes nothing. (It also means editing a
--     table here does NOT change an existing database: rebuild it instead.)
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

-- One row per driver who has started a race (that's who Jolpica knows).
-- Note: no team column here. A driver's team can change mid-season, so the
-- team is recorded on each result / session entry instead.
CREATE TABLE IF NOT EXISTS drivers (
    driver_id        TEXT PRIMARY KEY,  -- Jolpica id, e.g. 'hamilton'
    code             TEXT,              -- 3-letter code, e.g. 'HAM'
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
    -- Weekend format: 'conventional', 'sprint', 'sprint_shootout',
    -- 'sprint_qualifying'. Stored rather than hard-coded because the PRD says
    -- to always read the schedule from the calendar.
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
    -- loaders map every name onto these 7 codes:
    --   FP1/FP2/FP3 = practice, SQ = sprint qualifying/shootout,
    --   S = sprint race, Q = qualifying, R = race.
    session_type   TEXT NOT NULL
                   CHECK (session_type IN ('FP1', 'FP2', 'FP3', 'SQ', 'S', 'Q', 'R')),
    session_name   TEXT,                 -- the original name, kept for reference
    start_time_utc TIMESTAMP,            -- scheduled start; the scheduler uses this later
    -- OpenF1's id for the same session (e.g. 9662), once the OpenF1 loader has
    -- matched it. NULL before 2023 (OpenF1 has no data) and for future sessions.
    -- UNIQUE: two of our sessions can never claim the same OpenF1 session.
    openf1_session_key INTEGER UNIQUE,

    -- An event has at most one of each session type.
    UNIQUE (event_id, session_type)
);


-- -----------------------------------------------------------------------------
-- Session results. Source: Jolpica.
-- -----------------------------------------------------------------------------

-- Final classification for race (R) and sprint (S) sessions.
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
    race_time_ms   INTEGER,             -- total race time; NULL for non-finishers

    -- One result per driver per session.
    PRIMARY KEY (session_id, driver_id)
);

-- Qualifying results for Q sessions.
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
-- Who drove in each session. Source: OpenF1 (2023+).
-- -----------------------------------------------------------------------------

-- One row per car per session: "car #22 ran in this session, the timing
-- screens called the driver HIR / Ryo HIRAKAWA, and the team McLaren".
--
-- Why this table exists: OpenF1 identifies drivers by car number, and some
-- drivers who set laps (FP1 rookies, debut weekends before Jolpica updates)
-- aren't in our drivers table. So all lap-level data hangs off this entry
-- instead of off drivers. driver_id / constructor_id are filled in when we can
-- match them to Jolpica, and stay NULL when we can't (yet).
CREATE TABLE IF NOT EXISTS session_entries (
    session_id     INTEGER NOT NULL REFERENCES sessions (session_id),
    -- Car numbers are unique within a session, so (session_id, car_number)
    -- identifies one driver for the whole session.
    car_number     INTEGER NOT NULL,

    -- Exactly what OpenF1 said, kept as-is. If matching ever goes wrong, this
    -- is the evidence for fixing it.
    acronym        TEXT,                -- 'HIR'
    full_name      TEXT,                -- 'Ryo HIRAKAWA'
    team_name      TEXT,                -- 'McLaren' (OpenF1's name, not Jolpica's id)

    -- Our matched ids. NULL = not matched (yet). When not NULL, the foreign
    -- keys guarantee they point at real rows. (In SQL, a NULL foreign key is
    -- allowed and simply means "no reference".)
    driver_id      TEXT REFERENCES drivers (driver_id),
    constructor_id TEXT REFERENCES constructors (constructor_id),

    PRIMARY KEY (session_id, car_number)
);


-- -----------------------------------------------------------------------------
-- Lap-level data. Source: OpenF1 (2023+). Every table here points at a
-- session entry via (session_id, car_number).
-- -----------------------------------------------------------------------------

-- One row per car per lap.
CREATE TABLE IF NOT EXISTS laps (
    session_id      INTEGER NOT NULL,
    car_number      INTEGER NOT NULL,
    lap_number      INTEGER NOT NULL,

    lap_start_utc   TIMESTAMP,          -- when the lap began; lines up with weather / race control
    lap_time_ms     INTEGER,            -- NULL when timing failed or the lap wasn't completed
    sector1_ms      INTEGER,
    sector2_ms      INTEGER,
    sector3_ms      INTEGER,
    is_pit_out_lap  BOOLEAN,            -- driver left the pits at the start of this lap

    -- Speed traps in km/h: two intermediate points and the main speed trap.
    -- Cheap to keep; possibly useful for track-type / straight-line features.
    i1_speed_kph    INTEGER,
    i2_speed_kph    INTEGER,
    st_speed_kph    INTEGER,

    PRIMARY KEY (session_id, car_number, lap_number),
    -- A composite foreign key: the (session_id, car_number) PAIR must exist in
    -- session_entries. So a lap can never belong to a car that wasn't there.
    FOREIGN KEY (session_id, car_number) REFERENCES session_entries (session_id, car_number)
);

-- One row per tyre stint: a continuous run on one set of tyres.
CREATE TABLE IF NOT EXISTS stints (
    session_id        INTEGER NOT NULL,
    car_number        INTEGER NOT NULL,
    stint_number      INTEGER NOT NULL,   -- 1, 2, 3... within the session
    compound          TEXT,               -- 'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'
    lap_start         INTEGER,            -- first lap of the stint
    lap_end           INTEGER,            -- last lap of the stint
    -- Laps already on the set when the stint started (0 = new tyres).
    -- Includes laps from earlier sessions, e.g. a used set from qualifying.
    tyre_age_at_start INTEGER,

    PRIMARY KEY (session_id, car_number, stint_number),
    FOREIGN KEY (session_id, car_number) REFERENCES session_entries (session_id, car_number)
);

-- One row per pit stop.
CREATE TABLE IF NOT EXISTS pit_stops (
    session_id     INTEGER NOT NULL,
    car_number     INTEGER NOT NULL,
    lap_number     INTEGER NOT NULL,   -- lap the stop happened on
    stop_time_utc  TIMESTAMP,
    pit_lane_ms    INTEGER,            -- total time in the pit lane (entry to exit)
    stationary_ms  INTEGER,            -- time stopped in the box; NULL in older OpenF1 data

    PRIMARY KEY (session_id, car_number, lap_number),
    FOREIGN KEY (session_id, car_number) REFERENCES session_entries (session_id, car_number)
);


-- -----------------------------------------------------------------------------
-- Conditions during a session. Source: OpenF1 (2023+).
-- -----------------------------------------------------------------------------

-- Weather, sampled about once a minute during each session.
CREATE TABLE IF NOT EXISTS weather (
    session_id         INTEGER NOT NULL REFERENCES sessions (session_id),
    sample_time_utc    TIMESTAMP NOT NULL,
    air_temp_c         REAL,
    track_temp_c       REAL,
    humidity_pct       REAL,
    pressure_mbar      REAL,
    rainfall           BOOLEAN,            -- was it raining at this moment?
    wind_speed_ms      REAL,               -- metres per second
    wind_direction_deg INTEGER,

    PRIMARY KEY (session_id, sample_time_utc)
);

-- Every message from Race Control: flags, safety cars, VSCs, red flags,
-- penalties, deleted lap times... We store them raw and derive things like
-- "number of safety cars" with queries later.
CREATE TABLE IF NOT EXISTS race_control (
    session_id       INTEGER NOT NULL REFERENCES sessions (session_id),
    message_time_utc TIMESTAMP NOT NULL,
    category         TEXT,      -- 'Flag', 'SafetyCar', 'SessionStatus', 'Drs', 'Other'
    flag             TEXT,      -- 'GREEN', 'YELLOW', 'RED', 'BLUE', 'CHEQUERED'... or NULL
    scope            TEXT,      -- 'Track', 'Sector', 'Driver'
    sector           INTEGER,   -- for sector-scoped flags
    -- Car the message is about, if any. Deliberately NOT a foreign key:
    -- this is a raw log, and we don't want one odd message to reject the batch.
    car_number       INTEGER,
    lap_number       INTEGER,
    message          TEXT NOT NULL,  -- e.g. 'VIRTUAL SAFETY CAR DEPLOYED'

    -- There's no natural id for a message, so "same session, same time, same
    -- text" counts as the same message. This lets re-runs skip duplicates.
    PRIMARY KEY (session_id, message_time_utc, message)
);


-- -----------------------------------------------------------------------------
-- Pipeline bookkeeping.
-- -----------------------------------------------------------------------------

-- One row per thing we've tried to fetch, with how it went. The backfill reads
-- this to skip work already done (so it can be stopped and restarted), and
-- loaders write failures here instead of crashing.
CREATE TABLE IF NOT EXISTS ingestion_log (
    source      TEXT NOT NULL,       -- 'jolpica' or 'openf1'
    -- What was fetched, as a readable key, e.g. '2024-results' or '2024-06-FP1'.
    -- A text key keeps one table usable for every kind of fetch.
    item_key    TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('ok', 'failed', 'skipped')),
    row_count   INTEGER,             -- rows written, as a quick sanity number
    message     TEXT,                -- error details when status = 'failed'
    ingested_at TIMESTAMP NOT NULL,  -- when this attempt happened (UTC)

    -- Only the latest attempt per item is kept; retries overwrite it.
    PRIMARY KEY (source, item_key)
);


-- -----------------------------------------------------------------------------
-- Views: saved queries that behave like read-only tables.
-- -----------------------------------------------------------------------------

-- Every lap with its driver, team and tyre attached: the joins that nearly
-- every pace feature needs, written once here instead of in every query.
CREATE VIEW IF NOT EXISTS laps_detailed AS
SELECT
    l.session_id,
    l.car_number,
    e.driver_id,              -- NULL for drivers not matched to Jolpica
    e.constructor_id,
    e.acronym,
    l.lap_number,
    l.lap_start_utc,
    l.lap_time_ms,
    l.sector1_ms,
    l.sector2_ms,
    l.sector3_ms,
    l.is_pit_out_lap,
    st.stint_number,
    st.compound,
    -- Tyre age on THIS lap = age when the stint started + laps done since.
    -- E.g. a stint starting on lap 10 with 3-lap-old tyres: lap 12 -> 3 + 2 = 5.
    st.tyre_age_at_start + (l.lap_number - st.lap_start) AS tyre_age
FROM laps AS l
-- Every lap has an entry (the foreign key guarantees it), so a plain JOIN.
JOIN session_entries AS e
    ON e.session_id = l.session_id AND e.car_number = l.car_number
-- LEFT JOIN: keep the lap even if no stint covers it (OpenF1 occasionally has
-- gaps); its tyre columns are just NULL.
LEFT JOIN stints AS st
    ON st.session_id = l.session_id
   AND st.car_number = l.car_number
   AND l.lap_number BETWEEN st.lap_start AND st.lap_end;


-- -----------------------------------------------------------------------------
-- Indexes: speed up the lookups we know we'll do often. (Primary keys and
-- UNIQUE constraints already get an index automatically.)
-- -----------------------------------------------------------------------------

-- "All sessions for this event" and "all results for this driver" come up
-- constantly in feature building (e.g. a driver's last 5 finishes).
CREATE INDEX IF NOT EXISTS idx_sessions_event     ON sessions (event_id);
CREATE INDEX IF NOT EXISTS idx_results_driver     ON results (driver_id);
CREATE INDEX IF NOT EXISTS idx_qualifying_driver  ON qualifying (driver_id);
CREATE INDEX IF NOT EXISTS idx_events_circuit     ON events (circuit_id);
-- "All practice laps for this driver" goes through session_entries.
CREATE INDEX IF NOT EXISTS idx_entries_driver     ON session_entries (driver_id);
