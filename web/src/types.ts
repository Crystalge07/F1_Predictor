// The shapes of the exported JSON files, field for field.
//
// The single source of truth is the Python export
// (src/f1predictor/export/dashboard.py). These types must match it; the
// contract test (src/test/contract.test.ts) loads real files exported from
// the simulated test world, so a mismatch fails a test instead of the page.
//
// SCHEMA is bumped on the Python side whenever a file's shape changes; the
// app refuses files with a schema it doesn't know (see data.ts).

export const SCHEMA = 1;

export type Stage = "pre_weekend" | "practice" | "post_quali";
export type Mode = "live" | "backtest";
export type RaceStatus = "upcoming" | "predicted" | "finished";

// --- data/version.json --------------------------------------------------------
export interface Version {
  schema: number;
  hash: string;
  exported_at_utc: string; // when the data was exported (ISO, UTC, "Z")
  files: number;
}

// --- data/index.json ----------------------------------------------------------
export interface IndexRace {
  round: number;
  name: string;
  race_date: string; // "2026-11-08"
  race_start_utc: string | null;
  status: RaceStatus;
  live_stages: Stage[];
  backtest_stages: Stage[];
  file: string | null; // "races/2026-20.json", or null if nothing logged yet
}

export interface Index {
  schema: number;
  seasons: { season: number; races: IndexRace[] }[];
  next_race: { season: number; round: number } | null;
}

// --- data/races/<season>-<round>.json -------------------------------------------
export interface Factor {
  feature: string;
  value: number | null;
  contribution: number | null; // SHAP: + pushes the predicted position back, - forward
}

export interface DriverPrediction {
  driver_id: string;
  team_id: string | null;
  predicted_rank: number;
  anchor_position: number | null;
  expected_position: number | null;
  mean_position: number | null;
  p_win: number;
  p_podium: number;
  p_points: number;
  p_dnf: number;
  factors: Factor[]; // biggest effect first
}

export interface Score {
  model: number;
  baseline: number | null;
  baseline_name: string | null;
}

export interface Run {
  run_id: number;
  stage: Stage;
  mode: Mode;
  as_of_utc: string; // the cutoff: the prediction knew nothing after this
  logged_at_utc: string; // when it was written to the log
  lineup_source: string | null;
  model_version: string;
  n_training_races: number | null;
  drivers: DriverPrediction[];
  scores: Record<string, Score> | null; // null until the race is scored
}

export interface ResultRow {
  driver_id: string;
  team_id: string;
  car_number: number | null;
  position: number;
  position_text: string;
  grid: number | null;
  points: number | null;
  status: string | null;
  laps: number | null;
}

export interface DriverInfo {
  name: string;
  code: string | null;
  provisional: boolean; // a debutant's temporary id (no real id yet)
  resolved_as: string | null; // who that turned out to be, after the race
}

export interface Race {
  schema: number;
  season: number;
  round: number;
  name: string;
  race_date: string;
  circuit: string;
  locality: string | null;
  country: string | null;
  sessions: Record<string, string | null>; // {"FP1": "2026-11-06T14:30:00Z", ...}
  status: RaceStatus;
  runs: Run[];
  result: ResultRow[] | null;
  drivers: Record<string, DriverInfo>;
  teams: Record<string, string>;
  feature_labels: Record<string, string>; // "driver_form_finish" -> "recent finishes (last 5)"
}

// --- data/scorecard.json --------------------------------------------------------
export interface ScorecardMetric {
  races: number;
  model: number | null;
  baseline: number | null;
  baseline_name: string | null;
  improvement: number | null; // share better than the baseline (+ = better)
  higher_is_better: boolean | null;
}

export interface RaceScores {
  season: number;
  round: number;
  name: string;
  metrics: Record<string, { model: number | null; baseline: number | null }>;
}

export interface CalibrationBin {
  from: number; // predicted chance from..to (the last includes 1)
  to: number;
  n: number; // drivers in this group
  predicted: number | null; // their average predicted chance
  observed: number | null; // how often it happened
}

export interface ScorecardGroup {
  season: number;
  mode: Mode;
  label: string; // must be shown with the numbers (never mix groups)
  stages: Partial<Record<Stage, Record<string, ScorecardMetric>>>;
  per_race: Partial<Record<Stage, RaceScores[]>>; // calendar order
  calibration: Partial<Record<Stage, { win: CalibrationBin[]; podium: CalibrationBin[] }>>;
}

export interface Scorecard {
  schema: number;
  groups: ScorecardGroup[]; // most important first: live (newest season first), then backtest
}
