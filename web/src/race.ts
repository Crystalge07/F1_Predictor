// Reading a race file: helpers shared by the pages.

import type { Race, Run, Stage } from "./types";

/** "2026-05": a race's key in addresses and file names. */
export function raceKey(season: number, round: number) {
  return `${season}-${String(round).padStart(2, "0")}`;
}

export const STAGE_ORDER: Stage[] = ["pre_weekend", "practice", "post_quali"];

/** The race's LIVE runs, in weekend order (pre-weekend -> after qualifying).
 * Backtest runs (replays of past races, in-sample) are never shown as
 * predictions: only the Scorecard and Past races pages, labelled, use them. */
export function liveRuns(race: Race): Run[] {
  return race.runs
    .filter((run) => run.mode === "live")
    .sort((a, b) => STAGE_ORDER.indexOf(a.stage) - STAGE_ORDER.indexOf(b.stage));
}

/** "Max Verstappen"; falls back to the id if a name is missing. */
export function driverName(race: Race, driverId: string): string {
  return race.drivers[driverId]?.name ?? driverId;
}

/** "VER", or the first three letters of the family name (debutants have no code yet). */
export function driverCode(race: Race, driverId: string): string {
  const info = race.drivers[driverId];
  if (info?.code) return info.code;
  const family = (info?.name ?? driverId).split(" ").pop() ?? driverId;
  return family.slice(0, 3).toUpperCase();
}

export function teamName(race: Race, teamId: string | null): string {
  return (teamId && race.teams[teamId]) || "Team not known yet";
}

/** A probability as a percentage: "37%", "<1%" for tiny non-zero chances, "0%". */
export function formatPct(p: number): string {
  if (p > 0 && p < 0.005) return "<1%";
  return `${Math.round(p * 100)}%`;
}

/** What each stage's anchor (the order the model starts from) is. */
export const ANCHOR = {
  pre_weekend: { column: "Champ.", title: "Championship position before the weekend" },
  practice: { column: "Practice", title: "Position in the practice pace order" },
  post_quali: { column: "Grid", title: "Starting grid position" },
} as const;

/** The session each stage's cutoff is: the prediction knew nothing after it began. */
export const CUTOFF_SESSION = {
  pre_weekend: "the first session",
  practice: "qualifying",
  post_quali: "the race",
} as const;

/** A factor's value in its own unit, for the Why page: "avg P2.4", "1.23%",
 * "365 pts". null (no data: e.g. a debutant's history) -> "no data". */
export function formatFactorValue(feature: string, value: number | null): string {
  if (value === null) return "no data";
  if (feature === "pit_lane_start") return value ? "yes" : "no";
  if (feature === "grid") return `P${Math.round(value)}`;
  if (feature.endsWith("_pct")) return `${value.toFixed(2)}%`;
  if (feature.endsWith("_per_race")) return value.toFixed(1);
  if (feature.endsWith("_points")) return `${Math.round(value)} pts`;
  // The rest are average positions (form, team season, track record).
  return `avg P${value.toFixed(1)}`;
}

/** "a factor's places": 1.97 -> "2.0 places", 1 -> "1.0 place". */
export function formatPlaces(places: number): string {
  const rounded = Math.abs(places).toFixed(1);
  return `${rounded} ${rounded === "1.0" ? "place" : "places"}`;
}

/** A push in places, signed: "−3.1", "+0.4"; "0.0" when it rounds to nothing
 * (so a tiny push never reads as a "−0.0" sign error). decimals: 1 or 2. */
export function formatSignedPlaces(places: number, decimals = 1): string {
  const text = Math.abs(places).toFixed(decimals);
  if (Number(text) === 0) return (0).toFixed(decimals);
  return `${places < 0 ? "−" : "+"}${text}`;
}
