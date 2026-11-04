// Reading a race file: helpers shared by the pages.

import type { Race, Run, Stage } from "./types";

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
