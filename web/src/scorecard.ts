// What each scorecard measure means, and how it's shown.

import type { ScorecardMetric } from "./types";

export interface MetricInfo {
  key: string;
  name: string;
  explain: string;
  format: (value: number) => string;
}

const pct = (v: number) => `${Math.round(v * 100)}%`;

/** The measures in the summary table, in order. Direction ("lower is
 * better") comes from the file (higher_is_better), never assumed here. */
export const METRICS: MetricInfo[] = [
  { key: "position_mae", name: "Places off", format: (v) => v.toFixed(2),
    explain: "How many places each driver's predicted position was from where they finished, on average. Lower is better." },
  { key: "brier_podium", name: "Podium Brier score", format: (v) => v.toFixed(3),
    explain: "How good the podium chances were: 0 is perfect, and a confident wrong call costs more than a cautious one. Lower is better." },
  { key: "brier_win", name: "Win Brier score", format: (v) => v.toFixed(3),
    explain: "The same for the win chances. Lower is better." },
  { key: "winner_hit", name: "Winner picked", format: pct,
    explain: "How often the predicted winner won." },
  { key: "podium_hits", name: "Podium places right", format: pct,
    explain: "Of the three drivers predicted on the podium, the share who finished there." },
];

/** Below this many races, the page says the numbers can't be judged yet. */
export const SMALL_SAMPLE = 10;

export const BASELINE_NAMES: Record<string, string> = {
  championship_order: "championship order",
  practice_order: "practice order",
  grid_order: "starting grid",
};

/** "7% better" / "3% worse" / "the same", with ▲ / ▼, from the file's
 * improvement (positive = better, whichever way the measure runs). */
export function comparison(metric: ScorecardMetric): { text: string; better: boolean | null } {
  if (metric.improvement === null) return { text: "", better: null };
  const size = Math.round(Math.abs(metric.improvement) * 100);
  if (size === 0) return { text: "about the same", better: null };
  return metric.improvement > 0 ? { text: `▲ ${size}% better`, better: true } : { text: `▼ ${size}% worse`, better: false };
}
