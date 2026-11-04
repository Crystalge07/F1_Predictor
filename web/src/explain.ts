// One driver's SHAP breakdown, ready to show.
//
// The race file holds, per driver, every factor's push on the model's
// predicted finishing position (SHAP: negative = moved forward). They add
// up exactly:
//     predicted position = start + sum of every push
// where `start` is the model's average prediction (a typical driver). The
// file doesn't store `start`; it's the predicted position minus the pushes.

import type { DriverPrediction } from "./types";

export const TOP_FACTORS = 6;

export interface Push {
  feature: string | null; // null = "other factors" (the rest, combined)
  value: number | null;
  places: number; // negative = forward
}

export interface Breakdown {
  predicted: number; // the model's predicted position (expected_position)
  start: number; // a typical driver
  pushes: Push[]; // the biggest TOP_FACTORS, then "other factors" if any
}

export function breakdown(driver: DriverPrediction): Breakdown | null {
  if (driver.expected_position === null || driver.factors.length === 0) return null;
  const all = driver.factors.map((f) => ({ feature: f.feature, value: f.value, places: f.contribution ?? 0 }));
  // Already biggest-first in the file; sorted again so this never depends on it.
  all.sort((a, b) => Math.abs(b.places) - Math.abs(a.places) || a.feature.localeCompare(b.feature));
  const total = all.reduce((sum, p) => sum + p.places, 0);
  const pushes: Push[] = all.slice(0, TOP_FACTORS);
  const rest = all.slice(TOP_FACTORS);
  if (rest.length) pushes.push({ feature: null, value: null, places: rest.reduce((sum, p) => sum + p.places, 0) });
  return { predicted: driver.expected_position, start: driver.expected_position - total, pushes };
}
