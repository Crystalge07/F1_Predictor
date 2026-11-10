// The race weekend as a series of steps, for the tracker on the Next race page.
//
// A weekend produces four things, in this order:
//
//   1. Pre-weekend prediction   made once the previous race's result is in,
//                               logged before the first session (FP1)
//   2. After-practice           made once practice is over (on a sprint
//                               weekend: practice, sprint qualifying and the
//                               sprint), logged before qualifying
//   3. After-qualifying         made once the starting grid is known,
//                               logged before the race starts
//   4. Result + scoring         the official result, and every prediction
//                               above scored against it
//
// Everything here is worked out in the browser from the race file (which
// stages are logged, the session times) and the clock. Nothing new is
// exported: these are the same rules the pipeline's planner follows
// (src/f1predictor/pipeline/plan.py), just restated for display.
//
// Two clocks matter:
//   * `now`        the viewer's clock: is a session in the past yet?
//   * `dataTime`   when the data files were last exported. A prediction is
//                  only called "missed" if the data was exported AFTER its
//                  cutoff and still has no run. If the site simply hasn't
//                  been updated since (the pipeline runs hourly), we don't
//                  know yet, so it's shown as still due rather than missed.

import type { Race, Stage } from "./types";
import { liveRuns } from "./race";

export type StepKey = Stage | "result";

/**
 *  done      it happened (prediction logged / race scored)
 *  current   the next thing to happen: being made now, or the next one due
 *  upcoming  later in the weekend
 *  missed    its cutoff passed and no prediction was logged (rare: the
 *            pipeline was down). Never filled in afterwards, on purpose.
 */
export type StepState = "done" | "current" | "upcoming" | "missed";

export interface WeekendStep {
  key: StepKey;
  label: string; // "After practice"
  state: StepState;
  detail: string; // one short line under the label: when it was / will be
  time: string | null; // the time `detail` is about (ISO, UTC), if any
}

// How long sessions last, to say roughly when the next step can start.
// Same values as plan.SESSION_LENGTH (in minutes).
const SESSION_MINUTES: Record<string, number> = { FP1: 60, FP2: 60, FP3: 60, SQ: 45, S: 60, Q: 60, R: 120 };

// The sessions that feed the after-practice prediction (as_of.PRE_QUALI_SESSION_TYPES).
const PRE_QUALI = ["FP1", "FP2", "FP3", "SQ", "S"];

const LABELS: Record<StepKey, string> = {
  pre_weekend: "Pre-weekend",
  practice: "After practice",
  post_quali: "After qualifying",
  result: "Result + scoring",
};

const ms = (iso: string) => new Date(iso).getTime();
const addMinutes = (iso: string, minutes: number) => new Date(ms(iso) + minutes * 60_000).toISOString();

/** The race's sessions that have a time, as [type, start] sorted by start. */
function timedSessions(race: Race): [string, string][] {
  return Object.entries(race.sessions)
    .filter((entry): entry is [string, string] => entry[1] !== null)
    .sort((a, b) => ms(a[1]) - ms(b[1]));
}

/** When each stage closes (its cutoff), and roughly when it can first be made. */
function stageTimes(race: Race): Record<Stage, { opens: string | null; cutoff: string | null }> {
  const sessions = timedSessions(race);
  const start = (type: string) => race.sessions[type] ?? null;
  const first = sessions.length ? sessions[0][1] : null;
  const quali = start("Q");
  const raceStart = start("R");
  // The practice stage closes at qualifying, or at the race if qualifying
  // isn't on the calendar (cancelled): features/as_of.stage_cutoff.
  const practiceCutoff = quali ?? raceStart;
  // It can be made once the last session before that cutoff has ended.
  const before = sessions.filter(
    ([type, at]) => PRE_QUALI.includes(type) && (!practiceCutoff || ms(at) < ms(practiceCutoff)),
  );
  const lastPractice = before.length ? before[before.length - 1] : null;
  return {
    pre_weekend: { opens: null, cutoff: first }, // opens: when the previous result is in
    practice: {
      opens: lastPractice ? addMinutes(lastPractice[1], SESSION_MINUTES[lastPractice[0]] ?? 60) : null,
      cutoff: practiceCutoff,
    },
    post_quali: { opens: quali ? addMinutes(quali, SESSION_MINUTES.Q) : null, cutoff: raceStart },
  };
}

/** What each stage's cutoff is called in a sentence. */
const CUTOFF_NAME: Record<Stage, (race: Race) => string> = {
  pre_weekend: () => "practice starts",
  practice: (race) => (race.sessions.Q ? "qualifying" : "the race"),
  post_quali: () => "the race",
};

/** The weekend's four steps, each with its state and one line of detail. */
export function weekendSteps(race: Race, now: Date, dataTime: Date): WeekendStep[] {
  const runs = liveRuns(race);
  const times = stageTimes(race);
  const steps: WeekendStep[] = [];

  for (const stage of ["pre_weekend", "practice", "post_quali"] as Stage[]) {
    const run = runs.find((r) => r.stage === stage);
    const { opens, cutoff } = times[stage];
    const cutoffName = CUTOFF_NAME[stage](race);
    if (run) {
      steps.push({ key: stage, label: LABELS[stage], state: "done", detail: "Logged", time: run.logged_at_utc });
    } else if (cutoff && dataTime.getTime() >= ms(cutoff)) {
      // The data is newer than the cutoff and the run still isn't there.
      steps.push({ key: stage, label: LABELS[stage], state: "missed", detail: `Not logged before ${cutoffName}`, time: cutoff });
    } else if (stage === "post_quali" && !race.sessions.Q && cutoff && now.getTime() < ms(cutoff)) {
      // No qualifying on the calendar: there's no grid to predict from.
      steps.push({ key: stage, label: LABELS[stage], state: "upcoming", detail: "No qualifying on the calendar", time: null });
    } else {
      // Not logged yet. "current" vs "upcoming" is decided below, once we
      // know which step is the first one still to happen. The time shown is
      // always the deadline (the cutoff), so it can't be misread.
      const ready = !opens || now.getTime() >= ms(opens);
      steps.push({
        key: stage,
        label: LABELS[stage],
        state: "upcoming",
        // Ready: its inputs are in, so the next pipeline run makes it.
        detail: ready ? `Being made, due before ${cutoffName}` : `Due before ${cutoffName}`,
        time: cutoff,
      });
    }
  }

  // The result. The page switches to the next race once this one starts, so
  // it's normally seen as "upcoming"; the other states are for completeness
  // (and for the "how we did" card, which reuses them).
  const raceStart = race.sessions.R ?? null;
  if (race.status === "finished") {
    steps.push({ key: "result", label: LABELS.result, state: "done", detail: "Result in, predictions scored", time: null });
  } else if (raceStart && now.getTime() >= ms(raceStart)) {
    steps.push({ key: "result", label: LABELS.result, state: "upcoming", detail: "Waiting for the official result", time: null });
  } else {
    steps.push({ key: "result", label: LABELS.result, state: "upcoming", detail: "Once the official result is in", time: null });
  }

  // The first step that hasn't happened (and wasn't missed) is "current".
  const next = steps.find((s) => s.state === "upcoming");
  if (next) next.state = "current";
  return steps;
}

/**
 * "Still to come", as one sentence of the sessions and steps left, in time
 * order: "Qualifying Sat 18:00 → final prediction before the race Sun 17:00
 * → scored once the result is in." Empty when the weekend is over.
 */
export function stillToCome(race: Race, now: Date): { what: string; time: string | null }[] {
  const items: { what: string; time: string | null }[] = [];
  const runs = liveRuns(race);
  const logged = (stage: Stage) => runs.some((r) => r.stage === stage);
  const times = stageTimes(race);
  for (const [type, at] of timedSessions(race)) {
    if (ms(at) <= now.getTime()) continue;
    // Each prediction lands just before the session that is its cutoff.
    if (type === "R" && !logged("post_quali") && race.sessions.Q) items.push({ what: "Final prediction (after qualifying)", time: null });
    if (at === times.practice.cutoff && !logged("practice")) items.push({ what: "Prediction after practice", time: null });
    if (at === times.pre_weekend.cutoff && !logged("pre_weekend")) items.push({ what: "Pre-weekend prediction", time: null });
    items.push({ what: SESSION_NAMES[type] ?? type, time: at });
  }
  if (race.status !== "finished") items.push({ what: "Result and scoring", time: null });
  return items;
}

export const SESSION_NAMES: Record<string, string> = {
  FP1: "Practice 1", FP2: "Practice 2", FP3: "Practice 3", SQ: "Sprint qualifying",
  S: "Sprint", Q: "Qualifying", R: "Race",
};

// --- The "how we did" card ----------------------------------------------------

/** How long after the race starts the home page keeps showing its result
 * card: the race itself (~2h) plus two days. */
export const RECENT_RESULT_HOURS = 50;

/** The race that started most recently, if it started within
 * RECENT_RESULT_HOURS of `now` (any season): {season, round, file, ...}. */
export function recentRace<R extends { race_start_utc: string | null }>(
  seasons: { season: number; races: R[] }[],
  now: Date,
): (R & { season: number }) | null {
  let best: (R & { season: number }) | null = null;
  for (const season of seasons) {
    for (const race of season.races) {
      if (!race.race_start_utc) continue;
      const age = now.getTime() - ms(race.race_start_utc);
      if (age < 0 || age > RECENT_RESULT_HOURS * 3_600_000) continue;
      if (!best || ms(race.race_start_utc) > ms(best.race_start_utc!)) best = { ...race, season: season.season };
    }
  }
  return best;
}
