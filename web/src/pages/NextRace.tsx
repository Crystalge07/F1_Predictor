// Next race: the latest live prediction for the next race, every stage
// logged so far, and how the win chances moved between them.

import { useState } from "react";
import { useRace, useSiteData } from "../data";
import { formatUtc, STAGE_NAMES } from "../format";
import { CUTOFF_SESSION, liveRuns } from "../race";
import type { Race, Stage } from "../types";
import { ErrorBox, Loading } from "../components/Status";
import { PredictionTable } from "../components/PredictionTable";
import { OddsChart } from "../components/OddsChart";

const SESSION_NAMES: Record<string, string> = {
  FP1: "Practice 1", FP2: "Practice 2", FP3: "Practice 3", SQ: "Sprint qualifying",
  S: "Sprint", Q: "Qualifying", R: "Race",
};

export function NextRace() {
  const site = useSiteData();
  const next = site.state === "ready" ? site.data.index.next_race : null;
  const entry =
    site.state === "ready" && next
      ? site.data.index.seasons.find((s) => s.season === next.season)?.races.find((r) => r.round === next.round)
      : undefined;
  const race = useRace(entry?.file ?? null);

  if (site.state === "loading") return <Loading />;
  if (site.state === "error") return <ErrorBox message={site.message} />;
  if (!next || !entry) {
    return (
      <section>
        <h1>Next race</h1>
        <p className="muted">No race is scheduled yet. The next season's calendar appears here once it's published.</p>
      </section>
    );
  }
  if (!race) {
    // On the calendar, but nothing logged yet (no file).
    return (
      <section>
        <p className="eyebrow">{next.season} · Round {entry.round}</p>
        <h1>{entry.name}</h1>
        <p className="muted">Race start: {formatUtc(entry.race_start_utc)}</p>
        <p>No prediction has been logged yet. The first one is made once the previous race's result is in, before practice starts.</p>
      </section>
    );
  }
  if (race.state === "loading") return <Loading what="the race" />;
  if (race.state === "error") return <ErrorBox message={race.message} />;
  // key: a different race gets a fresh view (its own latest stage selected).
  return <RaceView key={entry.file} race={race.data} />;
}

function RaceView({ race }: { race: Race }) {
  const runs = liveRuns(race);
  // The latest stage is shown first; the tabs go back through the weekend.
  const [stage, setStage] = useState<Stage | null>(runs.length ? runs[runs.length - 1].stage : null);
  const run = runs.find((r) => r.stage === stage) ?? runs[runs.length - 1];
  const sessions = Object.entries(race.sessions).filter(([, start]) => start);

  return (
    <section>
      <p className="eyebrow">{race.season} · Round {race.round}</p>
      <h1>{race.name}</h1>
      <p className="muted race-meta">
        {race.circuit}
        {race.locality ? `, ${race.locality}` : ""}
        {race.country ? `, ${race.country}` : ""}
      </p>
      <ul className="sessions" aria-label="Sessions (UTC)">
        {sessions.map(([type, start]) => (
          <li key={type}>
            <strong>{SESSION_NAMES[type] ?? type}</strong> {formatUtc(start)}
          </li>
        ))}
      </ul>

      {!run ? (
        <p>No prediction has been logged yet. The first one is made before practice starts.</p>
      ) : (
        <>
          <div className="tabs" role="tablist" aria-label="Prediction stage">
            {runs.map((r) => (
              <button
                key={r.stage}
                type="button"
                role="tab"
                className="tab"
                id={`tab-${r.stage}`}
                aria-selected={r.stage === run.stage}
                aria-controls="stage-panel"
                onClick={() => setStage(r.stage)}
              >
                {STAGE_NAMES[r.stage]}
              </button>
            ))}
          </div>
          <div className="card" role="tabpanel" id="stage-panel" aria-labelledby={`tab-${run.stage}`}>
            <p className="muted small" style={{ marginTop: 0 }}>
              Logged {formatUtc(run.logged_at_utc)}, before {CUTOFF_SESSION[run.stage]} ({formatUtc(run.as_of_utc)}).
              Never edited since. Chances come from 10,000 simulated races.
            </p>
            <PredictionTable race={race} run={run} />
          </div>

          <div className="card">
            <h2>How the win chances moved</h2>
            {runs.length < 2 ? (
              <p className="muted">
                This chart appears once a second prediction is logged for this race (after practice), so you can see how
                the chances moved.
              </p>
            ) : (
              <OddsChart race={race} runs={runs} />
            )}
          </div>
        </>
      )}
    </section>
  );
}
