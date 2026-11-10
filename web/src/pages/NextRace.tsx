// Next race: the home page.
//
//   1. For ~2 days after a race: a "how we did" card for it (RecentResult).
//   2. The next race, with the weekend tracker (which predictions are logged,
//      which is next, and what's still to come), then the latest prediction,
//      tabs for the earlier stages, and how the win chances moved.

import { useState } from "react";
import { useRace, useSiteData } from "../data";
import { formatUtc, STAGE_NAMES } from "../format";
import { CUTOFF_SESSION, liveRuns, raceKey } from "../race";
import type { Race, Stage } from "../types";
import { recentRace } from "../weekend";
import { ErrorBox, Loading } from "../components/Status";
import { PredictionTable } from "../components/PredictionTable";
import { OddsChart } from "../components/OddsChart";
import { WeekendTracker } from "../components/WeekendTracker";
import { RecentResult } from "../components/RecentResult";

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
  // The viewer's clock (which sessions are over) and the data's clock (when
  // the files were exported): see weekend.ts for why both.
  const now = new Date();
  const dataTime = new Date(site.data.version.exported_at_utc);
  const recent = recentRace(site.data.index.seasons, now);
  return (
    <>
      {recent?.file && <RecentResult file={recent.file} raceKey={raceKey(recent.season, recent.round)} />}
      <NextRaceBody next={next} entry={entry} race={race} now={now} dataTime={dataTime} />
    </>
  );
}

interface BodyProps {
  next: { season: number; round: number } | null;
  entry: { round: number; name: string; race_start_utc: string | null; file: string | null } | undefined;
  race: ReturnType<typeof useRace>;
  now: Date;
  dataTime: Date;
}

function NextRaceBody({ next, entry, race, now, dataTime }: BodyProps) {
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
  return <RaceView key={entry.file} race={race.data} now={now} dataTime={dataTime} />;
}

function RaceView({ race, now, dataTime }: { race: Race; now: Date; dataTime: Date }) {
  const runs = liveRuns(race);
  // The latest stage is shown first; the tabs go back through the weekend.
  const [stage, setStage] = useState<Stage | null>(runs.length ? runs[runs.length - 1].stage : null);
  const run = runs.find((r) => r.stage === stage) ?? runs[runs.length - 1];
  const latest = runs.length ? runs[runs.length - 1] : null;

  return (
    <section>
      <p className="eyebrow">{race.season} · Round {race.round}</p>
      <h1>{race.name}</h1>
      <p className="muted race-meta">
        {race.circuit}
        {race.locality ? `, ${race.locality}` : ""}
        {race.country ? `, ${race.country}` : ""}
      </p>
      <WeekendTracker race={race} now={now} dataTime={dataTime} selected={run?.stage ?? null} onSelect={setStage} />

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
            <h2 className="stage-heading">
              {run === latest ? "Current prediction" : "Earlier prediction"}: {STAGE_NAMES[run.stage].toLowerCase()}
            </h2>
            <p className="muted small" style={{ marginTop: 0 }}>
              Logged {formatUtc(run.logged_at_utc)}, before {CUTOFF_SESSION[run.stage]} ({formatUtc(run.as_of_utc)}).
              Never edited since. Chances come from 10,000 simulated races. Click a driver to see why.
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
