// Why: the factors behind one driver's predicted position, for any race
// with live predictions, at any stage. The choice of race, stage and driver
// lives in the address (#/why?race=2026-20&stage=practice&driver=...), so an
// explanation can be shared as a link.

import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useRace, useSiteData } from "../data";
import { STAGE_NAMES } from "../format";
import { driverName, formatFactorValue, formatPlaces, formatSignedPlaces, liveRuns } from "../race";
import { breakdown, type Breakdown } from "../explain";
import type { Index, Race, Stage } from "../types";
import { ErrorBox, Loading } from "../components/Status";
import { FactorChart } from "../components/FactorChart";

interface RaceOption {
  key: string; // "2026-20"
  file: string;
  label: string;
}

/** Every race with a live prediction, newest first. */
function raceOptions(index: Index): RaceOption[] {
  return index.seasons
    .flatMap((s) =>
      s.races
        .filter((r) => r.file && r.live_stages.length > 0)
        .map((r) => ({ key: `${s.season}-${String(r.round).padStart(2, "0")}`, file: r.file!, label: `${s.season} R${r.round} · ${r.name}` })),
    )
    .reverse();
}

export function Why() {
  const site = useSiteData();
  const [params, setParams] = useSearchParams();
  const options = site.state === "ready" ? raceOptions(site.data.index) : [];
  // Default race: the next race if it has a prediction, else the latest one.
  const next = site.state === "ready" ? site.data.index.next_race : null;
  const nextKey = next ? `${next.season}-${String(next.round).padStart(2, "0")}` : null;
  const chosen =
    options.find((o) => o.key === params.get("race")) ?? options.find((o) => o.key === nextKey) ?? options[0];
  const race = useRace(chosen?.file ?? null);

  // Change one choice in the address, keeping the others (a new race resets stage and driver).
  const choose = (changes: Record<string, string | null>) => {
    const nextParams = new URLSearchParams(params);
    for (const [key, value] of Object.entries(changes)) {
      if (value === null) nextParams.delete(key);
      else nextParams.set(key, value);
    }
    setParams(nextParams, { replace: true });
  };

  if (site.state === "loading") return <Loading />;
  if (site.state === "error") return <ErrorBox message={site.message} />;
  return (
    <section>
      <h1>Why</h1>
      <p className="muted">
        What moved each driver's predicted finishing position, factor by factor. The pushes add up exactly to the
        prediction.
      </p>
      {!chosen ? (
        <p>No live prediction has been logged yet.</p>
      ) : (
        <>
          <div className="pickers">
            <label>
              Race
              <select value={chosen.key} onChange={(e) => choose({ race: e.target.value, stage: null, driver: null })}>
                {options.map((o) => (
                  <option key={o.key} value={o.key}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {!race || race.state === "loading" ? (
            <Loading what="the race" />
          ) : race.state === "error" ? (
            <ErrorBox message={race.message} />
          ) : (
            <RaceWhy race={race.data} stage={params.get("stage")} driver={params.get("driver")} choose={choose} />
          )}
        </>
      )}
    </section>
  );
}

function RaceWhy({ race, stage, driver, choose }: {
  race: Race;
  stage: string | null;
  driver: string | null;
  choose: (changes: Record<string, string | null>) => void;
}) {
  const runs = liveRuns(race);
  const run = runs.find((r) => r.stage === stage) ?? runs[runs.length - 1]; // default: the latest stage
  // Default driver: the predicted winner (rank 1). A driver not in this
  // stage's prediction (e.g. a debutant's temporary id) falls back to it too.
  const chosen = run.drivers.find((d) => d.driver_id === driver) ?? run.drivers[0];
  const data = breakdown(chosen);
  const [showTable, setShowTable] = useState(false);

  return (
    <>
      <div className="tabs" role="tablist" aria-label="Prediction stage">
        {runs.map((r) => (
          <button
            key={r.stage}
            type="button"
            role="tab"
            className="tab"
            aria-selected={r.stage === run.stage}
            onClick={() => choose({ stage: r.stage as Stage })}
          >
            {STAGE_NAMES[r.stage]}
          </button>
        ))}
      </div>
      <div className="pickers">
        <label>
          Driver
          <select value={chosen.driver_id} onChange={(e) => choose({ driver: e.target.value })}>
            {run.drivers.map((d) => (
              <option key={d.driver_id} value={d.driver_id}>
                {d.predicted_rank}. {driverName(race, d.driver_id)}
                {race.drivers[d.driver_id]?.provisional ? " (debut)" : ""}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="card">
        {!data ? (
          <p className="muted">No factor breakdown was logged for this driver.</p>
        ) : (
          <>
            <p className="summary">{summary(race, chosen.driver_id, data)}</p>
            <FactorChart data={data} labels={race.feature_labels} />
            <div className="chart-controls">
              <button type="button" className="link-button" onClick={() => setShowTable((v) => !v)} aria-expanded={showTable}>
                {showTable ? "Hide the table" : "Show as a table"}
              </button>
            </div>
            {showTable && <FactorTable race={race} data={data} />}
          </>
        )}
        <p className="muted small caveat">
          This explains the model's predicted finishing position. The win and podium chances on the Next race page also
          include simulated race randomness and retirements{run.stage === "post_quali" ? ", and a blend with the starting grid" : ""},
          so these factors explain where the model rates the driver, not the exact percentages.
        </p>
      </div>
    </>
  );
}

/** "Max Verstappen: predicted P3.5, against P9.6 for a typical driver.
 * Biggest reasons: starting position: P5, 2.3 places forward; ..." */
export function summary(race: Race, driverId: string, data: Breakdown): string {
  const reasons = data.pushes
    .filter((p) => p.feature !== null && Math.abs(p.places) >= 0.05)
    .slice(0, 3)
    .map((p) => {
      const label = race.feature_labels[p.feature!] ?? p.feature!;
      return `${label}: ${formatFactorValue(p.feature!, p.value)}, ${formatPlaces(p.places)} ${p.places < 0 ? "forward" : "back"}`;
    });
  const head = `${driverName(race, driverId)}: predicted P${data.predicted.toFixed(1)}, against P${data.start.toFixed(1)} for a typical driver.`;
  return reasons.length ? `${head} Biggest reasons: ${reasons.join("; ")}.` : head;
}

function FactorTable({ race, data }: { race: Race; data: Breakdown }) {
  return (
    <table className="predictions" style={{ marginTop: 12 }}>
      <caption className="visually-hidden">Factors and their effect, in places</caption>
      <thead>
        <tr>
          <th scope="col" className="left">Factor</th>
          <th scope="col">Value</th>
          <th scope="col">Effect</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td className="left">A typical driver (the starting point)</td>
          <td />
          <td>P{data.start.toFixed(1)}</td>
        </tr>
        {data.pushes.map((p) => (
          <tr key={p.feature ?? "other"}>
            <td className="left">{p.feature === null ? "Other factors combined" : race.feature_labels[p.feature] ?? p.feature}</td>
            <td>{p.feature === null ? "" : formatFactorValue(p.feature, p.value)}</td>
            <td>{formatSignedPlaces(p.places, 2)}</td>
          </tr>
        ))}
        <tr>
          <td className="left"><strong>Predicted position</strong></td>
          <td />
          <td><strong>P{data.predicted.toFixed(1)}</strong></td>
        </tr>
      </tbody>
    </table>
  );
}
