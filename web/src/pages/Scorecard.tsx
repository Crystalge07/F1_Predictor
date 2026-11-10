// The stats side of the Results tab (#/results/stats): how good the
// predictions were, against the simple baseline of each stage. One scorecard
// at a time (2027 live, the 2026 rehearsal, a backtest...): they're never
// added together, and each shows its label. The live one comes first in the
// file, so it's the default; backtests are picked from the menu.

import { useState } from "react";
import { useSiteData } from "../data";
import { STAGE_NAMES } from "../format";
import { STAGE_ORDER } from "../race";
import { METRICS, SMALL_SAMPLE } from "../scorecard";
import type { ScorecardGroup, Stage } from "../types";
import { ErrorBox, Loading } from "../components/Status";
import { ScorecardTable } from "../components/ScorecardTable";
import { RaceBars } from "../components/RaceBars";
import { CalibrationChart } from "../components/CalibrationChart";

const groupKey = (g: ScorecardGroup) => `${g.season}-${g.mode}`;

export function Stats() {
  const site = useSiteData();
  const [chosen, setChosen] = useState<string | null>(null);
  if (site.state === "loading") return <Loading />;
  if (site.state === "error") return <ErrorBox message={site.message} />;

  const groups = site.data.scorecard.groups;
  // The file lists the most important scorecard first (live before
  // backtest, newest season first), so that's the default.
  const group = groups.find((g) => groupKey(g) === chosen) ?? groups[0];
  return (
    <>
      <p className="muted">How close the predictions came, against the simple baseline each stage has to beat.</p>
      {!group ? (
        <p>Nothing has been scored yet: the first scores appear after the first predicted race.</p>
      ) : (
        <>
          {groups.length > 1 && (
            <div className="pickers">
              <label>
                Scorecard
                <select value={groupKey(group)} onChange={(e) => setChosen(e.target.value)}>
                  {groups.map((g) => (
                    <option key={groupKey(g)} value={groupKey(g)}>
                      {g.season} {g.mode}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          )}
          <GroupView key={groupKey(group)} group={group} />
        </>
      )}
    </>
  );
}

function GroupView({ group }: { group: ScorecardGroup }) {
  const stages = STAGE_ORDER.filter((s): s is Stage => group.stages[s] !== undefined);
  const races = Math.max(0, ...stages.map((s) => Object.values(group.stages[s]!)[0]?.races ?? 0));
  const [stage, setStage] = useState<Stage>(stages[stages.length - 1]);
  const [outcome, setOutcome] = useState<"podium" | "win">("podium");
  const calibration = group.calibration[stage];

  return (
    <>
      <p className={`banner ${group.mode === "live" ? "is-live" : "is-backtest"}`} role="note">
        {group.label}
      </p>
      {races < SMALL_SAMPLE && (
        <p className="small-sample" role="note">
          Only {races} {races === 1 ? "race" : "races"} so far: too few to judge. One lucky or unlucky weekend moves
          these numbers a lot.
        </p>
      )}

      <div className="card">
        <h2>Model against baseline</h2>
        <ScorecardTable group={group} />
        <dl className="definitions">
          {METRICS.map((m) => (
            <div key={m.key}>
              <dt>{m.name}</dt>
              <dd>{m.explain}</dd>
            </div>
          ))}
          <div>
            <dt>Driver list misses</dt>
            <dd>Starters a stage's prediction didn't have, and drivers it had who didn't start.</dd>
          </div>
        </dl>
      </div>

      <div className="tabs" role="tablist" aria-label="Stage">
        {stages.map((s) => (
          <button key={s} type="button" role="tab" className="tab" aria-selected={s === stage} onClick={() => setStage(s)}>
            {STAGE_NAMES[s]}
          </button>
        ))}
      </div>

      <div className="card">
        <h2>Race by race</h2>
        <p className="muted small">Places off, the model's predicted order against the baseline's, for each race.</p>
        <RaceBars races={group.per_race[stage] ?? []} />
      </div>

      <div className="card">
        <h2>Calibration</h2>
        <p className="muted small">
          When the model gave a chance of X%, how often did it happen? On the dashed line is perfect; below it, the
          model was too sure; above it, too cautious.
        </p>
        <div className="tabs compact" role="tablist" aria-label="Chances of">
          {(["podium", "win"] as const).map((o) => (
            <button key={o} type="button" role="tab" className="tab" aria-selected={o === outcome} onClick={() => setOutcome(o)}>
              {o === "podium" ? "Podium chances" : "Win chances"}
            </button>
          ))}
        </div>
        {calibration ? (
          <CalibrationChart key={`${stage}-${outcome}`} bins={calibration[outcome]} what={outcome} />
        ) : (
          <p className="muted">No scored predictions yet.</p>
        )}
      </div>
    </>
  );
}
