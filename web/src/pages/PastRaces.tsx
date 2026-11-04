// Past races: every race with a result, and for each one, what every stage
// predicted next to what actually happened, plus that race's scores.
//
//   #/races           the list, newest first
//   #/races/2026-19   one race

import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useRace, useSiteData } from "../data";
import { STAGE_NAMES } from "../format";
import { driverName, STAGE_ORDER, teamName } from "../race";
import type { Index, Mode, Race, Run } from "../types";
import { ErrorBox, Loading } from "../components/Status";

// Column headers on a phone, where the full names don't fit side by side.
const SHORT_STAGE = { pre_weekend: "Pre", practice: "Practice", post_quali: "Quali" } as const;

export function raceKey(season: number, round: number) {
  return `${season}-${String(round).padStart(2, "0")}`;
}

/** Races with a result and a logged prediction, newest first. */
function finishedRaces(index: Index) {
  return index.seasons
    .flatMap((s) => s.races.filter((r) => r.status === "finished" && r.file).map((r) => ({ season: s.season, ...r })))
    .reverse();
}

export function PastRaces() {
  const site = useSiteData();
  if (site.state === "loading") return <Loading />;
  if (site.state === "error") return <ErrorBox message={site.message} />;
  const races = finishedRaces(site.data.index);
  return (
    <section>
      <h1>Past races</h1>
      <p className="muted">Every logged prediction, stage by stage, next to the real result.</p>
      {races.length === 0 ? (
        <p>No predicted race has finished yet.</p>
      ) : (
        <ul className="race-list">
          {races.map((r) => (
            <li key={raceKey(r.season, r.round)}>
              <Link to={`/races/${raceKey(r.season, r.round)}`}>
                <span className="race-list-name">
                  {r.season} R{r.round} · {r.name}
                </span>
                <span className="race-list-badges">
                  {r.live_stages.length > 0 && <span className="badge is-live">live</span>}
                  {r.backtest_stages.length > 0 && <span className="badge">backtest</span>}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export function PastRace() {
  const { key } = useParams();
  const site = useSiteData();
  const entry =
    site.state === "ready"
      ? site.data.index.seasons.flatMap((s) => s.races.map((r) => ({ season: s.season, ...r }))).find((r) => raceKey(r.season, r.round) === key)
      : undefined;
  const race = useRace(entry?.file ?? null);

  if (site.state === "loading") return <Loading />;
  if (site.state === "error") return <ErrorBox message={site.message} />;
  if (!entry || !race) {
    return (
      <section>
        <h1>Race not found</h1>
        <p className="muted">
          There's no logged prediction for that race. <Link to="/races">All past races</Link>
        </p>
      </section>
    );
  }
  if (race.state === "loading") return <Loading what="the race" />;
  if (race.state === "error") return <ErrorBox message={race.message} />;
  return <RaceReview key={key} race={race.data} />;
}

function RaceReview({ race }: { race: Race }) {
  const modes = (["live", "backtest"] as Mode[]).filter((m) => race.runs.some((r) => r.mode === m));
  const [mode, setMode] = useState<Mode>(modes[0]);
  const runs = race.runs
    .filter((r) => r.mode === mode)
    .sort((a, b) => STAGE_ORDER.indexOf(a.stage) - STAGE_ORDER.indexOf(b.stage));

  return (
    <section>
      <p className="eyebrow">
        <Link to="/races">Past races</Link> · {race.season} · Round {race.round}
      </p>
      <h1>{race.name}</h1>
      <p className="muted race-meta">{race.circuit}</p>

      {modes.length > 1 && (
        <div className="tabs" role="tablist" aria-label="Which predictions">
          {modes.map((m) => (
            <button key={m} type="button" role="tab" className="tab" aria-selected={m === mode} onClick={() => setMode(m)}>
              {m === "live" ? "Live" : "Backtest"}
            </button>
          ))}
        </div>
      )}
      {mode === "backtest" && (
        <p className="banner is-backtest" role="note">
          Backtest: a replay of this race after the fact, as if live (in-sample: a pipeline check, not a test result).
        </p>
      )}

      {!race.result ? (
        <p>
          The result isn't in yet. <Link to="/">See the prediction</Link>
        </p>
      ) : (
        <>
          <div className="card">
            <h2>Predicted against the result</h2>
            <p className="muted small">
              Each stage's predicted position, and how far off it was (+2: finished two places lower than predicted).
            </p>
            <ComparisonTable race={race} runs={runs} />
            <NonStarters race={race} runs={runs} />
          </div>
          <div className="card">
            <h2>This race's scores</h2>
            <RaceScores runs={runs} />
          </div>
        </>
      )}
    </section>
  );
}

/** {real driver id: the run's prediction}. A debutant predicted under a
 * temporary id is filed under who he turned out to be (resolved_as). */
function byDriver(race: Race, run: Run) {
  return new Map(run.drivers.map((d) => [race.drivers[d.driver_id]?.resolved_as ?? d.driver_id, d] as const));
}

function ComparisonTable({ race, runs }: { race: Race; runs: Run[] }) {
  const predicted = runs.map((run) => byDriver(race, run));
  return (
    <div className="table-wrap scroll-inside">
      <table className="predictions comparison">
        <caption className="visually-hidden">Finishing order and each stage's predicted position</caption>
        <thead>
          <tr>
            <th scope="col" className="left rank">Pos</th>
            <th scope="col" className="left">Driver</th>
            <th scope="col" className="optional">Grid</th>
            {runs.map((run) => (
              <th scope="col" key={run.stage}>
                <span className="long">{STAGE_NAMES[run.stage]}</span>
                <span className="short" aria-hidden="true">{SHORT_STAGE[run.stage]}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {race.result!.map((row) => (
            <tr key={row.driver_id} data-driver={row.driver_id}>
              <td className="left rank">{row.position_text}</td>
              <td className="left">
                <span className="driver-name">{driverName(race, row.driver_id)}</span>
                <span className="team">{teamName(race, row.team_id)}{row.status && row.status !== "Finished" && !row.status.startsWith("+") ? ` · ${row.status}` : ""}</span>
              </td>
              <td className="optional">{row.grid ? `P${row.grid}` : row.grid === 0 ? "Pit" : "–"}</td>
              {predicted.map((map, i) => {
                const d = map.get(row.driver_id);
                if (!d) {
                  return (
                    <td key={runs[i].stage} data-stage={runs[i].stage} className="muted" title="Not in this stage's prediction">
                      missed
                    </td>
                  );
                }
                const off = row.position - d.predicted_rank;
                return (
                  <td key={runs[i].stage} data-stage={runs[i].stage}>
                    P{d.predicted_rank}
                    <span className="off">{off === 0 ? " ✓" : ` ${off > 0 ? "+" : "−"}${Math.abs(off)}`}</span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function NonStarters({ race, runs }: { race: Race; runs: Run[] }) {
  const started = new Set(race.result!.map((r) => r.driver_id));
  const lines = runs
    .map((run) => ({
      run,
      extra: [...byDriver(race, run).keys()].filter((id) => !started.has(id)),
    }))
    .filter((x) => x.extra.length > 0);
  if (!lines.length) return null;
  return (
    <ul className="non-starters">
      {lines.map(({ run, extra }) => (
        <li key={run.stage}>
          {STAGE_NAMES[run.stage]} also predicted {extra.map((id) => driverName(race, id)).join(", ")}, who didn't start.
        </li>
      ))}
    </ul>
  );
}

function RaceScores({ runs }: { runs: Run[] }) {
  const scored = runs.filter((r) => r.scores);
  if (!scored.length) return <p className="muted">Not scored yet.</p>;
  const yes = (v: number | undefined) => (v === undefined ? "–" : v >= 0.5 ? "yes" : "no");
  return (
    <div className="table-wrap">
      <table className="predictions race-scores">
        <caption className="visually-hidden">This race's scores, model against baseline</caption>
        <thead>
          <tr>
            <th scope="col" className="left">Stage</th>
            <th scope="col">Places off</th>
            <th scope="col">Winner picked</th>
            <th scope="col">Podium places right</th>
          </tr>
        </thead>
        <tbody>
          {scored.map((run) => {
            const s = run.scores!;
            return (
              <tr key={run.stage} data-stage={run.stage}>
                <th scope="row" className="left">{STAGE_NAMES[run.stage]}</th>
                <td>
                  {s.position_mae?.model.toFixed(2)}
                  <span className="score-baseline"> vs {s.position_mae?.baseline?.toFixed(2)}</span>
                </td>
                <td>
                  {yes(s.winner_hit?.model)}
                  <span className="score-baseline"> vs {yes(s.winner_hit?.baseline ?? undefined)}</span>
                </td>
                <td>
                  {Math.round((s.podium_hits?.model ?? 0) * 3)}/3
                  <span className="score-baseline"> vs {Math.round((s.podium_hits?.baseline ?? 0) * 3)}/3</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="muted small">Model first, then the stage's baseline (championship order, practice order or the starting grid).</p>
    </div>
  );
}
