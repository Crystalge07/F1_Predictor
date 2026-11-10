// The race side of the Results tab: every race with a result, and for each
// one, what every stage predicted next to what actually happened.
//
//   #/results          the list (RaceList), newest first: one line per race
//                      with how its final prediction did
//   #/results/2026-19  one race (PastRace)
//
// Live predictions only by default. Backtest replays (past races re-run
// after the fact, in-sample) can be switched on, and are always labelled.

import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useRace, useSiteData } from "../data";
import { STAGE_NAMES } from "../format";
import { driverName, raceKey, STAGE_ORDER, teamName } from "../race";
import { BASELINE_NAMES } from "../scorecard";
import type { Index, Mode, Race, RaceScores, Run, Scorecard, Stage } from "../types";
import { ErrorBox, Loading } from "../components/Status";

// Column headers on a phone, where the full names don't fit side by side.
const SHORT_STAGE = { pre_weekend: "Pre", practice: "Practice", post_quali: "Quali" } as const;

/** Races with a result and a logged prediction, newest first. */
function finishedRaces(index: Index) {
  return index.seasons
    .flatMap((s) => s.races.filter((r) => r.status === "finished" && r.file).map((r) => ({ season: s.season, ...r })))
    .reverse();
}

/** A race's scores from the scorecard file: the latest stage scored for it
 * (the final prediction before the race), in the given mode. */
function finalScores(scorecard: Scorecard, season: number, round: number, mode: Mode): { stage: Stage; race: RaceScores } | null {
  const group = scorecard.groups.find((g) => g.season === season && g.mode === mode);
  if (!group) return null;
  for (const stage of [...STAGE_ORDER].reverse()) {
    const race = group.per_race[stage]?.find((r) => r.round === round);
    if (race) return { stage, race };
  }
  return null;
}

/** One line about how the final prediction did: "4.6 places off (grid 4.6)
 * · winner ✓ · podium 1/3". */
function ScoreLine({ scores }: { scores: { stage: Stage; race: RaceScores } }) {
  const m = scores.race.metrics;
  const mae = m.position_mae;
  const winner = m.winner_hit?.model;
  const podium = m.podium_hits?.model;
  const baseline = { pre_weekend: "championship_order", practice: "practice_order", post_quali: "grid_order" }[scores.stage];
  return (
    <span className="race-list-score">
      {STAGE_NAMES[scores.stage]}:{" "}
      {mae?.model != null && (
        <>
          {mae.model.toFixed(1)} places off
          {mae.baseline != null && <span className="muted"> ({BASELINE_NAMES[baseline]} {mae.baseline.toFixed(1)})</span>}
        </>
      )}
      {winner != null && <> · winner {winner >= 0.5 ? "✓" : "✗"}</>}
      {podium != null && <> · podium {Math.round(podium * 3)}/3</>}
    </span>
  );
}

export function RaceList() {
  const site = useSiteData();
  const [withBacktest, setWithBacktest] = useState(false);
  if (site.state === "loading") return <Loading />;
  if (site.state === "error") return <ErrorBox message={site.message} />;
  const all = finishedRaces(site.data.index);
  const races = withBacktest ? all : all.filter((r) => r.live_stages.length > 0);
  const hasBacktest = all.some((r) => r.backtest_stages.length > 0);
  return (
    <>
      <p className="muted">
        Every race we predicted, newest first, with how the final prediction (the last one before the race) did against
        the simple baseline. Open a race to see every stage next to the full result.
      </p>
      {hasBacktest && (
        <label className="check">
          <input type="checkbox" checked={withBacktest} onChange={(e) => setWithBacktest(e.target.checked)} />
          Also show backtest replays (past races re-run after the fact: in-sample, a pipeline check, not a real test)
        </label>
      )}
      {races.length === 0 ? (
        <p>No live-predicted race has finished yet.</p>
      ) : (
        <ul className="race-list">
          {races.map((r) => {
            const mode: Mode = r.live_stages.length > 0 ? "live" : "backtest";
            const scores = finalScores(site.data.scorecard, r.season, r.round, mode);
            return (
              <li key={raceKey(r.season, r.round)}>
                <Link to={`/results/${raceKey(r.season, r.round)}`}>
                  <span className="race-list-text">
                    <span className="race-list-name">
                      {r.season} R{r.round} · {r.name}
                    </span>
                    {scores && <ScoreLine scores={scores} />}
                  </span>
                  <span className="race-list-badges">
                    {r.live_stages.length > 0 && <span className="badge is-live">live</span>}
                    {r.backtest_stages.length > 0 && <span className="badge">backtest</span>}
                  </span>
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </>
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
          There's no logged prediction for that race. <Link to="/results">All results</Link>
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
        <Link to="/results">Results</Link> · {race.season} · Round {race.round}
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
 * temporary id is filed under who they turned out to be (resolved_as). */
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
