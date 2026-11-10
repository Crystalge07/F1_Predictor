// "How we did": the card at the top of the Next race page for about two
// days after a race (weekend.RECENT_RESULT_HOURS). It puts the real podium
// next to the podium our final prediction (the last stage logged before the
// race) called, plus that prediction's headline scores, and links to the full
// comparison under Results.
//
// Before the official result is in, it says so instead.

import { Link } from "react-router-dom";
import { useRace } from "../data";
import { STAGE_NAMES } from "../format";
import { driverName, formatPct, liveRuns } from "../race";
import { BASELINE_NAMES } from "../scorecard";
import type { Race } from "../types";

interface Props {
  file: string;
  raceKey: string; // "2026-20", for the link to the full comparison
}

export function RecentResult({ file, raceKey }: Props) {
  const loaded = useRace(file);
  // Loading or broken: the card is a bonus, so it just doesn't appear
  // (the race's own page reports errors).
  if (!loaded || loaded.state !== "ready") return null;
  const race = loaded.data;
  const runs = liveRuns(race);
  if (runs.length === 0) return null; // nothing was predicted live: nothing to compare
  const final = runs[runs.length - 1];

  return (
    <div className="card recent-result">
      <p className="eyebrow">Last race · {race.season} Round {race.round}</p>
      <h2>{race.name}: how we did</h2>
      {race.status !== "finished" || !race.result ? (
        <p className="muted">
          The race has run. The result and the scores for every prediction appear here once the official result is in.
        </p>
      ) : (
        <ResultBody race={race} raceKey={raceKey} final={final} />
      )}
    </div>
  );
}

function ResultBody({ race, raceKey, final }: { race: Race; raceKey: string; final: ReturnType<typeof liveRuns>[number] }) {
  const actual = [...race.result!].sort((a, b) => a.position - b.position).slice(0, 3);
  const predicted = [...final.drivers].sort((a, b) => a.predicted_rank - b.predicted_rank).slice(0, 3);
  const predictedIds = new Set(predicted.map((d) => d.driver_id));
  const winner = actual[0];
  const winnerChance = final.drivers.find((d) => d.driver_id === winner?.driver_id)?.p_win;
  const scores = final.scores ?? {};
  const placesOff = scores.position_mae;
  const podiumRight = scores.podium_hits;

  return (
    <>
      <table className="predictions podium-compare">
        <caption className="visually-hidden">Real podium next to our final prediction's podium</caption>
        <thead>
          <tr>
            <th scope="col" className="left">Place</th>
            <th scope="col" className="left">Finished</th>
            <th scope="col" className="left">We predicted ({STAGE_NAMES[final.stage].toLowerCase()})</th>
          </tr>
        </thead>
        <tbody>
          {[0, 1, 2].map((i) => (
            <tr key={i}>
              <td className="left rank">P{i + 1}</td>
              <td className="left">
                {actual[i] ? (
                  <>
                    <span className="driver-name">{driverName(race, actual[i].driver_id)}</span>
                    {/* ✓: we had them on the podium (any of the three places). */}
                    {predictedIds.has(actual[i].driver_id) && (
                      <span className="hit" title="We predicted a podium for this driver">
                        {" "}✓<span className="visually-hidden"> predicted on the podium</span>
                      </span>
                    )}
                  </>
                ) : (
                  "—"
                )}
              </td>
              <td className="left">{predicted[i] ? driverName(race, predicted[i].driver_id) : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <ul className="recent-facts small">
        {winner && winnerChance !== undefined && (
          <li>
            We gave the winner, {driverName(race, winner.driver_id)}, a <strong>{formatPct(winnerChance)}</strong> chance to win.
          </li>
        )}
        {placesOff && (
          <li>
            Each driver's predicted position was <strong>{placesOff.model.toFixed(1)} places</strong> off on average
            {placesOff.baseline !== null &&
              ` (just following the ${BASELINE_NAMES[placesOff.baseline_name ?? ""] ?? "baseline"}: ${placesOff.baseline.toFixed(1)})`}
            .
          </li>
        )}
        {podiumRight && (
          <li>
            <strong>{Math.round(podiumRight.model * 3)} of 3</strong> predicted podium places were right.
          </li>
        )}
      </ul>
      <p className="small" style={{ marginBottom: 0 }}>
        <Link to={`/results/${raceKey}`}>Every stage next to the full result →</Link>
      </p>
    </>
  );
}
