// Race by race: how many places better (or worse) than the baseline the
// model's predicted order was, at one stage. Shows whether an average comes
// from steady results or one lucky weekend. Same diverging bars as Why:
// blue = model better, red = model worse; also the side and the signed number.

import type { RaceScores } from "../types";
import { formatSignedPlaces } from "../race";

export function RaceBars({ races }: { races: RaceScores[] }) {
  const rows = races
    .filter((r) => r.metrics.position_mae?.model != null && r.metrics.position_mae?.baseline != null)
    .map((r) => ({ race: r, gain: r.metrics.position_mae.baseline! - r.metrics.position_mae.model! })); // + = model better
  const biggest = Math.max(0.01, ...rows.map((r) => Math.abs(r.gain)));
  return (
    <div className="factor-chart race-bars">
      <div className="factor-scale" aria-hidden="true">
        <span>← baseline closer</span>
        <span>model closer →</span>
      </div>
      <ol className="factors" aria-label="Places off, model against baseline, race by race">
        {rows.map(({ race, gain }) => {
          const better = gain > 0;
          const width = `${(Math.abs(gain) / biggest) * 100}%`;
          return (
            <li key={`${race.season}-${race.round}`} className="factor" data-round={race.round} data-gain={gain}>
              <div className="factor-text">
                <span className="factor-name">R{race.round} · {race.name}</span>
                <span className="factor-value">
                  {race.metrics.position_mae.model!.toFixed(2)} vs {race.metrics.position_mae.baseline!.toFixed(2)} places off
                </span>
              </div>
              <div className="factor-bar" aria-hidden="true">
                <div className="half left">{!better && <span className="bar back-left" style={{ width }} />}</div>
                <div className="half right">{better && <span className="bar forward-right" style={{ width }} />}</div>
              </div>
              <div className={`factor-effect ${better ? "is-forward" : "is-back"}`}>
                {formatSignedPlaces(gain, 2)}
                <span className="visually-hidden"> ({better ? "model closer" : "baseline closer"})</span>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
