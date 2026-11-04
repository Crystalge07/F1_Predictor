// The prediction table: one row per driver, in predicted order, with the
// chances of each outcome. Percentages are text first; the thin bar under
// each is a quick visual of the same number (one hue: it's magnitude).

import type { Race, Run } from "../types";
import { ANCHOR, driverName, formatPct, teamName } from "../race";

function Pct({ value, label }: { value: number; label: string }) {
  return (
    <td className="pct" aria-label={`${label} ${formatPct(value)}`}>
      {formatPct(value)}
      <span className="pct-bar" aria-hidden="true">
        <span style={{ width: `${Math.min(100, value * 100)}%` }} />
      </span>
    </td>
  );
}

export function PredictionTable({ race, run }: { race: Race; run: Run }) {
  const anchor = ANCHOR[run.stage];
  return (
    <div className="table-wrap">
      <table className="predictions">
        <caption className="visually-hidden">Predicted finishing order and chances, {run.stage.replace("_", " ")}</caption>
        <thead>
          <tr>
            <th scope="col" className="left rank">Pos</th>
            <th scope="col" className="left">Driver</th>
            <th scope="col" className="optional" title={anchor.title}>{anchor.column}</th>
            <th scope="col">Win</th>
            <th scope="col">Podium</th>
            <th scope="col" className="optional">Points</th>
            <th scope="col" className="optional" title="Chance of not finishing">Retire</th>
          </tr>
        </thead>
        <tbody>
          {run.drivers.map((d) => {
            const info = race.drivers[d.driver_id];
            return (
              <tr key={d.driver_id}>
                <td className="left rank">{d.predicted_rank}</td>
                <td className="left">
                  <span className="driver-name">{driverName(race, d.driver_id)}</span>
                  {info?.provisional && (
                    <span className="tag" title="Making his debut: predicted under a temporary id until his first result">
                      debut
                    </span>
                  )}
                  <span className="team">{teamName(race, d.team_id)}</span>
                </td>
                <td className="optional">{d.anchor_position == null ? "–" : `P${Math.round(d.anchor_position)}`}</td>
                <Pct value={d.p_win} label="Win" />
                <Pct value={d.p_podium} label="Podium" />
                <td className="optional pct">{formatPct(d.p_points)}</td>
                <td className="optional pct">{formatPct(d.p_dnf)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
