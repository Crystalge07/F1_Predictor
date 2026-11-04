// The summary: each measure (rows) at each stage (columns), the model next to
// that stage's baseline. Better / worse is a word and an arrow, not colour alone.

import type { ScorecardGroup, Stage } from "../types";
import { STAGE_NAMES } from "../format";
import { STAGE_ORDER } from "../race";
import { BASELINE_NAMES, comparison, METRICS } from "../scorecard";

export function ScorecardTable({ group }: { group: ScorecardGroup }) {
  const stages = STAGE_ORDER.filter((s): s is Stage => group.stages[s] !== undefined);
  // On a phone the grid is stacked (styles.css): each cell then shows this
  // label, since the column headers are hidden.
  const label = (s: Stage) => {
    const any = Object.values(group.stages[s]!)[0];
    return `${STAGE_NAMES[s]} · vs ${BASELINE_NAMES[any?.baseline_name ?? ""] ?? "baseline"}`;
  };
  return (
    <div className="table-wrap">
      <table className="predictions scorecard">
        <caption className="visually-hidden">Model against baseline, by stage</caption>
        <thead>
          <tr>
            <th scope="col" className="left">Measure</th>
            {stages.map((s) => {
              const any = Object.values(group.stages[s]!)[0];
              return (
                <th scope="col" key={s}>
                  {STAGE_NAMES[s]}
                  <span className="th-note">
                    vs {BASELINE_NAMES[any?.baseline_name ?? ""] ?? "baseline"} · {any?.races ?? 0} races
                  </span>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {METRICS.map((m) => (
            <tr key={m.key}>
              <th scope="row" className="left">{m.name}</th>
              {stages.map((s) => {
                const metric = group.stages[s]![m.key];
                if (!metric || metric.model === null) return <td key={s} data-label={label(s)}>–</td>;
                const { text, better } = comparison(metric);
                return (
                  <td key={s} data-stage={s} data-metric={m.key} data-label={label(s)}>
                    <span className="score">
                      <span className="score-model">{m.format(metric.model)}</span>
                      {metric.baseline !== null && <span className="score-baseline"> vs {m.format(metric.baseline)}</span>}
                      {text && <span className={`chip ${better === true ? "is-better" : better === false ? "is-worse" : ""}`}>{text}</span>}
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
          <tr>
            <th scope="row" className="left">Driver list misses</th>
            {stages.map((s) => {
              const races = group.per_race[s] ?? [];
              const missing = races.reduce((n, r) => n + (r.metrics.lineup_missing?.model ?? 0), 0);
              const extra = races.reduce((n, r) => n + (r.metrics.lineup_extra?.model ?? 0), 0);
              return (
                <td key={s} data-stage={s} data-metric="lineup" data-label={label(s)}>
                  <span className="score">{missing} missed · {extra} didn't start</span>
                </td>
              );
            })}
          </tr>
        </tbody>
      </table>
    </div>
  );
}
