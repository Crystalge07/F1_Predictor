// The weekend tracker at the top of the Next race page:
//
//   ✓ Pre-weekend    ✓ After practice    ● After qualifying    ○ Result + scoring
//     Logged Thu…      Logged Sat…         Due before the race   Scored once…
//
// A row of four steps on a wide screen, a vertical list on a phone. The
// logged steps are buttons: clicking one shows that stage's prediction
// (the same as the tabs further down). Below it, one "Still to come" line.

import { formatUtc } from "../format";
import type { Race, Stage } from "../types";
import { stillToCome, weekendSteps, type StepState } from "../weekend";

// The marker in front of each step. The state is also said in words for
// screen readers (the symbols alone don't say anything).
const MARKS: Record<StepState, { symbol: string; words: string }> = {
  done: { symbol: "✓", words: "done" },
  current: { symbol: "●", words: "next" },
  upcoming: { symbol: "○", words: "later" },
  missed: { symbol: "✕", words: "missed" },
};

interface Props {
  race: Race;
  now: Date;
  dataTime: Date;
  selected: Stage | null; // the stage whose prediction is shown below
  onSelect: (stage: Stage) => void;
}

export function WeekendTracker({ race, now, dataTime, selected, onSelect }: Props) {
  const steps = weekendSteps(race, now, dataTime);
  const toCome = stillToCome(race, now);
  return (
    <div className="weekend">
      <h2 className="visually-hidden">This weekend</h2>
      <ol className="weekend-steps">
        {steps.map((step) => {
          const mark = MARKS[step.state];
          const body = (
            <>
              <span className="weekend-mark" aria-hidden="true">{mark.symbol}</span>
              <span className="weekend-text">
                <span className="weekend-label">
                  {step.label}
                  <span className="visually-hidden"> ({mark.words})</span>
                </span>
                <span className="weekend-detail">
                  {step.detail}
                  {step.time ? ` ${step.state === "done" ? "" : "· "}${formatUtc(step.time)}` : ""}
                </span>
              </span>
            </>
          );
          // A logged prediction can be opened; the others are just text.
          const canOpen = step.state === "done" && step.key !== "result";
          return (
            <li key={step.key} className={`weekend-step is-${step.state}`}>
              {canOpen ? (
                <button
                  type="button"
                  className={`weekend-open${selected === step.key ? " is-selected" : ""}`}
                  aria-pressed={selected === step.key}
                  onClick={() => onSelect(step.key as Stage)}
                >
                  {body}
                </button>
              ) : (
                <div className="weekend-open">{body}</div>
              )}
            </li>
          );
        })}
      </ol>
      {toCome.length > 0 && (
        <p className="still-to-come small">
          <strong>Still to come:</strong>{" "}
          {toCome.map((item, i) => (
            <span key={i}>
              {i > 0 && " → "}
              {item.what}
              {item.time && <span className="muted"> ({formatUtc(item.time)})</span>}
            </span>
          ))}
        </p>
      )}
    </div>
  );
}
