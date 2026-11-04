// The Why chart: one bar per factor, from a centre line. Left = the factor
// moved the driver FORWARD (a better predicted position), right = BACK.
// Blue / red is the palette's diverging pair; direction is also the side
// and the signed number, so colour is never the only cue. Plain HTML bars
// (no SVG needed): they reflow on a phone like the rest of the page.

import type { Breakdown } from "../explain";
import { formatFactorValue, formatPlaces, formatSignedPlaces } from "../race";

export function FactorChart({ data, labels }: { data: Breakdown; labels: Record<string, string> }) {
  // Every bar on one scale: the biggest push fills its half.
  const biggest = Math.max(0.01, ...data.pushes.map((p) => Math.abs(p.places)));
  return (
    <div className="factor-chart">
      <div className="factor-scale" aria-hidden="true">
        <span>← moved forward</span>
        <span>moved back →</span>
      </div>
      <ol className="factors" aria-label="Factors, biggest effect first">
        {data.pushes.map((p) => {
          const name = p.feature === null ? "other factors combined" : labels[p.feature] ?? p.feature;
          const forward = p.places < 0;
          const width = `${(Math.abs(p.places) / biggest) * 100}%`;
          const effect = Math.abs(p.places) < 0.05 ? "no real effect" : `${formatPlaces(p.places)} ${forward ? "forward" : "back"}`;
          return (
            <li key={p.feature ?? "other"} className="factor" data-feature={p.feature ?? "other"} data-places={p.places}>
              <div className="factor-text">
                <span className="factor-name">{name}</span>
                {p.feature !== null && <span className="factor-value">{formatFactorValue(p.feature, p.value)}</span>}
              </div>
              <div className="factor-bar" aria-hidden="true">
                <div className="half left">{forward && <span className="bar forward" style={{ width }} />}</div>
                <div className="half right">{!forward && <span className="bar back" style={{ width }} />}</div>
              </div>
              <div className={`factor-effect ${forward ? "is-forward" : "is-back"}`}>
                {formatSignedPlaces(p.places)}
                <span className="visually-hidden"> ({effect})</span>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
