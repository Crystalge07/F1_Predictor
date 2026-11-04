// Calibration: when the model said X%, how often did it happen? Each point
// is one group of predictions (by predicted chance): across, their average
// predicted chance; up, how often it happened. On the diagonal = perfectly
// calibrated; below it = too confident, above = too cautious.
//
// One series, so no legend box (the title says what's plotted); 8px+
// markers with a surface ring; hairline grid; the diagonal is a recessive
// reference line. Hover / arrow keys read each group out, and the table
// view has every number.

import { useLayoutEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import type { CalibrationBin } from "../types";

const HEIGHT = 300;
const MARGIN = { top: 12, right: 16, bottom: 40, left: 60 };

const pct = (v: number) => `${Math.round(v * 100)}%`;

export function CalibrationChart({ bins, what }: { bins: CalibrationBin[]; what: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(560);
  const [active, setActive] = useState<number | null>(null);
  const [showTable, setShowTable] = useState(false);
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const measured = element.getBoundingClientRect().width;
    if (measured > 0) setWidth(Math.max(280, Math.min(560, measured)));
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(280, Math.min(560, entry.contentRect.width))));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const points = bins.filter((b) => b.predicted !== null && b.observed !== null);
  // Both axes 0 to the same top, so the diagonal is a true 45°: 100%, or
  // 50% / 20% when every chance is small (win chances mostly are).
  const highest = Math.max(0.01, ...points.flatMap((b) => [b.predicted!, b.observed!]));
  const top = highest <= 0.2 ? 0.2 : highest <= 0.5 ? 0.5 : 1;
  const ticks = top === 1 ? [0, 0.25, 0.5, 0.75, 1] : top === 0.5 ? [0, 0.1, 0.2, 0.3, 0.4, 0.5] : [0, 0.05, 0.1, 0.15, 0.2];
  const plot = Math.min(width - MARGIN.left - MARGIN.right, HEIGHT - MARGIN.top - MARGIN.bottom);
  const x = (v: number) => MARGIN.left + (v / top) * plot;
  const y = (v: number) => MARGIN.top + plot - (v / top) * plot;
  const chartWidth = MARGIN.left + plot + MARGIN.right;

  const onPointerMove = (event: PointerEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const px = event.clientX - box.left;
    const py = event.clientY - box.top;
    let nearest = 0;
    points.forEach((b, i) => {
      const d = Math.hypot(x(b.predicted!) - px, y(b.observed!) - py);
      if (d < Math.hypot(x(points[nearest].predicted!) - px, y(points[nearest].observed!) - py)) nearest = i;
    });
    setActive(points.length ? nearest : null);
  };
  const onKeyDown = (event: KeyboardEvent<SVGSVGElement>) => {
    if (event.key === "ArrowLeft") setActive((a) => Math.max(0, (a ?? 0) - 1));
    else if (event.key === "ArrowRight") setActive((a) => Math.min(points.length - 1, (a ?? -1) + 1));
    else return;
    event.preventDefault();
  };
  const describe = (b: CalibrationBin) =>
    `predicted ${pct(b.from)}–${pct(b.to)} (average ${pct(b.predicted!)}): happened ${pct(b.observed!)} of the time, ${b.n} predictions`;

  if (points.length === 0) return <p className="muted">No scored predictions yet.</p>;
  return (
    <div ref={ref}>
      <svg
        className="chart"
        width={chartWidth}
        height={HEIGHT}
        viewBox={`0 0 ${chartWidth} ${HEIGHT}`}
        style={{ width: chartWidth }}
        role="img"
        aria-label={`Calibration of ${what} chances: predicted chance across, how often it happened up. Use the arrow keys to read each group.`}
        aria-describedby="calibration-readout"
        tabIndex={0}
        onPointerMove={onPointerMove}
        onPointerLeave={() => setActive(null)}
        onFocus={() => setActive((a) => a ?? 0)}
        onBlur={() => setActive(null)}
        onKeyDown={onKeyDown}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(0)} x2={x(top)} y1={y(t)} y2={y(t)} stroke="var(--grid)" strokeWidth={1} />
            <text x={x(0) - 8} y={y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--axis)">{pct(t)}</text>
            <text x={x(t)} y={y(0) + 16} textAnchor="middle" fontSize={11} fill="var(--axis)">{pct(t)}</text>
          </g>
        ))}
        <text x={x(top / 2)} y={HEIGHT - 6} textAnchor="middle" fontSize={12} fill="var(--muted)">predicted chance</text>
        <text transform={`translate(14 ${y(top / 2)}) rotate(-90)`} textAnchor="middle" fontSize={12} fill="var(--muted)">how often it happened</text>
        {/* The diagonal: perfectly calibrated. */}
        <line x1={x(0)} y1={y(0)} x2={x(top)} y2={y(top)} stroke="var(--axis)" strokeWidth={1} strokeDasharray="4 4" />
        {/* (No label on the line: wherever it sat, real data could put a dot
            under it. The text above the chart says what the dashed line is.) */}
        {points.map((b, i) => (
          <circle
            key={b.from}
            cx={x(b.predicted!)}
            cy={y(b.observed!)}
            r={active === i ? 7 : 5}
            fill="var(--series-1)"
            stroke="var(--surface)"
            strokeWidth={2}
            data-from={b.from}
            data-predicted={b.predicted!}
            data-observed={b.observed!}
          />
        ))}
      </svg>
      <p id="calibration-readout" className="chart-tooltip" aria-live="polite">
        {active !== null ? describe(points[active]) : "Point at a dot (or focus the chart and use the arrow keys) to read it."}
      </p>
      <div className="chart-controls">
        <button type="button" className="link-button" onClick={() => setShowTable((v) => !v)} aria-expanded={showTable}>
          {showTable ? "Hide the table" : "Show as a table"}
        </button>
      </div>
      {showTable && (
        <table className="predictions" style={{ marginTop: 12 }}>
          <caption className="visually-hidden">Calibration of {what} chances</caption>
          <thead>
            <tr>
              <th scope="col" className="left">Predicted chance</th>
              <th scope="col">Predictions</th>
              <th scope="col">Average predicted</th>
              <th scope="col">Happened</th>
            </tr>
          </thead>
          <tbody>
            {points.map((b) => (
              <tr key={b.from}>
                <td className="left">{pct(b.from)}–{pct(b.to)}</td>
                <td>{b.n}</td>
                <td>{pct(b.predicted!)}</td>
                <td>{pct(b.observed!)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
