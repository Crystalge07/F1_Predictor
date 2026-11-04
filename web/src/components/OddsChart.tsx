// How each driver's win chance moved through the weekend: one line per
// driver across the stages logged so far (pre-weekend -> after practice ->
// after qualifying). Hand-drawn SVG, no chart library.
//
// Rules it follows (the dataviz guidance this project uses):
//   * at most eight colours, in a fixed order, given to the top eight by the
//     LATEST win chance; nobody else gets a hue. "All drivers" adds the rest
//     in one recessive grey, and never repaints the coloured eight;
//   * identity is never colour alone: a legend, plus each line's driver code
//     at its end (text in the text colour, not the line colour);
//   * 2px lines; 8px end markers with a 2px ring in the surface colour;
//     hairline gridlines; one y axis (win chance, from 0);
//   * hover or arrow keys snap to the nearest stage and list every shown
//     driver's value there; the same numbers are in the table view, so
//     nothing needs the pointer.

import { useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import type { Race, Run } from "../types";
import { STAGE_NAMES } from "../format";
import { driverCode, driverName, formatPct } from "../race";

const COLOURED = 8;
const HEIGHT = 260;
const MARGIN = { top: 14, right: 56, bottom: 30, left: 40 };
const LABEL_GAP = 13; // minimum vertical distance between end labels
// Stage names along the bottom: short ones when the plot is too narrow for
// the full names to sit side by side (phones).
const SHORT_STAGE = { pre_weekend: "Pre-weekend", practice: "Practice", post_quali: "Qualifying" } as const;
const SHORT_BELOW = 380; // plot width in px

/** The chart's width in pixels, following its container (fixed text size at any width). */
function useWidth(fallback = 640) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(fallback);
  // useLayoutEffect: measured before the first paint, so the chart is never
  // drawn at the fallback width and squeezed; the observer follows resizes.
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const measured = element.getBoundingClientRect().width;
    if (measured > 0) setWidth(Math.max(280, measured)); // (tests: no layout, 0)
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(280, entry.contentRect.width)));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

/** A scale top that reads well: 10%, 20%, 30%... (5% steps below 10%). */
function niceTop(max: number): { top: number; step: number } {
  if (max <= 0.1) return { top: 0.1, step: 0.05 };
  const top = Math.min(1, Math.ceil(max * 10) / 10);
  return { top, step: top <= 0.5 ? 0.1 : 0.2 };
}

/** Move label positions (sorted top to bottom) apart by at least LABEL_GAP,
 * keeping them inside [top, bottom]: first push down, then, if the lowest
 * ran off the bottom (many drivers near 0%), push back up from the bottom.
 * Exported for the tests. */
export function spreadLabels(labels: { y: number }[], top: number, bottom: number) {
  for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + LABEL_GAP);
  for (let i = labels.length - 1; i >= 0; i--) {
    const limit = i === labels.length - 1 ? bottom : labels[i + 1].y - LABEL_GAP;
    labels[i].y = Math.max(top, Math.min(labels[i].y, limit));
  }
}

export interface Series {
  driverId: string;
  slot: number | null; // 1-8: its colour; null: grey ("other")
  values: (number | null)[]; // win chance at each run; null = not in that run
}

/** Every driver in any run, ordered by the latest run's win chance; the
 * first eight get colour slots 1-8. Exported for the tests. */
export function buildSeries(runs: Run[]): Series[] {
  const latest = runs[runs.length - 1];
  const latestRank = new Map(latest.drivers.map((d) => [d.driver_id, d] as const));
  const ids = [...new Set(runs.flatMap((run) => run.drivers.map((d) => d.driver_id)))];
  ids.sort((a, b) => {
    const da = latestRank.get(a);
    const db = latestRank.get(b);
    if (da && db) return db.p_win - da.p_win || da.predicted_rank - db.predicted_rank;
    if (da || db) return da ? -1 : 1; // drivers not in the latest run go last
    return a.localeCompare(b);
  });
  return ids.map((driverId, i) => ({
    driverId,
    slot: i < COLOURED ? i + 1 : null,
    values: runs.map((run) => run.drivers.find((d) => d.driver_id === driverId)?.p_win ?? null),
  }));
}

function colour(slot: number | null) {
  return slot ? `var(--series-${slot})` : "var(--other)";
}

export function OddsChart({ race, runs }: { race: Race; runs: Run[] }) {
  const [showAll, setShowAll] = useState(false);
  const [showTable, setShowTable] = useState(false);
  const [active, setActive] = useState<number | null>(null); // the stage under the pointer
  const [wrapRef, width] = useWidth();

  const all = useMemo(() => buildSeries(runs), [runs]);
  const shown = showAll ? all : all.filter((s) => s.slot !== null);
  const coloured = all.filter((s) => s.slot !== null);

  // --- Geometry ----------------------------------------------------------
  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = HEIGHT - MARGIN.top - MARGIN.bottom;
  const { top, step } = niceTop(Math.max(0.0001, ...shown.flatMap((s) => s.values.filter((v): v is number => v !== null))));
  const x = (i: number) => MARGIN.left + (runs.length === 1 ? plotW / 2 : (i * plotW) / (runs.length - 1));
  const y = (v: number) => MARGIN.top + plotH * (1 - v / top);
  const ticks = Array.from({ length: Math.round(top / step) + 1 }, (_, i) => i * step);

  // A line through the stages a driver is in (a gap where he isn't).
  const path = (s: Series) =>
    s.values
      .map((v, i) => (v === null ? null : `${x(i)},${y(v)}`))
      .reduce<string[]>((parts, point, i, points) => {
        if (point === null) return parts;
        parts.push(`${i > 0 && points[i - 1] !== null ? "L" : "M"}${point}`);
        return parts;
      }, [])
      .join(" ");

  // End labels for the coloured lines, spread so they never overlap.
  const last = runs.length - 1;
  const labels = coloured
    .filter((s) => s.values[last] !== null)
    .map((s) => ({ s, y: y(s.values[last]!) }))
    .sort((a, b) => a.y - b.y);
  spreadLabels(labels, MARGIN.top, MARGIN.top + plotH);

  // --- Pointer and keys: snap to the nearest stage ----------------------
  const onPointerMove = (event: PointerEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * width;
    let nearest = 0;
    runs.forEach((_, i) => {
      if (Math.abs(x(i) - px) < Math.abs(x(nearest) - px)) nearest = i;
    });
    setActive(nearest);
  };
  const onKeyDown = (event: KeyboardEvent<SVGSVGElement>) => {
    if (event.key === "ArrowLeft") setActive((a) => Math.max(0, (a ?? last) - 1));
    else if (event.key === "ArrowRight") setActive((a) => Math.min(last, (a ?? 0) + 1));
    else return;
    event.preventDefault();
  };
  const readout =
    active === null
      ? null
      : shown
          .filter((s) => s.values[active] !== null)
          .sort((a, b) => b.values[active]! - a.values[active]!)
          .map((s) => `${driverCode(race, s.driverId)} ${formatPct(s.values[active]!)}`);

  return (
    <div>
      <ul className="chart-legend" aria-label="Drivers in the chart">
        {coloured.map((s) => (
          <li key={s.driverId}>
            <span className="swatch" style={{ background: colour(s.slot) }} aria-hidden="true" />
            {driverName(race, s.driverId)}
          </li>
        ))}
        {showAll && all.length > COLOURED && (
          <li>
            <span className="swatch" style={{ background: colour(null) }} aria-hidden="true" />
            Other drivers
          </li>
        )}
      </ul>

      <div ref={wrapRef}>
        <svg
          className="chart"
          width={width}
          height={HEIGHT}
          viewBox={`0 0 ${width} ${HEIGHT}`}
          role="img"
          aria-label={`Win chance at each stage for ${showAll ? "every driver" : `the top ${coloured.length} drivers`}. Use the left and right arrow keys to read each stage.`}
          aria-describedby="odds-readout"
          tabIndex={0}
          onPointerMove={onPointerMove}
          onPointerLeave={() => setActive(null)}
          onFocus={() => setActive((a) => a ?? last)}
          onBlur={() => setActive(null)}
          onKeyDown={onKeyDown}
        >
          {/* Gridlines and the y axis labels (recessive). */}
          {ticks.map((t) => (
            <g key={t}>
              <line x1={MARGIN.left} x2={width - MARGIN.right} y1={y(t)} y2={y(t)} stroke="var(--grid)" strokeWidth={1} />
              <text x={MARGIN.left - 8} y={y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--axis)">
                {Math.round(t * 100)}%
              </text>
            </g>
          ))}
          {/* Stage names along the bottom. */}
          {runs.map((run, i) => (
            <text
              key={run.stage}
              x={x(i)}
              y={HEIGHT - 8}
              textAnchor={runs.length > 1 && i === 0 ? "start" : runs.length > 1 && i === last ? "end" : "middle"}
              fontSize={12}
              fill={active === i ? "var(--text)" : "var(--axis)"}
            >
              {plotW < SHORT_BELOW ? SHORT_STAGE[run.stage] : STAGE_NAMES[run.stage]}
            </text>
          ))}
          {/* The crosshair at the active stage. */}
          {active !== null && (
            <line x1={x(active)} x2={x(active)} y1={MARGIN.top} y2={MARGIN.top + plotH} stroke="var(--axis)" strokeWidth={1} />
          )}
          {/* Grey lines first (behind), then the coloured ones, slot 1 on top. */}
          {[...shown].reverse().map((s) => (
            <g key={s.driverId} data-driver={s.driverId} data-slot={s.slot ?? "other"}>
              <path d={path(s)} fill="none" stroke={colour(s.slot)} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
              {s.slot !== null &&
                s.values.map((v, i) =>
                  v === null ? null : (
                    <circle key={i} cx={x(i)} cy={y(v)} r={4} fill={colour(s.slot)} stroke="var(--surface)" strokeWidth={2} data-value={v} />
                  ),
                )}
            </g>
          ))}
          {/* Driver codes at the line ends. */}
          {labels.map(({ s, y: ly }) => (
            <text key={s.driverId} x={x(last) + 10} y={ly} dy="0.32em" fontSize={11} fontWeight={600} fill="var(--text)">
              {driverCode(race, s.driverId)}
            </text>
          ))}
        </svg>
      </div>

      <p id="odds-readout" className="chart-tooltip" aria-live="polite">
        {readout ? (
          <>
            <strong>{STAGE_NAMES[runs[active!].stage]}:</strong> {readout.join(" · ")}
          </>
        ) : (
          "Point at the chart (or focus it and use the arrow keys) to read each stage."
        )}
      </p>

      <div className="chart-controls">
        {all.length > COLOURED && (
          <button type="button" className="link-button" onClick={() => setShowAll((v) => !v)}>
            {showAll ? `Show the top ${COLOURED} only` : `Show all ${all.length} drivers`}
          </button>
        )}
        <button type="button" className="link-button" onClick={() => setShowTable((v) => !v)} aria-expanded={showTable}>
          {showTable ? "Hide the table" : "Show as a table"}
        </button>
      </div>

      {showTable && (
        <table className="predictions" style={{ marginTop: 12 }}>
          <caption className="visually-hidden">Win chance at each stage</caption>
          <thead>
            <tr>
              <th scope="col" className="left">Driver</th>
              {runs.map((run) => (
                <th scope="col" key={run.stage}>{STAGE_NAMES[run.stage]}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((s) => (
              <tr key={s.driverId}>
                <td className="left">{driverName(race, s.driverId)}</td>
                {s.values.map((v, i) => (
                  <td key={i}>{v === null ? "–" : formatPct(v)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
