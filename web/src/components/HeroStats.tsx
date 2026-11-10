// The row of stat tiles under the race name on the Next race page:
//
//   LIGHTS OUT       FAVOURITE           PREDICTED PODIUM     LATEST CALL
//   in 11d 14h       Verstappen 30%      VER · ANT · NOR      Pre-weekend
//   Sun 22 Nov…      to win              most likely order    logged Sun 8 Nov…
//
// It's a summary of what the table below says in full, so a visitor gets
// the headline without reading 22 rows. Four tiles in a row on a wide
// screen, two by two on a phone (CSS: .hero-stats).

import { formatUtc, STAGE_NAMES } from "../format";
import { driverCode, driverName, formatPct } from "../race";
import type { Race, Run } from "../types";

/** "11d 14h", "5h 20m", "40m": the time left until a start. Null once it's passed. */
export function countdown(startIso: string | null | undefined, now: Date): string | null {
  if (!startIso) return null;
  const ms = new Date(startIso).getTime() - now.getTime();
  if (Number.isNaN(ms) || ms <= 0) return null;
  const minutes = Math.floor(ms / 60000);
  const days = Math.floor(minutes / 1440);
  const hours = Math.floor((minutes % 1440) / 60);
  const mins = minutes % 60;
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${mins}m`;
  return `${mins}m`;
}

interface Props {
  race: Race;
  latest: Run | null; // the newest live prediction, if any
  now: Date;
}

export function HeroStats({ race, latest, now }: Props) {
  const raceStart = race.sessions.R ?? null;
  const left = countdown(raceStart, now);
  // The favourite: the highest win chance (not necessarily predicted P1:
  // the order is by expected position, the chance is from the simulation).
  const favourite = latest ? [...latest.drivers].sort((a, b) => b.p_win - a.p_win)[0] : undefined;
  const podium = latest ? [...latest.drivers].sort((a, b) => a.predicted_rank - b.predicted_rank).slice(0, 3) : [];

  return (
    <dl className="hero-stats">
      <div className="stat">
        <dt>Lights out</dt>
        <dd className="stat-value">{left ? `in ${left}` : race.status === "finished" ? "Finished" : "Underway"}</dd>
        <dd className="stat-note">{formatUtc(raceStart)}</dd>
      </div>
      <div className="stat">
        <dt>Favourite</dt>
        {favourite ? (
          <>
            <dd className="stat-value">
              {driverName(race, favourite.driver_id).split(" ").pop()} <span className="stat-accent">{formatPct(favourite.p_win)}</span>
            </dd>
            <dd className="stat-note">chance to win</dd>
          </>
        ) : (
          <dd className="stat-value muted">–</dd>
        )}
      </div>
      <div className="stat">
        <dt>Predicted podium</dt>
        {podium.length ? (
          <>
            <dd className="stat-value mono">{podium.map((d) => driverCode(race, d.driver_id)).join(" · ")}</dd>
            <dd className="stat-note">most likely top three</dd>
          </>
        ) : (
          <dd className="stat-value muted">–</dd>
        )}
      </div>
      <div className="stat">
        <dt>Latest call</dt>
        <dd className="stat-value">{latest ? STAGE_NAMES[latest.stage] : "Not yet"}</dd>
        <dd className="stat-note">{latest ? `logged ${formatUtc(latest.logged_at_utc)}` : "first one before practice"}</dd>
      </div>
    </dl>
  );
}
