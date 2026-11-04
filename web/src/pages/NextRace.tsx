// Next race (Phase 6, step 3 fills in the predictions). For now: which race
// is next, when, and which stages have been logged so far.

import { useSiteData } from "../data";
import { formatUtc, STAGE_NAMES } from "../format";
import { ErrorBox, Loading } from "../components/Status";

export function NextRace() {
  const site = useSiteData();
  if (site.state === "loading") return <Loading />;
  if (site.state === "error") return <ErrorBox message={site.message} />;

  const { index } = site.data;
  const next = index.next_race;
  const race = next && index.seasons.find((s) => s.season === next.season)?.races.find((r) => r.round === next.round);
  if (!race || !next) {
    return (
      <section>
        <h1>Next race</h1>
        <p className="muted">No race is scheduled yet. The next season's calendar appears here once it's published.</p>
      </section>
    );
  }
  return (
    <section>
      <p className="eyebrow">
        {next.season} · Round {race.round}
      </p>
      <h1>{race.name}</h1>
      <p className="muted">Race start: {formatUtc(race.race_start_utc)}</p>
      <h2>Predictions logged so far</h2>
      {race.live_stages.length === 0 ? (
        <p className="muted">None yet: the first prediction is logged before practice starts.</p>
      ) : (
        <ul className="pills">
          {race.live_stages.map((stage) => (
            <li key={stage}>{STAGE_NAMES[stage]}</li>
          ))}
        </ul>
      )}
      <p className="muted small">The full prediction table, win and podium chances, and how they moved arrive in the next update.</p>
    </section>
  );
}
