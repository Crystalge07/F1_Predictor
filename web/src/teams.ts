// Team colours: the thin stripe next to each driver's name, the way the TV
// graphics mark who drives for whom.
//
// These are IDENTITY colours only (which team), never a data encoding: no
// chart reads meaning from them, so they don't have to pass the chart
// palette's colour-blind checks. The team name is always written next to
// the stripe, so nothing depends on seeing the colour.
//
// Keys are the export's team ids (race.teams). An unknown team (a new
// entry, or the simulated test world's T01..T10) falls back to grey.

const TEAM_COLOURS: Record<string, string> = {
  red_bull: "#3671c6",
  mercedes: "#27f4d2",
  mclaren: "#ff8000",
  ferrari: "#e8002d",
  rb: "#6692ff",
  alpine: "#0093cc",
  aston_martin: "#229971",
  audi: "#f50537",
  sauber: "#52e252", // Audi's name before 2026, for older races
  williams: "#64c4ff",
  haas: "#b6babd",
  cadillac: "#d4af37",
};

/** The team's colour, or a neutral grey when we don't know the team. */
export function teamColour(teamId: string | null | undefined): string {
  return (teamId && TEAM_COLOURS[teamId]) || "#52525b";
}
