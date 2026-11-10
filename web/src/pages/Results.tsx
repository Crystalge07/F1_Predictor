// Results: our predictions against what actually happened.
//
//   #/results            Races: every predicted race, newest first (RaceList)
//   #/results/stats      Stats: the scorecard over all scored races (Stats)
//   #/results/2026-20    one race in full (PastRace, its own page)
//
// The two sub-tabs are links, not buttons, so each has its own address and
// can be shared or bookmarked.

import { NavLink, Outlet } from "react-router-dom";

export function Results() {
  return (
    <section>
      <h1>Results</h1>
      <nav className="tabs" aria-label="Results">
        <NavLink to="/results" end className="tab">
          Races
        </NavLink>
        <NavLink to="/results/stats" className="tab">
          Stats
        </NavLink>
      </nav>
      <Outlet />
    </section>
  );
}
