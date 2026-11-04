// The frame around every page: header with the four pages, footer with
// when the data last changed.

import { NavLink, Outlet } from "react-router-dom";
import { useSiteData } from "../data";
import { formatUtc } from "../format";

const PAGES = [
  { to: "/", label: "Next race", end: true },
  { to: "/why", label: "Why" },
  { to: "/scorecard", label: "Scorecard" },
  { to: "/races", label: "Past races" },
];

export function Layout() {
  const site = useSiteData();
  return (
    <div className="shell">
      <header className="header">
        <NavLink to="/" className="brand">
          F1 Predictor
        </NavLink>
        <nav aria-label="Pages">
          {PAGES.map((page) => (
            <NavLink key={page.to} to={page.to} end={page.end} className="nav-link">
              {page.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="main">
        <Outlet />
      </main>
      <footer className="footer">
        <span>Predictions are logged before each session and never edited, then scored after the race.</span>
        {site.state === "ready" && <span>Data updated {formatUtc(site.data.version.exported_at_utc)}</span>}
      </footer>
    </div>
  );
}
