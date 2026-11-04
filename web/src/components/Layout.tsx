// The frame around every page: header with the four pages, footer with
// when the data last changed.

import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useSiteData } from "../data";
import { formatUtc } from "../format";
import { usePageTitle } from "../pageTitle";

const PAGES = [
  { to: "/", label: "Next race", end: true },
  { to: "/why", label: "Why" },
  { to: "/scorecard", label: "Scorecard" },
  { to: "/races", label: "Past races" },
  { to: "/about", label: "About" },
];

/** The tab title for the current address. */
function titleFor(path: string): string | null {
  if (path === "/") return null;
  if (path.startsWith("/races/")) return "Past race";
  return PAGES.find((p) => p.to === path)?.label ?? null;
}

export function Layout() {
  const site = useSiteData();
  usePageTitle(titleFor(useLocation().pathname));
  return (
    <div className="shell">
      {/* For keyboard users: jump past the menu (a plain link would change the
          hash route, so it moves focus instead). */}
      <a
        href="#main"
        className="skip-link"
        onClick={(event) => {
          event.preventDefault();
          document.getElementById("main")?.focus();
        }}
      >
        Skip to content
      </a>
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
      <main className="main" id="main" tabIndex={-1}>
        <Outlet />
      </main>
      <footer className="footer">
        <span>Predictions are logged before each session and never edited, then scored after the race.</span>
        {site.state === "ready" && <span>Data updated {formatUtc(site.data.version.exported_at_utc)}</span>}
      </footer>
    </div>
  );
}
