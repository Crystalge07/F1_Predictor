// The frame around every page: header with the pages, footer with when the
// data last changed.

import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useSiteData } from "../data";
import { formatUtc } from "../format";
import { usePageTitle } from "../pageTitle";

const PAGES = [
  { to: "/", label: "Next race", end: true },
  { to: "/results", label: "Results" },
  { to: "/about", label: "About" },
];

/** The tab title for the current address. */
function titleFor(path: string): string | null {
  if (path === "/") return null;
  if (path === "/why") return "Why";
  if (path === "/results/stats") return "Stats";
  if (path.startsWith("/results/")) return "Race result";
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
      {/* Sticky, see-through header: brand mark, the pages as a segmented
          control, and a live chip saying how fresh the data is. */}
      <header className="header">
        <div className="header-inner">
          <NavLink to="/" className="brand" aria-label="F1 Predictor, home">
            {/* The mark: two slanted "speed" bars in F1 red (decorative). */}
            <span className="brand-mark" aria-hidden="true">
              <span />
              <span />
            </span>
            <span className="brand-text">
              F1 <span className="brand-light">Predictor</span>
            </span>
          </NavLink>
          <nav aria-label="Pages" className="nav">
            {PAGES.map((page) => (
              <NavLink key={page.to} to={page.to} end={page.end} className="nav-link">
                {page.label}
              </NavLink>
            ))}
          </nav>
          {site.state === "ready" && (
            <span className="data-chip" title="When the data was last exported">
              <span className="live-dot" aria-hidden="true" />
              <span className="data-chip-text">Updated {formatUtc(site.data.version.exported_at_utc)}</span>
            </span>
          )}
        </div>
      </header>
      <main className="main" id="main" tabIndex={-1}>
        <Outlet />
      </main>
      <footer className="footer">
        <div className="footer-inner">
          {/* A strip of chequered flag, the race-control way to say "the end". */}
          <span className="chequer" aria-hidden="true" />
          <span>Predictions are logged before each session and never edited, then scored after the race.</span>
          {site.state === "ready" && <span className="mono">Data updated {formatUtc(site.data.version.exported_at_utc)}</span>}
        </div>
      </footer>
    </div>
  );
}
