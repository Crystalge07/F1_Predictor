// Routes. Hash routing (".../F1_Predictor/#/scorecard"): GitHub Pages only
// serves files, so a path like /scorecard would 404 on refresh or when
// shared; everything after "#" stays in the browser, so every link works.

import { HashRouter, Navigate, Route, Routes, useParams } from "react-router-dom";
import { SiteDataProvider } from "./data";
import { Layout } from "./components/Layout";
import { NextRace } from "./pages/NextRace";
import { Why } from "./pages/Why";
import { Stats } from "./pages/Scorecard";
import { PastRace, RaceList } from "./pages/PastRaces";
import { Results } from "./pages/Results";
import { About } from "./pages/About";

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<NextRace />} />
        {/* Why: no longer in the menu; opened from a driver in a prediction table. */}
        <Route path="why" element={<Why />} />
        <Route path="results" element={<Results />}>
          <Route index element={<RaceList />} />
          <Route path="stats" element={<Stats />} />
        </Route>
        <Route path="results/:key" element={<PastRace />} />
        {/* The old addresses (before Scorecard and Past races became Results)
            still work: links shared before the change land on the new page. */}
        <Route path="scorecard" element={<Navigate to="/results/stats" replace />} />
        <Route path="races" element={<Navigate to="/results" replace />} />
        <Route path="races/:key" element={<OldRaceAddress />} />
        <Route path="about" element={<About />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}

/** #/races/2026-20 -> #/results/2026-20 */
function OldRaceAddress() {
  const { key } = useParams();
  return <Navigate to={`/results/${key}`} replace />;
}

function NotFound() {
  return (
    <section>
      <h1>Page not found</h1>
      <p className="muted">That page doesn't exist. Use the menu above.</p>
    </section>
  );
}

export function App() {
  return (
    <SiteDataProvider>
      <HashRouter>
        <AppRoutes />
      </HashRouter>
    </SiteDataProvider>
  );
}
