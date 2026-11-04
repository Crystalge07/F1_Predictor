// Routes. Hash routing (".../F1_Predictor/#/scorecard"): GitHub Pages only
// serves files, so a path like /scorecard would 404 on refresh or when
// shared; everything after "#" stays in the browser, so every link works.

import { HashRouter, Route, Routes } from "react-router-dom";
import { SiteDataProvider } from "./data";
import { Layout } from "./components/Layout";
import { NextRace } from "./pages/NextRace";
import { Why } from "./pages/Why";
import { Scorecard } from "./pages/Scorecard";
import { PastRace, PastRaces } from "./pages/PastRaces";
import { About } from "./pages/About";

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<NextRace />} />
        <Route path="why" element={<Why />} />
        <Route path="scorecard" element={<Scorecard />} />
        <Route path="races" element={<PastRaces />} />
        <Route path="races/:key" element={<PastRace />} />
        <Route path="about" element={<About />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
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
