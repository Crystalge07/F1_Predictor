// The app shell: pages, navigation, and what happens when data is missing.

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { AppRoutes } from "../App";
import { SiteDataProvider } from "../data";
import { serveTestData } from "./serve";

function renderApp(path = "/") {
  // MemoryRouter instead of the app's HashRouter: same routes, but the
  // test controls the address.
  return render(
    <SiteDataProvider>
      <MemoryRouter initialEntries={[path]}>
        <AppRoutes />
      </MemoryRouter>
    </SiteDataProvider>,
  );
}

describe("the shell", () => {
  it("shows the next race from the data, and when the data last changed", async () => {
    serveTestData();
    renderApp();
    // The simulated world ends after round 2; round 3 is next, with its
    // pre-weekend prediction already logged.
    expect(await screen.findByRole("heading", { level: 1, name: "Test GP 3" })).toBeInTheDocument();
    expect(screen.getByText("2027 · Round 3")).toBeInTheDocument();
    expect(screen.getByText(/Still to come:/).parentElement).toHaveTextContent("Race (Sun 21 Mar, 15:00 UTC)");
    expect(screen.getByRole("tab", { name: "Pre-weekend" })).toHaveAttribute("aria-selected", "true");
    // The export time of the test data (scripts/export_test_data.py: the end of the simulation).
    expect(screen.getByText("Data updated Tue 16 Mar, 12:00 UTC")).toBeInTheDocument();
  });

  it("says 'loading' until the data arrives, not an empty page", () => {
    serveTestData();
    renderApp();
    expect(screen.getByRole("status")).toHaveTextContent("Loading data");
  });

  it("moves between the pages and marks the current one", async () => {
    serveTestData();
    renderApp();
    const user = userEvent.setup();
    expect(screen.getAllByRole("link", { name: /^(Next race|Results|About)$/ })).toHaveLength(3);
    for (const [link, heading] of [["Results", "Results"], ["About", "How it works"]]) {
      await user.click(screen.getByRole("link", { name: link }));
      expect(screen.getByRole("heading", { level: 1, name: heading })).toBeInTheDocument();
      expect(screen.getByRole("link", { name: link })).toHaveClass("active");
    }
    await user.click(screen.getByRole("link", { name: "Next race" }));
    expect(await screen.findByRole("heading", { level: 1, name: "Test GP 3" })).toBeInTheDocument();
  });

  it("has a not-found page for unknown addresses", () => {
    serveTestData();
    renderApp("/no-such-page");
    expect(screen.getByRole("heading", { level: 1, name: "Page not found" })).toBeInTheDocument();
  });
});

describe("when the data can't be used, the page says so", () => {
  // Each case: a file answered badly, and the message the visitor sees.
  const cases: [string, Parameters<typeof serveTestData>[0], RegExp][] = [
    ["a missing file", { "index.json": { status: 404 } }, /Couldn't load index\.json \(HTTP 404\)/],
    ["a broken file", { "scorecard.json": { body: "{not json" } }, /scorecard\.json isn't valid JSON/],
    ["a file in a newer format", { "version.json": { body: '{"schema": 2}' } },
      /version\.json is in format 2; this page reads format 1/],
    ["no connection", { "index.json": { networkError: true } }, /Couldn't reach index\.json \(offline\?\)/],
  ];
  it.each(cases)("%s", async (_name, overrides, message) => {
    serveTestData(overrides);
    renderApp();
    expect(await screen.findByRole("alert")).toHaveTextContent(message);
    // The rest of the page still works: the menu is there.
    expect(screen.getByRole("link", { name: "Results" })).toBeInTheDocument();
  });
});
