// The Next race page, against the simulated test world's export.
//
// The world's index says round 3 is next (only its pre-weekend prediction
// is logged). To test a full weekend, the tests point "next race" at round
// 2, which has all three stages and two debutants.

import { render, screen, within, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { AppRoutes } from "../App";
import { SiteDataProvider } from "../data";
import { buildSeries } from "../components/OddsChart";
import { formatPct, liveRuns } from "../race";
import type { Index, Race } from "../types";
import { serveTestData, testFile } from "./serve";

/** Serve the test data with "next race" = `round` (and any other overrides). */
function serveWithNext(round: number, overrides: Parameters<typeof serveTestData>[0] = {}) {
  const index = testFile<Index>("index.json");
  index.next_race = { season: 2027, round };
  return serveTestData({ "index.json": { body: JSON.stringify(index) }, ...overrides });
}

function renderPage() {
  return render(
    <SiteDataProvider>
      <MemoryRouter>
        <AppRoutes />
      </MemoryRouter>
    </SiteDataProvider>,
  );
}

/** The table's rows as [name, win%] pairs (the name without team / tag). */
function tableRows() {
  const table = screen.getAllByRole("table")[0];
  return within(table)
    .getAllByRole("row")
    .slice(1)
    .map((row) => {
      const cells = within(row).getAllByRole("cell");
      return [row.querySelector(".driver-name")!.textContent, cells[3].textContent!.replace(/[^0-9<%]/g, "")];
    });
}

function expectedRows(race: Race, stage: string) {
  const run = race.runs.find((r) => r.mode === "live" && r.stage === stage)!;
  return run.drivers.map((d) => [race.drivers[d.driver_id].name, formatPct(d.p_win)]);
}

const round2 = testFile<Race>("races/2027-02.json");

describe("a race with only its pre-weekend prediction (round 3)", () => {
  it("shows that prediction and explains the chart comes later", async () => {
    serveTestData();
    renderPage();
    expect(await screen.findByRole("tab", { name: "Pre-weekend" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getAllByRole("tab")).toHaveLength(1);
    expect(tableRows()).toEqual(expectedRows(testFile<Race>("races/2027-03.json"), "pre_weekend"));
    expect(screen.getByText(/appears once a second prediction is logged/)).toBeInTheDocument();
  });
});

describe("a full weekend (round 2)", () => {
  it("opens on the latest stage, with every row exactly as logged", async () => {
    serveWithNext(2);
    renderPage();
    const tabs = await screen.findAllByRole("tab");
    expect(tabs.map((t) => t.textContent)).toEqual(["Pre-weekend", "After practice", "After qualifying"]);
    expect(screen.getByRole("tab", { name: "After qualifying" })).toHaveAttribute("aria-selected", "true");
    expect(tableRows()).toEqual(expectedRows(round2, "post_quali"));
    // When it was logged, and that it was before its cutoff (the race start).
    expect(screen.getByRole("tabpanel")).toHaveTextContent(/before the race \(Sun 14 Mar, 15:00 UTC\)/);
    expect(screen.getByRole("columnheader", { name: "Grid" })).toBeInTheDocument();
  });

  it("switches stage with the tabs, and tags the debutants at practice", async () => {
    serveWithNext(2);
    renderPage();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("tab", { name: "After practice" }));
    expect(tableRows()).toEqual(expectedRows(round2, "practice"));
    expect(screen.getByRole("columnheader", { name: "Practice" })).toBeInTheDocument();
    // The two debutants, under their temporary ids: OpenF1's names, tagged.
    const debuts = screen.getAllByText("debut").map((tag) => tag.closest("tr")!.querySelector(".driver-name")!.textContent);
    expect(debuts.sort()).toEqual(["Rookie N50", "Rookie N51"]);
    await user.click(screen.getByRole("tab", { name: "Pre-weekend" }));
    expect(tableRows()).toEqual(expectedRows(round2, "pre_weekend"));
    expect(screen.queryByText("debut")).toBeNull();
  });

  it("never shows a backtest run as the prediction", async () => {
    // A backtest replay of the same race, with an absurd 99% favourite.
    const race = structuredClone(round2);
    const fake = structuredClone(race.runs.find((r) => r.stage === "post_quali")!);
    fake.mode = "backtest";
    fake.run_id = 9999;
    fake.drivers[0].p_win = 0.99;
    race.runs.push(fake);
    serveWithNext(2, { "races/2027-02.json": { body: JSON.stringify(race) } });
    renderPage();
    expect(await screen.findAllByRole("tab")).toHaveLength(3);
    expect(tableRows()).toEqual(expectedRows(round2, "post_quali"));
  });
});

describe("the odds chart (round 2)", () => {
  const runs = liveRuns(round2);
  const series = buildSeries(runs);

  function lines(container: HTMLElement) {
    return [...container.querySelectorAll("svg.chart g[data-driver]")].map((g) => ({
      driver: g.getAttribute("data-driver"),
      slot: g.getAttribute("data-slot"),
      values: [...g.querySelectorAll("circle")].map((c) => Number(c.getAttribute("data-value"))),
    }));
  }

  it("draws the top eight by latest win chance, one point per stage, values as logged", async () => {
    serveWithNext(2);
    const { container } = renderPage();
    await screen.findByText("How the win chances moved");
    const drawn = lines(container);
    expect(drawn).toHaveLength(8);
    // Slots 1-8 go to the eight best by the latest (post-quali) win chance.
    const latest = runs[2].drivers.slice().sort((a, b) => b.p_win - a.p_win || a.predicted_rank - b.predicted_rank);
    expect(drawn.filter((l) => l.slot !== "other").sort((a, b) => Number(a.slot) - Number(b.slot)).map((l) => l.driver))
      .toEqual(latest.slice(0, 8).map((d) => d.driver_id));
    for (const line of drawn) {
      const expected = runs.map((r) => r.drivers.find((d) => d.driver_id === line.driver)?.p_win).filter((v) => v !== undefined);
      expect(line.values).toEqual(expected);
    }
    // Identity isn't colour alone: a legend with their names.
    expect(within(screen.getByRole("list", { name: "Drivers in the chart" })).getAllByRole("listitem")).toHaveLength(8);
  });

  it("adds every other driver in grey without repainting the eight", async () => {
    serveWithNext(2);
    const { container } = renderPage();
    const before = Object.fromEntries(lines(await screen.findByText("How the win chances moved").then(() => container)).map((l) => [l.driver, l.slot]));
    // 18 regulars + the two replaced (pre-weekend) + two temporary ids
    // (practice) + the two debutants' real ids (after qualifying).
    expect(series).toHaveLength(24);
    await userEvent.setup().click(screen.getByRole("button", { name: "Show all 24 drivers" }));
    const after = lines(container);
    expect(after).toHaveLength(24);
    expect(after.filter((l) => l.slot === "other")).toHaveLength(16);
    for (const [driver, slot] of Object.entries(before)) expect(after.find((l) => l.driver === driver)?.slot).toBe(slot);
    expect(screen.getByText("Other drivers")).toBeInTheDocument();
  });

  it("reads out each stage from the keyboard", async () => {
    serveWithNext(2);
    renderPage();
    const chart = await screen.findByRole("img", { name: /Win chance at each stage/ });
    const readout = screen.getByText(/Point at the chart/);
    fireEvent.focus(chart);
    const top = series[0];
    expect(readout).toHaveTextContent(`After qualifying: ${round2.drivers[top.driverId].code} ${formatPct(top.values[2]!)}`);
    fireEvent.keyDown(chart, { key: "ArrowLeft" });
    expect(readout).toHaveTextContent(/^After practice:/);
    fireEvent.keyDown(chart, { key: "ArrowLeft" });
    fireEvent.keyDown(chart, { key: "ArrowLeft" });              // stays on the first stage
    expect(readout).toHaveTextContent(/^Pre-weekend:/);
  });

  it("has a table view with the same numbers", async () => {
    serveWithNext(2);
    renderPage();
    await userEvent.setup().click(await screen.findByRole("button", { name: "Show as a table" }));
    const table = screen.getByRole("table", { name: "Win chance at each stage" });
    const rows = within(table).getAllByRole("row").slice(1);
    expect(rows).toHaveLength(8);
    const first = within(rows[0]).getAllByRole("cell").map((c) => c.textContent);
    expect(first).toEqual([round2.drivers[series[0].driverId].name, ...series[0].values.map((v) => (v === null ? "–" : formatPct(v)))]);
  });
});

describe("before anything is logged", () => {
  it("says when the first prediction comes", async () => {
    const index = testFile<Index>("index.json");
    index.seasons[0].races[2].file = null;            // round 3: nothing logged yet
    serveTestData({ "index.json": { body: JSON.stringify(index) } });
    renderPage();
    expect(await screen.findByText(/No prediction has been logged yet/)).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 1, name: "Test GP 3" })).toBeInTheDocument();
  });
});

describe("formatPct", () => {
  it("rounds, and never shows a real chance as 0%", () => {
    expect([formatPct(0.3712), formatPct(0.004), formatPct(0), formatPct(1)]).toEqual(["37%", "<1%", "0%", "100%"]);
  });
});

describe("spreadLabels (the line-end driver codes)", () => {
  it("keeps labels apart and inside the plot, even when many end near 0%", async () => {
    const { spreadLabels } = await import("../components/OddsChart");
    // Six drivers ending at almost the same height, at the very bottom.
    const labels = [10, 195, 198, 199, 200, 200, 200].map((y) => ({ y }));
    spreadLabels(labels, 0, 200);
    const ys = labels.map((l) => l.y);
    expect(Math.max(...ys)).toBeLessThanOrEqual(200);
    expect(Math.min(...ys)).toBeGreaterThanOrEqual(0);
    for (let i = 1; i < ys.length; i++) expect(ys[i] - ys[i - 1]).toBeGreaterThanOrEqual(13);
    expect(ys[0]).toBe(10); // a label with room stays where its line ends
  });
});

describe("a driver's name", () => {
  it("opens the Why page for that driver at that stage", async () => {
    serveWithNext(2);
    renderPage();
    const run = round2.runs.find((r) => r.mode === "live" && r.stage === "post_quali")!;
    const first = run.drivers[0];
    const link = await screen.findByRole("link", { name: round2.drivers[first.driver_id].name });
    expect(link).toHaveAttribute("href", `/why?race=2027-02&stage=post_quali&driver=${first.driver_id}`);
    await userEvent.setup().click(link);
    expect(await screen.findByRole("heading", { level: 1, name: "Why" })).toBeInTheDocument();
  });
});
