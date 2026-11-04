// The Why page and the breakdown behind it, against the simulated test world.

import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { AppRoutes } from "../App";
import { SiteDataProvider } from "../data";
import { breakdown, TOP_FACTORS } from "../explain";
import { formatFactorValue, liveRuns } from "../race";
import { summary } from "../pages/Why";
import type { Race } from "../types";
import { serveTestData, testFile } from "./serve";

const round2 = testFile<Race>("races/2027-02.json");
const round3 = testFile<Race>("races/2027-03.json");

function renderWhy(path = "/why") {
  serveTestData();
  return render(
    <SiteDataProvider>
      <MemoryRouter initialEntries={[path]}>
        <AppRoutes />
      </MemoryRouter>
    </SiteDataProvider>,
  );
}

function bars(container: HTMLElement) {
  return [...container.querySelectorAll("li.factor")].map((li) => ({
    feature: li.getAttribute("data-feature"),
    places: Number(li.getAttribute("data-places")),
    side: li.querySelector(".bar")?.classList.contains("forward") ? "forward" : "back",
  }));
}

describe("the breakdown", () => {
  it("has one starting point per run: the data's SHAP values are complete", () => {
    // SHAP's starting point (the model's average prediction) is the same
    // number for every driver of a run. Worked out per driver as
    // predicted - sum(pushes), it must agree across the field; if a factor
    // were missing or wrong in the export, it wouldn't.
    for (const run of liveRuns(round2)) {
      const starts = run.drivers.map((d) => breakdown(d)!.start);
      expect(Math.max(...starts) - Math.min(...starts)).toBeLessThan(0.01);
    }
  });

  it("keeps the biggest factors, folds the rest into one, and still adds up", () => {
    const driver = liveRuns(round2)[1].drivers[0];
    const data = breakdown(driver)!;
    expect(data.pushes).toHaveLength(Math.min(driver.factors.length, TOP_FACTORS + 1));
    const named = data.pushes.filter((p) => p.feature !== null).map((p) => Math.abs(p.places));
    expect(named).toEqual([...named].sort((a, b) => b - a));
    const sum = data.pushes.reduce((s, p) => s + p.places, 0);
    expect(data.start + sum).toBeCloseTo(driver.expected_position!, 6);
  });
});

describe("the Why page", () => {
  it("opens on the next race's predicted winner, with every bar as logged", async () => {
    const { container } = renderWhy();
    const winner = liveRuns(round3)[0].drivers[0];
    const data = breakdown(winner)!;
    expect(await screen.findByText(summary(round3, winner.driver_id, data))).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Race" })).toHaveValue("2027-03");
    expect(screen.getByRole("combobox", { name: "Driver" })).toHaveValue(winner.driver_id);
    // Bars: same factors, same numbers, forward on the left for negative pushes.
    expect(bars(container)).toEqual(
      data.pushes.map((p) => ({ feature: p.feature ?? "other", places: p.places, side: p.places < 0 ? "forward" : "back" })),
    );
  });

  it("switches race, stage and driver", async () => {
    renderWhy();
    const user = userEvent.setup();
    await user.selectOptions(await screen.findByRole("combobox", { name: "Race" }), "2027-02");
    // A new race opens on its latest stage.
    expect(await screen.findByRole("tab", { name: "After qualifying" })).toHaveAttribute("aria-selected", "true");
    await user.click(screen.getByRole("tab", { name: "After practice" }));
    const practice = liveRuns(round2)[1];
    const third = practice.drivers[2];
    await user.selectOptions(screen.getByRole("combobox", { name: "Driver" }), third.driver_id);
    expect(screen.getByText(summary(round2, third.driver_id, breakdown(third)!))).toBeInTheDocument();
  });

  it("can be linked to directly", async () => {
    renderWhy("/why?race=2027-02&stage=practice&driver=d03");
    expect(await screen.findByText(/^Test D03: predicted P/)).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "After practice" })).toHaveAttribute("aria-selected", "true");
  });

  it("shows a debutant's missing history as 'no data', not as zero", async () => {
    renderWhy("/why?race=2027-02&stage=practice&driver=provisional_2027_r02_car50");
    expect(await screen.findByText(/^Rookie N50: predicted P/)).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /Rookie N50 \(debut\)/ })).toBeInTheDocument();
    const debutant = liveRuns(round2)[1].drivers.find((d) => d.driver_id === "provisional_2027_r02_car50")!;
    const shownWithoutData = breakdown(debutant)!.pushes.filter((p) => p.feature !== null && p.value === null);
    expect(shownWithoutData.length).toBeGreaterThan(0);                 // (the test means something)
    expect(screen.getAllByText("no data")).toHaveLength(shownWithoutData.length);
  });

  it("has a table view that adds up to the prediction", async () => {
    renderWhy();
    await userEvent.setup().click(await screen.findByRole("button", { name: "Show as a table" }));
    const table = screen.getByRole("table", { name: "Factors and their effect, in places" });
    const rows = within(table).getAllByRole("row").slice(1);
    const data = breakdown(liveRuns(round3)[0].drivers[0])!;
    expect(rows).toHaveLength(data.pushes.length + 2);               // start, factors, predicted
    expect(rows[0]).toHaveTextContent(`P${data.start.toFixed(1)}`);
    expect(rows[rows.length - 1]).toHaveTextContent(`P${data.predicted.toFixed(1)}`);
  });

  it("says what the factors do and don't explain", async () => {
    renderWhy("/why?race=2027-02&stage=post_quali");
    expect(await screen.findByText(/not the exact percentages/)).toHaveTextContent("a blend with the starting grid");
  });
});

describe("wording", () => {
  it("names the biggest real reasons, with the right direction", () => {
    const race = round2;
    const text = summary(race, "d01", {
      predicted: 2.5,
      start: 9.6,
      pushes: [
        { feature: "driver_form_finish", value: 2.4, places: -1.97 },
        { feature: "team_season_points", value: 607, places: 1.08 },
        { feature: "grid", value: 3, places: 0.01 },                 // no real effect: left out
        { feature: null, value: null, places: -0.2 },
      ],
    });
    expect(text).toBe(
      "Test D01: predicted P2.5, against P9.6 for a typical driver. Biggest reasons: " +
        "recent finishes (last 5): avg P2.4, 2.0 places forward; " +
        "team championship points: 607 pts, 1.1 places back.",
    );
  });

  it("never shows a tiny push as -0.0", async () => {
    const { formatSignedPlaces } = await import("../race");
    expect([formatSignedPlaces(-3.08), formatSignedPlaces(0.4), formatSignedPlaces(-0.03), formatSignedPlaces(0.004, 2)])
      .toEqual(["−3.1", "+0.4", "0.0", "0.00"]);
  });

  it("shows each factor's value in its own unit", () => {
    expect([
      formatFactorValue("grid", 5), formatFactorValue("quali_gap_pct", 0.456), formatFactorValue("driver_season_points", 365),
      formatFactorValue("driver_points_per_race", 20.28), formatFactorValue("driver_form_finish", 2.4),
      formatFactorValue("pit_lane_start", 1), formatFactorValue("driver_track_finish", null),
    ]).toEqual(["P5", "0.46%", "365 pts", "20.3", "avg P2.4", "yes", "no data"]);
  });
});
