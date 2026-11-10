// The Results tab's Stats (the scorecard), against the simulated test world's export (one
// scorecard: 2027 live, two races) and a hand-made second one.

import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { AppRoutes } from "../App";
import { SiteDataProvider } from "../data";
import { comparison, METRICS } from "../scorecard";
import type { Scorecard, ScorecardGroup } from "../types";
import { serveTestData, testFile } from "./serve";

const card = testFile<Scorecard>("scorecard.json");
const live = card.groups[0];

function renderScorecard(scorecard: Scorecard = card) {
  serveTestData({ "scorecard.json": { body: JSON.stringify(scorecard) } });
  return render(
    <SiteDataProvider>
      <MemoryRouter initialEntries={["/results/stats"]}>
        <AppRoutes />
      </MemoryRouter>
    </SiteDataProvider>,
  );
}

/** The test world's live scorecard, relabelled as a backtest of another season. */
function backtestCopy(): ScorecardGroup {
  return { ...structuredClone(live), season: 2026, mode: "backtest", label: "2026 backtest (in-sample: a pipeline check, not a test result)" };
}

describe("which scorecard", () => {
  it("shows the first (most important) one by default, with its label up front", async () => {
    renderScorecard({ ...card, groups: [live, backtestCopy()] });
    expect(await screen.findByText(live.label)).toHaveClass("banner");
    expect(screen.getByRole("combobox", { name: "Scorecard" })).toHaveValue("2027-live");
  });

  it("switches to another scorecard, and its label comes with it", async () => {
    renderScorecard({ ...card, groups: [live, backtestCopy()] });
    await userEvent.setup().selectOptions(await screen.findByRole("combobox", { name: "Scorecard" }), "2026-backtest");
    expect(screen.getByText(/2026 backtest \(in-sample/)).toHaveClass("banner");
    expect(screen.queryByText(live.label)).toBeNull();
  });

  it("warns that two races are too few to judge", async () => {
    renderScorecard();
    expect(await screen.findByText(/Only 2 races so far: too few to judge/)).toBeInTheDocument();
  });

  it("says when nothing is scored yet", async () => {
    renderScorecard({ ...card, groups: [] });
    expect(await screen.findByText(/Nothing has been scored yet/)).toBeInTheDocument();
  });
});

describe("the summary table", () => {
  it("shows every measure at every stage exactly as in the file", async () => {
    const { container } = renderScorecard();
    await screen.findByText("Model against baseline");
    for (const stage of ["pre_weekend", "practice", "post_quali"] as const) {
      for (const m of METRICS) {
        const metric = live.stages[stage]![m.key];
        const cell = container.querySelector(`td[data-stage="${stage}"][data-metric="${m.key}"]`)!;
        expect(cell).toHaveTextContent(`${m.format(metric.model!)} vs ${m.format(metric.baseline!)}`);
        // "% better" needs a non-zero baseline: in the fake world one
        // baseline scores a perfect 0, and then no chip is shown.
        const { text } = comparison(metric);
        if (text) expect(cell).toHaveTextContent(text);
        else expect(cell.querySelector(".chip")).toBeNull();
      }
    }
    // Each column says which baseline and how many races.
    expect(screen.getByRole("columnheader", { name: /After qualifying\s*vs starting grid · 2 races/ })).toBeInTheDocument();
  });

  it("calls better and worse by each measure's own direction", () => {
    // Places off: lower is better. improvement is already signed so that +
    // means better (Python's scorer.summary), whichever way the measure runs.
    const base = { races: 5, baseline_name: "grid_order" };
    expect(comparison({ ...base, model: 3.0, baseline: 3.3, improvement: 0.0909, higher_is_better: false }).text).toBe("▲ 9% better");
    expect(comparison({ ...base, model: 0.5, baseline: 0.6, improvement: -0.1667, higher_is_better: true }).text).toBe("▼ 17% worse");
    expect(comparison({ ...base, model: 0.5, baseline: 0.5, improvement: 0.001, higher_is_better: true }).text).toBe("about the same");
  });

  it("counts the driver-list misses from the race-by-race scores", async () => {
    const { container } = renderScorecard();
    await screen.findByText("Model against baseline");
    // Round 2's pre-weekend prediction had d05 and d12 (replaced by the debutants).
    expect(container.querySelector('td[data-stage="pre_weekend"][data-metric="lineup"]')).toHaveTextContent("2 missed · 2 didn't start");
    expect(container.querySelector('td[data-stage="practice"][data-metric="lineup"]')).toHaveTextContent("0 missed · 0 didn't start");
  });
});

describe("race by race", () => {
  it("has a bar per scored race, on the right side, for the chosen stage", async () => {
    const { container } = renderScorecard();
    const user = userEvent.setup();
    await screen.findByText("Race by race");
    for (const [tab, stage] of [["After qualifying", "post_quali"], ["Pre-weekend", "pre_weekend"]] as const) {
      await user.click(screen.getByRole("tab", { name: tab }));
      const bars = [...container.querySelectorAll(".race-bars li.factor")];
      const races = live.per_race[stage]!;
      expect(bars.map((b) => Number(b.getAttribute("data-round")))).toEqual(races.map((r) => r.round));
      bars.forEach((bar, i) => {
        const gain = races[i].metrics.position_mae.baseline! - races[i].metrics.position_mae.model!;
        expect(Number(bar.getAttribute("data-gain"))).toBeCloseTo(gain, 9);
        // Model closer = bar on the right; baseline closer = on the left.
        expect(bar.querySelector(gain > 0 ? ".right .bar" : ".left .bar")).not.toBeNull();
      });
    }
  });
});

describe("calibration", () => {
  it("plots each group of predictions where the file puts it, podium then win", async () => {
    const { container } = renderScorecard();
    const user = userEvent.setup();
    await screen.findByText("Calibration");
    const dots = () => [...container.querySelectorAll("circle[data-from]")].map((c) => [
      Number(c.getAttribute("data-from")), Number(c.getAttribute("data-predicted")), Number(c.getAttribute("data-observed"))]);
    const expected = (o: "podium" | "win") => live.calibration.post_quali![o].map((b) => [b.from, b.predicted, b.observed]);
    expect(dots()).toEqual(expected("podium"));
    await user.click(screen.getByRole("tab", { name: "Win chances" }));
    expect(dots()).toEqual(expected("win"));
  });

  it("reads each group out from the keyboard, and has a table view", async () => {
    renderScorecard();
    const user = userEvent.setup();
    const chart = await screen.findByRole("img", { name: /Calibration of podium chances/ });
    fireEvent.focus(chart);
    const first = live.calibration.post_quali!.podium[0];
    expect(screen.getByText(new RegExp(`^predicted 0%–5%`))).toHaveTextContent(`${first.n} predictions`);
    await user.click(screen.getAllByRole("button", { name: "Show as a table" })[0]);
    const table = screen.getByRole("table", { name: "Calibration of podium chances" });
    expect(within(table).getAllByRole("row")).toHaveLength(live.calibration.post_quali!.podium.length + 1);
  });
});
