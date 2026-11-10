// The weekend tracker and the "how we did" card.
//
// The test world's round 3 (Test GP 3) has only its pre-weekend prediction:
//   FP1 Fri 19 Mar 12:00, FP2 16:00, FP3 Sat 20 Mar 11:00,
//   Q Sat 20 Mar 15:00, R Sun 21 Mar 15:00 (UTC, 2027).
// Round 2 (a week earlier, race Sun 14 Mar 15:00) has all three stages and
// its result. Tests walk the clock through the weekend with fake dates.

import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AppRoutes } from "../App";
import { SiteDataProvider } from "../data";
import { recentRace, stillToCome, weekendSteps } from "../weekend";
import type { Index, Race, Stage } from "../types";
import { serveTestData, testFile } from "./serve";

const round2 = testFile<Race>("races/2027-02.json");
const round3 = testFile<Race>("races/2027-03.json");

/** A copy of `race` with only these live stages logged, and not finished. */
function withStages(race: Race, stages: Stage[]): Race {
  return {
    ...race,
    status: "predicted",
    result: null,
    runs: race.runs.filter((r) => r.mode === "live" && stages.includes(r.stage)),
  };
}

const at = (iso: string) => new Date(iso);
const states = (steps: ReturnType<typeof weekendSteps>) => steps.map((s) => `${s.key}:${s.state}`);

describe("weekendSteps", () => {
  it("before practice: pre-weekend done, after-practice is next", () => {
    const now = at("2027-03-18T10:00:00Z");
    const steps = weekendSteps(round3, now, now);
    expect(states(steps)).toEqual(["pre_weekend:done", "practice:current", "post_quali:upcoming", "result:upcoming"]);
    // Not ready yet (FP3 hasn't ended): just the deadline.
    expect(steps[1]).toMatchObject({ detail: "Due before qualifying", time: "2027-03-20T15:00:00Z" });
  });

  it("after FP3, before qualifying: after-practice is due before qualifying", () => {
    const now = at("2027-03-20T13:00:00Z");
    const steps = weekendSteps(round3, now, now);
    expect(steps[1]).toMatchObject({ state: "current", detail: "Being made, due before qualifying", time: "2027-03-20T15:00:00Z" });
  });

  it("practice logged, qualifying running: after-qualifying is next", () => {
    const race = withStages(round2, ["pre_weekend", "practice"]);
    const now = at("2027-03-13T15:30:00Z");
    expect(states(weekendSteps(race, now, now))).toEqual([
      "pre_weekend:done", "practice:done", "post_quali:current", "result:upcoming",
    ]);
  });

  it("calls a stage missed only when the data is newer than its cutoff", () => {
    const race = withStages(round2, ["pre_weekend"]); // practice never logged
    const now = at("2027-03-13T16:00:00Z"); // qualifying (15:00) has started
    // Data exported before qualifying: we can't tell yet, so it's still due.
    expect(weekendSteps(race, now, at("2027-03-13T14:00:00Z"))[1].state).toBe("current");
    // Data exported after it: missed, and the next step is current.
    const steps = weekendSteps(race, now, at("2027-03-13T15:10:00Z"));
    expect(states(steps)).toEqual(["pre_weekend:done", "practice:missed", "post_quali:current", "result:upcoming"]);
  });

  it("a finished race: everything done", () => {
    const now = at("2027-03-15T10:00:00Z");
    expect(states(weekendSteps(round2, now, now))).toEqual([
      "pre_weekend:done", "practice:done", "post_quali:done", "result:done",
    ]);
  });

  it("no qualifying on the calendar: practice closes at the race, no post-quali", () => {
    const race = withStages({ ...round3, sessions: { ...round3.sessions, Q: null } }, ["pre_weekend"]);
    const now = at("2027-03-20T13:00:00Z");
    const steps = weekendSteps(race, now, now);
    expect(steps[1]).toMatchObject({ state: "current", detail: "Being made, due before the race" });
    expect(steps[2].detail).toBe("No qualifying on the calendar");
  });
});

describe("stillToCome", () => {
  it("lists what's left in time order, each prediction before its cutoff session", () => {
    const items = stillToCome(round3, at("2027-03-20T12:30:00Z"));
    expect(items.map((i) => i.what)).toEqual([
      "Prediction after practice", "Qualifying", "Final prediction (after qualifying)", "Race", "Result and scoring",
    ]);
  });

  it("is empty once the race has run and been scored", () => {
    expect(stillToCome(round2, at("2027-03-15T10:00:00Z"))).toEqual([]);
  });
});

describe("recentRace", () => {
  const index = testFile<Index>("index.json");
  it("is the race that started in the last ~2 days", () => {
    expect(recentRace(index.seasons, at("2027-03-15T10:00:00Z"))?.round).toBe(2);
  });
  it("is nothing before a race or long after it", () => {
    // An hour before round 2 starts: round 1 (a week ago) is long over.
    expect(recentRace(index.seasons, at("2027-03-14T14:00:00Z"))).toBeNull();
    // 51h after round 2 started: its window (50h) has closed.
    expect(recentRace(index.seasons, at("2027-03-16T18:00:00Z"))).toBeNull();
  });
});

// --- The page ------------------------------------------------------------------

function renderAt(iso: string) {
  // Fake only Date: the page reads the clock, the test library's timers stay real.
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(new Date(iso));
  return render(
    <SiteDataProvider>
      <MemoryRouter>
        <AppRoutes />
      </MemoryRouter>
    </SiteDataProvider>,
  );
}

afterEach(() => {
  vi.useRealTimers();
});

describe("the Next race page during a weekend", () => {
  it("shows the tracker and what's still to come", async () => {
    serveTestData();
    renderAt("2027-03-18T10:00:00Z");
    const tracker = (await screen.findByRole("heading", { name: "This weekend" })).parentElement!;
    const steps = within(tracker).getAllByRole("listitem");
    expect(steps.map((s) => s.className)).toEqual([
      "weekend-step is-done", "weekend-step is-current", "weekend-step is-upcoming", "weekend-step is-upcoming",
    ]);
    expect(steps[1]).toHaveTextContent("After practice (next)");
    expect(tracker).toHaveTextContent(/Still to come: Practice 1 \(Fri 19 Mar, 12:00 UTC\) →/);
    expect(screen.getByRole("heading", { name: "Current prediction: pre-weekend" })).toBeInTheDocument();
  });

  it("opens an earlier stage from the tracker", async () => {
    // Round 2 as the next race, with all three stages logged.
    const index = testFile<Index>("index.json");
    index.next_race = { season: 2027, round: 2 };
    serveTestData({ "index.json": { body: JSON.stringify(index) } });
    renderAt("2027-03-14T13:00:00Z");
    const pre = await screen.findByRole("button", { name: /Pre-weekend/ });
    expect(screen.getByRole("heading", { name: "Current prediction: after qualifying" })).toBeInTheDocument();
    await userEvent.click(pre);
    expect(pre).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("heading", { name: "Earlier prediction: pre-weekend" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Pre-weekend" })).toHaveAttribute("aria-selected", "true");
  });
});

describe("the how-we-did card", () => {
  it("shows the last race's real podium next to our final prediction, for ~2 days", async () => {
    serveTestData();
    renderAt("2027-03-15T10:00:00Z");
    const card = (await screen.findByRole("heading", { name: "Test GP 2: how we did" })).closest(".card")!;
    const final = round2.runs.find((r) => r.mode === "live" && r.stage === "post_quali")!;
    const podium = [...round2.result!].sort((a, b) => a.position - b.position).slice(0, 3);
    const rows = within(card as HTMLElement).getAllByRole("row").slice(1);
    rows.forEach((row, i) => {
      expect(row).toHaveTextContent(round2.drivers[podium[i].driver_id].name);
    });
    const winnerChance = final.drivers.find((d) => d.driver_id === podium[0].driver_id)!.p_win;
    expect(card).toHaveTextContent(`a ${Math.round(winnerChance * 100)}% chance to win`);
    expect(within(card as HTMLElement).getByRole("link", { name: /full result/ })).toHaveAttribute("href", "/races/2027-02");
  });

  it("is gone two days later", async () => {
    serveTestData();
    renderAt("2027-03-17T10:00:00Z");
    await screen.findByRole("heading", { name: "This weekend" });
    expect(screen.queryByText(/how we did/)).not.toBeInTheDocument();
  });
});
