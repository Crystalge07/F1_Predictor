// Past races and the launch polish (About, tab titles, skip link), against
// the simulated test world: rounds 1 and 2 finished, round 3 not yet.

import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { AppRoutes } from "../App";
import { SiteDataProvider } from "../data";
import { liveRuns } from "../race";
import type { Race } from "../types";
import { serveTestData, testFile } from "./serve";

const round2 = testFile<Race>("races/2027-02.json");

function renderAt(path: string, overrides: Parameters<typeof serveTestData>[0] = {}) {
  serveTestData(overrides);
  return render(
    <SiteDataProvider>
      <MemoryRouter initialEntries={[path]}>
        <AppRoutes />
      </MemoryRouter>
    </SiteDataProvider>,
  );
}

describe("the list", () => {
  it("lists the finished races, newest first, and opens one", async () => {
    renderAt("/races");
    const links = await screen.findAllByRole("link", { name: /^2027 R\d/ });
    // Round 3 hasn't been raced: not a past race yet.
    expect(links.map((l) => l.textContent)).toEqual(["2027 R2 · Test GP 2live", "2027 R1 · Test GP 1live"]);
    await userEvent.setup().click(links[0]);
    expect(await screen.findByRole("heading", { level: 1, name: "Test GP 2" })).toBeInTheDocument();
  });
});

describe("one race", () => {
  it("lines every stage's prediction up with the result, debutants included", async () => {
    const { container } = renderAt("/races/2027-02");
    await screen.findByText("Predicted against the result");
    const rows = [...container.querySelectorAll("table.comparison tbody tr")];
    // In finishing order.
    expect(rows.map((r) => r.getAttribute("data-driver"))).toEqual(round2.result!.map((r) => r.driver_id));
    for (const run of liveRuns(round2)) {
      for (const result of round2.result!) {
        // The run's prediction for this driver: a debutant's temporary id
        // counts as who he turned out to be.
        const d = run.drivers.find((p) => (round2.drivers[p.driver_id].resolved_as ?? p.driver_id) === result.driver_id);
        const cell = container.querySelector(`tr[data-driver="${result.driver_id}"] td[data-stage="${run.stage}"]`)!;
        if (!d) {
          expect(cell).toHaveTextContent("missed");
          continue;
        }
        const off = result.position - d.predicted_rank;
        expect(cell).toHaveTextContent(`P${d.predicted_rank}${off === 0 ? " ✓" : ` ${off > 0 ? "+" : "−"}${Math.abs(off)}`}`);
      }
    }
    // The debutants were in the practice prediction (temporary ids), not the pre-weekend one.
    expect(container.querySelector('tr[data-driver="n50"] td[data-stage="practice"]')).not.toHaveTextContent("missed");
    expect(container.querySelector('tr[data-driver="n50"] td[data-stage="pre_weekend"]')).toHaveTextContent("missed");
  });

  it("names the predicted drivers who didn't start", async () => {
    renderAt("/races/2027-02");
    // The pre-weekend prediction copied the last race's lineup: d05 and d12.
    expect(await screen.findByText("Pre-weekend also predicted Test D05, Test D12, who didn't start.")).toBeInTheDocument();
  });

  it("shows the race's scores, model against baseline", async () => {
    renderAt("/races/2027-02");
    const table = await screen.findByRole("table", { name: "This race's scores, model against baseline" });
    const post = liveRuns(round2).find((r) => r.stage === "post_quali")!.scores!;
    const row = within(table).getByRole("row", { name: /After qualifying/ });
    expect(row).toHaveTextContent(`${post.position_mae.model.toFixed(2)} vs ${post.position_mae.baseline!.toFixed(2)}`);
    expect(row).toHaveTextContent(`${Math.round(post.podium_hits.model * 3)}/3`);
  });

  it("keeps a backtest replay apart, under its own label", async () => {
    const race = structuredClone(round2);
    const replay = structuredClone(race.runs.find((r) => r.stage === "post_quali")!);
    replay.mode = "backtest";
    race.runs.push(replay);
    renderAt("/races/2027-02", { "races/2027-02.json": { body: JSON.stringify(race) } });
    await userEvent.setup().click(await screen.findByRole("tab", { name: "Backtest" }));
    expect(screen.getByRole("note")).toHaveTextContent(/in-sample: a pipeline check, not a test result/);
    expect(screen.getByRole("columnheader", { name: "After qualifying" })).toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "Pre-weekend" })).toBeNull();
  });

  it("handles a race that hasn't finished, and one that doesn't exist", async () => {
    renderAt("/races/2027-03");
    expect(await screen.findByText(/The result isn't in yet/)).toBeInTheDocument();
    renderAt("/races/1999-01");
    expect(await screen.findByRole("heading", { level: 1, name: "Race not found" })).toBeInTheDocument();
  });
});

describe("launch polish", () => {
  it("has an About page in the menu", async () => {
    renderAt("/");
    await userEvent.setup().click(await screen.findByRole("link", { name: "About" }));
    expect(screen.getByRole("heading", { level: 1, name: "How it works" })).toBeInTheDocument();
  });

  it("names each page in the browser tab", async () => {
    renderAt("/scorecard");
    await screen.findByRole("heading", { level: 1, name: "Scorecard" });
    expect(document.title).toBe("Scorecard · F1 Predictor");
    renderAt("/races/2027-01");
    await screen.findByRole("heading", { level: 1, name: "Test GP 1" });
    expect(document.title).toBe("Past race · F1 Predictor");
  });

  it("lets keyboard users skip the menu", async () => {
    renderAt("/about");
    const user = userEvent.setup();
    await user.tab();
    const skip = screen.getByRole("link", { name: "Skip to content" });
    expect(skip).toHaveFocus();                     // the first thing on the page
    await user.keyboard("{Enter}");
    expect(screen.getByRole("main")).toHaveFocus();
  });
});
