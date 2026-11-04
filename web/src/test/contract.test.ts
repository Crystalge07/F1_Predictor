// The contract: the files the Python export writes are files this app reads.
//
// web/src/test/data is a real export (of the simulated test world), and a
// Python test (tests/test_export_dashboard.py) keeps it equal to what the
// export writes today. So if either side changes the format without the
// other, one of the two test suites fails.

import { describe, expect, it } from "vitest";
import { loadFile } from "../data";
import { SCHEMA, type Index, type Race, type Scorecard, type Version } from "../types";
import { serveTestData, testFile } from "./serve";

describe("the exported files", () => {
  it("all load and carry the schema this app reads", async () => {
    serveTestData();
    const index = await loadFile<Index>("index.json");
    await loadFile<Version>("version.json");
    await loadFile<Scorecard>("scorecard.json");
    // Every race file the index points to exists and loads.
    const files = index.seasons.flatMap((s) => s.races.map((r) => r.file)).filter((f): f is string => f !== null);
    expect(files).toEqual(["races/2027-01.json", "races/2027-02.json", "races/2027-03.json"]);
    for (const file of files) expect((await loadFile<Race>(file)).schema).toBe(SCHEMA);
  });

  it("have the fields the types promise, with the right kinds of values", () => {
    // TypeScript can't check JSON at runtime; this spot-checks one of each
    // shape against the declared types (src/types.ts).
    const race = testFile<Race>("races/2027-01.json");
    expect(race.status).toBe("finished");
    const run = race.runs[0];
    expect(run.stage).toBe("pre_weekend");
    expect(run.mode).toBe("live");
    const driver = run.drivers[0];
    for (const key of ["p_win", "p_podium", "p_points", "p_dnf"] as const) {
      expect(typeof driver[key]).toBe("number");
      expect(driver[key]).toBeGreaterThanOrEqual(0);
      expect(driver[key]).toBeLessThanOrEqual(1);
    }
    expect(typeof driver.factors[0].contribution).toBe("number");
    expect(typeof run.scores?.position_mae.model).toBe("number");
    expect(race.result?.[0].position).toBe(1);
    // Every id in the file has a name.
    for (const r of race.runs) for (const d of r.drivers) expect(race.drivers[d.driver_id]?.name).toBeTruthy();

    const scorecard = testFile<Scorecard>("scorecard.json");
    expect(scorecard.groups[0].mode).toBe("live");
    expect(scorecard.groups[0].label).toMatch(/^2027 live/);
    expect(typeof scorecard.groups[0].stages.post_quali?.position_mae.races).toBe("number");
  });

  it("describe a debutant's temporary id, and who he turned out to be", () => {
    const race = testFile<Race>("races/2027-02.json");
    expect(race.drivers["provisional_2027_r02_car50"]).toEqual({
      name: "Rookie N50", code: null, provisional: true, resolved_as: "n50",
    });
  });
});
