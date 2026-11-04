// Loading race files (useRace): each file once per visit, errors not cached.

import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useRace } from "../data";
import { serveTestData } from "./serve";

describe("useRace", () => {
  it("loads a race file, and only once however many places use it", async () => {
    const fetchMock = serveTestData();
    const first = renderHook(() => useRace("races/2027-01.json"));
    const second = renderHook(() => useRace("races/2027-01.json"));
    await waitFor(() => expect(first.result.current?.state).toBe("ready"));
    await waitFor(() => expect(second.result.current?.state).toBe("ready"));
    const race = first.result.current;
    expect(race?.state === "ready" && race.data.name).toBe("Test GP 1");
    expect(fetchMock.mock.calls.filter(([url]) => String(url).endsWith("races/2027-01.json"))).toHaveLength(1);
  });

  it("gives null for a race with no file (nothing logged yet)", () => {
    serveTestData();
    expect(renderHook(() => useRace(null)).result.current).toBeNull();
  });

  it("doesn't remember a failure: the next visit tries again", async () => {
    serveTestData({ "races/2027-01.json": { status: 500 } });
    const failed = renderHook(() => useRace("races/2027-01.json"));
    await waitFor(() => expect(failed.result.current?.state).toBe("error"));
    serveTestData();                                   // the file is back
    const retried = renderHook(() => useRace("races/2027-01.json"));
    await waitFor(() => expect(retried.result.current?.state).toBe("ready"));
  });
});
