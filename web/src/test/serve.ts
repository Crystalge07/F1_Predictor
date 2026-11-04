// A stand-in for the browser's fetch that serves the test data files
// (web/src/test/data, written by scripts/export_test_data.py from the
// simulated test world). Tests can replace any file's answer, to see how
// the app copes with a missing, broken or unknown file.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { vi } from "vitest";

// Tests run from web/ (npm test), so the data folder is found from there.
const DATA = join(process.cwd(), "src", "test", "data");

/** The test data file at `path` ("races/2027-01.json"), parsed. */
export function testFile<T = unknown>(path: string): T {
  return JSON.parse(readFileSync(join(DATA, path), "utf8")) as T;
}

type Answer = { status?: number; body?: string; networkError?: boolean };

/** Serve the test data through fetch. overrides: {"index.json": {status: 404}}.
 * Returns the mock, so tests can count requests. */
export function serveTestData(overrides: Record<string, Answer> = {}) {
  const fetchMock = vi.fn(async (url: string | URL) => {
    const path = String(url).split("data/").pop()!;
    const answer = overrides[path];
    if (answer?.networkError) throw new TypeError("Failed to fetch");
    if (answer) return new Response(answer.body ?? "", { status: answer.status ?? 200 });
    try {
      return new Response(readFileSync(join(DATA, path), "utf8"), { status: 200 });
    } catch {
      return new Response("", { status: 404 });
    }
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
