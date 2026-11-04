import { describe, expect, it } from "vitest";
import { formatUtc } from "../format";

describe("formatUtc", () => {
  it("shows the UTC time whatever the visitor's timezone", () => {
    // 23:30 UTC on a Saturday is already Sunday in Tokyo; the page still
    // says Saturday 23:30 UTC, the same instant the pipeline's cutoffs use.
    expect(formatUtc("2026-11-07T23:30:00Z")).toBe("Sat 7 Nov, 23:30 UTC");
    expect(formatUtc("2027-03-01T05:05:00Z")).toBe("Mon 1 Mar, 05:05 UTC");
  });
  it("copes with a missing or unreadable time", () => {
    expect(formatUtc(null)).toBe("time to be confirmed");
    expect(formatUtc("soon")).toBe("soon");
  });
});
