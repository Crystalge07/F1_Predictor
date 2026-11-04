// Runs before every test file: adds DOM matchers like toBeInTheDocument().
import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";
import { clearRaceCache } from "../data";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  clearRaceCache();
});
