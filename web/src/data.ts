// Loading the exported data files.
//
// The app never talks to a server or the database: it reads the static
// JSON files deployed next to it (data/index.json etc., written by
// scripts/export_dashboard.py after every pipeline run).
//
// Every file is checked before use: it must parse, be an object, and carry
// the schema version this app was built for. A file the app doesn't
// understand is reported as an error on the page, never shown half-read.

import { createContext, createElement, useContext, useEffect, useState, type ReactNode } from "react";
import { SCHEMA, type Index, type Race, type Scorecard, type Version } from "./types";

/** Where the data folder is, relative to the page (works on any host / path). */
export const DATA_URL = `${import.meta.env.BASE_URL}data/`;

export class DataError extends Error {}

/** Fetch one data file and check it's one this app can read. */
export async function loadFile<T extends { schema: number }>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(DATA_URL + path);
  } catch {
    throw new DataError(`Couldn't reach ${path} (offline?)`);
  }
  if (!response.ok) throw new DataError(`Couldn't load ${path} (HTTP ${response.status})`);
  let data: unknown;
  try {
    data = await response.json();
  } catch {
    throw new DataError(`${path} isn't valid JSON`);
  }
  if (typeof data !== "object" || data === null) throw new DataError(`${path} isn't a data file`);
  const schema = (data as { schema?: unknown }).schema;
  if (schema !== SCHEMA) {
    // The export changed shape and this build of the app doesn't know it
    // (or the other way round). Better a clear message than wrong numbers.
    throw new DataError(`${path} is in format ${String(schema)}; this page reads format ${SCHEMA}`);
  }
  return data as T;
}

// --- The site-wide files: loaded once, shared by every page --------------------

export type Loadable<T> =
  | { state: "loading" }
  | { state: "error"; message: string }
  | { state: "ready"; data: T };

export interface SiteData {
  index: Index;
  version: Version;
  scorecard: Scorecard;
}

const SiteDataContext = createContext<Loadable<SiteData>>({ state: "loading" });

function errorMessage(error: unknown): string {
  return error instanceof DataError ? error.message : "Something went wrong loading the data";
}

export function SiteDataProvider({ children }: { children: ReactNode }) {
  const [site, setSite] = useState<Loadable<SiteData>>({ state: "loading" });
  useEffect(() => {
    let cancelled = false;
    Promise.all([loadFile<Index>("index.json"), loadFile<Version>("version.json"), loadFile<Scorecard>("scorecard.json")])
      .then(([index, version, scorecard]) => !cancelled && setSite({ state: "ready", data: { index, version, scorecard } }))
      .catch((error) => !cancelled && setSite({ state: "error", message: errorMessage(error) }));
    return () => {
      cancelled = true; // the page went away before the files arrived
    };
  }, []);
  return createElement(SiteDataContext.Provider, { value: site }, children);
}

export function useSiteData(): Loadable<SiteData> {
  return useContext(SiteDataContext);
}

// --- One race file: loaded when a page needs it, kept for the visit ------------

const raceCache = new Map<string, Promise<Race>>();

/** For tests: forget loaded races. */
export function clearRaceCache() {
  raceCache.clear();
}

/** A race file by its path from index.json ("races/2026-20.json"); null = none. */
export function useRace(file: string | null): Loadable<Race> | null {
  const [race, setRace] = useState<Loadable<Race>>({ state: "loading" });
  useEffect(() => {
    if (!file) return;
    let cancelled = false;
    setRace({ state: "loading" });
    if (!raceCache.has(file)) raceCache.set(file, loadFile<Race>(file));
    raceCache
      .get(file)!
      .then((data) => !cancelled && setRace({ state: "ready", data }))
      .catch((error) => {
        raceCache.delete(file); // let a later visit try again
        if (!cancelled) setRace({ state: "error", message: errorMessage(error) });
      });
    return () => {
      cancelled = true;
    };
  }, [file]);
  return file ? race : null;
}
