// Formatting shared by every page.

const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "Sun 8 Nov, 17:00 UTC": times are stored in UTC and shown in UTC, so
 * everyone sees the same thing (the pipeline's cutoffs are UTC). Built by
 * hand, not with toLocaleString, whose punctuation differs between browsers. */
export function formatUtc(iso: string | null | undefined): string {
  if (!iso) return "time to be confirmed";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const two = (n: number) => String(n).padStart(2, "0");
  return `${DAYS[date.getUTCDay()]} ${date.getUTCDate()} ${MONTHS[date.getUTCMonth()]}, ` +
    `${two(date.getUTCHours())}:${two(date.getUTCMinutes())} UTC`;
}

export const STAGE_NAMES = {
  pre_weekend: "Pre-weekend",
  practice: "After practice",
  post_quali: "After qualifying",
} as const;
