/**
 * Display helpers for run metrics.
 *
 * A metric the backend could not compute (Ragas failed, or no eval item it
 * applies to) is null. Null renders as "—" and is never treated as 0.
 */

export const NOT_SCORED = "—";

export function isScored(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function formatMetric(value: number | null | undefined, unit?: string): string {
  if (!isScored(value)) return NOT_SCORED;
  if (unit === "ms") return `${Math.round(value)}ms`;
  if (unit === "$") return `$${value.toFixed(4)}`;
  return value.toFixed(3);
}

/** Sort comparator on a metric: unscored values always go last. */
export function compareMetric(
  a: number | null | undefined,
  b: number | null | undefined,
  dir: "asc" | "desc",
): number {
  const aOk = isScored(a);
  const bOk = isScored(b);
  if (!aOk && !bOk) return 0;
  if (!aOk) return 1;
  if (!bOk) return -1;
  return dir === "desc" ? b - a : a - b;
}
