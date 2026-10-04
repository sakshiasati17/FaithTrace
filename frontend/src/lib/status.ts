import type { Experiment, ExperimentStatus } from "@/types";

/**
 * Experiment statuses that are still moving: pending → running (generating)
 * → evaluating → diagnosing → done | failed. Pages poll while any is active.
 */
export const ACTIVE_EXPERIMENT_STATUSES: readonly ExperimentStatus[] = [
  "pending",
  "running",
  "evaluating",
  "diagnosing",
];

export const EXPERIMENT_POLL_MS = 5000;

export function isExperimentActive(status: string | null | undefined): boolean {
  return ACTIVE_EXPERIMENT_STATUSES.includes(status as ExperimentStatus);
}

/** react-query refetchInterval for a single experiment. */
export function experimentRefetchInterval(data: Experiment | undefined): number | false {
  return data && isExperimentActive(data.status) ? EXPERIMENT_POLL_MS : false;
}

/** react-query refetchInterval for the experiment list. */
export function experimentsRefetchInterval(data: Experiment[] | undefined): number | false {
  return data && data.some((e) => isExperimentActive(e.status)) ? EXPERIMENT_POLL_MS : false;
}
