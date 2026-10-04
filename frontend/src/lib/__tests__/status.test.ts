import { describe, it, expect } from "vitest";
import {
  ACTIVE_EXPERIMENT_STATUSES,
  EXPERIMENT_POLL_MS,
  experimentRefetchInterval,
  experimentsRefetchInterval,
  isExperimentActive,
} from "../status";
import type { Experiment, ExperimentStatus } from "@/types";

function exp(status: ExperimentStatus | string): Experiment {
  return {
    id: "e1", name: "e", description: "", status: status as ExperimentStatus,
    created_at: "2026-01-01T00:00:00", completed_at: null, runs: [],
  };
}

describe("experiment polling", () => {
  it.each(["pending", "running", "evaluating", "diagnosing"])("polls while %s", (status) => {
    expect(isExperimentActive(status)).toBe(true);
    expect(experimentRefetchInterval(exp(status))).toBe(EXPERIMENT_POLL_MS);
    expect(experimentsRefetchInterval([exp("done"), exp(status)])).toBe(EXPERIMENT_POLL_MS);
  });

  it.each(["done", "failed"])("stops polling when %s", (status) => {
    expect(isExperimentActive(status)).toBe(false);
    expect(experimentRefetchInterval(exp(status))).toBe(false);
    expect(experimentsRefetchInterval([exp(status), exp("done")])).toBe(false);
  });

  it("includes the post-generation stages", () => {
    expect(ACTIVE_EXPERIMENT_STATUSES).toEqual(
      expect.arrayContaining(["evaluating", "diagnosing"])
    );
  });

  it("does not poll before data loads", () => {
    expect(experimentRefetchInterval(undefined)).toBe(false);
    expect(experimentsRefetchInterval(undefined)).toBe(false);
    expect(experimentsRefetchInterval([])).toBe(false);
    expect(isExperimentActive(undefined)).toBe(false);
  });
});
