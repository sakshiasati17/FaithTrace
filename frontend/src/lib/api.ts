/**
 * FaithTrace API client.
 *
 * Thin wrapper around fetch/axios for type-safe calls to the FastAPI backend.
 */

import axios from "axios";
import type {
  Document,
  Experiment,
  Run,
  RunMetrics,
  QueryDiagnosis,
  Recommendation,
} from "@/types";

const client = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1",
  timeout: 30_000,
});

// ─── Corpus ───────────────────────────────────────────────────────────────────

export const corpusApi = {
  list: (): Promise<Document[]> =>
    client.get("/corpus/").then((r) => r.data),

  get: (id: string): Promise<Document> =>
    client.get(`/corpus/${id}`).then((r) => r.data),

  upload: (form: FormData): Promise<Document> =>
    client.post("/corpus/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
    }).then((r) => r.data),

  delete: (id: string): Promise<void> =>
    client.delete(`/corpus/${id}`).then((r) => r.data),
};

// ─── Experiments ─────────────────────────────────────────────────────────────

export const experimentsApi = {
  list: (): Promise<Experiment[]> =>
    client.get("/experiments/").then((r) => r.data),

  get: (id: string): Promise<Experiment> =>
    client.get(`/experiments/${id}`).then((r) => r.data),

  create: (payload: Partial<Experiment>): Promise<Experiment> =>
    client.post("/experiments/", payload).then((r) => r.data),

  getRuns: (experimentId: string): Promise<Run[]> =>
    client.get(`/experiments/${experimentId}/runs`).then((r) => r.data),

  getRunTrace: (experimentId: string, runId: string) =>
    client.get(`/experiments/${experimentId}/runs/${runId}/trace`).then((r) => r.data),
};

// ─── Evaluation ───────────────────────────────────────────────────────────────

export const evaluationApi = {
  getMetrics: (runId: string): Promise<RunMetrics> =>
    client.get(`/evaluation/run/${runId}/metrics`).then((r) => r.data),

  getLeaderboard: (experimentId?: string): Promise<Run[]> =>
    client.get("/evaluation/leaderboard", { params: { experiment_id: experimentId } })
      .then((r) => r.data),
};

// ─── Diagnostics ─────────────────────────────────────────────────────────────

export const diagnosticsApi = {
  getRunDiagnostics: (runId: string): Promise<QueryDiagnosis[]> =>
    client.get(`/diagnostics/run/${runId}`).then((r) => r.data),

  getQueryDiagnosis: (runId: string, queryId: string): Promise<QueryDiagnosis> =>
    client.get(`/diagnostics/run/${runId}/query/${queryId}`).then((r) => r.data),

  getFailureSummary: (experimentId: string) =>
    client.get("/diagnostics/summary", { params: { experiment_id: experimentId } })
      .then((r) => r.data),
};

// ─── Recommendations ─────────────────────────────────────────────────────────

export const recommendationsApi = {
  get: (experimentId: string): Promise<Recommendation[]> =>
    client.get("/recommendations/", { params: { experiment_id: experimentId } })
      .then((r) => r.data),

  explain: (runId: string): Promise<{ rationale: string }> =>
    client.get(`/recommendations/explain/${runId}`).then((r) => r.data),
};
