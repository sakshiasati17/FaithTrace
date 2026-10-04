/**
 * FaithTrace API client.
 *
 * Thin wrapper around fetch/axios for type-safe calls to the FastAPI backend.
 */

import axios, { AxiosError } from "axios";
import type {
  Document,
  Experiment,
  ExperimentCreatePayload,
  Run,
  RunMetrics,
  QueryDiagnosis,
  DiagnosticReasoning,
  Recommendation,
  QueryFeedback,
  FeedbackSummary,
  SystemStatus,
  EvalSetSummary,
  EvalSetDetail,
  EvalSetUploadResult,
  EvalSetRowError,
} from "@/types";

// Must match DEFAULT_EVAL_SET_PATH in backend/app/services/evaluation/eval_sets.py
export const DEFAULT_EVAL_SET_PATH = "eval_sets/faithtrace_v1.json";

/** Error thrown by the API client; `errors` carries per-row eval set validation errors. */
export class ApiError extends Error {
  errors: EvalSetRowError[];
  constructor(message: string, errors: EvalSetRowError[] = []) {
    super(message);
    this.name = "ApiError";
    this.errors = errors;
  }
}

const client = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1",
  timeout: 30_000,
});

// Normalise backend error detail into a clean Error message for all callers
client.interceptors.response.use(
  (res) => res,
  (
    err: AxiosError<{
      detail?: string | { msg: string }[] | { message: string; errors?: EvalSetRowError[] };
    }>
  ) => {
    const detail = err.response?.data?.detail;
    if (detail && typeof detail === "object" && !Array.isArray(detail)) {
      return Promise.reject(new ApiError(detail.message ?? err.message, detail.errors ?? []));
    }
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
        ? detail.map((d) => d.msg).join(", ")
        : err.message;
    return Promise.reject(new Error(message));
  }
);

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

  create: (payload: ExperimentCreatePayload): Promise<Experiment> =>
    client.post("/experiments/", payload).then((r) => r.data),

  getRuns: (experimentId: string): Promise<Run[]> =>
    client.get(`/experiments/${experimentId}/runs`).then((r) => r.data),

  getRunTrace: (experimentId: string, runId: string) =>
    client.get(`/experiments/${experimentId}/runs/${runId}/trace`).then((r) => r.data),
};

// ─── Eval sets ────────────────────────────────────────────────────────────────

export const evalSetsApi = {
  list: (): Promise<EvalSetSummary[]> =>
    client.get("/eval-sets/").then((r) => r.data),

  get: (id: string): Promise<EvalSetDetail> =>
    client.get(`/eval-sets/${encodeURIComponent(id)}`).then((r) => r.data),

  upload: (file: File, name = "", description = ""): Promise<EvalSetUploadResult> => {
    const form = new FormData();
    form.append("file", file);
    form.append("name", name);
    form.append("description", description);
    return client.post("/eval-sets/", form, {
      headers: { "Content-Type": "multipart/form-data" },
    }).then((r) => r.data);
  },

  delete: (id: string): Promise<void> =>
    client.delete(`/eval-sets/${encodeURIComponent(id)}`).then((r) => r.data),
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

  getReasoning: (runId: string, queryId: string): Promise<{ query_id: string; cached: boolean; reasoning: DiagnosticReasoning }> =>
    client.post(`/diagnostics/run/${runId}/query/${queryId}/reason`).then((r) => r.data),
};

// ─── Feedback ─────────────────────────────────────────────────────────────────

export const feedbackApi = {
  submit: (queryResultId: string, payload: { rating: string; correct_label?: string; comment?: string }) =>
    client.post(`/feedback/${queryResultId}`, payload).then((r) => r.data),

  get: (queryResultId: string) =>
    client.get(`/feedback/${queryResultId}`).then((r) => r.data),

  getRunSummary: (runId: string) =>
    client.get(`/feedback/run/${runId}/summary`).then((r) => r.data),
};

// ─── Recommendations ─────────────────────────────────────────────────────────

export const recommendationsApi = {
  get: (experimentId: string): Promise<Recommendation[]> =>
    client.get("/recommendations/", { params: { experiment_id: experimentId } })
      .then((r) => r.data),

  explain: (runId: string): Promise<{ rationale: string }> =>
    client.get(`/recommendations/explain/${runId}`).then((r) => r.data),
};


// ─── System ───────────────────────────────────────────────────────────────────

export const systemApi = {
  // Short timeout so an unreachable API shows as offline quickly.
  getStatus: (): Promise<SystemStatus> =>
    client.get("/system/status", { timeout: 10_000 }).then((r) => r.data),
};
