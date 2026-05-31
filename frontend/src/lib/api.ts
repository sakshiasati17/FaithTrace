/**
 * FaithTrace API client.
 *
 * Thin wrapper around fetch/axios for type-safe calls to the FastAPI backend.
 */

import axios, { AxiosError } from "axios";
import type {
  Document,
  Experiment,
  Run,
  RunMetrics,
  QueryDiagnosis,
  DiagnosticReasoning,
  Recommendation,
  QueryFeedback,
  FeedbackSummary,
  OptimizerJob,
  GPUProfile,
  BenchmarkReport,
  VoiceCommandResult,
} from "@/types";

const client = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1",
  timeout: 30_000,
});

// Normalize backend error messages so callers get a clean string
client.interceptors.response.use(
  (res) => res,
  (err: AxiosError<{ detail?: string | { msg: string }[] }>) => {
    const detail = err.response?.data?.detail;
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

// ─── Optimizer ───────────────────────────────────────────────────────────────

export const optimizerApi = {
  create: (payload: {
    name?: string;
    target_metric?: string;
    target_threshold?: number;
    max_iterations?: number;
    max_cost_usd?: number;
    eval_set_path?: string;
  }): Promise<OptimizerJob> =>
    client.post("/optimizer/", payload).then((r) => r.data),

  list: (): Promise<OptimizerJob[]> =>
    client.get("/optimizer/").then((r) => r.data),

  get: (id: string): Promise<OptimizerJob> =>
    client.get(`/optimizer/${id}`).then((r) => r.data),
};

// ─── Inference Optimization ───────────────────────────────────────────────────

export const optimizationApi = {
  gpuProfile: (): Promise<GPUProfile> =>
    client.get("/optimization/gpu-profile").then((r) => r.data),

  exportClassifier: (modelPath?: string): Promise<{ status: string; onnx_path: string }> =>
    client.post("/optimization/export/classifier", null, {
      params: { model_path: modelPath },
    }).then((r) => r.data),

  runBenchmark: (): Promise<{ task_id: string; status: string }> =>
    client.post("/optimization/benchmark").then((r) => r.data),

  getLatestBenchmark: (): Promise<BenchmarkReport> =>
    client.get("/optimization/benchmark/latest").then((r) => r.data),

  trainPytorch: (experimentId: string, numEpochs?: number): Promise<{ task_id: string; status: string }> =>
    client.post("/optimization/train/pytorch", null, {
      params: { experiment_id: experimentId, num_epochs: numEpochs },
    }).then((r) => r.data),

  tritonHealth: (): Promise<{ server_ready: boolean; model_ready: boolean }> =>
    client.get("/optimization/triton/health").then((r) => r.data),
};

// ─── Voice ────────────────────────────────────────────────────────────────────

export const voiceApi = {
  command: (audioBlob: Blob): Promise<VoiceCommandResult> => {
    const form = new FormData();
    form.append("audio", audioBlob, "command.wav");
    return client.post("/voice/command", form, {
      headers: { "Content-Type": "multipart/form-data" },
    }).then((r) => r.data);
  },

  speak: (text: string): Promise<Blob> =>
    client.post("/voice/speak", null, {
      params: { text },
      responseType: "blob",
    }).then((r) => r.data),

  health: (): Promise<{ stt_loaded: boolean; tts_loaded: boolean; whisper_model: string }> =>
    client.get("/voice/health").then((r) => r.data),
};
