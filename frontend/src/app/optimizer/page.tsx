"use client";

import { useEffect, useState, useCallback } from "react";
import { optimizerApi } from "@/lib/api";
import type { OptimizerJob, OptimizerStatus } from "@/types";

/* ─── Metric options for the dropdown ─────────────────────────────────────── */
const METRIC_OPTIONS = [
  { value: "faithfulness", label: "Faithfulness" },
  { value: "answer_correctness", label: "Answer Correctness" },
  { value: "context_recall", label: "Context Recall" },
  { value: "context_precision", label: "Context Precision" },
  { value: "answer_relevance", label: "Answer Relevance" },
  { value: "freshness_validity", label: "Freshness Validity" },
  { value: "multimodal_grounding_rate", label: "Multimodal Grounding" },
];

const STATUS_STYLES: Record<string, { bg: string; text: string; label: string }> = {
  pending:          { bg: "rgba(148,163,184,0.15)", text: "#94a3b8", label: "Pending" },
  running:          { bg: "rgba(59,130,246,0.15)",  text: "#60a5fa", label: "Running" },
  converged:        { bg: "rgba(34,197,94,0.15)",   text: "#4ade80", label: "Converged ✓" },
  budget_exceeded:  { bg: "rgba(251,191,36,0.15)",  text: "#fbbf24", label: "Budget Hit" },
  max_iterations:   { bg: "rgba(251,191,36,0.15)",  text: "#fbbf24", label: "Max Iterations" },
  failed:           { bg: "rgba(239,68,68,0.15)",   text: "#f87171", label: "Failed" },
};

export default function OptimizerPage() {
  const [jobs, setJobs] = useState<OptimizerJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [expandedJob, setExpandedJob] = useState<string | null>(null);

  /* ─── Form state ──────────────────────────────────────────────────────── */
  const [formMetric, setFormMetric] = useState("faithfulness");
  const [formThreshold, setFormThreshold] = useState(0.85);
  const [formMaxIter, setFormMaxIter] = useState(5);
  const [formName, setFormName] = useState("");

  const fetchJobs = useCallback(async () => {
    try {
      const data = await optimizerApi.list();
      setJobs(data);
    } catch {
      /* ignore fetch errors during polling */
    } finally {
      setLoading(false);
    }
  }, []);

  /* Initial load + polling for active jobs */
  useEffect(() => {
    fetchJobs();
    const interval = setInterval(() => {
      const hasActive = jobs.some((j) => j.status === "running" || j.status === "pending");
      if (hasActive) fetchJobs();
    }, 5000);
    return () => clearInterval(interval);
  }, [fetchJobs, jobs]);

  const handleCreate = async () => {
    setCreating(true);
    try {
      await optimizerApi.create({
        name: formName || "Auto-Optimizer",
        target_metric: formMetric,
        target_threshold: formThreshold,
        max_iterations: formMaxIter,
      });
      await fetchJobs();
      setFormName("");
    } catch (err) {
      console.error("Failed to create optimizer job:", err);
    } finally {
      setCreating(false);
    }
  };

  /* ─── Render ──────────────────────────────────────────────────────────── */
  return (
    <div style={{ maxWidth: 960, margin: "0 auto", padding: "32px 20px" }}>
      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ fontSize: 28, fontWeight: 700, color: "#f1f5f9", margin: 0 }}>
          ⚡ Autonomous Optimizer
        </h1>
        <p style={{ color: "#94a3b8", marginTop: 6, fontSize: 14 }}>
          AI agent that iteratively searches for the best RAG pipeline configuration
        </p>
      </div>

      {/* ─── Create Form ─────────────────────────────────────────────────── */}
      <div style={{
        background: "linear-gradient(135deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95))",
        border: "1px solid rgba(99,102,241,0.25)",
        borderRadius: 12,
        padding: 24,
        marginBottom: 32,
      }}>
        <h2 style={{ fontSize: 16, fontWeight: 600, color: "#e2e8f0", margin: "0 0 16px 0" }}>
          Launch New Optimizer
        </h2>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 12, marginBottom: 16 }}>
          <div>
            <label style={labelStyle}>Name</label>
            <input
              style={inputStyle}
              placeholder="Auto-Optimizer"
              value={formName}
              onChange={(e) => setFormName(e.target.value)}
            />
          </div>
          <div>
            <label style={labelStyle}>Target Metric</label>
            <select style={inputStyle} value={formMetric} onChange={(e) => setFormMetric(e.target.value)}>
              {METRIC_OPTIONS.map((m) => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label style={labelStyle}>Threshold</label>
            <input
              style={inputStyle}
              type="number"
              min={0} max={1} step={0.05}
              value={formThreshold}
              onChange={(e) => setFormThreshold(Number(e.target.value))}
            />
          </div>
          <div>
            <label style={labelStyle}>Max Iterations</label>
            <input
              style={inputStyle}
              type="number"
              min={1} max={10}
              value={formMaxIter}
              onChange={(e) => setFormMaxIter(Number(e.target.value))}
            />
          </div>
        </div>
        <button
          onClick={handleCreate}
          disabled={creating}
          style={{
            background: creating ? "#334155" : "linear-gradient(135deg, #6366f1, #8b5cf6)",
            color: "#fff",
            border: "none",
            borderRadius: 8,
            padding: "10px 24px",
            fontSize: 14,
            fontWeight: 600,
            cursor: creating ? "not-allowed" : "pointer",
            transition: "all 0.2s",
          }}
        >
          {creating ? "Starting..." : "🚀 Start Optimizer Agent"}
        </button>
      </div>

      {/* ─── Jobs List ───────────────────────────────────────────────────── */}
      {loading ? (
        <p style={{ color: "#64748b", textAlign: "center", padding: 40 }}>Loading...</p>
      ) : jobs.length === 0 ? (
        <div style={{
          textAlign: "center",
          padding: 60,
          color: "#64748b",
          border: "1px dashed rgba(100,116,139,0.3)",
          borderRadius: 12,
        }}>
          <p style={{ fontSize: 18 }}>No optimizer jobs yet</p>
          <p style={{ fontSize: 13 }}>Create one above to start searching for the best pipeline config</p>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {jobs.map((job) => (
            <JobCard
              key={job.id}
              job={job}
              expanded={expandedJob === job.id}
              onToggle={() => setExpandedJob(expandedJob === job.id ? null : job.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/* ─── Job Card Component ─────────────────────────────────────────────────── */

function JobCard({ job, expanded, onToggle }: { job: OptimizerJob; expanded: boolean; onToggle: () => void }) {
  const state = job.state;
  const goal = job.goal;
  const statusInfo = STATUS_STYLES[job.status] || STATUS_STYLES.pending;
  const progress = state?.iteration && goal?.max_iterations
    ? Math.min((state.iteration / goal.max_iterations) * 100, 100)
    : 0;

  return (
    <div style={{
      background: "rgba(30,41,59,0.7)",
      border: `1px solid ${job.status === "converged" ? "rgba(34,197,94,0.3)" : "rgba(51,65,85,0.5)"}`,
      borderRadius: 12,
      overflow: "hidden",
      transition: "border-color 0.3s",
    }}>
      {/* Header row */}
      <div
        onClick={onToggle}
        style={{
          padding: "16px 20px",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 20 }}>
            {job.status === "running" ? "⚡" : job.status === "converged" ? "🏆" : "🔬"}
          </span>
          <div>
            <div style={{ fontWeight: 600, color: "#e2e8f0", fontSize: 15 }}>{job.name}</div>
            <div style={{ color: "#64748b", fontSize: 12, marginTop: 2 }}>
              {goal?.target_metric} ≥ {goal?.target_threshold} · Max {goal?.max_iterations} iterations
            </div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {/* Best score */}
          {state?.best_score > 0 && (
            <div style={{
              background: "rgba(34,197,94,0.12)",
              color: "#4ade80",
              padding: "4px 10px",
              borderRadius: 6,
              fontSize: 13,
              fontWeight: 600,
              fontFamily: "monospace",
            }}>
              Best: {state.best_score.toFixed(3)}
            </div>
          )}
          {/* Status badge */}
          <span style={{
            background: statusInfo.bg,
            color: statusInfo.text,
            padding: "4px 10px",
            borderRadius: 6,
            fontSize: 12,
            fontWeight: 600,
          }}>
            {statusInfo.label}
          </span>
          <span style={{ color: "#64748b", fontSize: 18 }}>{expanded ? "▲" : "▼"}</span>
        </div>
      </div>

      {/* Progress bar */}
      {job.status === "running" && (
        <div style={{ height: 3, background: "rgba(51,65,85,0.5)" }}>
          <div style={{
            height: "100%",
            width: `${progress}%`,
            background: "linear-gradient(90deg, #6366f1, #8b5cf6)",
            transition: "width 0.5s ease",
          }} />
        </div>
      )}

      {/* Expanded details */}
      {expanded && (
        <div style={{ padding: "0 20px 20px", borderTop: "1px solid rgba(51,65,85,0.3)" }}>
          {/* Stats row */}
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: 12,
            marginTop: 16,
            marginBottom: 16,
          }}>
            <StatBox label="Iteration" value={`${state?.iteration || 0} / ${goal?.max_iterations}`} />
            <StatBox label="Best Score" value={(state?.best_score || 0).toFixed(3)} highlight />
            <StatBox label="Cost" value={`$${(state?.total_cost_usd || 0).toFixed(3)}`} />
            <StatBox label="Threshold" value={goal?.target_threshold?.toFixed(2) || "—"} />
          </div>

          {/* Message */}
          {state?.message && (
            <div style={{
              background: job.status === "converged"
                ? "rgba(34,197,94,0.08)"
                : "rgba(99,102,241,0.08)",
              border: `1px solid ${job.status === "converged" ? "rgba(34,197,94,0.2)" : "rgba(99,102,241,0.2)"}`,
              borderRadius: 8,
              padding: "10px 14px",
              fontSize: 13,
              color: "#cbd5e1",
              marginBottom: 16,
            }}>
              {state.message}
            </div>
          )}

          {/* Winner config */}
          {state?.best_config && Object.keys(state.best_config).length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <h3 style={{ fontSize: 13, fontWeight: 600, color: "#94a3b8", margin: "0 0 8px" }}>
                🏆 Best Configuration
              </h3>
              <div style={{
                display: "grid",
                gridTemplateColumns: "repeat(2, 1fr)",
                gap: 6,
                background: "rgba(15,23,42,0.5)",
                borderRadius: 8,
                padding: 12,
              }}>
                {Object.entries(state.best_config)
                  .filter(([k]) => !["embedding_model", "llm_model", "top_k", "prompt_template"].includes(k))
                  .map(([key, val]) => (
                    <div key={key} style={{ fontSize: 12 }}>
                      <span style={{ color: "#64748b" }}>{formatKey(key)}: </span>
                      <span style={{ color: "#e2e8f0", fontFamily: "monospace" }}>{String(val)}</span>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Iteration timeline */}
          {state?.history && state.history.length > 0 && (
            <div>
              <h3 style={{ fontSize: 13, fontWeight: 600, color: "#94a3b8", margin: "0 0 8px" }}>
                Iteration History
              </h3>
              {/* Convergence mini-chart */}
              <div style={{
                display: "flex",
                alignItems: "flex-end",
                gap: 3,
                height: 48,
                marginBottom: 12,
                padding: "0 4px",
              }}>
                {state.history.map((h, i) => {
                  const pct = Math.max((h.best_score / (goal?.target_threshold || 1)) * 100, 5);
                  const isAtGoal = h.best_score >= (goal?.target_threshold || 1);
                  return (
                    <div
                      key={i}
                      style={{
                        flex: 1,
                        height: `${Math.min(pct, 100)}%`,
                        background: isAtGoal
                          ? "linear-gradient(180deg, #4ade80, #22c55e)"
                          : "linear-gradient(180deg, #818cf8, #6366f1)",
                        borderRadius: "4px 4px 0 0",
                        transition: "height 0.3s",
                        position: "relative",
                      }}
                      title={`Iter ${h.iteration}: ${h.best_score.toFixed(3)}`}
                    />
                  );
                })}
              </div>
              {/* Threshold line label */}
              <div style={{ fontSize: 11, color: "#64748b", marginBottom: 8, textAlign: "right" }}>
                Goal: {goal?.target_threshold}
              </div>

              {/* Table */}
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr>
                    {["Iter", "Configs", "Best Score", "Experiment"].map((h) => (
                      <th key={h} style={{
                        textAlign: "left",
                        padding: "6px 8px",
                        color: "#64748b",
                        borderBottom: "1px solid rgba(51,65,85,0.4)",
                        fontWeight: 500,
                      }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {state.history.map((h) => (
                    <tr key={h.iteration}>
                      <td style={cellStyle}>{h.iteration}</td>
                      <td style={cellStyle}>{h.configs_tested}</td>
                      <td style={{
                        ...cellStyle,
                        color: h.best_score >= (goal?.target_threshold || 1) ? "#4ade80" : "#e2e8f0",
                        fontFamily: "monospace",
                        fontWeight: 600,
                      }}>
                        {h.best_score.toFixed(3)}
                      </td>
                      <td style={{ ...cellStyle, fontFamily: "monospace", color: "#64748b" }}>
                        {h.experiment_id?.slice(0, 8)}…
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ─── Stat Box ───────────────────────────────────────────────────────────── */

function StatBox({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div style={{
      background: "rgba(15,23,42,0.6)",
      borderRadius: 8,
      padding: "10px 12px",
      textAlign: "center",
    }}>
      <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4 }}>{label}</div>
      <div style={{
        fontSize: 18,
        fontWeight: 700,
        fontFamily: "monospace",
        color: highlight ? "#4ade80" : "#e2e8f0",
      }}>
        {value}
      </div>
    </div>
  );
}

/* ─── Helpers ─────────────────────────────────────────────────────────────── */

function formatKey(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

const labelStyle: React.CSSProperties = {
  display: "block",
  fontSize: 11,
  color: "#94a3b8",
  marginBottom: 4,
  fontWeight: 500,
};

const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "8px 10px",
  background: "rgba(15,23,42,0.8)",
  border: "1px solid rgba(51,65,85,0.5)",
  borderRadius: 6,
  color: "#e2e8f0",
  fontSize: 13,
  outline: "none",
  boxSizing: "border-box",
};

const cellStyle: React.CSSProperties = {
  padding: "6px 8px",
  color: "#e2e8f0",
  borderBottom: "1px solid rgba(51,65,85,0.2)",
};
