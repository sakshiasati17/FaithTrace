"use client";

import { useEffect, useState, useCallback } from "react";
import { optimizerApi } from "@/lib/api";
import { Zap, Loader2, ChevronDown, ChevronUp, Trophy, Target, DollarSign, Hash } from "lucide-react";
import { clsx } from "clsx";
import type { OptimizerJob, OptimizerStatus } from "@/types";

const METRIC_OPTIONS = [
  { value: "faithfulness", label: "Faithfulness" },
  { value: "answer_correctness", label: "Answer Correctness" },
  { value: "context_recall", label: "Context Recall" },
  { value: "context_precision", label: "Context Precision" },
  { value: "answer_relevance", label: "Answer Relevance" },
  { value: "freshness_validity", label: "Freshness Validity" },
  { value: "multimodal_grounding_rate", label: "Multimodal Grounding" },
];

const STATUS_STYLES: Record<string, { cls: string; label: string }> = {
  pending:         { cls: "bg-zinc-800 text-zinc-400 border-zinc-700", label: "Pending" },
  running:         { cls: "bg-blue-500/10 text-blue-400 border-blue-500/30", label: "Running" },
  converged:       { cls: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30", label: "Converged" },
  budget_exceeded: { cls: "bg-amber-500/10 text-amber-400 border-amber-500/30", label: "Budget Hit" },
  max_iterations:  { cls: "bg-amber-500/10 text-amber-400 border-amber-500/30", label: "Max Iterations" },
  failed:          { cls: "bg-red-500/10 text-red-400 border-red-500/30", label: "Failed" },
};

export default function OptimizerPage() {
  const [jobs, setJobs] = useState<OptimizerJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [expandedJob, setExpandedJob] = useState<string | null>(null);

  const [formMetric, setFormMetric] = useState("faithfulness");
  const [formThreshold, setFormThreshold] = useState(0.85);
  const [formMaxIter, setFormMaxIter] = useState(5);
  const [formName, setFormName] = useState("");

  const fetchJobs = useCallback(async () => {
    try {
      const data = await optimizerApi.list();
      setJobs(data);
    } catch {
    } finally {
      setLoading(false);
    }
  }, []);

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

  return (
    <div className="max-w-5xl mx-auto px-8 py-10">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white mb-1">Autonomous Optimizer</h1>
        <p className="text-sm text-zinc-500">
          AI agent that iteratively searches for the best RAG pipeline configuration
        </p>
      </div>

      {/* Create Form */}
      <div className="rounded-xl p-6 mb-8 border border-violet-500/20 bg-gradient-to-br from-zinc-900 to-zinc-950 gradient-border">
        <h2 className="text-sm font-semibold text-zinc-200 mb-4">Launch New Optimizer</h2>
        <div className="grid grid-cols-4 gap-3 mb-4">
          <div>
            <label className="block text-[11px] text-zinc-500 mb-1.5 font-medium">Name</label>
            <input
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-violet-500"
              placeholder="Auto-Optimizer"
              value={formName}
              onChange={(e) => setFormName(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-[11px] text-zinc-500 mb-1.5 font-medium">Target Metric</label>
            <select
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-violet-500"
              value={formMetric}
              onChange={(e) => setFormMetric(e.target.value)}
            >
              {METRIC_OPTIONS.map((m) => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-[11px] text-zinc-500 mb-1.5 font-medium">Threshold</label>
            <input
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-violet-500 font-mono"
              type="number"
              min={0} max={1} step={0.05}
              value={formThreshold}
              onChange={(e) => setFormThreshold(Number(e.target.value))}
            />
          </div>
          <div>
            <label className="block text-[11px] text-zinc-500 mb-1.5 font-medium">Max Iterations</label>
            <input
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-violet-500 font-mono"
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
          className="press inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all glow-violet-sm hover:glow-violet-md"
          style={{ background: "linear-gradient(135deg,#7c3aed,#6d28d9)" }}
        >
          {creating ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <Zap className="w-3.5 h-3.5" />
          )}
          {creating ? "Starting..." : "Start Optimizer Agent"}
        </button>
      </div>

      {/* Jobs List */}
      {loading ? (
        <div className="flex items-center justify-center gap-2 py-20 text-sm text-zinc-600">
          <Loader2 className="w-4 h-4 animate-spin" /> Loading...
        </div>
      ) : jobs.length === 0 ? (
        <div className="rounded-xl bg-zinc-900 border border-zinc-800 py-20 text-center">
          <Zap className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
          <p className="text-sm font-medium text-zinc-500 mb-1">No optimizer jobs yet</p>
          <p className="text-xs text-zinc-600">Create one above to start searching for the best pipeline config</p>
        </div>
      ) : (
        <div className="space-y-4 stagger-in">
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

function JobCard({ job, expanded, onToggle }: { job: OptimizerJob; expanded: boolean; onToggle: () => void }) {
  const state = job.state;
  const goal = job.goal;
  const statusInfo = STATUS_STYLES[job.status] || STATUS_STYLES.pending;
  const progress = state?.iteration && goal?.max_iterations
    ? Math.min((state.iteration / goal.max_iterations) * 100, 100)
    : 0;

  return (
    <div className={clsx(
      "rounded-xl overflow-hidden transition-all card-lift",
      job.status === "converged"
        ? "bg-zinc-900 border border-emerald-500/20"
        : "bg-zinc-900 border border-zinc-800"
    )}>
      {/* Header row */}
      <div
        onClick={onToggle}
        className="px-5 py-4 cursor-pointer flex items-center justify-between hover:bg-zinc-800/20 transition-colors"
      >
        <div className="flex items-center gap-3 min-w-0">
          <div className={clsx(
            "w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0",
            job.status === "running" ? "bg-blue-500/10" : job.status === "converged" ? "bg-emerald-500/10" : "bg-zinc-800"
          )}>
            <Zap className={clsx(
              "w-4 h-4",
              job.status === "running" ? "text-blue-400" : job.status === "converged" ? "text-emerald-400" : "text-zinc-500"
            )} />
          </div>
          <div className="min-w-0">
            <p className="text-sm font-semibold text-zinc-200 truncate">{job.name}</p>
            <p className="text-xs text-zinc-600 mt-0.5">
              {goal?.target_metric} &ge; {goal?.target_threshold} &middot; Max {goal?.max_iterations} iterations
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          {state?.best_score > 0 && (
            <span className="text-xs font-mono font-semibold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-1 rounded-md">
              Best: {state.best_score.toFixed(3)}
            </span>
          )}
          <span className={clsx("inline-flex items-center px-2.5 py-1 rounded-md text-xs font-semibold border", statusInfo.cls)}>
            {statusInfo.label}
          </span>
          {expanded ? (
            <ChevronUp className="w-4 h-4 text-zinc-600" />
          ) : (
            <ChevronDown className="w-4 h-4 text-zinc-600" />
          )}
        </div>
      </div>

      {/* Progress bar */}
      {job.status === "running" && (
        <div className="h-[3px] bg-zinc-800">
          <div
            className="h-full rounded-full transition-all duration-700 ease-out"
            style={{ width: `${progress}%`, background: "linear-gradient(90deg, #7c3aed, #a855f7)" }}
          />
        </div>
      )}

      {/* Expanded details */}
      {expanded && (
        <div className="px-5 pb-5 border-t border-zinc-800/60">
          <div className="grid grid-cols-4 gap-3 mt-4 mb-4 stagger-in">
            <StatBox icon={Hash} label="Iteration" value={`${state?.iteration || 0} / ${goal?.max_iterations}`} />
            <StatBox icon={Trophy} label="Best Score" value={(state?.best_score || 0).toFixed(3)} highlight />
            <StatBox icon={DollarSign} label="Cost" value={`$${(state?.total_cost_usd || 0).toFixed(3)}`} />
            <StatBox icon={Target} label="Threshold" value={goal?.target_threshold?.toFixed(2) || "—"} />
          </div>

          {state?.message && (
            <div className={clsx(
              "rounded-lg px-4 py-3 text-sm mb-4 border",
              job.status === "converged"
                ? "bg-emerald-500/5 border-emerald-500/20 text-emerald-300"
                : "bg-violet-500/5 border-violet-500/20 text-zinc-300"
            )}>
              {state.message}
            </div>
          )}

          {state?.best_config && Object.keys(state.best_config).length > 0 && (
            <div className="mb-4">
              <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">Best Configuration</p>
              <div className="grid grid-cols-2 gap-2 p-3 rounded-lg bg-zinc-800/40 border border-zinc-700/30">
                {Object.entries(state.best_config)
                  .filter(([k]) => !["embedding_model", "llm_model", "top_k", "prompt_template"].includes(k))
                  .map(([key, val]) => (
                    <div key={key} className="flex items-center gap-2 text-xs">
                      <span className="text-zinc-600">{formatKey(key)}</span>
                      <span className="text-zinc-200 font-mono">{String(val)}</span>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {state?.history && state.history.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-3">Iteration History</p>

              {/* Convergence mini-chart */}
              <div className="flex items-end gap-1 h-12 mb-3 px-1">
                {state.history.map((h, i) => {
                  const pct = Math.max((h.best_score / (goal?.target_threshold || 1)) * 100, 5);
                  const isAtGoal = h.best_score >= (goal?.target_threshold || 1);
                  return (
                    <div
                      key={i}
                      className="flex-1 rounded-t transition-all duration-500"
                      style={{
                        height: `${Math.min(pct, 100)}%`,
                        background: isAtGoal
                          ? "linear-gradient(180deg, #4ade80, #22c55e)"
                          : "linear-gradient(180deg, #a78bfa, #7c3aed)",
                      }}
                      title={`Iter ${h.iteration}: ${h.best_score.toFixed(3)}`}
                    />
                  );
                })}
              </div>
              <p className="text-[10px] text-zinc-600 text-right mb-3 font-mono">
                Goal: {goal?.target_threshold}
              </p>

              {/* Table */}
              <div className="rounded-lg bg-zinc-800/30 border border-zinc-700/30 overflow-hidden">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-zinc-700/30">
                      {["Iter", "Configs", "Best Score", "Experiment"].map((h) => (
                        <th key={h} className="px-3 py-2 text-left text-[10px] font-medium text-zinc-600 uppercase tracking-wider">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {state.history.map((h) => (
                      <tr key={h.iteration} className="border-b border-zinc-800/40 last:border-0">
                        <td className="px-3 py-2 text-zinc-300">{h.iteration}</td>
                        <td className="px-3 py-2 text-zinc-300">{h.configs_tested}</td>
                        <td className={clsx(
                          "px-3 py-2 font-mono font-semibold",
                          h.best_score >= (goal?.target_threshold || 1) ? "text-emerald-400" : "text-zinc-200"
                        )}>
                          {h.best_score.toFixed(3)}
                        </td>
                        <td className="px-3 py-2 font-mono text-zinc-600">
                          {h.experiment_id?.slice(0, 8)}...
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function StatBox({ icon: Icon, label, value, highlight }: { icon: React.ElementType; label: string; value: string; highlight?: boolean }) {
  return (
    <div className="rounded-lg bg-zinc-800/40 border border-zinc-700/30 p-3 text-center">
      <Icon className={clsx("w-3.5 h-3.5 mx-auto mb-1.5", highlight ? "text-emerald-400" : "text-zinc-600")} />
      <p className="text-[10px] text-zinc-600 mb-1">{label}</p>
      <p className={clsx(
        "text-lg font-bold font-mono",
        highlight ? "text-emerald-400 num-glow-green" : "text-zinc-200"
      )}>
        {value}
      </p>
    </div>
  );
}

function formatKey(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
