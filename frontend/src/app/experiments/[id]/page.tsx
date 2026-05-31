"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { experimentsApi, evaluationApi } from "@/lib/api";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { MetricCard } from "@/components/ui/MetricCard";
import { ArrowLeft, ChevronRight, Clock } from "lucide-react";
import { clsx } from "clsx";
import type { Experiment, Run } from "@/types";

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function ConfigPill({ label, value }: { label: string; value: string }) {
  return (
    <span className="text-[10px] bg-zinc-800 border border-zinc-700 text-zinc-400 rounded px-1.5 py-0.5 font-mono">
      {label}: {value}
    </span>
  );
}

export default function ExperimentDetailPage({ params }: { params: { id: string } }) {
  const { id } = params;

  const { data: exp, isLoading } = useQuery<Experiment>({
    queryKey: ["experiment", id],
    queryFn: () => experimentsApi.get(id),
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return false;
      return data.status === "running" || data.status === "pending" ? 5000 : false;
    },
  });

  const { data: leaderboard } = useQuery({
    queryKey: ["leaderboard", id],
    queryFn: () => evaluationApi.getLeaderboard(id),
    enabled: exp?.status === "done",
  });

  if (isLoading) {
    return (
      <div className="max-w-6xl mx-auto px-8 py-10">
        <div className="py-20 text-center text-sm text-zinc-600">Loading experiment…</div>
      </div>
    );
  }

  if (!exp) {
    return (
      <div className="max-w-6xl mx-auto px-8 py-10">
        <p className="text-red-400">Experiment not found.</p>
      </div>
    );
  }

  const runs = (exp?.runs ?? []) as Run[];
  const doneRuns = runs.filter((r) => r.status === "done").length;

  return (
    <div className="max-w-6xl mx-auto px-8 py-10">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-xs text-zinc-600 mb-6">
        <Link href="/experiments" className="hover:text-zinc-400 flex items-center gap-1">
          <ArrowLeft className="w-3 h-3" /> Experiments
        </Link>
        <ChevronRight className="w-3 h-3" />
        <span className="text-zinc-400 truncate max-w-64">{exp.name}</span>
      </div>

      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <h1 className="text-2xl font-bold text-white">{exp.name}</h1>
            <StatusBadge status={exp.status} />
          </div>
          {exp.description && <p className="text-sm text-zinc-500">{exp.description}</p>}
          <p className="text-xs text-zinc-600 mt-1 flex items-center gap-1">
            <Clock className="w-3 h-3" /> Created {formatDate(exp.created_at)}
            {exp.completed_at && ` · Completed ${formatDate(exp.completed_at)}`}
          </p>
        </div>
        <div className="text-right">
          <p className="text-2xl font-bold text-white">{doneRuns}/{runs.length}</p>
          <p className="text-xs text-zinc-500">runs complete</p>
        </div>
      </div>

      {/* Top metrics from leaderboard */}
      {leaderboard && leaderboard.length > 0 && (
        <div className="mb-8">
          <p className="text-xs text-zinc-500 font-medium uppercase tracking-wider mb-3">Best Run Metrics</p>
          <div className="grid grid-cols-5 gap-3 stagger-in">
            <MetricCard label="Faithfulness" value={leaderboard[0].metrics?.faithfulness} />
            <MetricCard label="Context Recall" value={leaderboard[0].metrics?.context_recall} />
            <MetricCard label="Answer Correctness" value={leaderboard[0].metrics?.answer_correctness} />
            <MetricCard label="Freshness Validity" value={leaderboard[0].metrics?.freshness_validity} />
            <MetricCard label="Latency p50" value={leaderboard[0].metrics?.latency_p50_ms} unit="ms" lowerIsBetter />
          </div>
        </div>
      )}

      {/* Runs table */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden card-lift">
        <div className="px-6 py-4 border-b border-zinc-800">
          <h2 className="text-sm font-semibold text-zinc-300">Pipeline Runs</h2>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-800">
              {["Config", "Status", "Faithfulness", "Context Recall", "Freshness", "Latency p50", "Cost/q", ""].map((h) => (
                <th key={h} className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {runs.map((run, i) => {
              const m = run.metrics;
              const lbEntry = leaderboard?.find((e) => e.run_id === run.id);
              const metrics = m || lbEntry?.metrics;
              const cfg = run.config;

              return (
                <tr
                  key={run.id}
                  className={clsx(
                    "border-b border-zinc-800/50 hover:bg-zinc-800/30 transition-colors",
                    i === runs.length - 1 && "border-b-0"
                  )}
                >
                  <td className="px-4 py-3">
                    <div className="flex flex-wrap gap-1 max-w-xs">
                      <ConfigPill label="ret" value={cfg?.retrieval_strategy ?? "—"} />
                      <ConfigPill label="chunk" value={cfg?.chunking_strategy ?? "—"} />
                      <ConfigPill label="parse" value={cfg?.parsing_strategy ?? "—"} />
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={run.status} />
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-zinc-300">
                    {metrics?.faithfulness != null ? metrics.faithfulness.toFixed(3) : "—"}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-zinc-300">
                    {metrics?.context_recall != null ? metrics.context_recall.toFixed(3) : "—"}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-zinc-300">
                    {metrics?.freshness_validity != null ? metrics.freshness_validity.toFixed(3) : "—"}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-zinc-300">
                    {metrics?.latency_p50_ms != null ? `${Math.round(metrics.latency_p50_ms)}ms` : "—"}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-zinc-300">
                    {metrics?.avg_cost_usd != null ? `$${metrics.avg_cost_usd.toFixed(4)}` : "—"}
                  </td>
                  <td className="px-4 py-3">
                    {run.status === "done" && (
                      <Link
                        href={`/experiments/${id}/runs/${run.id}`}
                        className="flex items-center gap-1 text-xs text-violet-400 hover:text-violet-300 transition-colors whitespace-nowrap"
                      >
                        Trace <ChevronRight className="w-3 h-3" />
                      </Link>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
