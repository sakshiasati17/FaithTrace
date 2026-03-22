"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { experimentsApi, diagnosticsApi } from "@/lib/api";
import { Stethoscope, AlertCircle } from "lucide-react";
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { clsx } from "clsx";

const FAILURE_COLORS: Record<string, string> = {
  NO_FAILURE: "#10b981",
  STALE_ANSWER: "#f59e0b",
  TABLE_RETRIEVAL_MISS: "#3b82f6",
  LOW_RECALL_RETRIEVAL: "#f97316",
  UNSUPPORTED_SYNTHESIS: "#ef4444",
  IRRELEVANT_CONTEXT_POLLUTION: "#a855f7",
  CHUNKING_BOUNDARY_ERROR: "#eab308",
  WRONG_VERSION: "#ec4899",
  CHART_LAYOUT_BLINDNESS: "#06b6d4",
};

const FAILURE_DESCRIPTIONS: Record<string, string> = {
  NO_FAILURE: "Query answered correctly with appropriate evidence",
  STALE_ANSWER: "Retrieved outdated content invalid for the query date",
  TABLE_RETRIEVAL_MISS: "Failed to retrieve table/spreadsheet evidence for structured data queries",
  LOW_RECALL_RETRIEVAL: "Insufficient relevant content retrieved (context_recall < 0.3)",
  UNSUPPORTED_SYNTHESIS: "Answer not grounded in retrieved context (faithfulness < 0.4)",
  IRRELEVANT_CONTEXT_POLLUTION: "Retrieved irrelevant chunks contaminating the context",
  CHUNKING_BOUNDARY_ERROR: "Answer content present but split across chunk boundaries",
  WRONG_VERSION: "Retrieved content from incorrect document version",
  CHART_LAYOUT_BLINDNESS: "Visual/chart content not recognized or interpreted",
};

export default function DiagnosticsPage() {
  const [selectedExperiment, setSelectedExperiment] = useState<string>("");

  const { data: experiments = [] } = useQuery({
    queryKey: ["experiments"],
    queryFn: experimentsApi.list,
  });

  const completedExps = experiments.filter((e) => e.status === "done");

  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: ["failure-summary", selectedExperiment],
    queryFn: () => diagnosticsApi.getFailureSummary(selectedExperiment),
    enabled: !!selectedExperiment,
  });

  const chartData = summary
    ? Object.entries(summary.failure_counts as Record<string, number>)
        .filter(([, count]) => count > 0)
        .map(([name, value]) => ({
          name: name || "NO_FAILURE",
          value,
          color: FAILURE_COLORS[name] ?? "#6b7280",
        }))
    : [];

  const totalQueries = summary?.total_queries ?? 0;
  const noFailureCount = (summary?.failure_counts as any)?.NO_FAILURE ?? 0;
  const failureRate = totalQueries > 0 ? ((totalQueries - noFailureCount) / totalQueries) * 100 : 0;

  return (
    <div className="max-w-6xl mx-auto px-8 py-10">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white mb-1">Diagnostics</h1>
        <p className="text-sm text-zinc-500">Root-cause failure analysis across pipeline configurations</p>
      </div>

      {/* Experiment selector */}
      <div className="mb-6">
        <label className="block text-xs text-zinc-500 mb-1.5">Select Experiment</label>
        <select
          value={selectedExperiment}
          onChange={(e) => setSelectedExperiment(e.target.value)}
          className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-violet-500 min-w-64"
        >
          <option value="">— choose an experiment —</option>
          {completedExps.map((exp) => (
            <option key={exp.id} value={exp.id}>
              {exp.name} ({exp.runs?.length ?? 0} runs)
            </option>
          ))}
        </select>
      </div>

      {!selectedExperiment ? (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl py-20 text-center">
          <Stethoscope className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
          <p className="text-sm font-medium text-zinc-500">Select a completed experiment above</p>
        </div>
      ) : summaryLoading ? (
        <div className="py-20 text-center text-sm text-zinc-600">Analyzing…</div>
      ) : (
        <>
          {/* Summary stats */}
          <div className="grid grid-cols-3 gap-4 mb-8">
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
              <p className="text-xs text-zinc-500 mb-1">Total Queries</p>
              <p className="text-3xl font-bold text-white">{totalQueries}</p>
              <p className="text-xs text-zinc-600 mt-1">{summary?.run_count} runs analyzed</p>
            </div>
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
              <p className="text-xs text-zinc-500 mb-1">Failure Rate</p>
              <p className={clsx("text-3xl font-bold", failureRate > 30 ? "text-red-400" : failureRate > 10 ? "text-amber-400" : "text-emerald-400")}>
                {failureRate.toFixed(1)}%
              </p>
              <p className="text-xs text-zinc-600 mt-1">{totalQueries - noFailureCount} queries with failures</p>
            </div>
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
              <p className="text-xs text-zinc-500 mb-1">Most Common Failure</p>
              {chartData.filter((d) => d.name !== "NO_FAILURE").length > 0 ? (
                <>
                  <p className="text-sm font-bold text-white mt-1">
                    {chartData
                      .filter((d) => d.name !== "NO_FAILURE")
                      .sort((a, b) => b.value - a.value)[0]?.name ?? "—"}
                  </p>
                  <p className="text-xs text-zinc-600 mt-1">
                    {chartData
                      .filter((d) => d.name !== "NO_FAILURE")
                      .sort((a, b) => b.value - a.value)[0]?.value ?? 0}{" "}
                    occurrences
                  </p>
                </>
              ) : (
                <p className="text-sm font-bold text-emerald-400 mt-1">None — all passing!</p>
              )}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-6">
            {/* Pie chart */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6">
              <h2 className="text-sm font-semibold text-zinc-300 mb-4">Failure Distribution</h2>
              {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={280}>
                  <PieChart>
                    <Pie
                      data={chartData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={2}
                      dataKey="value"
                    >
                      {chartData.map((entry, index) => (
                        <Cell key={index} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        background: "#18181b",
                        border: "1px solid #3f3f46",
                        borderRadius: "8px",
                        fontSize: "12px",
                        color: "#d4d4d8",
                      }}
                    />
                    <Legend
                      formatter={(value) => (
                        <span style={{ fontSize: "11px", color: "#71717a" }}>{value}</span>
                      )}
                    />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-64 flex items-center justify-center text-sm text-zinc-600">No data</div>
              )}
            </div>

            {/* Failure table */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6">
              <h2 className="text-sm font-semibold text-zinc-300 mb-4">Category Breakdown</h2>
              <div className="space-y-2">
                {chartData
                  .sort((a, b) => b.value - a.value)
                  .map((item) => {
                    const pct = totalQueries > 0 ? (item.value / totalQueries) * 100 : 0;
                    return (
                      <div key={item.name} className="flex items-start gap-3">
                        <div
                          className="w-2 h-2 rounded-full flex-shrink-0 mt-1"
                          style={{ backgroundColor: item.color }}
                        />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between mb-0.5">
                            <p className="text-xs font-medium text-zinc-300 truncate">{item.name}</p>
                            <span className="text-xs text-zinc-500 flex-shrink-0 ml-2">
                              {item.value} ({pct.toFixed(0)}%)
                            </span>
                          </div>
                          <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all"
                              style={{ width: `${pct}%`, backgroundColor: item.color }}
                            />
                          </div>
                          <p className="text-[10px] text-zinc-600 mt-0.5 line-clamp-1">
                            {FAILURE_DESCRIPTIONS[item.name] ?? ""}
                          </p>
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
