"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { evaluationApi } from "@/lib/api";
import { BarChart3, ArrowUp, ArrowDown, ExternalLink } from "lucide-react";
import { clsx } from "clsx";
import type { Experiment, Run } from "@/types";

const METRIC_COLS = [
  { key: "faithfulness", label: "Faithfulness", lowerBetter: false },
  { key: "context_recall", label: "Context Recall", lowerBetter: false },
  { key: "context_precision", label: "Ctx Precision", lowerBetter: false },
  { key: "answer_correctness", label: "Ans Correctness", lowerBetter: false },
  { key: "answer_relevance", label: "Ans Relevance", lowerBetter: false },
  { key: "freshness_validity", label: "Freshness", lowerBetter: false },
  { key: "multimodal_grounding_rate", label: "Multimodal", lowerBetter: false },
  { key: "latency_p50_ms", label: "Latency p50", lowerBetter: true, unit: "ms" },
  { key: "avg_cost_usd", label: "Cost/q", lowerBetter: true, unit: "$" },
];

function metricColor(value: number | null | undefined, lowerBetter: boolean): string {
  if (value == null) return "text-zinc-600";
  const v = lowerBetter ? 1 - Math.min(value / 5000, 1) : value;
  if (v >= 0.7) return "text-emerald-400";
  if (v >= 0.4) return "text-amber-400";
  return "text-red-400";
}

function formatMetric(value: number | null | undefined, unit?: string): string {
  if (value == null) return "—";
  if (unit === "ms") return `${Math.round(value)}ms`;
  if (unit === "$") return `$${value.toFixed(4)}`;
  return value.toFixed(3);
}

export default function LeaderboardPage() {
  const [sortBy, setSortBy] = useState("faithfulness");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const { data: entries = [] as any[], isLoading } = useQuery<any[]>({
    queryKey: ["leaderboard", sortBy],
    queryFn: () => evaluationApi.getLeaderboard(undefined),
  });

  const col = METRIC_COLS.find((c) => c.key === sortBy);
  const entriesArray = (entries || []) as any[];

  const sorted = [...entriesArray].sort((a, b) => {
    const av = (a.metrics as any)?.[sortBy] ?? null;
    const bv = (b.metrics as any)?.[sortBy] ?? null;
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    return sortDir === "desc" ? bv - av : av - bv;
  });

  function toggleSort(key: string) {
    if (sortBy === key) {
      setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setSortBy(key);
      const c = METRIC_COLS.find((mc) => mc.key === key);
      setSortDir(c?.lowerBetter ? "asc" : "desc");
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-8 py-10">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white mb-1">Leaderboard</h1>
        <p className="text-sm text-zinc-500">
          Ranked pipeline configurations across all completed runs — click column headers to sort
        </p>
      </div>

      {isLoading ? (
        <div className="py-20 text-center text-sm text-zinc-600">Loading leaderboard…</div>
      ) : entriesArray.length === 0 ? (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl py-20 text-center">
          <BarChart3 className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
          <p className="text-sm font-medium text-zinc-500 mb-1">No results yet</p>
          <p className="text-xs text-zinc-600">Complete an experiment to see rankings</p>
        </div>
      ) : (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-x-auto">
          <table className="w-full text-sm min-w-[1200px]">
            <thead>
              <tr className="border-b border-zinc-800">
                <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 w-7">#</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500">Config</th>
                {METRIC_COLS.map((col) => (
                  <th
                    key={col.key}
                    className="px-3 py-3 text-right cursor-pointer hover:text-zinc-300 transition-colors select-none"
                    onClick={() => toggleSort(col.key)}
                  >
                    <div className="flex items-center justify-end gap-1">
                      <span className="text-xs font-medium text-zinc-500 hover:text-zinc-300 whitespace-nowrap">
                        {col.label}
                      </span>
                      {sortBy === col.key ? (
                        sortDir === "desc" ? (
                          <ArrowDown className="w-3 h-3 text-violet-400" />
                        ) : (
                          <ArrowUp className="w-3 h-3 text-violet-400" />
                        )
                      ) : null}
                    </div>
                  </th>
                ))}
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody>
              {sorted.map((entry, i) => {
                const cfg = entry.config as any;
                return (
                  <tr
                    key={entry.run_id}
                    className={clsx(
                      "border-b border-zinc-800/50 hover:bg-zinc-800/20 transition-colors",
                      i === sorted.length - 1 && "border-b-0"
                    )}
                  >
                    <td className="px-4 py-3">
                      <span className={clsx(
                        "text-xs font-bold",
                        i === 0 ? "text-amber-400" : i === 1 ? "text-zinc-400" : i === 2 ? "text-amber-700" : "text-zinc-600"
                      )}>
                        {i + 1}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex flex-wrap gap-1 max-w-xs">
                        {cfg?.retrieval_strategy && (
                          <span className="text-[10px] bg-violet-500/10 text-violet-400 border border-violet-500/20 px-1.5 py-0.5 rounded font-mono">
                            {cfg.retrieval_strategy}
                          </span>
                        )}
                        {cfg?.chunking_strategy && (
                          <span className="text-[10px] bg-zinc-800 text-zinc-400 border border-zinc-700 px-1.5 py-0.5 rounded font-mono">
                            {cfg.chunking_strategy}
                          </span>
                        )}
                        {cfg?.parsing_strategy && (
                          <span className="text-[10px] bg-zinc-800 text-zinc-400 border border-zinc-700 px-1.5 py-0.5 rounded font-mono">
                            {cfg.parsing_strategy}
                          </span>
                        )}
                        {cfg?.freshness_policy && cfg.freshness_policy !== "none" && (
                          <span className="text-[10px] bg-blue-500/10 text-blue-400 border border-blue-500/20 px-1.5 py-0.5 rounded font-mono">
                            {cfg.freshness_policy}
                          </span>
                        )}
                      </div>
                    </td>
                    {METRIC_COLS.map((col) => {
                      const value = (entry.metrics as any)?.[col.key];
                      return (
                        <td key={col.key} className="px-3 py-3 text-right">
                          <span className={clsx("font-mono text-xs", metricColor(value, col.lowerBetter))}>
                            {formatMetric(value, col.unit)}
                          </span>
                        </td>
                      );
                    })}
                    <td className="px-4 py-3">
                      <Link
                        href={`/experiments/${entry.experiment_id}/runs/${entry.run_id}`}
                        className="text-zinc-600 hover:text-violet-400 transition-colors"
                      >
                        <ExternalLink className="w-3.5 h-3.5" />
                      </Link>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
