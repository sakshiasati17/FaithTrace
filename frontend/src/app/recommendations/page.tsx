"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { experimentsApi, recommendationsApi } from "@/lib/api";
import { Lightbulb, CheckCircle2, ChevronDown, ChevronUp } from "lucide-react";
import { clsx } from "clsx";
import type { Experiment, Recommendation } from "@/types";

const OBJECTIVE_LABELS: Record<string, string> = {
  best_overall: "Best Overall",
  lowest_cost: "Lowest Cost",
  best_latency: "Best Latency",
  best_faithfulness: "Best Faithfulness",
  best_for_tables: "Best for Tables",
  best_for_drift: "Best for Drift",
  best_for_long_pdfs: "Best for Long PDFs",
};

const OBJECTIVE_DESCRIPTIONS: Record<string, string> = {
  best_overall: "Highest composite score across all quality metrics",
  lowest_cost: "Minimum token cost per query",
  best_latency: "Fastest p50 response time",
  best_faithfulness: "Highest answer grounding in retrieved context",
  best_for_tables: "Best retrieval accuracy for structured/tabular queries",
  best_for_drift: "Most robust against temporal content changes",
  best_for_long_pdfs: "Best performance on large multi-page documents",
};

const OBJECTIVE_ICONS: Record<string, string> = {
  best_overall: "⭐",
  lowest_cost: "💰",
  best_latency: "⚡",
  best_faithfulness: "🎯",
  best_for_tables: "📊",
  best_for_drift: "📅",
  best_for_long_pdfs: "📄",
};

function ConfigBadges({ config }: { config: Record<string, unknown> }) {
  const badges = [
    { key: "retrieval_strategy", color: "violet" },
    { key: "chunking_strategy", color: "zinc" },
    { key: "parsing_strategy", color: "zinc" },
    { key: "freshness_policy", color: "blue", hide: "none" },
  ];

  return (
    <div className="flex flex-wrap gap-1.5">
      {badges.map(({ key, color, hide }) => {
        const val = config[key] as string | undefined;
        if (!val || val === hide) return null;
        return (
          <span
            key={key}
            className={clsx(
              "text-[10px] px-1.5 py-0.5 rounded font-mono border",
              color === "violet"
                ? "bg-violet-500/10 text-violet-400 border-violet-500/20"
                : color === "blue"
                ? "bg-blue-500/10 text-blue-400 border-blue-500/20"
                : "bg-zinc-800 text-zinc-400 border-zinc-700"
            )}
          >
            {val}
          </span>
        );
      })}
      {Boolean(config.reranker_enabled) && (
        <span className="text-[10px] px-1.5 py-0.5 rounded font-mono border bg-emerald-500/10 text-emerald-400 border-emerald-500/20">
          reranker
        </span>
      )}
    </div>
  );
}

function RecommendationCard({ rec, rank }: { rec: Recommendation; rank: number }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={clsx(
        "bg-zinc-900 border rounded-xl overflow-hidden transition-all card-lift",
        rank === 0 ? "border-violet-500/40 gradient-border" : "border-zinc-800"
      )}
    >
      {rank === 0 && (
        <div
          className="h-0.5"
          style={{
            background:
              "linear-gradient(90deg, transparent, rgba(124,58,237,0.8), rgba(168,85,247,0.4), transparent)",
          }}
        />
      )}
      <div className="p-5">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-base leading-none">
                {OBJECTIVE_ICONS[rec.objective] ?? "📌"}
              </span>
              <h3 className="text-sm font-semibold text-white">
                {OBJECTIVE_LABELS[rec.objective] ?? rec.objective}
              </h3>
              {rank === 0 && (
                <span className="text-[10px] bg-violet-500/15 text-violet-400 border border-violet-500/25 px-1.5 py-0.5 rounded font-medium">
                  Top Pick
                </span>
              )}
            </div>
            <p className="text-xs text-zinc-500 mb-3">
              {OBJECTIVE_DESCRIPTIONS[rec.objective] ?? ""}
            </p>
            <ConfigBadges config={rec.best_config as unknown as Record<string, unknown>} />
          </div>

          <div className="text-right flex-shrink-0">
            <p className="text-2xl font-bold text-white tabular-nums">
              {(rec.score * 100).toFixed(0)}
            </p>
            <p className="text-[10px] text-zinc-600 font-mono">score</p>
          </div>
        </div>

        {/* Rationale toggle */}
        <button
          onClick={() => setExpanded((e) => !e)}
          className="mt-3 flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
        >
          {expanded ? (
            <ChevronUp className="w-3.5 h-3.5" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5" />
          )}
          {expanded ? "Hide rationale" : "Show rationale"}
        </button>

        {expanded && (
          <div className="mt-3 p-3 bg-zinc-800/50 border border-zinc-700/50 rounded-lg">
            <p className="text-xs text-zinc-400 leading-relaxed">{rec.rationale}</p>
            <p className="text-[10px] text-zinc-600 mt-2 font-mono">run: {rec.run_id}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default function RecommendationsPage() {
  const [selectedExperiment, setSelectedExperiment] = useState<string>("");

  const { data: experiments = [] as Experiment[] } = useQuery<Experiment[]>({
    queryKey: ["experiments"],
    queryFn: experimentsApi.list,
  });

  const completedExps = (experiments as Experiment[]).filter((e) => e.status === "done");

  const {
    data: recommendations = [],
    isLoading,
    isError,
  } = useQuery<Recommendation[]>({
    queryKey: ["recommendations", selectedExperiment],
    queryFn: () => recommendationsApi.get(selectedExperiment),
    enabled: !!selectedExperiment,
  });

  const recs = recommendations as Recommendation[];

  return (
    <div className="max-w-4xl mx-auto px-8 py-10">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white mb-1">Recommendations</h1>
        <p className="text-sm text-zinc-500">
          Best pipeline configurations per objective, derived from completed experiment runs
        </p>
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
          <Lightbulb className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
          <p className="text-sm font-medium text-zinc-500 mb-1">
            Select a completed experiment above
          </p>
          <p className="text-xs text-zinc-600">
            Recommendations are generated after all runs have been evaluated
          </p>
        </div>
      ) : isLoading ? (
        <div className="py-20 text-center text-sm text-zinc-600">Computing recommendations…</div>
      ) : isError ? (
        <div className="bg-zinc-900 border border-red-500/20 rounded-xl py-12 text-center">
          <p className="text-sm text-red-400">Failed to load recommendations</p>
        </div>
      ) : recs.length === 0 ? (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl py-20 text-center">
          <CheckCircle2 className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
          <p className="text-sm font-medium text-zinc-500">No recommendations available yet</p>
          <p className="text-xs text-zinc-600 mt-1">
            Ensure at least one run has completed evaluation
          </p>
        </div>
      ) : (
        <>
          <p className="text-xs text-zinc-600 mb-4">
            {recs.length} recommendation{recs.length !== 1 ? "s" : ""} across {recs.length} objectives
          </p>
          <div className="grid gap-4 stagger-in">
            {recs.map((rec, i) => (
              <RecommendationCard key={rec.objective} rec={rec} rank={i} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
