"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { clsx } from "clsx";
import type { Recommendation } from "@/types";
import { formatMetric, isScored, NOT_SCORED } from "@/lib/metrics";

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

// lowest_cost scores are $/query and best_latency scores are p50 ms; the
// rest are 0–1 scores shown out of 100.
function formatScore(rec: Recommendation): string {
  if (!isScored(rec.score)) return NOT_SCORED;
  if (rec.objective === "lowest_cost") return formatMetric(rec.score, "$");
  if (rec.objective === "best_latency") return formatMetric(rec.score, "ms");
  return (rec.score * 100).toFixed(0);
}

function scoreLabel(objective: string): string {
  if (objective === "lowest_cost") return "cost / query";
  if (objective === "best_latency") return "p50 latency";
  return "score";
}

export function RecommendationCard({ rec, rank }: { rec: Recommendation; rank: number }) {
  const [expanded, setExpanded] = useState(false);
  // No run qualified for this objective (e.g. its metric was never scored).
  const unavailable = (rec.status ?? "ok") !== "ok" || !rec.run_id;

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
            {unavailable ? (
              <p className="text-xs text-amber-400/80" data-testid="rec-unavailable">
                {rec.status === "error" ? "Could not be scored" : "No eligible runs"}: {rec.rationale}
              </p>
            ) : (
              <ConfigBadges config={rec.best_config as unknown as Record<string, unknown>} />
            )}
          </div>

          <div className="text-right flex-shrink-0">
            <p className={clsx("text-2xl font-bold tabular-nums", isScored(rec.score) ? "text-white" : "text-zinc-600")}>
              {formatScore(rec)}
            </p>
            <p className="text-[10px] text-zinc-600 font-mono">{scoreLabel(rec.objective)}</p>
          </div>
        </div>

        {/* Rationale toggle */}
        {!unavailable && (
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
        )}

        {expanded && !unavailable && (
          <div className="mt-3 p-3 bg-zinc-800/50 border border-zinc-700/50 rounded-lg">
            <p className="text-xs text-zinc-400 leading-relaxed">{rec.rationale}</p>
            <p className="text-[10px] text-zinc-600 mt-2 font-mono">run: {rec.run_id}</p>
          </div>
        )}
      </div>
    </div>
  );
}
