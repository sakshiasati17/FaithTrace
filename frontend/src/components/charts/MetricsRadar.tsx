"use client";

import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import type { RunMetrics } from "@/types";

interface MetricsRadarProps {
  metrics: Partial<RunMetrics>;
  runLabel?: string;
}

export function MetricsRadar({ metrics, runLabel }: MetricsRadarProps) {
  const data = [
    { metric: "Faithfulness", value: metrics.faithfulness ?? 0 },
    { metric: "Ctx Recall", value: metrics.context_recall ?? 0 },
    { metric: "Ctx Precision", value: metrics.context_precision ?? 0 },
    { metric: "Ans Relevance", value: metrics.answer_relevance ?? 0 },
    { metric: "Ans Correctness", value: metrics.answer_correctness ?? 0 },
    { metric: "Freshness", value: metrics.freshness_validity ?? 0 },
    { metric: "Multimodal", value: metrics.multimodal_grounding_rate ?? 0 },
  ];

  return (
    <div className="w-full">
      {runLabel && (
        <p className="text-xs text-zinc-500 mb-2 text-center">{runLabel}</p>
      )}
      <ResponsiveContainer width="100%" height={260}>
        <RadarChart data={data}>
          <PolarGrid stroke="#3f3f46" />
          <PolarAngleAxis
            dataKey="metric"
            tick={{ fill: "#71717a", fontSize: 11 }}
          />
          <Radar
            name="Score"
            dataKey="value"
            stroke="#7c3aed"
            fill="#7c3aed"
            fillOpacity={0.25}
            strokeWidth={2}
          />
          <Tooltip
            contentStyle={{
              background: "#18181b",
              border: "1px solid #3f3f46",
              borderRadius: "8px",
              fontSize: "12px",
              color: "#d4d4d8",
            }}
            formatter={(value: number) => [value.toFixed(3), "Score"]}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
