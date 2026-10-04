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
import { isScored } from "@/lib/metrics";

interface MetricsRadarProps {
  metrics: Partial<RunMetrics>;
  runLabel?: string;
}

export function MetricsRadar({ metrics, runLabel }: MetricsRadarProps) {
  // Unscored (null) metrics are left off the chart rather than drawn as 0.
  const data = [
    { metric: "Faithfulness", value: metrics.faithfulness },
    { metric: "Ctx Recall", value: metrics.context_recall },
    { metric: "Ctx Precision", value: metrics.context_precision },
    { metric: "Ans Relevance", value: metrics.answer_relevance },
    { metric: "Ans Correctness", value: metrics.answer_correctness },
    { metric: "Freshness", value: metrics.freshness_validity },
    { metric: "Multimodal", value: metrics.multimodal_grounding_rate },
  ].filter((d): d is { metric: string; value: number } => isScored(d.value));

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
