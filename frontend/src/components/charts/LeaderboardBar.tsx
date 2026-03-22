"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface LeaderboardBarProps {
  entries: Array<{
    run_id: string;
    config: any;
    metrics: any;
  }>;
  metric: string;
  metricLabel?: string;
  topK?: number;
}

export function LeaderboardBar({
  entries,
  metric,
  metricLabel,
  topK = 10,
}: LeaderboardBarProps) {
  const data = entries
    .filter((e) => e.metrics?.[metric] != null)
    .slice(0, topK)
    .map((e, i) => ({
      name: `#${i + 1} ${e.config?.retrieval_strategy ?? ""}`,
      value: e.metrics?.[metric] ?? 0,
      run_id: e.run_id,
    }));

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} layout="vertical" margin={{ left: 0, right: 16 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" horizontal={false} />
        <XAxis
          type="number"
          domain={[0, 1]}
          tick={{ fill: "#71717a", fontSize: 10 }}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          type="category"
          dataKey="name"
          tick={{ fill: "#71717a", fontSize: 10 }}
          tickLine={false}
          axisLine={false}
          width={100}
        />
        <Tooltip
          contentStyle={{
            background: "#18181b",
            border: "1px solid #3f3f46",
            borderRadius: "8px",
            fontSize: "12px",
            color: "#d4d4d8",
          }}
          formatter={(value: number) => [value.toFixed(3), metricLabel ?? metric]}
        />
        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
          {data.map((_, index) => (
            <Cell
              key={index}
              fill={index === 0 ? "#7c3aed" : index === 1 ? "#6d28d9" : "#4c1d95"}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
