import { clsx } from "clsx";

interface MetricCardProps {
  label: string;
  value: number | null | undefined;
  unit?: string;
  description?: string;
  lowerIsBetter?: boolean;
  className?: string;
}

function scoreColor(value: number, lowerIsBetter: boolean): string {
  const v = lowerIsBetter ? 1 - Math.min(value, 1) : value;
  if (v >= 0.7) return "text-emerald-400";
  if (v >= 0.4) return "text-amber-400";
  return "text-red-400";
}

export function MetricCard({
  label,
  value,
  unit,
  description,
  lowerIsBetter = false,
  className,
}: MetricCardProps) {
  const display =
    value == null
      ? "—"
      : unit === "ms"
      ? `${Math.round(value)}ms`
      : unit === "$"
      ? `$${value.toFixed(4)}`
      : value.toFixed(3);

  const color = value != null ? scoreColor(unit === "ms" ? value / 5000 : value, lowerIsBetter) : "text-zinc-500";

  return (
    <div className={clsx("bg-zinc-900 border border-zinc-800 rounded-xl p-4 card-lift gradient-border", className)}>
      <p className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider mb-1">{label}</p>
      <p className={clsx("text-2xl font-bold tabular-nums", color)}>{display}</p>
      {description && <p className="text-[10px] text-zinc-600 mt-1 leading-snug">{description}</p>}
    </div>
  );
}
