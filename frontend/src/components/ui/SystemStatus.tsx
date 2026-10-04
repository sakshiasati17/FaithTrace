"use client";

import { useQuery } from "@tanstack/react-query";
import { clsx } from "clsx";
import { systemApi } from "@/lib/api";
import type { ComponentStatus, SystemComponent, SystemStatus } from "@/types";

const POLL_MS = 30_000;

export function useSystemStatus() {
  return useQuery<SystemStatus>({
    queryKey: ["system-status"],
    queryFn: systemApi.getStatus,
    refetchInterval: POLL_MS,
    staleTime: POLL_MS,
    retry: 0,
  });
}

type Tone = "ok" | "warn" | "down" | "unknown";

const DOT: Record<Tone, string> = {
  ok: "bg-emerald-500",
  warn: "bg-amber-500",
  down: "bg-red-500",
  unknown: "bg-zinc-600",
};

const TEXT: Record<Tone, string> = {
  ok: "text-emerald-400",
  warn: "text-amber-400",
  down: "text-red-400",
  unknown: "text-zinc-500",
};

// ─── Sidebar footer indicator ────────────────────────────────────────────────

export function SystemStatusIndicator() {
  const { data, isLoading, isError } = useSystemStatus();

  let tone: Tone;
  let label: string;
  if (isLoading) {
    tone = "unknown";
    label = "Checking status…";
  } else if (isError || !data) {
    tone = "down";
    label = "API offline";
  } else if (data.ok) {
    tone = "ok";
    label = "All systems online";
  } else {
    const down = Object.values(data.components).filter((c) => c && !c.ok).length;
    tone = "warn";
    label = `Degraded: ${down} component${down === 1 ? "" : "s"} down`;
  }

  return (
    <div data-testid="system-status-indicator" data-tone={tone}>
      <div className="flex items-center gap-2 mb-2">
        <span className="relative flex h-2 w-2">
          {tone === "ok" && (
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-40" />
          )}
          <span className={clsx("relative inline-flex rounded-full h-2 w-2", DOT[tone])} />
        </span>
        <p className="text-[10px] text-zinc-500 font-medium">{label}</p>
      </div>
      <p className="text-[9px] text-zinc-700 font-mono tracking-wide">
        {data?.version ? `v${data.version}` : "version unknown"}
      </p>
    </div>
  );
}

// ─── Home page component list ────────────────────────────────────────────────

const ROWS: { key: SystemComponent; label: string }[] = [
  { key: "database",      label: "Database" },
  { key: "qdrant",        label: "Vector DB (Qdrant)" },
  { key: "redis",         label: "Broker (Redis)" },
  { key: "workers",       label: "Worker Queue" },
  { key: "ml_classifier", label: "ML Classifier" },
];

function componentState(key: SystemComponent, c: ComponentStatus | undefined): { tone: Tone; text: string } {
  if (!c) return { tone: "unknown", text: "Unknown" };
  if (!c.ok) return { tone: "down", text: "Down" };
  if (key === "ml_classifier") {
    return c.trained ? { tone: "ok", text: "Trained" } : { tone: "warn", text: "Heuristic" };
  }
  return { tone: "ok", text: "Ready" };
}

function StatusRow({ label, tone, text, title }: { label: string; tone: Tone; text: string; title?: string }) {
  return (
    <div className="flex items-center justify-between" title={title}>
      <span className="text-xs text-zinc-500">{label}</span>
      <span className="flex items-center gap-1.5">
        <span className={clsx("w-1.5 h-1.5 rounded-full", DOT[tone])} />
        <span className={clsx("text-[11px] font-medium", TEXT[tone])}>{text}</span>
      </span>
    </div>
  );
}

export function SystemStatusList() {
  const { data, isLoading, isError } = useSystemStatus();
  const reachable = !!data && !isError;

  const api = isLoading
    ? { tone: "unknown" as Tone, text: "Checking…" }
    : reachable
    ? { tone: "ok" as Tone, text: "Online" }
    : { tone: "down" as Tone, text: "Offline" };

  return (
    <div className="px-4 py-3 space-y-2.5" data-testid="system-status-list">
      <StatusRow label="API" tone={api.tone} text={api.text} />
      {ROWS.map(({ key, label }) => {
        const c = reachable ? data.components[key] : undefined;
        const { tone, text } = reachable ? componentState(key, c) : { tone: "unknown" as Tone, text: "Unknown" };
        return <StatusRow key={key} label={label} tone={tone} text={text} title={c?.detail} />;
      })}
    </div>
  );
}
