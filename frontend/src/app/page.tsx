"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { corpusApi, evaluationApi, experimentsApi } from "@/lib/api";
import {
  ArrowRight, ArrowUpRight, Upload, FlaskConical,
  BarChart3, Stethoscope, Circle, CheckCircle2,
  Loader2, AlertCircle, FileText, Table2, Eye, Layers,
  Cpu, GitBranch, Search, Zap, Star,
} from "lucide-react";
import { clsx } from "clsx";
import type { Document, Experiment } from "@/types";

// ─── Tiny sparkline SVG ───────────────────────────────────────────────────────
function Sparkline({ values, color = "#a78bfa" }: { values: number[]; color?: string }) {
  if (values.length < 2) return null;
  const w = 64, h = 24, pad = 2;
  const min = Math.min(...values), max = Math.max(...values);
  const range = max - min || 1;
  const pts = values.map((v, i) => {
    const x = pad + (i / (values.length - 1)) * (w - pad * 2);
    const y = h - pad - ((v - min) / range) * (h - pad * 2);
    return `${x},${y}`;
  }).join(" ");
  return (
    <svg width={w} height={h} className="overflow-visible">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ─── Status dot ──────────────────────────────────────────────────────────────
function StatusDot({ status }: { status: string }) {
  const map: Record<string, { color: string; label: string }> = {
    done:     { color: "bg-emerald-500", label: "Done" },
    running:  { color: "bg-blue-500 animate-pulse", label: "Running" },
    pending:  { color: "bg-zinc-500", label: "Pending" },
    failed:   { color: "bg-red-500", label: "Failed" },
  };
  const s = map[status] ?? { color: "bg-zinc-600", label: status };
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={clsx("w-1.5 h-1.5 rounded-full flex-shrink-0", s.color)} />
      <span className="text-xs text-zinc-400">{s.label}</span>
    </span>
  );
}

// ─── Metric pill ─────────────────────────────────────────────────────────────
function MetricPill({ label, value, good }: { label: string; value: string; good: boolean }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-zinc-800/60 last:border-0">
      <span className="text-xs text-zinc-500">{label}</span>
      <span className={clsx("text-xs font-mono font-semibold", good ? "text-emerald-400" : "text-amber-400")}>
        {value}
      </span>
    </div>
  );
}

// ─── Pipeline step ────────────────────────────────────────────────────────────
const PIPELINE_STEPS = [
  { icon: FileText,   label: "Parse",    sub: "PDF · DOCX · XLSX" },
  { icon: Layers,     label: "Chunk",    sub: "4 strategies" },
  { icon: Cpu,        label: "Embed",    sub: "text-embedding-3" },
  { icon: Table2,     label: "Index",    sub: "Qdrant" },
  { icon: Search,     label: "Retrieve", sub: "Hybrid + BM25" },
  { icon: Star,       label: "Rerank",   sub: "Cross-encoder" },
  { icon: Zap,        label: "Generate", sub: "GPT-4o" },
  { icon: BarChart3,  label: "Evaluate", sub: "Ragas + custom" },
];

export default function HomePage() {
  const { data: docs = [] }        = useQuery<Document[]>({ queryKey: ["corpus"],      queryFn: corpusApi.list });
  const { data: experiments = [] } = useQuery<Experiment[]>({ queryKey: ["experiments"], queryFn: experimentsApi.list });
  const { data: leaderboard = [] } = useQuery<any[]>({ queryKey: ["leaderboard"], queryFn: () => evaluationApi.getLeaderboard() });

  const totalDocs      = docs.length;
  const totalExps      = experiments.length;
  const completedExps  = experiments.filter((e) => e.status === "done");
  const completedRuns  = experiments.flatMap((e) => e.runs || []).filter((r) => r.status === "done");
  const recentExps     = [...experiments].sort((a, b) => new Date(b.created_at ?? 0).getTime() - new Date(a.created_at ?? 0).getTime()).slice(0, 6);
  const topRuns        = (leaderboard as any[]).slice(0, 3);
  const bestRun        = topRuns[0];

  const faithScores    = topRuns.map((r) => r.metrics?.faithfulness ?? 0).filter(Boolean);
  const recallScores   = topRuns.map((r) => r.metrics?.context_recall ?? 0).filter(Boolean);

  return (
    <div className="min-h-screen bg-[#09090b] bg-noise">

      {/* ── Top bar ─────────────────────────────────────────────────────────── */}
      <div className="border-b border-zinc-800/50 px-8 py-5 flex items-center justify-between relative bg-dot-grid">
        <div>
          <h1 className="text-lg font-bold text-white tracking-tight">Overview</h1>
          <p className="text-xs text-zinc-500 mt-0.5">RAG pipeline diagnostics & benchmarking</p>
        </div>
        <div className="flex items-center gap-2.5">
          <Link
            href="/corpus"
            className="press inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-zinc-300 bg-zinc-800/80 border border-zinc-700/50 rounded-lg hover:bg-zinc-700/80 hover:text-white hover:border-zinc-600 transition-all"
          >
            <Upload className="w-3.5 h-3.5" />
            Upload Docs
          </Link>
          <Link
            href="/experiments"
            className="press inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-white rounded-lg transition-all glow-violet-sm hover:glow-violet-md"
            style={{ background: "linear-gradient(135deg,#7c3aed,#6d28d9)" }}
          >
            <FlaskConical className="w-3.5 h-3.5" />
            New Experiment
          </Link>
        </div>
      </div>

      <div className="px-8 py-7 max-w-7xl relative z-10">

        {/* ── KPI strip ────────────────────────────────────────────────────── */}
        <div className="grid grid-cols-4 gap-3 mb-7 stagger-in">
          {[
            {
              label: "Documents",
              value: totalDocs,
              sub: `${docs.filter((d) => d.parse_status === "done").length} indexed`,
              color: "text-blue-400",
              glow: "num-glow-blue",
              bar: docs.filter((d) => d.parse_status === "done").length / Math.max(totalDocs, 1),
              barColor: "#3b82f6",
            },
            {
              label: "Experiments",
              value: totalExps,
              sub: `${completedExps.length} completed`,
              color: "text-violet-400",
              glow: "num-glow-violet",
              bar: completedExps.length / Math.max(totalExps, 1),
              barColor: "#7c3aed",
            },
            {
              label: "Evaluated Runs",
              value: completedRuns.length,
              sub: "across all experiments",
              color: "text-emerald-400",
              glow: "num-glow-green",
              bar: 1,
              barColor: "#10b981",
            },
            {
              label: "Best Faithfulness",
              value: bestRun?.metrics?.faithfulness != null
                ? bestRun.metrics.faithfulness.toFixed(3)
                : "—",
              sub: bestRun ? `${bestRun.config?.retrieval_strategy ?? ""}` : "No data yet",
              color: "text-amber-400",
              glow: "num-glow-amber",
              bar: bestRun?.metrics?.faithfulness ?? 0,
              barColor: "#f59e0b",
            },
          ].map(({ label, value, sub, color, glow, bar, barColor }) => (
            <div
              key={label}
              className="rounded-xl p-4 bg-zinc-900/80 border border-zinc-800/60 card-lift gradient-border"
            >
              <p className="text-[11px] font-medium text-zinc-500 uppercase tracking-widest mb-3">{label}</p>
              <p className={clsx("text-3xl font-bold tracking-tight mb-1 tabular-nums", color, glow)}>{value}</p>
              <p className="text-[11px] text-zinc-600 mb-3">{sub}</p>
              <div className="h-[3px] bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-1000 ease-out"
                  style={{ width: `${Math.min(bar * 100, 100)}%`, background: `linear-gradient(90deg, ${barColor}, ${barColor}99)` }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* ── Main two-column layout ────────────────────────────────────────── */}
        <div className="grid grid-cols-3 gap-5 mb-5">

          {/* Left 2/3 — Recent Experiments */}
          <div className="col-span-2 rounded-xl bg-zinc-900 border border-zinc-800">
            <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800">
              <h2 className="text-sm font-semibold text-zinc-200">Recent Experiments</h2>
              <Link href="/experiments" className="flex items-center gap-1 text-xs text-violet-400 hover:text-violet-300 transition-colors font-medium">
                All experiments <ArrowRight className="w-3 h-3" />
              </Link>
            </div>
            {recentExps.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <FlaskConical className="w-8 h-8 text-zinc-700 mb-3" />
                <p className="text-sm text-zinc-500 mb-1">No experiments yet</p>
                <p className="text-xs text-zinc-600 mb-4">Upload documents and run your first benchmark</p>
                <Link href="/experiments" className="text-xs text-violet-400 hover:text-violet-300 transition-colors font-medium">
                  Create experiment →
                </Link>
              </div>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-zinc-800/60">
                    {["Name", "Status", "Runs", "Created"].map((h) => (
                      <th key={h} className="px-5 py-2.5 text-left text-[11px] font-medium text-zinc-600 uppercase tracking-wider">{h}</th>
                    ))}
                    <th className="px-5 py-2.5" />
                  </tr>
                </thead>
                <tbody>
                  {recentExps.map((exp, i) => {
                    const runs = exp.runs ?? [];
                    const done = runs.filter((r) => r.status === "done").length;
                    return (
                      <tr
                        key={exp.id}
                        className={clsx(
                          "hover:bg-zinc-800/30 transition-colors group",
                          i < recentExps.length - 1 && "border-b border-zinc-800/40"
                        )}
                      >
                        <td className="px-5 py-3">
                          <p className="text-sm font-medium text-zinc-200 truncate max-w-[180px]">{exp.name}</p>
                          {exp.description && (
                            <p className="text-[11px] text-zinc-600 truncate max-w-[180px] mt-0.5">{exp.description}</p>
                          )}
                        </td>
                        <td className="px-5 py-3"><StatusDot status={exp.status ?? "pending"} /></td>
                        <td className="px-5 py-3">
                          <span className="text-sm text-zinc-300 tabular-nums">{done}</span>
                          <span className="text-zinc-600 text-xs"> / {runs.length}</span>
                        </td>
                        <td className="px-5 py-3 text-xs text-zinc-500">
                          {exp.created_at ? new Date(exp.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric" }) : "—"}
                        </td>
                        <td className="px-5 py-3 text-right">
                          <Link
                            href={`/experiments/${exp.id}`}
                            className="opacity-0 group-hover:opacity-100 transition-opacity text-zinc-500 hover:text-violet-400"
                          >
                            <ArrowUpRight className="w-3.5 h-3.5" />
                          </Link>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>

          {/* Right 1/3 — Quick Actions + Status */}
          <div className="flex flex-col gap-4">

            {/* Quick actions */}
            <div className="rounded-xl bg-zinc-900 border border-zinc-800">
              <div className="px-4 py-3.5 border-b border-zinc-800">
                <h2 className="text-sm font-semibold text-zinc-200">Quick Start</h2>
              </div>
              <div className="p-3 space-y-1.5">
                {[
                  { href: "/corpus",      icon: Upload,      label: "Upload documents",    sub: "PDF, DOCX, XLSX, HTML" },
                  { href: "/experiments", icon: FlaskConical, label: "Run experiment",       sub: "Benchmark 24 configs" },
                  { href: "/leaderboard", icon: BarChart3,    label: "View leaderboard",    sub: "Compare all results" },
                  { href: "/diagnostics", icon: Stethoscope,  label: "Diagnose failures",   sub: "Root-cause analysis" },
                  { href: "/optimizer",   icon: Zap,          label: "Auto-optimize",        sub: "Hill-climb config search" },
                ].map(({ href, icon: Icon, label, sub }) => (
                  <Link
                    key={href}
                    href={href}
                    className="flex items-center gap-3 p-2.5 rounded-lg hover:bg-zinc-800/60 transition-colors group"
                  >
                    <div className="w-8 h-8 rounded-lg bg-zinc-800 border border-zinc-700 flex items-center justify-center flex-shrink-0 group-hover:border-zinc-600 transition-colors">
                      <Icon className="w-3.5 h-3.5 text-zinc-400 group-hover:text-violet-400 transition-colors" />
                    </div>
                    <div className="min-w-0">
                      <p className="text-xs font-medium text-zinc-300 group-hover:text-white transition-colors">{label}</p>
                      <p className="text-[10px] text-zinc-600">{sub}</p>
                    </div>
                    <ArrowRight className="w-3 h-3 text-zinc-700 group-hover:text-zinc-500 ml-auto flex-shrink-0 transition-colors" />
                  </Link>
                ))}
              </div>
            </div>

            {/* System status */}
            <div className="rounded-xl bg-zinc-900 border border-zinc-800">
              <div className="px-4 py-3.5 border-b border-zinc-800">
                <h2 className="text-sm font-semibold text-zinc-200">System</h2>
              </div>
              <div className="px-4 py-3 space-y-2.5">
                {[
                  { label: "API",               ok: true  },
                  { label: "Vector DB (Qdrant)", ok: totalDocs > 0 },
                  { label: "Worker Queue",       ok: true  },
                  { label: "Evaluation Engine",  ok: completedRuns.length > 0 },
                  { label: "ML Classifier",      ok: completedRuns.length > 0 },
                ].map(({ label, ok }) => (
                  <div key={label} className="flex items-center justify-between">
                    <span className="text-xs text-zinc-500">{label}</span>
                    <span className="flex items-center gap-1.5">
                      <span className={clsx("w-1.5 h-1.5 rounded-full", ok ? "bg-emerald-500" : "bg-zinc-600")} />
                      <span className={clsx("text-[11px] font-medium", ok ? "text-emerald-400" : "text-zinc-600")}>
                        {ok ? "Ready" : "Idle"}
                      </span>
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* ── Leaderboard snapshot ──────────────────────────────────────────── */}
        {topRuns.length > 0 && (
          <div className="rounded-xl bg-zinc-900 border border-zinc-800 mb-5">
            <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800">
              <h2 className="text-sm font-semibold text-zinc-200">Top Configurations</h2>
              <Link href="/leaderboard" className="flex items-center gap-1 text-xs text-violet-400 hover:text-violet-300 transition-colors font-medium">
                Full leaderboard <ArrowRight className="w-3 h-3" />
              </Link>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm min-w-[700px]">
                <thead>
                  <tr className="border-b border-zinc-800/60">
                    <th className="px-5 py-2.5 text-left text-[11px] font-medium text-zinc-600 uppercase tracking-wider w-6">#</th>
                    <th className="px-5 py-2.5 text-left text-[11px] font-medium text-zinc-600 uppercase tracking-wider">Config</th>
                    {["Faithfulness", "Ctx Recall", "Correctness", "Latency", "Cost"].map((h) => (
                      <th key={h} className="px-4 py-2.5 text-right text-[11px] font-medium text-zinc-600 uppercase tracking-wider">{h}</th>
                    ))}
                    <th className="px-5 py-2.5 text-right text-[11px] font-medium text-zinc-600 uppercase tracking-wider">Trend</th>
                  </tr>
                </thead>
                <tbody>
                  {topRuns.map((run, i) => {
                    const cfg = run.config ?? {};
                    const m = run.metrics ?? {};
                    const rank = i + 1;
                    return (
                      <tr key={run.run_id} className="border-b border-zinc-800/40 hover:bg-zinc-800/20 transition-colors last:border-0">
                        <td className="px-5 py-3.5">
                          <span className={clsx("text-sm font-bold tabular-nums",
                            rank === 1 ? "text-amber-400" : rank === 2 ? "text-zinc-400" : "text-zinc-600"
                          )}>{rank}</span>
                        </td>
                        <td className="px-5 py-3.5">
                          <div className="flex flex-wrap gap-1">
                            {[cfg.retrieval_strategy, cfg.chunking_strategy, cfg.parsing_strategy].filter(Boolean).map((v: string) => (
                              <span key={v} className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 border border-zinc-700 text-zinc-400 font-mono">{v}</span>
                            ))}
                          </div>
                        </td>
                        {[
                          { v: m.faithfulness,      fmt: (x: number) => x.toFixed(3), good: (x: number) => x >= 0.7 },
                          { v: m.context_recall,    fmt: (x: number) => x.toFixed(3), good: (x: number) => x >= 0.6 },
                          { v: m.answer_correctness,fmt: (x: number) => x.toFixed(3), good: (x: number) => x >= 0.6 },
                          { v: m.latency_p50_ms,    fmt: (x: number) => `${Math.round(x)}ms`, good: (x: number) => x < 2000 },
                          { v: m.avg_cost_usd,      fmt: (x: number) => `$${x.toFixed(4)}`, good: (x: number) => x < 0.01 },
                        ].map(({ v, fmt, good }, ci) => (
                          <td key={ci} className="px-4 py-3.5 text-right">
                            {v != null ? (
                              <span className={clsx("text-xs font-mono font-semibold tabular-nums",
                                good(v) ? "text-emerald-400" : "text-amber-400"
                              )}>{fmt(v)}</span>
                            ) : (
                              <span className="text-xs text-zinc-700">—</span>
                            )}
                          </td>
                        ))}
                        <td className="px-5 py-3.5 text-right">
                          <Sparkline
                            values={[m.context_recall, m.faithfulness, m.answer_correctness].filter((x) => x != null)}
                            color={rank === 1 ? "#a78bfa" : "#52525b"}
                          />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── Pipeline visualization ────────────────────────────────────────── */}
        <div className="rounded-xl bg-zinc-900/80 border border-zinc-800/60 p-5 mb-5 relative overflow-hidden">
          <div className="absolute inset-0 bg-dot-grid opacity-40" />
          <p className="text-[11px] font-semibold text-zinc-500 uppercase tracking-[0.15em] mb-5 relative z-10">Pipeline Architecture</p>
          <div className="flex items-center gap-0 overflow-x-auto pb-1 relative z-10 stagger-in">
            {PIPELINE_STEPS.map(({ icon: Icon, label, sub }, i) => (
              <div key={label} className="flex items-center min-w-0 flex-shrink-0">
                <div className="flex flex-col items-center gap-1.5 group cursor-default px-1">
                  <div className="w-10 h-10 rounded-xl bg-zinc-800/80 border border-zinc-700/50 flex items-center justify-center group-hover:border-violet-500/40 group-hover:bg-violet-500/10 transition-all duration-300 group-hover:glow-violet-sm">
                    <Icon className="w-4 h-4 text-zinc-400 group-hover:text-violet-400 transition-colors duration-300" />
                  </div>
                  <p className="text-[11px] font-semibold text-zinc-300 text-center">{label}</p>
                  <p className="text-[9px] text-zinc-600 text-center whitespace-nowrap">{sub}</p>
                </div>
                {i < PIPELINE_STEPS.length - 1 && (
                  <div className="flex-1 mx-1.5 min-w-[24px] flex items-center">
                    <div className="h-px flex-1 bg-gradient-to-r from-zinc-700/60 to-zinc-800/40" />
                    <div className="w-1 h-1 rounded-full bg-zinc-600 mx-0.5" />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* ── Tech stack ───────────────────────────────────────────────────── */}
        <div className="flex items-center gap-2 flex-wrap stagger-in">
          <span className="text-[10px] text-zinc-600 font-semibold uppercase tracking-[0.15em] mr-1">Stack</span>
          {[
            "FastAPI", "Celery", "PostgreSQL", "Qdrant",
            "LangChain", "OpenAI", "Ragas", "XGBoost",
            "PyTorch", "TensorRT", "Triton", "Whisper",
            "sentence-transformers", "Next.js", "Docker",
          ].map((tech) => (
            <span
              key={tech}
              className="text-[10px] px-2 py-0.5 rounded-md bg-zinc-900/80 border border-zinc-800/60 text-zinc-500 font-mono hover:border-zinc-600 hover:text-zinc-300 transition-all cursor-default"
            >
              {tech}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
