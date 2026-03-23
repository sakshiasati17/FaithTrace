"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { clsx } from "clsx";
import { corpusApi, evaluationApi, experimentsApi } from "@/lib/api";
import {
  Database,
  FlaskConical,
  BarChart3,
  Upload,
  ArrowRight,
  Zap,
  Shield,
  Clock,
  GitBranch,
  TrendingUp,
  CheckCircle2,
  Sparkles,
} from "lucide-react";
import type { Document, Experiment } from "@/types";

function StatCard({ label, value, sub, icon: Icon, gradient }: {
  label: string;
  value: string | number;
  sub?: string;
  icon: React.ElementType;
  gradient: string;
}) {
  return (
    <div
      className="relative rounded-2xl p-5 transition-all duration-300 hover:scale-[1.01] group cursor-default"
      style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}
    >
      <div className="flex items-start justify-between mb-4">
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
          style={{ background: gradient }}
        >
          <Icon className="w-5 h-5 text-white" />
        </div>
        <span className="text-[10px] text-zinc-600 font-mono uppercase tracking-widest">{label}</span>
      </div>
      <p className="text-3xl font-bold text-white mb-1 tracking-tight">{value}</p>
      {sub && <p className="text-xs text-zinc-600">{sub}</p>}
    </div>
  );
}

function FeatureCard({ icon: Icon, title, description, highlighted = false }: {
  icon: React.ElementType;
  title: string;
  description: string;
  highlighted?: boolean;
}) {
  return (
    <div
      className={clsx(
        "group relative rounded-2xl p-5 transition-all duration-300",
        highlighted ? "" : ""
      )}
      style={{
        background: highlighted ? "rgba(124,58,237,0.04)" : "rgba(255,255,255,0.02)",
        border: highlighted ? "1px solid rgba(124,58,237,0.2)" : "1px solid rgba(255,255,255,0.06)",
      }}
    >
      <div
        className="w-9 h-9 rounded-xl flex items-center justify-center mb-4"
        style={{
          background: "rgba(124,58,237,0.12)",
          border: "1px solid rgba(124,58,237,0.2)",
        }}
      >
        <Icon className="w-4 h-4 text-violet-400" />
      </div>
      <h3 className="text-sm font-semibold text-zinc-100 mb-2">{title}</h3>
      <p className="text-xs text-zinc-500 leading-relaxed">{description}</p>
    </div>
  );
}

export default function HomePage() {
  const { data: docs = [] as Document[] } = useQuery<Document[]>({ queryKey: ["corpus"], queryFn: corpusApi.list });
  const { data: experiments = [] as Experiment[] } = useQuery<Experiment[]>({ queryKey: ["experiments"], queryFn: experimentsApi.list });
  const { data: leaderboard = [] as any[] } = useQuery<any[]>({
    queryKey: ["leaderboard"],
    queryFn: () => evaluationApi.getLeaderboard(),
  });

  const docCount = (docs as Document[]).length;
  const expCount = (experiments as Experiment[]).length;
  const completedRuns = (experiments as Experiment[]).flatMap((e) => e.runs || []).filter((r) => r.status === "done").length;
  const bestRun = leaderboard?.[0];

  return (
    <div className="min-h-screen">
      {/* Hero */}
      <div className="relative overflow-hidden" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
        {/* Background grid */}
        <div className="absolute inset-0 bg-dot-grid opacity-60" />

        {/* Ambient glow blobs */}
        <div
          className="absolute -top-32 left-1/4 w-[500px] h-[400px] rounded-full animate-glow-pulse"
          style={{ background: "radial-gradient(ellipse, rgba(124,58,237,0.12) 0%, transparent 70%)" }}
        />
        <div
          className="absolute -top-20 right-1/4 w-[350px] h-[300px] rounded-full animate-glow-pulse"
          style={{ background: "radial-gradient(ellipse, rgba(168,85,247,0.08) 0%, transparent 70%)", animationDelay: "2s" }}
        />

        {/* Top border glow */}
        <div className="absolute top-0 left-0 right-0 h-px" style={{ background: "linear-gradient(90deg, transparent, rgba(124,58,237,0.5), rgba(168,85,247,0.2), transparent)" }} />

        <div className="relative max-w-6xl mx-auto px-8 py-20">
          {/* Badge */}
          <div className="flex items-center gap-2 mb-7">
            <span
              className="inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 text-violet-300 rounded-full"
              style={{ background: "rgba(124,58,237,0.12)", border: "1px solid rgba(124,58,237,0.25)" }}
            >
              <Sparkles className="w-3 h-3" />
              Master&apos;s Research Platform
            </span>
          </div>

          {/* Heading */}
          <h1 className="text-6xl font-bold mb-5 leading-[1.05] tracking-tight">
            <span className="gradient-text">FaithTrace</span>
          </h1>
          <p className="text-xl text-zinc-300 max-w-lg leading-relaxed mb-3 font-medium">
            Temporal + Multimodal RAG Diagnostics
          </p>
          <p className="text-sm text-zinc-500 max-w-2xl leading-relaxed mb-10">
            Benchmark, diagnose, and compare RAG pipeline configurations across enterprise document corpora — with
            built-in temporal validity scoring and multimodal grounding analysis.
          </p>

          {/* CTAs */}
          <div className="flex items-center gap-3">
            <Link
              href="/corpus"
              className="inline-flex items-center gap-2 px-5 py-2.5 text-white text-sm font-semibold rounded-xl transition-all duration-200 shadow-glow-sm hover:shadow-glow-md hover:scale-[1.02]"
              style={{ background: "linear-gradient(135deg, #7c3aed, #a855f7)" }}
            >
              <Upload className="w-4 h-4" />
              Upload Documents
            </Link>
            <Link
              href="/experiments"
              className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-medium rounded-xl transition-all duration-200 text-zinc-200 hover:text-white hover:scale-[1.01]"
              style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.09)" }}
            >
              <FlaskConical className="w-4 h-4" />
              Run Experiments
            </Link>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-8 py-10">
        {/* Stats */}
        <div className="grid grid-cols-4 gap-4 mb-10">
          <StatCard
            label="Documents"
            value={docCount}
            sub={`${(docs as Document[]).filter((d) => d.parse_status === "done").length} indexed`}
            icon={Database}
            gradient="linear-gradient(135deg, #2563eb, #1d4ed8)"
          />
          <StatCard
            label="Experiments"
            value={expCount}
            sub={`${(experiments as Experiment[]).filter((e) => e.status === "done").length} completed`}
            icon={FlaskConical}
            gradient="linear-gradient(135deg, #7c3aed, #6d28d9)"
          />
          <StatCard
            label="Runs Completed"
            value={completedRuns}
            sub="across all experiments"
            icon={CheckCircle2}
            gradient="linear-gradient(135deg, #059669, #047857)"
          />
          <StatCard
            label="Best Faithfulness"
            value={bestRun?.metrics?.faithfulness != null
              ? bestRun.metrics.faithfulness.toFixed(3)
              : "—"
            }
            sub={bestRun ? `Run ${bestRun.run_id.slice(0, 8)}` : "No runs yet"}
            icon={TrendingUp}
            gradient="linear-gradient(135deg, #d97706, #b45309)"
          />
        </div>

        {/* Best Configuration */}
        {bestRun && (
          <div
            className="relative rounded-2xl p-6 mb-10 overflow-hidden"
            style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}
          >
            {/* Top glow line */}
            <div className="absolute top-0 left-0 right-0 h-px" style={{ background: "linear-gradient(90deg, transparent, rgba(124,58,237,0.4), transparent)" }} />

            <div className="flex items-center justify-between mb-5">
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                <h2 className="text-xs font-semibold text-zinc-400 uppercase tracking-widest">
                  Top Performing Configuration
                </h2>
              </div>
              <Link
                href="/leaderboard"
                className="flex items-center gap-1.5 text-xs text-violet-400 hover:text-violet-300 transition-colors font-semibold"
              >
                Full leaderboard <ArrowRight className="w-3 h-3" />
              </Link>
            </div>

            <div className="grid grid-cols-4 gap-3 mb-5">
              {[
                { label: "Retrieval", value: bestRun.config?.retrieval_strategy },
                { label: "Chunking", value: bestRun.config?.chunking_strategy },
                { label: "Parsing", value: bestRun.config?.parsing_strategy },
                { label: "Freshness", value: bestRun.config?.freshness_policy },
              ].map(({ label, value }) => (
                <div
                  key={label}
                  className="rounded-xl px-3 py-2.5"
                  style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}
                >
                  <p className="text-[10px] text-zinc-600 uppercase tracking-widest mb-1.5 font-medium">{label}</p>
                  <p className="text-xs font-semibold text-zinc-200 font-mono">{value ?? "—"}</p>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-5 gap-3 pt-5" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
              {[
                { label: "Faithfulness", value: bestRun.metrics?.faithfulness },
                { label: "Context Recall", value: bestRun.metrics?.context_recall },
                { label: "Answer Correctness", value: bestRun.metrics?.answer_correctness },
                { label: "Freshness Validity", value: bestRun.metrics?.freshness_validity },
                { label: "Latency p50", value: bestRun.metrics?.latency_p50_ms, unit: "ms", isMs: true },
              ].map(({ label, value, unit, isMs }) => (
                <div key={label} className="text-center">
                  <p className="text-[10px] text-zinc-600 mb-2 font-medium tracking-wide">{label}</p>
                  <p className="text-lg font-bold text-white tracking-tight">
                    {value != null
                      ? isMs
                        ? `${Math.round(value as number)}${unit}`
                        : (value as number).toFixed(3)
                      : "—"}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Platform Capabilities */}
        <div className="mb-8">
          <div className="flex items-center gap-4 mb-6">
            <div className="h-px flex-1" style={{ background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.06))" }} />
            <h2 className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest whitespace-nowrap">Platform Capabilities</h2>
            <div className="h-px flex-1" style={{ background: "linear-gradient(90deg, rgba(255,255,255,0.06), transparent)" }} />
          </div>
          <div className="grid grid-cols-3 gap-4">
            <FeatureCard
              icon={GitBranch}
              title="Multi-Config Benchmarking"
              description="Compare up to 256 pipeline variants across retrieval strategies, chunking methods, parsing approaches, and freshness policies in a single experiment."
            />
            <FeatureCard
              icon={Clock}
              title="Temporal Validity Scoring"
              description="Evaluate whether retrieved chunks are version-correct for the query date using effective_from / effective_to metadata across document versions."
            />
            <FeatureCard
              icon={Shield}
              title="Multimodal Grounding"
              description="Score how well the pipeline grounds answers in table, spreadsheet, and chart content — not just plain text — for complex enterprise documents."
              highlighted
            />
            <FeatureCard
              icon={Zap}
              title="Root-Cause Diagnostics"
              description="Classify each query failure into one of 8 categories: stale answer, wrong version, table miss, low recall, irrelevant context, and more."
              highlighted
            />
            <FeatureCard
              icon={BarChart3}
              title="Ragas Evaluation Suite"
              description="Automatic evaluation with Faithfulness, Context Precision, Context Recall, Answer Relevance, and Answer Correctness via the Ragas framework."
            />
            <FeatureCard
              icon={TrendingUp}
              title="Config Recommendations"
              description="Get objective-specific recommendations: best overall, lowest cost, fastest latency, best for tables, and best temporal drift handling."
            />
          </div>
        </div>

        {/* Quick Start */}
        <div
          className="relative rounded-2xl p-6 overflow-hidden"
          style={{ background: "rgba(255,255,255,0.015)", border: "1px solid rgba(255,255,255,0.06)" }}
        >
          {/* Ambient glow */}
          <div
            className="absolute bottom-0 right-0 w-80 h-80 rounded-full pointer-events-none"
            style={{ background: "radial-gradient(ellipse, rgba(124,58,237,0.06) 0%, transparent 70%)", transform: "translate(30%, 30%)" }}
          />

          <h2 className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-5">Get Started in 3 Steps</h2>
          <div className="grid grid-cols-3 gap-3">
            {[
              { step: "01", title: "Upload Corpus", desc: "Add PDFs, DOCX, XLSX or HTML documents to your knowledge base", href: "/corpus", cta: "Open Corpus" },
              { step: "02", title: "Run Experiment", desc: "Benchmark 24 MVP pipeline configurations automatically", href: "/experiments", cta: "New Experiment" },
              { step: "03", title: "View Results", desc: "Analyze the leaderboard and drill into failure diagnostics", href: "/leaderboard", cta: "View Leaderboard" },
            ].map(({ step, title, desc, href, cta }) => (
              <Link key={step} href={href} className="block group">
                <div
                  className="rounded-xl p-4 transition-all duration-200 h-full"
                  style={{
                    background: "rgba(255,255,255,0.02)",
                    border: "1px solid rgba(255,255,255,0.06)",
                  }}
                >
                  <div className="flex items-center gap-2.5 mb-3">
                    <span className="text-[10px] font-bold text-violet-600 font-mono tracking-widest">{step}</span>
                    <p className="text-sm font-semibold text-zinc-200">{title}</p>
                  </div>
                  <p className="text-xs text-zinc-600 mb-4 leading-relaxed">{desc}</p>
                  <div className="inline-flex items-center gap-1.5 text-xs text-violet-400 group-hover:text-violet-300 transition-colors font-semibold">
                    {cta}
                    <ArrowRight className="w-3 h-3 group-hover:translate-x-0.5 transition-transform" />
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
