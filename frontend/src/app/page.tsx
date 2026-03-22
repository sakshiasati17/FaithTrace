"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
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
} from "lucide-react";

function StatCard({ label, value, sub, icon: Icon, color }: {
  label: string;
  value: string | number;
  sub?: string;
  icon: React.ElementType;
  color: string;
}) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-zinc-500 font-medium uppercase tracking-wider">{label}</p>
          <p className="text-3xl font-bold text-white mt-1.5">{value}</p>
          {sub && <p className="text-xs text-zinc-500 mt-1">{sub}</p>}
        </div>
        <div className={`w-10 h-10 rounded-lg ${color} flex items-center justify-center`}>
          <Icon className="w-5 h-5 text-white" />
        </div>
      </div>
    </div>
  );
}

function FeatureCard({ icon: Icon, title, description }: {
  icon: React.ElementType;
  title: string;
  description: string;
}) {
  return (
    <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-5 hover:border-zinc-700 transition-colors">
      <div className="w-9 h-9 rounded-lg bg-violet-600/20 flex items-center justify-center mb-3">
        <Icon className="w-4.5 h-4.5 text-violet-400" />
      </div>
      <h3 className="text-sm font-semibold text-white mb-1.5">{title}</h3>
      <p className="text-xs text-zinc-500 leading-relaxed">{description}</p>
    </div>
  );
}

export default function HomePage() {
  const { data: docs } = useQuery({ queryKey: ["corpus"], queryFn: corpusApi.list });
  const { data: experiments } = useQuery({ queryKey: ["experiments"], queryFn: experimentsApi.list });
  const { data: leaderboard } = useQuery({
    queryKey: ["leaderboard"],
    queryFn: () => evaluationApi.getLeaderboard(),
  });

  const docCount = docs?.length ?? 0;
  const expCount = experiments?.length ?? 0;
  const completedRuns = experiments?.flatMap((e) => e.runs).filter((r) => r.status === "done").length ?? 0;
  const bestRun = leaderboard?.[0];

  return (
    <div className="min-h-screen">
      {/* Hero */}
      <div className="border-b border-zinc-800 bg-gradient-to-b from-violet-950/20 to-transparent">
        <div className="max-w-6xl mx-auto px-8 py-16">
          <div className="flex items-center gap-2 mb-5">
            <span className="text-xs font-medium px-2.5 py-1 bg-violet-500/10 text-violet-400 rounded-full border border-violet-500/20">
              Master&apos;s Research Platform
            </span>
          </div>
          <h1 className="text-5xl font-bold text-white mb-4 leading-tight tracking-tight">
            FaithTrace
          </h1>
          <p className="text-xl text-zinc-400 max-w-2xl leading-relaxed mb-2">
            Temporal + Multimodal RAG Diagnostics Platform
          </p>
          <p className="text-sm text-zinc-600 max-w-xl leading-relaxed mb-8">
            Benchmark, diagnose, and compare RAG pipeline configurations across enterprise document corpora — with
            built-in temporal validity scoring and multimodal grounding analysis.
          </p>

          <div className="flex items-center gap-3">
            <Link
              href="/corpus"
              className="flex items-center gap-2 px-5 py-2.5 bg-violet-600 hover:bg-violet-500 text-white text-sm font-medium rounded-lg transition-colors"
            >
              <Upload className="w-4 h-4" />
              Upload Documents
            </Link>
            <Link
              href="/experiments"
              className="flex items-center gap-2 px-5 py-2.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-sm font-medium rounded-lg transition-colors"
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
            sub={`${docs?.filter((d) => d.parse_status === "done").length ?? 0} indexed`}
            icon={Database}
            color="bg-blue-600"
          />
          <StatCard
            label="Experiments"
            value={expCount}
            sub={`${experiments?.filter((e) => e.status === "done").length ?? 0} completed`}
            icon={FlaskConical}
            color="bg-violet-600"
          />
          <StatCard
            label="Runs Completed"
            value={completedRuns}
            sub="across all experiments"
            icon={CheckCircle2}
            color="bg-emerald-600"
          />
          <StatCard
            label="Best Faithfulness"
            value={bestRun?.metrics?.faithfulness != null
              ? bestRun.metrics.faithfulness.toFixed(3)
              : "—"
            }
            sub={bestRun ? `Run ${bestRun.run_id.slice(0, 8)}` : "No runs yet"}
            icon={TrendingUp}
            color="bg-amber-600"
          />
        </div>

        {/* Best Configuration */}
        {bestRun && (
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 mb-10">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider">
                Top Performing Configuration
              </h2>
              <Link
                href="/leaderboard"
                className="flex items-center gap-1 text-xs text-violet-400 hover:text-violet-300 transition-colors"
              >
                Full leaderboard <ArrowRight className="w-3 h-3" />
              </Link>
            </div>
            <div className="grid grid-cols-4 gap-4 mb-4">
              {[
                { label: "Retrieval", value: bestRun.config?.retrieval_strategy },
                { label: "Chunking", value: bestRun.config?.chunking_strategy },
                { label: "Parsing", value: bestRun.config?.parsing_strategy },
                { label: "Freshness", value: bestRun.config?.freshness_policy },
              ].map(({ label, value }) => (
                <div key={label} className="bg-zinc-800/50 rounded-lg px-3 py-2">
                  <p className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">{label}</p>
                  <p className="text-xs font-medium text-zinc-200">{value ?? "—"}</p>
                </div>
              ))}
            </div>
            <div className="grid grid-cols-5 gap-3">
              {[
                { label: "Faithfulness", value: bestRun.metrics?.faithfulness },
                { label: "Context Recall", value: bestRun.metrics?.context_recall },
                { label: "Answer Correctness", value: bestRun.metrics?.answer_correctness },
                { label: "Freshness Validity", value: bestRun.metrics?.freshness_validity },
                { label: "Latency p50", value: bestRun.metrics?.latency_p50_ms, unit: "ms", isMs: true },
              ].map(({ label, value, unit, isMs }) => (
                <div key={label} className="text-center">
                  <p className="text-[10px] text-zinc-500 mb-1">{label}</p>
                  <p className="text-base font-semibold text-white">
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

        {/* Features */}
        <div className="mb-8">
          <h2 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-5">Platform Capabilities</h2>
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
            />
            <FeatureCard
              icon={Zap}
              title="Root-Cause Diagnostics"
              description="Classify each query failure into one of 8 categories: stale answer, wrong version, table miss, low recall, irrelevant context, and more."
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

        {/* Quick actions */}
        <div className="bg-zinc-900/40 border border-zinc-800 rounded-xl p-6">
          <h2 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-4">Quick Start</h2>
          <div className="grid grid-cols-3 gap-3">
            {[
              {
                step: "1",
                title: "Upload corpus",
                desc: "Add PDFs, DOCX, XLSX or HTML documents",
                href: "/corpus",
                cta: "Open Corpus →",
              },
              {
                step: "2",
                title: "Run experiment",
                desc: "Benchmark 6 MVP pipeline configurations",
                href: "/experiments",
                cta: "New Experiment →",
              },
              {
                step: "3",
                title: "View results",
                desc: "Analyze leaderboard and failure diagnostics",
                href: "/leaderboard",
                cta: "View Leaderboard →",
              },
            ].map(({ step, title, desc, href, cta }) => (
              <Link key={step} href={href} className="block group">
                <div className="bg-zinc-800/40 hover:bg-zinc-800/80 border border-zinc-700/50 hover:border-zinc-600 rounded-lg p-4 transition-all">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="w-5 h-5 rounded-full bg-violet-600/30 text-violet-400 text-xs font-bold flex items-center justify-center">
                      {step}
                    </span>
                    <p className="text-sm font-medium text-zinc-200">{title}</p>
                  </div>
                  <p className="text-xs text-zinc-500 mb-3 leading-relaxed">{desc}</p>
                  <p className="text-xs text-violet-400 group-hover:text-violet-300 transition-colors font-medium">{cta}</p>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
