"use client";

import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { optimizationApi } from "@/lib/api";
import { Cpu, Zap, BarChart3, Loader2, AlertTriangle, CheckCircle2, Mic } from "lucide-react";
import { clsx } from "clsx";
import type { BenchmarkResult } from "@/types";
import { VoicePanel } from "@/components/ui/VoicePanel";

// ─── Precision badge ──────────────────────────────────────────────────────────
function PrecisionBadge({ precision }: { precision: string }) {
  const colors: Record<string, string> = {
    fp32: "bg-zinc-800 text-zinc-400 border-zinc-700",
    fp16: "bg-blue-900/40 text-blue-300 border-blue-700/50",
    int8: "bg-emerald-900/40 text-emerald-300 border-emerald-700/50",
  };
  return (
    <span className={clsx("text-[10px] font-mono px-1.5 py-0.5 rounded border", colors[precision] ?? colors.fp32)}>
      {precision.toUpperCase()}
    </span>
  );
}

// ─── Speedup bar ─────────────────────────────────────────────────────────────
function SpeedupBar({ speedup }: { speedup: number }) {
  const pct = Math.min((speedup / 5) * 100, 100);
  const color = speedup >= 3 ? "#10b981" : speedup >= 1.5 ? "#3b82f6" : "#71717a";
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 bg-zinc-800 rounded-full flex-1 overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-xs font-mono font-semibold tabular-nums" style={{ color }}>
        {speedup.toFixed(1)}x
      </span>
    </div>
  );
}

export default function InferencePage() {
  const [benchmarkQueued, setBenchmarkQueued] = useState(false);
  const [trainExperimentId, setTrainExperimentId] = useState("");
  const [trainQueued, setTrainQueued] = useState(false);

  const { data: gpuProfile, isLoading: gpuLoading } = useQuery({
    queryKey: ["gpu-profile"],
    queryFn: optimizationApi.gpuProfile,
    retry: false,
  });

  const { data: benchmark, isLoading: benchLoading, refetch: refetchBench } = useQuery({
    queryKey: ["benchmark-latest"],
    queryFn: optimizationApi.getLatestBenchmark,
    retry: false,
  });

  const benchmarkMut = useMutation({
    mutationFn: optimizationApi.runBenchmark,
    onSuccess: () => { setBenchmarkQueued(true); },
  });

  const trainMut = useMutation({
    mutationFn: ({ id }: { id: string }) => optimizationApi.trainPytorch(id),
    onSuccess: () => { setTrainQueued(true); },
  });

  const precisionColors: Record<string, string> = {
    fp32: "text-zinc-400",
    fp16: "text-blue-400",
    int8: "text-emerald-400",
  };

  return (
    <div className="max-w-6xl mx-auto px-8 py-10">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white mb-1">Inference Optimization</h1>
        <p className="text-sm text-zinc-500">
          GPU profiling · ONNX export · TensorRT benchmarks · PyTorch classifier training
        </p>
      </div>

      <div className="grid grid-cols-3 gap-5 mb-6">

        {/* ── GPU Profile ─────────────────────────────────────────────────── */}
        <div className="col-span-1 rounded-xl bg-zinc-900 border border-zinc-800 p-5">
          <div className="flex items-center gap-2 mb-4">
            <Cpu className="w-4 h-4 text-violet-400" />
            <h2 className="text-sm font-semibold text-zinc-200">GPU Profile</h2>
          </div>
          {gpuLoading ? (
            <div className="flex items-center gap-2 text-zinc-500 text-sm">
              <Loader2 className="w-4 h-4 animate-spin" /> Detecting…
            </div>
          ) : gpuProfile ? (
            <div className="space-y-3">
              <div>
                <p className="text-xs text-zinc-500 mb-0.5">Device</p>
                <p className="text-sm font-medium text-zinc-200">{gpuProfile.name}</p>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <p className="text-xs text-zinc-500 mb-0.5">VRAM Total</p>
                  <p className="text-sm font-mono text-zinc-300">{gpuProfile.vram_total_gb} GB</p>
                </div>
                <div>
                  <p className="text-xs text-zinc-500 mb-0.5">Available</p>
                  <p className="text-sm font-mono text-zinc-300">{gpuProfile.vram_available_gb} GB</p>
                </div>
              </div>
              <div>
                <p className="text-xs text-zinc-500 mb-1">Recommended Precision</p>
                <PrecisionBadge precision={gpuProfile.recommended_precision} />
              </div>
              <div>
                <p className="text-xs text-zinc-500 mb-0.5">Max Batch Size</p>
                <p className="text-sm font-mono text-zinc-300">{gpuProfile.max_batch_size}</p>
              </div>
              <div className="p-2.5 rounded-lg bg-zinc-800/60 border border-zinc-700/50">
                <p className="text-[11px] text-zinc-500 leading-relaxed">{gpuProfile.reasoning}</p>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-2 text-zinc-600 text-sm">
              <AlertTriangle className="w-4 h-4" /> No GPU data
            </div>
          )}
        </div>

        {/* ── Actions ──────────────────────────────────────────────────────── */}
        <div className="col-span-2 space-y-4">

          {/* Benchmark card */}
          <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-5">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-4 h-4 text-blue-400" />
              <h2 className="text-sm font-semibold text-zinc-200">Run Inference Benchmark</h2>
            </div>
            <p className="text-xs text-zinc-500 mb-4">
              Compares raw PyTorch vs TensorRT FP32 / FP16 / INT8.
              Measures p50, p95, p99 latency and throughput. Requires a trained classifier checkpoint.
            </p>
            <div className="flex items-center gap-3">
              <button
                onClick={() => benchmarkMut.mutate()}
                disabled={benchmarkMut.isPending || benchmarkQueued}
                className="inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                style={{ background: "linear-gradient(135deg,#2563eb,#1d4ed8)" }}
              >
                {benchmarkMut.isPending ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <BarChart3 className="w-3.5 h-3.5" />}
                {benchmarkQueued ? "Queued" : "Run Benchmark"}
              </button>
              {benchmarkQueued && (
                <button onClick={() => refetchBench()} className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors">
                  Refresh results
                </button>
              )}
            </div>
          </div>

          {/* Train PyTorch classifier card */}
          <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-5">
            <div className="flex items-center gap-2 mb-2">
              <Zap className="w-4 h-4 text-violet-400" />
              <h2 className="text-sm font-semibold text-zinc-200">Train PyTorch Classifier</h2>
            </div>
            <p className="text-xs text-zinc-500 mb-4">
              Fine-tunes DistilBERT on labelled query results from a completed experiment.
              Uses focal loss + label smoothing to handle class imbalance.
              Replaces the XGBoost model for failure classification.
            </p>
            <div className="flex items-center gap-3">
              <input
                type="text"
                value={trainExperimentId}
                onChange={(e) => setTrainExperimentId(e.target.value)}
                placeholder="Experiment ID"
                className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-violet-500 w-64"
              />
              <button
                onClick={() => trainMut.mutate({ id: trainExperimentId })}
                disabled={!trainExperimentId || trainMut.isPending || trainQueued}
                className="inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                style={{ background: "linear-gradient(135deg,#7c3aed,#6d28d9)" }}
              >
                {trainMut.isPending ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Zap className="w-3.5 h-3.5" />}
                {trainQueued ? "Queued" : "Train"}
              </button>
            </div>
            {trainQueued && (
              <div className="mt-3 flex items-center gap-2 text-xs text-emerald-400">
                <CheckCircle2 className="w-3.5 h-3.5" /> Training started — check Celery logs for progress
              </div>
            )}
          </div>

          <VoicePanel />
        </div>
      </div>

      {/* ── Benchmark Results ───────────────────────────────────────────────── */}
      {benchLoading ? (
        <div className="flex items-center gap-2 text-zinc-500 text-sm py-4">
          <Loader2 className="w-4 h-4 animate-spin" /> Loading benchmark…
        </div>
      ) : benchmark ? (
        <div className="rounded-xl bg-zinc-900 border border-zinc-800">
          <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800">
            <div>
              <h2 className="text-sm font-semibold text-zinc-200">Benchmark Results</h2>
              <p className="text-xs text-zinc-600 mt-0.5">
                {benchmark.gpu.name} · {new Date(benchmark.timestamp).toLocaleString()}
              </p>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm min-w-[700px]">
              <thead>
                <tr className="border-b border-zinc-800/60">
                  {["Engine", "Precision", "p50", "p95", "p99", "Throughput", "Speedup"].map((h) => (
                    <th key={h} className="px-5 py-2.5 text-left text-[11px] font-medium text-zinc-600 uppercase tracking-wider">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {benchmark.results.map((r: BenchmarkResult, i: number) => {
                  const parts = r.label.split("_");
                  const engine = parts[0];
                  const precision = parts[1] ?? "fp32";
                  return (
                    <tr key={r.label} className={clsx("border-b border-zinc-800/40 last:border-0 hover:bg-zinc-800/20 transition-colors", i === 0 && "bg-zinc-800/10")}>
                      <td className="px-5 py-3.5">
                        <span className="text-sm font-medium text-zinc-200 capitalize">{engine}</span>
                      </td>
                      <td className="px-5 py-3.5">
                        <PrecisionBadge precision={precision} />
                      </td>
                      {[r.p50_ms, r.p95_ms, r.p99_ms].map((v, ci) => (
                        <td key={ci} className="px-5 py-3.5">
                          <span className="text-xs font-mono text-zinc-300 tabular-nums">{v} ms</span>
                        </td>
                      ))}
                      <td className="px-5 py-3.5">
                        <span className="text-xs font-mono text-zinc-300 tabular-nums">{r.throughput_qps} qps</span>
                      </td>
                      <td className="px-5 py-3.5 min-w-[120px]">
                        {r.speedup_vs_pytorch != null ? (
                          <SpeedupBar speedup={r.speedup_vs_pytorch} />
                        ) : (
                          <span className="text-xs text-zinc-600">baseline</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-8 text-center">
          <BarChart3 className="w-8 h-8 text-zinc-700 mx-auto mb-3" />
          <p className="text-sm text-zinc-500 mb-1">No benchmark results yet</p>
          <p className="text-xs text-zinc-600">Train a classifier then run the benchmark above</p>
        </div>
      )}
    </div>
  );
}
