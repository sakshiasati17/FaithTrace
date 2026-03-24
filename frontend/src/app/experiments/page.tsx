"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { experimentsApi } from "@/lib/api";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { FlaskConical, Plus, X, ChevronRight, Clock } from "lucide-react";
import type { Experiment } from "@/types";

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function NewExperimentModal({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [form, setForm] = useState({
    name: "",
    description: "",
    eval_set_path: "eval_sets/procurement_policy_eval.json",
    config_preset: "mvp",
  });
  const [error, setError] = useState<string | null>(null);

  const createMut = useMutation({
    mutationFn: experimentsApi.create,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["experiments"] });
      onClose();
    },
    onError: (err: any) => {
      setError(err?.response?.data?.detail ?? "Failed to create experiment");
    },
  });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-zinc-900 border border-zinc-700 rounded-2xl p-6 w-full max-w-md shadow-2xl">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-base font-semibold text-white">New Experiment</h2>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300 transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-xs text-zinc-500 mb-1.5">Experiment Name *</label>
            <input
              type="text"
              value={form.name}
              onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
              placeholder="e.g. Procurement Policy Analysis Q1"
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-violet-500"
            />
          </div>
          <div>
            <label className="block text-xs text-zinc-500 mb-1.5">Description</label>
            <textarea
              value={form.description}
              onChange={(e) => setForm((f) => ({ ...f, description: e.target.value }))}
              placeholder="What are you testing in this experiment?"
              rows={2}
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-violet-500 resize-none"
            />
          </div>
          <div>
            <label className="block text-xs text-zinc-500 mb-1.5">Eval Set Path</label>
            <input
              type="text"
              value={form.eval_set_path}
              onChange={(e) => setForm((f) => ({ ...f, eval_set_path: e.target.value }))}
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 font-mono focus:outline-none focus:border-violet-500"
            />
          </div>
          <div>
            <label className="block text-xs text-zinc-500 mb-1.5">Config Preset</label>
            <select
              value={form.config_preset}
              onChange={(e) => setForm((f) => ({ ...f, config_preset: e.target.value }))}
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-violet-500"
            >
              <option value="mvp">MVP (24 configs — recommended)</option>
            </select>
            <p className="text-[10px] text-zinc-600 mt-1">
              MVP runs: 3 retrieval × 2 chunking × 2 parsing × 2 freshness = 24 pipeline configurations
            </p>
          </div>

          {error && (
            <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
              {error}
            </div>
          )}

          <div className="flex gap-2 pt-1">
            <button
              onClick={onClose}
              className="flex-1 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-sm rounded-lg transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={() => createMut.mutate(form as any)}
              disabled={!form.name || createMut.isPending}
              className="flex-1 px-4 py-2 bg-violet-600 hover:bg-violet-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
            >
              {createMut.isPending ? "Creating…" : "Create & Run"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ExperimentsPage() {
  const [showModal, setShowModal] = useState(false);

  const { data: experiments = [] as Experiment[], isLoading } = useQuery<Experiment[]>({
    queryKey: ["experiments"],
    queryFn: experimentsApi.list,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return false;
      return (data as Experiment[]).some(
        (e) => e.status === "running" || e.status === "pending"
      )
        ? 5000
        : false;
    },
  });

  return (
    <div className="max-w-6xl mx-auto px-8 py-10">
      {showModal && <NewExperimentModal onClose={() => setShowModal(false)} />}

      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white mb-1">Experiments</h1>
          <p className="text-sm text-zinc-500">Benchmark RAG pipeline configurations against your corpus</p>
        </div>
        <button
          onClick={() => setShowModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white text-sm font-medium rounded-lg transition-colors"
        >
          <Plus className="w-4 h-4" />
          New Experiment
        </button>
      </div>

      {isLoading ? (
        <div className="py-20 text-center text-sm text-zinc-600">Loading…</div>
      ) : experiments.length === 0 ? (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl py-20 text-center">
          <FlaskConical className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
          <p className="text-sm font-medium text-zinc-500 mb-1">No experiments yet</p>
          <p className="text-xs text-zinc-600 mb-4">Upload documents first, then create an experiment</p>
          <button
            onClick={() => setShowModal(true)}
            className="px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white text-sm rounded-lg transition-colors"
          >
            Create First Experiment
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          {(experiments as Experiment[]).map((exp: Experiment) => {
            const totalRuns = exp.runs?.length ?? 0;
            const doneRuns = exp.runs?.filter((r) => r.status === "done").length ?? 0;
            const failedRuns = exp.runs?.filter((r) => r.status === "failed").length ?? 0;

            return (
              <Link key={exp.id} href={`/experiments/${exp.id}`}>
                <div className="bg-zinc-900 border border-zinc-800 hover:border-zinc-700 rounded-xl p-5 transition-all cursor-pointer">
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-3 mb-1">
                        <h3 className="text-sm font-semibold text-white truncate">{exp.name}</h3>
                        <StatusBadge status={exp.status} />
                      </div>
                      {exp.description && (
                        <p className="text-xs text-zinc-500 mb-2 line-clamp-1">{exp.description}</p>
                      )}
                      <div className="flex items-center gap-4 text-xs text-zinc-600">
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {formatDate(exp.created_at)}
                        </span>
                        <span>{totalRuns} runs</span>
                        {doneRuns > 0 && <span className="text-emerald-600">{doneRuns} done</span>}
                        {failedRuns > 0 && <span className="text-red-600">{failedRuns} failed</span>}
                      </div>
                    </div>
                    <ChevronRight className="w-4 h-4 text-zinc-600 flex-shrink-0 mt-0.5" />
                  </div>

                  {/* Run progress bar */}
                  {totalRuns > 0 && (
                    <div className="mt-3">
                      <div className="flex gap-0.5 h-1 rounded-full overflow-hidden bg-zinc-800">
                        <div
                          className="bg-emerald-500 transition-all"
                          style={{ width: `${(doneRuns / totalRuns) * 100}%` }}
                        />
                        <div
                          className="bg-red-500/70 transition-all"
                          style={{ width: `${(failedRuns / totalRuns) * 100}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}
