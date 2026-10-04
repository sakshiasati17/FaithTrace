"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { experimentsApi, recommendationsApi } from "@/lib/api";
import { Lightbulb, CheckCircle2 } from "lucide-react";
import type { Experiment, Recommendation } from "@/types";
import { RecommendationCard } from "@/components/recommendations/RecommendationCard";

export default function RecommendationsPage() {
  const [selectedExperiment, setSelectedExperiment] = useState<string>("");

  const { data: experiments = [] as Experiment[] } = useQuery<Experiment[]>({
    queryKey: ["experiments"],
    queryFn: experimentsApi.list,
  });

  const completedExps = (experiments as Experiment[]).filter((e) => e.status === "done");

  const {
    data: recommendations = [],
    isLoading,
    isError,
  } = useQuery<Recommendation[]>({
    queryKey: ["recommendations", selectedExperiment],
    queryFn: () => recommendationsApi.get(selectedExperiment),
    enabled: !!selectedExperiment,
  });

  const recs = recommendations as Recommendation[];

  return (
    <div className="max-w-4xl mx-auto px-8 py-10">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white mb-1">Recommendations</h1>
        <p className="text-sm text-zinc-500">
          Best pipeline configurations per objective, derived from completed experiment runs
        </p>
      </div>

      {/* Experiment selector */}
      <div className="mb-6">
        <label className="block text-xs text-zinc-500 mb-1.5">Select Experiment</label>
        <select
          value={selectedExperiment}
          onChange={(e) => setSelectedExperiment(e.target.value)}
          className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-violet-500 min-w-64"
        >
          <option value="">— choose an experiment —</option>
          {completedExps.map((exp) => (
            <option key={exp.id} value={exp.id}>
              {exp.name} ({exp.runs?.length ?? 0} runs)
            </option>
          ))}
        </select>
      </div>

      {!selectedExperiment ? (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl py-20 text-center">
          <Lightbulb className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
          <p className="text-sm font-medium text-zinc-500 mb-1">
            Select a completed experiment above
          </p>
          <p className="text-xs text-zinc-600">
            Recommendations are generated after all runs have been evaluated
          </p>
        </div>
      ) : isLoading ? (
        <div className="py-20 text-center text-sm text-zinc-600">Computing recommendations…</div>
      ) : isError ? (
        <div className="bg-zinc-900 border border-red-500/20 rounded-xl py-12 text-center">
          <p className="text-sm text-red-400">Failed to load recommendations</p>
        </div>
      ) : recs.length === 0 ? (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl py-20 text-center">
          <CheckCircle2 className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
          <p className="text-sm font-medium text-zinc-500">No recommendations available yet</p>
          <p className="text-xs text-zinc-600 mt-1">
            Ensure at least one run has completed evaluation
          </p>
        </div>
      ) : (
        <>
          <p className="text-xs text-zinc-600 mb-4">
            {recs.length} recommendation{recs.length !== 1 ? "s" : ""} across {recs.length} objectives
          </p>
          <div className="grid gap-4 stagger-in">
            {recs.map((rec, i) => (
              <RecommendationCard key={rec.objective} rec={rec} rank={i} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
