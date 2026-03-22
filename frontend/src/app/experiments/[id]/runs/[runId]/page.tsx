"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { experimentsApi, diagnosticsApi } from "@/lib/api";
import { ArrowLeft, ChevronRight, ChevronDown, FileText, Table2, Sheet, AlertCircle } from "lucide-react";
import { clsx } from "clsx";
import type { QueryDiagnosis } from "@/types";

const FAILURE_COLORS: Record<string, string> = {
  NO_FAILURE: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
  STALE_ANSWER: "bg-amber-500/10 text-amber-400 border-amber-500/20",
  TABLE_RETRIEVAL_MISS: "bg-blue-500/10 text-blue-400 border-blue-500/20",
  LOW_RECALL_RETRIEVAL: "bg-orange-500/10 text-orange-400 border-orange-500/20",
  UNSUPPORTED_SYNTHESIS: "bg-red-500/10 text-red-400 border-red-500/20",
  IRRELEVANT_CONTEXT_POLLUTION: "bg-purple-500/10 text-purple-400 border-purple-500/20",
  CHUNKING_BOUNDARY_ERROR: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
  WRONG_VERSION: "bg-pink-500/10 text-pink-400 border-pink-500/20",
};

const CHUNK_TYPE_STYLES: Record<string, { color: string; icon: React.ElementType; label: string }> = {
  text: { color: "border-zinc-600", icon: FileText, label: "Text" },
  table: { color: "border-blue-500/50", icon: Table2, label: "Table" },
  spreadsheet_cell: { color: "border-emerald-500/50", icon: Sheet, label: "Spreadsheet" },
  image: { color: "border-violet-500/50", icon: FileText, label: "Image" },
};

function ChunkBadge({ chunkType }: { chunkType: string }) {
  const style = CHUNK_TYPE_STYLES[chunkType] ?? CHUNK_TYPE_STYLES.text;
  const Icon = style.icon;
  return (
    <span className={clsx(
      "inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded border font-medium",
      chunkType === "table" ? "bg-blue-500/10 text-blue-400 border-blue-500/30" :
      chunkType === "spreadsheet_cell" ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30" :
      "bg-zinc-800 text-zinc-400 border-zinc-700"
    )}>
      <Icon className="w-2.5 h-2.5" />
      {style.label}
    </span>
  );
}

export default function RunTracePage({ params }: { params: { id: string; runId: string } }) {
  const { id, runId } = params;
  const [expandedRow, setExpandedRow] = useState<string | null>(null);

  const { data: trace = [] as any[], isLoading: traceLoading } = useQuery<any[]>({
    queryKey: ["run-trace", id, runId],
    queryFn: () => experimentsApi.getRunTrace(id, runId),
  });

  const { data: diagnostics = [] as QueryDiagnosis[] } = useQuery<QueryDiagnosis[]>({
    queryKey: ["run-diagnostics", runId],
    queryFn: () => diagnosticsApi.getRunDiagnostics(runId),
  });

  const diagByQueryId = Object.fromEntries(
    diagnostics.map((d: any) => [d.query_id, d])
  );

  return (
    <div className="max-w-6xl mx-auto px-8 py-10">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-xs text-zinc-600 mb-6">
        <Link href="/experiments" className="hover:text-zinc-400 flex items-center gap-1">
          <ArrowLeft className="w-3 h-3" /> Experiments
        </Link>
        <ChevronRight className="w-3 h-3" />
        <Link href={`/experiments/${id}`} className="hover:text-zinc-400">
          {id.slice(0, 8)}…
        </Link>
        <ChevronRight className="w-3 h-3" />
        <span className="text-zinc-400">Run Trace</span>
      </div>

      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white mb-1">Query Trace</h1>
        <p className="text-sm text-zinc-500">
          Run <code className="font-mono text-zinc-400 text-xs">{runId.slice(0, 16)}…</code>
          {" · "}{trace.length} queries
        </p>
      </div>

      {traceLoading ? (
        <div className="py-20 text-center text-sm text-zinc-600">Loading trace…</div>
      ) : trace.length === 0 ? (
        <div className="py-20 text-center text-sm text-zinc-600">No query results yet</div>
      ) : (
        <div className="space-y-2">
          {trace.map((qr: any) => {
            const diag = diagByQueryId[qr.query_id];
            const failureStyle = FAILURE_COLORS[diag?.failure_category ?? "NO_FAILURE"] ?? FAILURE_COLORS.NO_FAILURE;
            const isExpanded = expandedRow === qr.query_id;

            return (
              <div key={qr.query_id} className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
                {/* Row header */}
                <button
                  className="w-full text-left px-5 py-4 flex items-start gap-4 hover:bg-zinc-800/30 transition-colors"
                  onClick={() => setExpandedRow(isExpanded ? null : qr.query_id)}
                >
                  <span className="text-xs text-zinc-600 font-mono w-12 flex-shrink-0 pt-0.5">{qr.query_id}</span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-zinc-200 mb-1 line-clamp-2">{qr.question}</p>
                    <p className="text-xs text-zinc-500 line-clamp-1">{qr.generated_answer}</p>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    {diag?.failure_category && (
                      <span className={clsx(
                        "text-[10px] font-medium px-2 py-0.5 rounded border",
                        failureStyle
                      )}>
                        {diag.failure_category}
                      </span>
                    )}
                    <span className="text-xs text-zinc-600">{Math.round(qr.latency_ms)}ms</span>
                    <span className="text-xs text-zinc-600">${qr.cost_usd.toFixed(4)}</span>
                    {isExpanded
                      ? <ChevronDown className="w-3.5 h-3.5 text-zinc-500" />
                      : <ChevronRight className="w-3.5 h-3.5 text-zinc-500" />
                    }
                  </div>
                </button>

                {/* Expanded view */}
                {isExpanded && (
                  <div className="border-t border-zinc-800 px-5 py-4 space-y-4">
                    {/* Q&A */}
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">Question</p>
                        <p className="text-sm text-zinc-300">{qr.question}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">Generated Answer</p>
                        <p className="text-sm text-zinc-300">{qr.generated_answer}</p>
                      </div>
                    </div>

                    {/* Diagnosis */}
                    {diag && (
                      <div className="bg-zinc-800/50 rounded-lg p-3">
                        <p className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2 flex items-center gap-1">
                          <AlertCircle className="w-3 h-3" /> Diagnosis
                        </p>
                        <div className="flex flex-wrap gap-3 text-xs">
                          <span>
                            <span className="text-zinc-500">Primary:</span>{" "}
                            <span className={clsx("font-medium px-1.5 py-0.5 rounded border text-[10px]", failureStyle)}>
                              {diag.failure_category}
                            </span>
                          </span>
                          <span className="text-zinc-500">Confidence: <span className="text-zinc-300">{(diag.confidence * 100).toFixed(0)}%</span></span>
                          {Object.entries(diag.evidence || {}).map(([k, v]) =>
                            typeof v === "number" ? (
                              <span key={k} className="text-zinc-500">
                                {k}: <span className="text-zinc-300">{(v as number).toFixed(3)}</span>
                              </span>
                            ) : null
                          )}
                        </div>
                      </div>
                    )}

                    {/* Retrieved chunks */}
                    <div>
                      <p className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">
                        Retrieved Chunks ({qr.retrieved_chunks?.length ?? 0})
                      </p>
                      <div className="space-y-2">
                        {(qr.retrieved_chunks ?? []).map((chunk: any, ci: number) => (
                          <div
                            key={ci}
                            className={clsx(
                              "border-l-2 pl-3 py-1.5 rounded-r",
                              chunk.chunk_type === "table"
                                ? "border-blue-500/50 bg-blue-500/5"
                                : chunk.chunk_type === "spreadsheet_cell"
                                ? "border-emerald-500/50 bg-emerald-500/5"
                                : "border-zinc-700 bg-zinc-800/30"
                            )}
                          >
                            <div className="flex items-center gap-2 mb-1">
                              <ChunkBadge chunkType={chunk.chunk_type ?? "text"} />
                              {chunk.filename && (
                                <span className="text-[10px] text-zinc-600 font-mono">{chunk.filename}</span>
                              )}
                              {chunk.doc_version && (
                                <span className="text-[10px] text-zinc-600">v: {chunk.doc_version}</span>
                              )}
                              {chunk.page && (
                                <span className="text-[10px] text-zinc-600">p.{chunk.page}</span>
                              )}
                            </div>
                            <p className="text-xs text-zinc-400 line-clamp-3">{chunk.content}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
