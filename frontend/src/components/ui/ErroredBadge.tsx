import { XCircle } from "lucide-react";
import type { QueryResult } from "@/types";

export function isErroredQuery(qr: Pick<QueryResult, "status">): boolean {
  return qr.status === "error";
}

export function ErroredBadge() {
  return (
    <span className="inline-flex items-center gap-1 text-[10px] font-medium px-2 py-0.5 rounded border bg-red-500/10 text-red-400 border-red-500/20">
      <XCircle className="w-2.5 h-2.5" />
      Errored
    </span>
  );
}

export function QueryErrorMessage({ message }: { message?: string | null }) {
  return (
    <div className="text-xs text-red-300 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
      <p className="text-[10px] text-red-400/80 uppercase tracking-wider mb-1">Query errored</p>
      <p className="font-mono break-words">{message || "Unknown error"}</p>
    </div>
  );
}
