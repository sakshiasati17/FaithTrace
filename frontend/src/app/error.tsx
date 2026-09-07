"use client";

import { useEffect } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="min-h-screen bg-[#09090b] flex flex-col items-center justify-center text-center px-6">
      <AlertTriangle className="w-10 h-10 text-red-500 mb-4" />
      <h2 className="text-lg font-semibold text-white mb-1">Something went wrong</h2>
      <p className="text-zinc-500 text-xs mb-6 max-w-xs font-mono">
        {error.message || "An unexpected error occurred"}
      </p>
      <button
        onClick={reset}
        className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-zinc-200 bg-zinc-800 border border-zinc-700 rounded-lg hover:bg-zinc-700 transition-all"
      >
        <RefreshCw className="w-3.5 h-3.5" />
        Try again
      </button>
    </div>
  );
}
