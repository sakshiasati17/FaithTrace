import { AlertTriangle, RotateCw } from "lucide-react";
import { clsx } from "clsx";

/** Best human-readable message from a failed request. */
export function errorMessage(error: unknown): string {
  const e = error as { response?: { data?: { detail?: unknown } }; message?: unknown } | null;
  const detail = e?.response?.data?.detail;
  if (typeof detail === "string" && detail) return detail;
  if (typeof e?.message === "string" && e.message) return e.message;
  return "Unknown error";
}

interface ErrorStateProps {
  title: string;
  error: unknown;
  onRetry: () => void;
  className?: string;
}

/**
 * Shown when an API request fails. Distinct from an empty state, so a failed
 * request is never mistaken for "no data".
 */
export function ErrorState({ title, error, onRetry, className }: ErrorStateProps) {
  return (
    <div
      role="alert"
      className={clsx(
        "bg-red-500/5 border border-red-500/30 rounded-xl py-10 px-6 text-center",
        className
      )}
    >
      <AlertTriangle className="w-8 h-8 text-red-400 mx-auto mb-3" />
      <p className="text-sm font-medium text-red-300 mb-1">{title}</p>
      <p className="text-xs text-red-400/80 font-mono mb-4 break-words">{errorMessage(error)}</p>
      <button
        type="button"
        onClick={onRetry}
        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-zinc-200 bg-zinc-800 border border-zinc-700 rounded-lg hover:bg-zinc-700 transition-colors"
      >
        <RotateCw className="w-3.5 h-3.5" />
        Retry
      </button>
    </div>
  );
}
