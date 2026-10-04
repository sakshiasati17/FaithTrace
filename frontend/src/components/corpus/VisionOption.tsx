"use client";

import { Eye } from "lucide-react";

export interface UploadFields {
  version_label: string;
  effective_from: string;
  effective_to: string;
  enable_vision: boolean;
}

/** Multipart body for POST /corpus/upload. enable_vision is sent only when on. */
export function buildUploadForm(file: File, fields: UploadFields): FormData {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("version_label", fields.version_label);
  if (fields.effective_from) fd.append("effective_from", fields.effective_from);
  if (fields.effective_to) fd.append("effective_to", fields.effective_to);
  if (fields.enable_vision) fd.append("enable_vision", "true");
  return fd;
}

/** Opt-in checkbox for GPT-4o page parsing of PDFs (costs money, so off by default). */
export function VisionOption({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex items-start gap-2 text-sm text-zinc-300 cursor-pointer select-none">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 accent-violet-500"
      />
      <span>
        <span className="inline-flex items-center gap-1.5">
          <Eye className="w-3.5 h-3.5 text-zinc-500" />
          Enable vision parsing (PDF charts, uses GPT-4o; extra cost)
        </span>
        <span className="block text-xs text-zinc-600 mt-0.5">
          Sends rendered PDF pages (up to the server&apos;s page limit) to a vision model so charts become
          searchable. Ignored for non-PDF files.
        </span>
      </span>
    </label>
  );
}

const STRATEGY_LABELS: Record<string, string> = {
  text_only: "text",
  text_table: "text + tables",
  text_table_vision: "text + tables + vision",
  spreadsheet_aware: "spreadsheet",
};

/** Number of pages whose vision call failed, from doc_metadata.vision.errors. */
export function visionErrorCount(docMetadata?: Record<string, unknown>): number {
  const vision = docMetadata?.vision as { errors?: unknown[] } | undefined;
  return Array.isArray(vision?.errors) ? vision.errors.length : 0;
}

/** Parsing strategy a document was ingested with. */
export function StrategyBadge({
  strategy,
  visionErrors = 0,
}: {
  strategy?: string | null;
  visionErrors?: number;
}) {
  if (!strategy) return <span className="text-zinc-600 text-xs">—</span>;
  const vision = strategy === "text_table_vision";
  const badge = (
    <span
      title={strategy}
      className={
        vision
          ? "text-xs bg-violet-500/10 text-violet-300 border border-violet-500/20 px-2 py-0.5 rounded font-mono"
          : "text-xs bg-zinc-800 text-zinc-400 px-2 py-0.5 rounded font-mono"
      }
    >
      {STRATEGY_LABELS[strategy] ?? strategy}
    </span>
  );
  if (!visionErrors) return badge;
  return (
    <span className="inline-flex items-center gap-1.5">
      {badge}
      <span className="text-xs text-amber-400">
        {visionErrors} vision page error{visionErrors === 1 ? "" : "s"}
      </span>
    </span>
  );
}
