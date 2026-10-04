"use client";

import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Upload } from "lucide-react";
import { ApiError, DEFAULT_EVAL_SET_PATH, evalSetsApi } from "@/lib/api";
import type { EvalSetRowError, EvalSetSummary } from "@/types";

const MAX_ERRORS_SHOWN = 20;

function label(set: EvalSetSummary) {
  const origin = set.source === "builtin" ? "built-in" : "uploaded";
  return `${set.name} (${set.item_count} items, ${origin})`;
}

/**
 * Eval set selector for the New Experiment form: lists built-in and uploaded
 * sets and uploads a new JSON/CSV set, showing validation errors and warnings.
 * `value` is an eval set id (built-ins are "builtin:<file>.json").
 */
export function EvalSetPicker({
  value,
  onChange,
}: {
  value: string;
  onChange: (id: string) => void;
}) {
  const qc = useQueryClient();
  const fileInput = useRef<HTMLInputElement>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [rowErrors, setRowErrors] = useState<EvalSetRowError[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);

  const { data: sets = [], isLoading, error: listError } = useQuery<EvalSetSummary[]>({
    queryKey: ["eval-sets"],
    queryFn: evalSetsApi.list,
  });

  // Pre-select the default built-in set (or the first one) once the list loads.
  useEffect(() => {
    if (value || sets.length === 0) return;
    const preferred = sets.find((s) => s.path === DEFAULT_EVAL_SET_PATH) ?? sets[0];
    onChange(preferred.id);
  }, [sets, value, onChange]);

  const uploadMut = useMutation({
    mutationFn: (file: File) => evalSetsApi.upload(file, file.name.replace(/\.(json|csv)$/i, "")),
    onMutate: () => {
      setUploadError(null);
      setRowErrors([]);
      setWarnings([]);
    },
    onSuccess: (created) => {
      setWarnings(created.warnings ?? []);
      qc.invalidateQueries({ queryKey: ["eval-sets"] });
      onChange(created.id);
    },
    onError: (err: Error) => {
      setUploadError(err.message || "Upload failed");
      setRowErrors(err instanceof ApiError ? err.errors : []);
    },
  });

  const builtin = sets.filter((s) => s.source === "builtin");
  const uploaded = sets.filter((s) => s.source === "upload");

  return (
    <div>
      <label htmlFor="eval-set-select" className="block text-xs text-zinc-500 mb-1.5">
        Eval Set
      </label>
      <div className="flex gap-2">
        <select
          id="eval-set-select"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={isLoading}
          className="flex-1 min-w-0 bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-violet-500"
        >
          {isLoading && <option value="">Loading eval sets…</option>}
          {!isLoading && sets.length === 0 && <option value="">No eval sets available</option>}
          {builtin.length > 0 && (
            <optgroup label="Built-in">
              {builtin.map((s) => (
                <option key={s.id} value={s.id}>{label(s)}</option>
              ))}
            </optgroup>
          )}
          {uploaded.length > 0 && (
            <optgroup label="Uploaded">
              {uploaded.map((s) => (
                <option key={s.id} value={s.id}>{label(s)}</option>
              ))}
            </optgroup>
          )}
        </select>
        <button
          type="button"
          onClick={() => fileInput.current?.click()}
          disabled={uploadMut.isPending}
          className="flex items-center gap-1.5 px-3 py-2 bg-zinc-800 hover:bg-zinc-700 disabled:opacity-50 text-zinc-300 text-xs rounded-lg transition-colors whitespace-nowrap"
        >
          <Upload className="w-3.5 h-3.5" />
          {uploadMut.isPending ? "Uploading…" : "Upload eval set"}
        </button>
        <input
          ref={fileInput}
          type="file"
          accept=".json,.csv"
          aria-label="Eval set file"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) uploadMut.mutate(file);
            e.target.value = "";
          }}
        />
      </div>
      <p className="text-[10px] text-zinc-600 mt-1">
        JSON list or CSV with id, question, ground_truth (source_docs separated by ;)
      </p>

      {listError && (
        <div className="mt-2 text-xs text-red-400">Could not load eval sets: {(listError as Error).message}</div>
      )}

      {uploadError && (
        <div role="alert" className="mt-2 text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
          <p>{uploadError}</p>
          {rowErrors.length > 0 && (
            <ul className="mt-1 space-y-0.5 max-h-32 overflow-y-auto">
              {rowErrors.slice(0, MAX_ERRORS_SHOWN).map((e, i) => (
                <li key={i}>
                  {e.row != null ? `Row ${e.row}` : "File"}
                  {e.id ? ` (${e.id})` : ""}: {e.message}
                </li>
              ))}
              {rowErrors.length > MAX_ERRORS_SHOWN && (
                <li>…and {rowErrors.length - MAX_ERRORS_SHOWN} more</li>
              )}
            </ul>
          )}
        </div>
      )}

      {warnings.length > 0 && (
        <div className="mt-2 text-xs text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2">
          <p>Uploaded with {warnings.length} warning(s):</p>
          <ul className="mt-1 space-y-0.5 max-h-24 overflow-y-auto">
            {warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
