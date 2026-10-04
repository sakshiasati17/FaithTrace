"use client";

import { useQuery } from "@tanstack/react-query";
import { corpusApi } from "@/lib/api";
import type { Document } from "@/types";

/**
 * Document scope for the New Experiment form. `value` null = every document
 * (default); otherwise the ids of the documents retrieval may search.
 */
export function DocumentScopePicker({
  value,
  onChange,
}: {
  value: string[] | null;
  onChange: (ids: string[] | null) => void;
}) {
  const { data: docs = [], isLoading, error } = useQuery<Document[]>({
    queryKey: ["corpus"],
    queryFn: corpusApi.list,
  });

  const scoped = value !== null;
  const toggle = (id: string) => {
    const current = value ?? [];
    onChange(current.includes(id) ? current.filter((d) => d !== id) : [...current, id]);
  };

  return (
    <div>
      <label className="block text-xs text-zinc-500 mb-1.5">Documents</label>
      <div className="flex gap-4 text-xs text-zinc-300 mb-2">
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input
            type="radio"
            name="document-scope"
            checked={!scoped}
            onChange={() => onChange(null)}
          />
          All documents
        </label>
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input
            type="radio"
            name="document-scope"
            checked={scoped}
            onChange={() => onChange(value ?? [])}
            disabled={docs.length === 0}
          />
          Selected documents
        </label>
      </div>

      {scoped && (
        <div
          role="group"
          aria-label="Documents to search"
          className="max-h-36 overflow-y-auto bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 space-y-1"
        >
          {docs.map((d) => (
            <label key={d.id} className="flex items-center gap-2 text-xs text-zinc-300 cursor-pointer">
              <input type="checkbox" checked={value.includes(d.id)} onChange={() => toggle(d.id)} />
              <span className="truncate">{d.filename}</span>
              <span className="text-zinc-600 flex-shrink-0">{d.version_label}</span>
            </label>
          ))}
        </div>
      )}

      <p className="text-[10px] text-zinc-600 mt-1">
        {isLoading
          ? "Loading documents…"
          : error
          ? "Could not load documents; the experiment will search all documents."
          : !scoped
          ? `Retrieval searches every document (${docs.length} uploaded).`
          : value.length === 0
          ? "Select at least one document."
          : `Retrieval searches ${value.length} of ${docs.length} documents.`}
      </p>
    </div>
  );
}
