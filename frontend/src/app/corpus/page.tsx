"use client";

import { useState, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { corpusApi } from "@/lib/api";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { Upload, Trash2, FileText, File, RefreshCw, X, Database } from "lucide-react";
import { clsx } from "clsx";

const FILE_ICONS: Record<string, React.ElementType> = {
  pdf: FileText,
  docx: File,
  xlsx: File,
  csv: File,
  html: File,
};

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export default function CorpusPage() {
  const qc = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [form, setForm] = useState({ version_label: "v1", effective_from: "", effective_to: "" });

  const { data: docs = [], isLoading } = useQuery({
    queryKey: ["corpus"],
    queryFn: corpusApi.list,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return false;
      return (data as typeof docs).some((d) => d.parse_status === "running" || d.index_status === "running")
        ? 3000
        : false;
    },
  });

  const deleteMut = useMutation({
    mutationFn: corpusApi.delete,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["corpus"] }),
  });

  async function handleUpload(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const file = fileInputRef.current?.files?.[0];
    if (!file) return;

    const fd = new FormData();
    fd.append("file", file);
    fd.append("version_label", form.version_label);
    if (form.effective_from) fd.append("effective_from", form.effective_from);
    if (form.effective_to) fd.append("effective_to", form.effective_to);

    setUploading(true);
    setUploadError(null);
    try {
      await corpusApi.upload(fd);
      qc.invalidateQueries({ queryKey: ["corpus"] });
      if (fileInputRef.current) fileInputRef.current.value = "";
      setForm({ version_label: "v1", effective_from: "", effective_to: "" });
    } catch (err: any) {
      setUploadError(err?.response?.data?.detail ?? "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-8 py-10">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white mb-1">Document Corpus</h1>
        <p className="text-sm text-zinc-500">Upload and manage enterprise documents for RAG evaluation</p>
      </div>

      {/* Upload form */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 mb-8">
        <h2 className="text-sm font-semibold text-zinc-300 mb-4">Upload Document</h2>
        <form onSubmit={handleUpload} className="space-y-4">
          {/* Drop zone */}
          <div
            className="border-2 border-dashed border-zinc-700 hover:border-violet-600 rounded-lg p-8 text-center cursor-pointer transition-colors"
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload className="w-8 h-8 text-zinc-600 mx-auto mb-2" />
            <p className="text-sm text-zinc-400">
              Drop a file or <span className="text-violet-400 font-medium">browse</span>
            </p>
            <p className="text-xs text-zinc-600 mt-1">PDF, DOCX, XLSX, CSV, HTML</p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.doc,.xlsx,.csv,.html,.htm"
              className="hidden"
              required
            />
          </div>

          {/* Metadata */}
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-xs text-zinc-500 mb-1.5">Version Label</label>
              <input
                type="text"
                value={form.version_label}
                onChange={(e) => setForm((f) => ({ ...f, version_label: e.target.value }))}
                placeholder="e.g. v1, v2.1"
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-violet-500"
              />
            </div>
            <div>
              <label className="block text-xs text-zinc-500 mb-1.5">Effective From</label>
              <input
                type="date"
                value={form.effective_from}
                onChange={(e) => setForm((f) => ({ ...f, effective_from: e.target.value }))}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-violet-500"
              />
            </div>
            <div>
              <label className="block text-xs text-zinc-500 mb-1.5">Effective To</label>
              <input
                type="date"
                value={form.effective_to}
                onChange={(e) => setForm((f) => ({ ...f, effective_to: e.target.value }))}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-violet-500"
              />
            </div>
          </div>

          {uploadError && (
            <div className="flex items-center gap-2 text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
              <X className="w-3.5 h-3.5 flex-shrink-0" />
              {uploadError}
            </div>
          )}

          <button
            type="submit"
            disabled={uploading}
            className="flex items-center gap-2 px-5 py-2 bg-violet-600 hover:bg-violet-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
          >
            {uploading ? (
              <><RefreshCw className="w-4 h-4 animate-spin" /> Uploading…</>
            ) : (
              <><Upload className="w-4 h-4" /> Upload Document</>
            )}
          </button>
        </form>
      </div>

      {/* Documents table */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
        <div className="px-6 py-4 border-b border-zinc-800 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-zinc-300">
            Documents
            <span className="ml-2 text-xs text-zinc-600 font-normal">({docs.length})</span>
          </h2>
        </div>

        {isLoading ? (
          <div className="py-16 text-center text-sm text-zinc-600">Loading…</div>
        ) : docs.length === 0 ? (
          <div className="py-16 text-center">
            <Database className="w-8 h-8 text-zinc-700 mx-auto mb-2" />
            <p className="text-sm text-zinc-600">No documents yet — upload one above</p>
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-800">
                {["Filename", "Type", "Version", "Effective From", "Parse", "Index", "Uploaded", ""].map((h) => (
                  <th key={h} className="px-4 py-3 text-left text-xs font-medium text-zinc-500 uppercase tracking-wider">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {docs.map((doc, i) => {
                const Icon = FILE_ICONS[doc.file_type] ?? File;
                return (
                  <tr
                    key={doc.id}
                    className={clsx(
                      "border-b border-zinc-800/50 hover:bg-zinc-800/30 transition-colors",
                      i === docs.length - 1 && "border-b-0"
                    )}
                  >
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <Icon className="w-3.5 h-3.5 text-zinc-500 flex-shrink-0" />
                        <span className="text-zinc-200 font-medium truncate max-w-48">{doc.filename}</span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <span className="text-xs bg-zinc-800 text-zinc-400 px-2 py-0.5 rounded font-mono">
                        {doc.file_type}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-zinc-400 text-xs font-mono">{doc.version_label}</td>
                    <td className="px-4 py-3 text-zinc-500 text-xs">
                      {doc.effective_from ? formatDate(doc.effective_from) : "—"}
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge status={doc.parse_status} />
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge status={doc.index_status} />
                    </td>
                    <td className="px-4 py-3 text-zinc-500 text-xs">{formatDate(doc.created_at)}</td>
                    <td className="px-4 py-3">
                      <button
                        onClick={() => deleteMut.mutate(doc.id)}
                        disabled={deleteMut.isPending}
                        className="text-zinc-600 hover:text-red-400 transition-colors"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

