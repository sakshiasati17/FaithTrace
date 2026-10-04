import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

const list = vi.fn();

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return { ...actual, corpusApi: { list: () => list() } };
});

import { DocumentScopePicker } from "../DocumentScopePicker";

const DOCS = [
  { id: "d1", filename: "policy_v1.pdf", file_type: "pdf", version_label: "v1", effective_from: null,
    effective_to: null, parse_status: "done", index_status: "done", created_at: "2026-01-01T00:00:00" },
  { id: "d2", filename: "rates.xlsx", file_type: "xlsx", version_label: "v2", effective_from: null,
    effective_to: null, parse_status: "done", index_status: "done", created_at: "2026-01-01T00:00:00" },
];

let lastValue: string[] | null = null;

function Harness() {
  const [value, setValue] = useState<string[] | null>(null);
  lastValue = value;
  return <DocumentScopePicker value={value} onChange={setValue} />;
}

function renderPicker() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <Harness />
    </QueryClientProvider>
  );
}

describe("DocumentScopePicker", () => {
  beforeEach(() => {
    list.mockReset();
    list.mockResolvedValue(DOCS);
    lastValue = null;
  });

  it("defaults to all documents (null)", async () => {
    renderPicker();
    await screen.findByText(/every document \(2 uploaded\)/);
    expect(screen.getByLabelText("All documents")).toBeChecked();
    expect(screen.queryByRole("group", { name: "Documents to search" })).toBeNull();
    expect(lastValue).toBeNull();
  });

  it("selects a subset of documents", async () => {
    renderPicker();
    await screen.findByText(/every document/);
    fireEvent.click(screen.getByLabelText("Selected documents"));
    expect(lastValue).toEqual([]);
    expect(screen.getByText("Select at least one document.")).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText(/rates\.xlsx/));
    expect(lastValue).toEqual(["d2"]);
    fireEvent.click(screen.getByLabelText(/policy_v1\.pdf/));
    expect(lastValue).toEqual(["d2", "d1"]);
    fireEvent.click(screen.getByLabelText(/rates\.xlsx/));
    expect(lastValue).toEqual(["d1"]);
    expect(screen.getByText("Retrieval searches 1 of 2 documents.")).toBeInTheDocument();
  });

  it("switching back to all documents resets to null", async () => {
    renderPicker();
    await screen.findByText(/every document/);
    fireEvent.click(screen.getByLabelText("Selected documents"));
    fireEvent.click(screen.getByLabelText(/rates\.xlsx/));
    fireEvent.click(screen.getByLabelText("All documents"));
    expect(lastValue).toBeNull();
  });

  it("cannot scope when there are no documents", async () => {
    list.mockResolvedValue([]);
    renderPicker();
    await waitFor(() => expect(screen.getByText(/every document \(0 uploaded\)/)).toBeInTheDocument());
    expect(screen.getByLabelText("Selected documents")).toBeDisabled();
  });
});
