import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

const list = vi.fn();
const upload = vi.fn();

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    evalSetsApi: { list: () => list(), upload: (...args: unknown[]) => upload(...args) },
  };
});

import { ApiError } from "@/lib/api";
import { EvalSetPicker } from "../EvalSetPicker";

const SETS = [
  {
    id: "builtin:faithtrace_v1.json", name: "faithtrace_v1", description: "", source: "builtin",
    filename: "faithtrace_v1.json", path: "eval_sets/faithtrace_v1.json", item_count: 86,
    created_at: null, is_default: true,
  },
  {
    id: "builtin:golden_regression_set.json", name: "golden_regression_set", description: "",
    source: "builtin", filename: "golden_regression_set.json",
    path: "eval_sets/golden_regression_set.json", item_count: 5, created_at: null, is_default: false,
  },
  {
    id: "u1", name: "My questions", description: "", source: "upload", filename: "mine.csv",
    path: null, item_count: 12, created_at: "2026-01-01T00:00:00", is_default: false,
  },
];

let lastValue = "";

function Harness() {
  const [value, setValue] = useState("");
  lastValue = value;
  return <EvalSetPicker value={value} onChange={setValue} />;
}

function renderPicker() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <Harness />
    </QueryClientProvider>
  );
}

function chooseFile(name: string, content: string) {
  const input = screen.getByLabelText("Eval set file") as HTMLInputElement;
  const file = new File([content], name, { type: "application/json" });
  fireEvent.change(input, { target: { files: [file] } });
  return file;
}

describe("EvalSetPicker", () => {
  beforeEach(() => {
    list.mockReset();
    upload.mockReset();
    lastValue = "";
  });

  it("lists built-in and uploaded sets with item counts and selects the default", async () => {
    list.mockResolvedValue(SETS);
    renderPicker();

    expect(await screen.findByText("faithtrace_v1 (86 items, built-in)")).toBeInTheDocument();
    expect(screen.getByText("golden_regression_set (5 items, built-in)")).toBeInTheDocument();
    expect(screen.getByText("My questions (12 items, uploaded)")).toBeInTheDocument();
    await waitFor(() => expect(lastValue).toBe("builtin:faithtrace_v1.json"));

    fireEvent.change(screen.getByLabelText("Eval Set"), { target: { value: "u1" } });
    expect(lastValue).toBe("u1");
  });

  it("uploads a file, selects the new set and shows warnings", async () => {
    list.mockResolvedValue(SETS);
    upload.mockResolvedValue({
      ...SETS[2], id: "u2", name: "new", item_count: 3,
      warnings: ["source_docs not found in uploaded documents: missing.pdf"],
    });
    renderPicker();
    await screen.findByText("faithtrace_v1 (86 items, built-in)");

    const file = chooseFile("new.json", "[]");

    expect(await screen.findByText("source_docs not found in uploaded documents: missing.pdf"))
      .toBeInTheDocument();
    expect(upload).toHaveBeenCalledWith(file, "new");
    await waitFor(() => expect(lastValue).toBe("u2"));
  });

  it("shows per-row validation errors when the upload is rejected", async () => {
    list.mockResolvedValue(SETS);
    upload.mockRejectedValue(
      new ApiError("2 validation error(s)", [
        { row: 2, id: "q1", field: "id", message: "duplicate id 'q1' (first seen in row 1)" },
        { row: 3, id: "q3", field: "modality", message: "'modality' must be one of [...]" },
      ])
    );
    renderPicker();
    await screen.findByText("faithtrace_v1 (86 items, built-in)");

    chooseFile("bad.json", "[]");

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("2 validation error(s)");
    expect(alert).toHaveTextContent("Row 2 (q1): duplicate id 'q1' (first seen in row 1)");
    expect(alert).toHaveTextContent("Row 3 (q3): 'modality' must be one of [...]");
    await waitFor(() => expect(lastValue).toBe("builtin:faithtrace_v1.json"));
  });
});
