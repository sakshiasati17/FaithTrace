import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const list = vi.fn();
const upload = vi.fn();

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    corpusApi: {
      list: () => list(),
      upload: (fd: FormData) => upload(fd),
      delete: vi.fn(),
    },
  };
});

import CorpusPage from "@/app/corpus/page";
import { StrategyBadge, buildUploadForm, visionErrorCount } from "../VisionOption";

const LABEL = /Enable vision parsing \(PDF charts, uses GPT-4o; extra cost\)/;

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <CorpusPage />
    </QueryClientProvider>
  );
}

async function submitWithFile(container: HTMLElement) {
  const input = container.querySelector('input[type="file"]') as HTMLInputElement;
  const file = new File(["%PDF"], "charts.pdf", { type: "application/pdf" });
  fireEvent.change(input, { target: { files: [file] } });
  fireEvent.submit(container.querySelector("form")!);
  await waitFor(() => expect(upload).toHaveBeenCalledTimes(1));
  return upload.mock.calls[0][0] as FormData;
}

describe("corpus upload vision option", () => {
  beforeEach(() => {
    list.mockReset().mockResolvedValue([]);
    upload.mockReset().mockResolvedValue({});
  });

  it("is off by default and not sent", async () => {
    const { container } = renderPage();
    const box = screen.getByLabelText(LABEL) as HTMLInputElement;
    expect(box.checked).toBe(false);
    const fd = await submitWithFile(container);
    expect(fd.get("enable_vision")).toBeNull();
    expect(fd.get("version_label")).toBe("v1");
  });

  it("sends enable_vision=true when checked and resets after upload", async () => {
    const { container } = renderPage();
    const box = screen.getByLabelText(LABEL) as HTMLInputElement;
    fireEvent.click(box);
    expect(box.checked).toBe(true);
    const fd = await submitWithFile(container);
    expect(fd.get("enable_vision")).toBe("true");
    expect((fd.get("file") as File).name).toBe("charts.pdf");
    await waitFor(() => expect(box.checked).toBe(false));
  });

  it("shows the parsing strategy per document", async () => {
    list.mockResolvedValue([
      {
        id: "d1", filename: "charts.pdf", file_type: "pdf", version_label: "v1",
        effective_from: null, effective_to: null, parse_status: "done", index_status: "done",
        created_at: "2026-01-01T00:00:00", parsing_strategy: "text_table_vision",
        doc_metadata: { strategy: "text_table_vision", vision: { errors: [{ page: 6, error: "x" }] } },
      },
      {
        id: "d2", filename: "data.csv", file_type: "csv", version_label: "v1",
        effective_from: null, effective_to: null, parse_status: "done", index_status: "done",
        created_at: "2026-01-01T00:00:00", parsing_strategy: "spreadsheet_aware",
      },
    ]);
    renderPage();
    expect(await screen.findByText("text + tables + vision")).toBeInTheDocument();
    expect(screen.getByText("spreadsheet")).toBeInTheDocument();
    expect(screen.getByText("1 vision page error")).toBeInTheDocument();
    expect(screen.getByText("Strategy")).toBeInTheDocument();
  });
});

describe("helpers", () => {
  it("buildUploadForm omits empty dates and vision when off", () => {
    const file = new File(["x"], "a.pdf");
    const fd = buildUploadForm(file, { version_label: "v2", effective_from: "", effective_to: "", enable_vision: false });
    expect(Array.from(fd.keys())).toEqual(["file", "version_label"]);
  });

  it("StrategyBadge shows a dash without a strategy", () => {
    render(<StrategyBadge strategy={null} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("visionErrorCount tolerates missing metadata", () => {
    expect(visionErrorCount(undefined)).toBe(0);
    expect(visionErrorCount({ vision: { errors: [] } })).toBe(0);
    expect(visionErrorCount({ vision: { errors: [{}, {}] } })).toBe(2);
  });
});
