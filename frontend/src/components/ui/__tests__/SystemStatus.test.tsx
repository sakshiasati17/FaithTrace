import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";

const getStatus = vi.fn();
vi.mock("@/lib/api", () => ({ systemApi: { getStatus: () => getStatus() } }));

import { SystemStatusIndicator, SystemStatusList } from "../SystemStatus";

function renderWithClient(ui: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

const HEALTHY = {
  ok: true,
  version: "0.1.0",
  env: "test",
  checked_at: "2026-01-01T00:00:00Z",
  components: {
    database: { ok: true },
    qdrant: { ok: true },
    redis: { ok: true },
    workers: { ok: true },
    ml_classifier: { ok: true, trained: false },
  },
};

describe("SystemStatus", () => {
  it("shows offline (never green) when the API is unreachable", async () => {
    getStatus.mockImplementation(() => Promise.reject(new Error("Network Error")));
    const { container } = renderWithClient(
      <>
        <SystemStatusIndicator />
        <SystemStatusList />
      </>
    );

    expect(await screen.findByText("API offline")).toBeInTheDocument();
    expect(screen.getByText("Offline")).toBeInTheDocument();
    expect(screen.getAllByText("Unknown")).toHaveLength(5);
    expect(screen.getByText("version unknown")).toBeInTheDocument();
    expect(screen.queryByText("All systems online")).not.toBeInTheDocument();
    expect(container.querySelector(".bg-emerald-500")).toBeNull();
    expect(container.querySelector(".text-emerald-400")).toBeNull();
  });

  it("shows online and the backend version when all components are ok", async () => {
    getStatus.mockResolvedValue(HEALTHY);
    renderWithClient(<SystemStatusIndicator />);

    expect(await screen.findByText("All systems online")).toBeInTheDocument();
    expect(screen.getByText("v0.1.0")).toBeInTheDocument();
  });

  it("shows degraded and per-component state when some components fail", async () => {
    getStatus.mockResolvedValue({
      ...HEALTHY,
      ok: false,
      components: { ...HEALTHY.components, qdrant: { ok: false, detail: "refused" }, workers: { ok: false } },
    });
    renderWithClient(
      <>
        <SystemStatusIndicator />
        <SystemStatusList />
      </>
    );

    expect(await screen.findByText("Degraded: 2 components down")).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByText("Down")).toHaveLength(2));
    expect(screen.getByText("Heuristic")).toBeInTheDocument();
    expect(screen.getByText("Online")).toBeInTheDocument();
  });
});
