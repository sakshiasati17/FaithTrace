import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const listExperiments = vi.fn();
vi.mock("@/lib/api", () => ({
  experimentsApi: { list: () => listExperiments() },
  diagnosticsApi: { getFailureSummary: () => Promise.resolve({}) },
}));

import DiagnosticsPage from "../page";

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}><DiagnosticsPage /></QueryClientProvider>);
}

describe("DiagnosticsPage", () => {
  it("shows an error state when experiments fail to load", async () => {
    listExperiments.mockRejectedValue(new Error("Network Error"));
    renderPage();

    expect(await screen.findByRole("alert")).toHaveTextContent("Failed to load experiments");
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });

  it("shows no error state when experiments load", async () => {
    listExperiments.mockResolvedValue([]);
    renderPage();

    expect(await screen.findByText("Select a completed experiment above")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
