import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const listDocs = vi.fn();
const listExperiments = vi.fn();
const getLeaderboard = vi.fn();
vi.mock("@/lib/api", () => ({
  corpusApi: { list: () => listDocs() },
  experimentsApi: { list: () => listExperiments() },
  evaluationApi: { getLeaderboard: () => getLeaderboard() },
}));
vi.mock("@/components/ui/SystemStatus", () => ({ SystemStatusList: () => null }));
vi.mock("next/link", () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) => <a href={href}>{children}</a>,
}));

import HomePage from "../page";

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}><HomePage /></QueryClientProvider>);
}

describe("HomePage", () => {
  it("shows an error state, not 'No experiments yet', when the API fails", async () => {
    listDocs.mockResolvedValue([]);
    listExperiments.mockRejectedValue(new Error("Network Error"));
    getLeaderboard.mockResolvedValue([]);
    renderPage();

    expect(await screen.findByRole("alert")).toHaveTextContent("Network Error");
    expect(screen.queryByText("No experiments yet")).not.toBeInTheDocument();

    listExperiments.mockResolvedValue([]);
    fireEvent.click(screen.getByRole("button", { name: /retry/i }));
    expect(await screen.findByText("No experiments yet")).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
  });

  it("shows the empty state when the API returns no data", async () => {
    listDocs.mockResolvedValue([]);
    listExperiments.mockResolvedValue([]);
    getLeaderboard.mockResolvedValue([]);
    renderPage();

    expect(await screen.findByText("No experiments yet")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
