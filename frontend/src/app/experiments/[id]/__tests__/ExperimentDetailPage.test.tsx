import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const getExperiment = vi.fn();
vi.mock("@/lib/api", () => ({
  experimentsApi: { get: () => getExperiment() },
  evaluationApi: { getLeaderboard: () => Promise.resolve([]) },
}));
vi.mock("next/link", () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) => <a href={href}>{children}</a>,
}));

import ExperimentDetailPage from "../page";

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ExperimentDetailPage params={{ id: "e1" }} />
    </QueryClientProvider>
  );
}

describe("ExperimentDetailPage", () => {
  it("shows an error state, not 'not found', when the API fails", async () => {
    getExperiment.mockRejectedValue({ message: "Request failed", response: { data: { detail: "DB down" } } });
    renderPage();

    expect(await screen.findByRole("alert")).toHaveTextContent("DB down");
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
    expect(screen.queryByText("Experiment not found.")).not.toBeInTheDocument();
  });
});
