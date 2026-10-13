import { describe, it, expect, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const getExperiment = vi.fn();
const getLeaderboard = vi.fn(() => Promise.resolve([] as unknown[]));
vi.mock("@/lib/api", () => ({
  experimentsApi: { get: () => getExperiment() },
  evaluationApi: { getLeaderboard: () => getLeaderboard() },
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

  it("shows the freshness policy so runs differing only by it are distinguishable", async () => {
    getExperiment.mockResolvedValue(experiment([
      run("r1", { freshness_policy: "none" }),
      run("r2", { freshness_policy: "version_aware" }),
    ]));
    renderPage();

    expect(await screen.findByText("fresh: none")).toBeInTheDocument();
    expect(screen.getByText("fresh: version_aware")).toBeInTheDocument();
  });

  it("labels the top run with its ranking metric and config", async () => {
    getExperiment.mockResolvedValue(experiment([
      run("r1", { freshness_policy: "none" }),
      run("r2", { freshness_policy: "version_aware" }),
    ]));
    getLeaderboard.mockResolvedValue([
      lbEntry("r2", { freshness_policy: "version_aware" }, 0.9),
      lbEntry("r1", { freshness_policy: "none" }, 0.5),
    ]);
    renderPage();

    const heading = await screen.findByText(/top run by faithfulness/i);
    expect(heading).toHaveTextContent("(of 2 completed runs)");
    expect(screen.queryByText(/best run/i)).not.toBeInTheDocument();
    // Run table + top-run box both show r2's freshness policy.
    expect(screen.getAllByText("fresh: version_aware")).toHaveLength(2);
  });

  it("does not call a single run 'best'", async () => {
    getExperiment.mockResolvedValue(experiment([run("r1", {})]));
    getLeaderboard.mockResolvedValue([lbEntry("r1", {}, 0.7)]);
    renderPage();

    expect(await screen.findByText("Run Metrics")).toBeInTheDocument();
    expect(screen.queryByText(/top run|best/i)).not.toBeInTheDocument();
  });

  it("shows per-run query counts and highlights incomplete ones", async () => {
    getExperiment.mockResolvedValue(experiment([
      run("r1", {}, { n_queries: 41, n_errored: 2, n_unscored: 1, n_undiagnosed: 0 }),
    ]));
    renderPage();

    const counts = await screen.findByTestId("query-counts");
    expect(counts).toHaveTextContent("39 answered · 2 errored · 1 unscored · 0 undiagnosed");
    expect(within(counts).getByText("2 errored")).toHaveClass("text-amber-400");
    expect(within(counts).getByText("1 unscored")).toHaveClass("text-amber-400");
    expect(within(counts).getByText("0 undiagnosed")).not.toHaveClass("text-amber-400");
  });

  it("omits counts the API did not compute instead of showing 0", async () => {
    getExperiment.mockResolvedValue(experiment([
      run("r1", {}, { n_queries: 5, n_errored: 0, n_unscored: null, n_undiagnosed: null }),
      run("r2", {}),
    ]));
    renderPage();

    const counts = await screen.findAllByTestId("query-counts");
    expect(counts).toHaveLength(1);
    expect(counts[0]).toHaveTextContent("5 answered · 0 errored");
    expect(counts[0]).not.toHaveTextContent(/unscored|undiagnosed/);
  });
});

const BASE_CONFIG = {
  retrieval_strategy: "vector_only",
  chunking_strategy: "recursive",
  parsing_strategy: "text_only",
  freshness_policy: "none",
};

function run(id: string, config: Record<string, string>, extra: Record<string, unknown> = {}) {
  return {
    id,
    experiment_id: "e1",
    config: { ...BASE_CONFIG, ...config },
    status: "done",
    created_at: "2026-01-01T00:00:00Z",
    completed_at: "2026-01-01T00:00:00Z",
    ...extra,
  };
}

function experiment(runs: unknown[]) {
  return {
    id: "e1",
    name: "exp",
    description: "",
    status: "done",
    created_at: "2026-01-01T00:00:00Z",
    completed_at: "2026-01-01T00:00:00Z",
    document_ids: null,
    runs,
  };
}

function lbEntry(runId: string, config: Record<string, string>, faithfulness: number) {
  return {
    run_id: runId,
    experiment_id: "e1",
    config: { ...BASE_CONFIG, ...config },
    metrics: { id: `m-${runId}`, run_id: runId, faithfulness },
  };
}
