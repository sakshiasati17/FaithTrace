import { describe, it, expect, vi } from "vitest";
import { render, screen, within, fireEvent } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const getLeaderboard = vi.fn();
vi.mock("@/lib/api", () => ({ evaluationApi: { getLeaderboard: () => getLeaderboard() } }));
vi.mock("next/link", () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) => <a href={href}>{children}</a>,
}));

import LeaderboardPage from "../page";

function metrics(overrides: Record<string, number | null>) {
  return {
    faithfulness: 0.5, context_recall: 0.5, context_precision: 0.5, answer_correctness: 0.5,
    answer_relevance: 0.5, freshness_validity: 0.5, multimodal_grounding_rate: 0.5,
    latency_p50_ms: 1000, avg_cost_usd: 0.01, ...overrides,
  };
}

const ENTRIES = [
  { run_id: "unscored", experiment_id: "e1", config: { retrieval_strategy: "unscored_cfg" },
    metrics: metrics({ faithfulness: null, freshness_validity: null, avg_cost_usd: null }) },
  { run_id: "zero", experiment_id: "e1", config: { retrieval_strategy: "zero_cfg" },
    metrics: metrics({ faithfulness: 0, avg_cost_usd: 0 }) },
  { run_id: "good", experiment_id: "e1", config: { retrieval_strategy: "good_cfg" },
    metrics: metrics({ faithfulness: 0.9 }) },
];

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <LeaderboardPage />
    </QueryClientProvider>
  );
}

async function rowOrder() {
  // Changing the sort column refetches (sortBy is in the query key).
  await screen.findByText("unscored_cfg");
  return screen.getAllByRole("row").slice(1).map((row) =>
    ["unscored_cfg", "zero_cfg", "good_cfg"].find((c) => within(row).queryByText(c))
  );
}

describe("LeaderboardPage null metrics", () => {
  it("renders null metrics as a dash and real zeros as 0", async () => {
    getLeaderboard.mockResolvedValue(ENTRIES);
    renderPage();

    const unscoredRow = (await screen.findByText("unscored_cfg")).closest("tr")!;
    // faithfulness, freshness and cost are null for this run
    expect(within(unscoredRow).getAllByText("—")).toHaveLength(3);
    expect(within(unscoredRow).queryByText("0.000")).not.toBeInTheDocument();
    expect(within(unscoredRow).queryByText("$0.0000")).not.toBeInTheDocument();

    const zeroRow = screen.getByText("zero_cfg").closest("tr")!;
    expect(within(zeroRow).getByText("0.000")).toBeInTheDocument();
    expect(within(zeroRow).getByText("$0.0000")).toBeInTheDocument();
    expect(within(zeroRow).queryByText("—")).not.toBeInTheDocument();
  });

  it("sorts null metrics last, in both directions", async () => {
    getLeaderboard.mockResolvedValue(ENTRIES);
    renderPage();
    await screen.findByText("unscored_cfg");

    // Default: faithfulness descending.
    expect(await rowOrder()).toEqual(["good_cfg", "zero_cfg", "unscored_cfg"]);
    fireEvent.click(screen.getByText("Faithfulness"));
    expect(await rowOrder()).toEqual(["zero_cfg", "good_cfg", "unscored_cfg"]);

    // Cost ascending: the null-cost run is not "cheapest".
    fireEvent.click(screen.getByText("Cost/q"));
    expect(await rowOrder()).toEqual(["zero_cfg", "good_cfg", "unscored_cfg"]);
  });
});
