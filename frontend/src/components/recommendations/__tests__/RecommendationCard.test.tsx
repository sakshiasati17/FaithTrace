import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { RecommendationCard } from "../RecommendationCard";
import type { Recommendation } from "@/types";

const CONFIG = {
  retrieval_strategy: "hybrid", chunking_strategy: "recursive",
  parsing_strategy: "text_table", freshness_policy: "none",
} as unknown as Recommendation["best_config"];

describe("RecommendationCard", () => {
  it("shows the score and config for a normal recommendation", () => {
    render(
      <RecommendationCard
        rank={0}
        rec={{ objective: "best_faithfulness", best_config: CONFIG, run_id: "r1",
               score: 0.87, rationale: "why", status: "ok" }}
      />
    );
    expect(screen.getByText("87")).toBeInTheDocument();
    expect(screen.getByText("hybrid")).toBeInTheDocument();
    expect(screen.queryByTestId("rec-unavailable")).not.toBeInTheDocument();
  });

  it("renders a zero score as 0, not as missing", () => {
    render(
      <RecommendationCard
        rank={1}
        rec={{ objective: "lowest_cost", best_config: CONFIG, run_id: "r1", score: 0, rationale: "free" }}
      />
    );
    expect(screen.getByText("0")).toBeInTheDocument();
  });

  it("renders a null score as a dash with the no-eligible-runs reason", () => {
    render(
      <RecommendationCard
        rank={0}
        rec={{ objective: "best_for_drift", best_config: {} as Recommendation["best_config"],
               run_id: "", score: null, status: "no_eligible_runs",
               rationale: "none of 2 runs has freshness_validity scored" }}
      />
    );
    expect(screen.getByText("—")).toBeInTheDocument();
    expect(screen.queryByText("0")).not.toBeInTheDocument();
    expect(screen.getByTestId("rec-unavailable")).toHaveTextContent(
      "No eligible runs: none of 2 runs has freshness_validity scored"
    );
    expect(screen.queryByText("Show rationale")).not.toBeInTheDocument();
  });

  it("labels an objective that errored", () => {
    render(
      <RecommendationCard
        rank={0}
        rec={{ objective: "best_overall", best_config: {} as Recommendation["best_config"],
               run_id: "", score: null, status: "error", rationale: "boom" }}
      />
    );
    expect(screen.getByTestId("rec-unavailable")).toHaveTextContent("Could not be scored: boom");
  });
});
