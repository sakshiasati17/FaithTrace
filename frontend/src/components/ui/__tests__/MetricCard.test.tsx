import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MetricCard } from "../MetricCard";

describe("MetricCard", () => {
  it("renders label", () => {
    render(<MetricCard label="Faithfulness" value={0.85} />);
    expect(screen.getByText("FAITHFULNESS")).toBeInTheDocument();
  });

  it("displays null value as dash", () => {
    render(<MetricCard label="Score" value={null} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("formats value as decimal", () => {
    render(<MetricCard label="Score" value={0.753} />);
    expect(screen.getByText("0.753")).toBeInTheDocument();
  });

  it("formats ms unit correctly", () => {
    render(<MetricCard label="Latency" value={1234.6} unit="ms" />);
    expect(screen.getByText("1235ms")).toBeInTheDocument();
  });

  it("formats dollar unit correctly", () => {
    render(<MetricCard label="Cost" value={0.00123} unit="$" />);
    expect(screen.getByText("$0.0012")).toBeInTheDocument();
  });

  it("renders optional description", () => {
    render(<MetricCard label="Score" value={0.9} description="Higher is better" />);
    expect(screen.getByText("Higher is better")).toBeInTheDocument();
  });

  it("applies emerald color for high values", () => {
    const { container } = render(<MetricCard label="Score" value={0.9} />);
    const valueEl = container.querySelector(".text-emerald-400");
    expect(valueEl).toBeInTheDocument();
  });

  it("applies red color for low values", () => {
    const { container } = render(<MetricCard label="Score" value={0.2} />);
    const valueEl = container.querySelector(".text-red-400");
    expect(valueEl).toBeInTheDocument();
  });

  it("applies amber color for mid-range values", () => {
    const { container } = render(<MetricCard label="Score" value={0.55} />);
    const valueEl = container.querySelector(".text-amber-400");
    expect(valueEl).toBeInTheDocument();
  });

  it("inverts color logic when lowerIsBetter", () => {
    // low value → emerald when lowerIsBetter
    const { container } = render(<MetricCard label="Latency" value={0.1} lowerIsBetter />);
    const valueEl = container.querySelector(".text-emerald-400");
    expect(valueEl).toBeInTheDocument();
  });
});
