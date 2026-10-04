import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { StatusBadge } from "../StatusBadge";

describe("StatusBadge", () => {
  it("renders status text", () => {
    render(<StatusBadge status="done" />);
    expect(screen.getByText("done")).toBeInTheDocument();
  });

  it("applies done styles", () => {
    const { container } = render(<StatusBadge status="done" />);
    const badge = container.querySelector("span");
    expect(badge?.className).toContain("text-emerald-400");
  });

  it("applies failed styles", () => {
    const { container } = render(<StatusBadge status="failed" />);
    const badge = container.querySelector("span");
    expect(badge?.className).toContain("text-red-400");
  });

  it("applies running styles", () => {
    const { container } = render(<StatusBadge status="running" />);
    const badge = container.querySelector("span");
    expect(badge?.className).toContain("text-blue-400");
  });

  it("applies pending styles", () => {
    const { container } = render(<StatusBadge status="pending" />);
    const badge = container.querySelector("span");
    expect(badge?.className).toContain("text-zinc-400");
  });

  it("falls back to zinc styles for unknown status", () => {
    const { container } = render(<StatusBadge status="unknown_status" />);
    const badge = container.querySelector("span");
    expect(badge?.className).toContain("text-zinc-400");
  });

  it("renders the status dot", () => {
    const { container } = render(<StatusBadge status="done" />);
    const dot = container.querySelector(".w-1\\.5.h-1\\.5.rounded-full");
    expect(dot).toBeInTheDocument();
  });

  it("applies animate-pulse for running status dot", () => {
    const { container } = render(<StatusBadge status="running" />);
    const dot = container.querySelector(".animate-pulse");
    expect(dot).toBeInTheDocument();
  });

  it.each([
    ["evaluating", "text-amber-400"],
    ["diagnosing", "text-violet-400"],
  ])("renders %s with its own style, label and pulsing dot", (status, color) => {
    const { container } = render(<StatusBadge status={status} />);
    expect(screen.getByText(status)).toBeInTheDocument();
    const badge = container.querySelector("span");
    expect(badge?.className).toContain(color);
    expect(badge?.className).not.toContain("text-zinc-400");
    expect(badge?.getAttribute("title")).toBeTruthy();
    expect(container.querySelector(".animate-pulse")).toBeInTheDocument();
  });
});
