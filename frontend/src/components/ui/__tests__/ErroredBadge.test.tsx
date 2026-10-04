import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ErroredBadge, QueryErrorMessage, isErroredQuery } from "../ErroredBadge";

describe("ErroredBadge", () => {
  it("renders the Errored label with error styles", () => {
    const { container } = render(<ErroredBadge />);
    expect(screen.getByText("Errored")).toBeInTheDocument();
    expect(container.querySelector("span")?.className).toContain("text-red-400");
  });
});

describe("QueryErrorMessage", () => {
  it("shows the error message", () => {
    render(<QueryErrorMessage message="TimeoutError: retriever timed out" />);
    expect(screen.getByText("TimeoutError: retriever timed out")).toBeInTheDocument();
  });

  it("falls back when no message is recorded", () => {
    render(<QueryErrorMessage message={null} />);
    expect(screen.getByText("Unknown error")).toBeInTheDocument();
  });
});

describe("isErroredQuery", () => {
  it("is true only for status 'error'", () => {
    expect(isErroredQuery({ status: "error" })).toBe(true);
    expect(isErroredQuery({ status: "ok" })).toBe(false);
    expect(isErroredQuery({ status: null })).toBe(false);
    expect(isErroredQuery({})).toBe(false);
  });
});
