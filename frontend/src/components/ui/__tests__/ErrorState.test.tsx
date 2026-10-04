import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ErrorState, errorMessage } from "../ErrorState";

describe("ErrorState", () => {
  it("shows the title, the error message and a working Retry button", () => {
    const onRetry = vi.fn();
    render(<ErrorState title="Failed to load" error={new Error("Network Error")} onRetry={onRetry} />);

    expect(screen.getByRole("alert")).toHaveTextContent("Failed to load");
    expect(screen.getByText("Network Error")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /retry/i }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("prefers the API's detail message", () => {
    expect(errorMessage({ message: "Request failed with status code 500",
                          response: { data: { detail: "DB down" } } })).toBe("DB down");
    expect(errorMessage(new Error("boom"))).toBe("boom");
    expect(errorMessage(null)).toBe("Unknown error");
  });
});
