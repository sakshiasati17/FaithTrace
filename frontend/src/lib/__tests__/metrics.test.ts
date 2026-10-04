import { describe, it, expect } from "vitest";
import { compareMetric, formatMetric, isScored } from "../metrics";

describe("metric display helpers", () => {
  it.each([null, undefined, NaN, Infinity])("renders %s as a dash, not 0", (v) => {
    expect(formatMetric(v as number | null | undefined)).toBe("—");
    expect(formatMetric(v as number | null | undefined, "$")).toBe("—");
    expect(formatMetric(v as number | null | undefined, "ms")).toBe("—");
    expect(isScored(v)).toBe(false);
  });

  it("keeps a real zero", () => {
    expect(isScored(0)).toBe(true);
    expect(formatMetric(0)).toBe("0.000");
    expect(formatMetric(0, "$")).toBe("$0.0000");
    expect(formatMetric(0, "ms")).toBe("0ms");
  });

  it("sorts unscored values last in both directions", () => {
    const values = [0.5, null, 0, undefined, 0.9];
    expect([...values].sort((a, b) => compareMetric(a, b, "desc"))).toEqual([0.9, 0.5, 0, null, undefined]);
    expect([...values].sort((a, b) => compareMetric(a, b, "asc"))).toEqual([0, 0.5, 0.9, null, undefined]);
  });
});
