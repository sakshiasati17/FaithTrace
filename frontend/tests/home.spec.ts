import { test, expect } from "@playwright/test";

test.describe("Home page", () => {
  test("displays FaithTrace title and hero section", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "FaithTrace" })).toBeVisible();
    await expect(page.getByText("Temporal + Multimodal RAG Diagnostics Platform")).toBeVisible();
  });

  test("shows Upload Documents CTA button", async ({ page }) => {
    await page.goto("/");
    const uploadBtn = page.getByRole("link", { name: /Upload Documents/i });
    await expect(uploadBtn).toBeVisible();
    await expect(uploadBtn).toHaveAttribute("href", "/corpus");
  });

  test("shows Run Experiments CTA button", async ({ page }) => {
    await page.goto("/");
    const expBtn = page.getByRole("link", { name: /Run Experiments/i });
    await expect(expBtn).toBeVisible();
    await expect(expBtn).toHaveAttribute("href", "/experiments");
  });

  test("displays Platform Capabilities section", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("Platform Capabilities")).toBeVisible();
    await expect(page.getByText("Multi-Config Benchmarking")).toBeVisible();
    await expect(page.getByText("Temporal Validity Scoring")).toBeVisible();
    await expect(page.getByText("Multimodal Grounding")).toBeVisible();
  });

  test("displays Quick Start steps", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("Quick Start")).toBeVisible();
    await expect(page.getByText("Upload corpus")).toBeVisible();
    await expect(page.getByText("Run experiment")).toBeVisible();
    await expect(page.getByText("View results")).toBeVisible();
  });

  test("navigates to corpus page from quick start", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("link", { name: /Open Corpus/i }).click();
    await expect(page).toHaveURL("/corpus");
  });
});
