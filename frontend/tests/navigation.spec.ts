import { test, expect } from "@playwright/test";

test.describe("Navigation", () => {
  test("corpus page is reachable", async ({ page }) => {
    await page.goto("/corpus");
    await expect(page).toHaveURL("/corpus");
    // Should not be a 404
    await expect(page.locator("body")).not.toContainText("404");
  });

  test("experiments page is reachable", async ({ page }) => {
    await page.goto("/experiments");
    await expect(page).toHaveURL("/experiments");
    await expect(page.locator("body")).not.toContainText("404");
  });

  test("leaderboard page is reachable", async ({ page }) => {
    await page.goto("/leaderboard");
    await expect(page).toHaveURL("/leaderboard");
    await expect(page.locator("body")).not.toContainText("404");
  });

  test("diagnostics page is reachable", async ({ page }) => {
    await page.goto("/diagnostics");
    await expect(page).toHaveURL("/diagnostics");
    await expect(page.locator("body")).not.toContainText("404");
  });
});
