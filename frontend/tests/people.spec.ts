import { startE2EServer } from "./support/server";
import { expect, test } from "./support/test";

test("people and person views sort by added time and name", async ({
  page,
}) => {
  const server = await startE2EServer();

  try {
    await page.goto(`${server.baseURL}/people`);

    await expect(page.getByRole("heading", { name: "Alpha" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Zulu" })).toBeVisible();

    await page.getByRole("link", { name: "Added" }).click();
    await expect(page).toHaveURL(/\/people\?sort=added$/);
    await expect(page.locator("article").first()).toContainText("Zulu");

    await page
      .locator("article")
      .filter({ hasText: "Zulu" })
      .locator("a.card-link")
      .click();

    await expect(page).toHaveURL(/\/people\/\d+(\?.*)?$/);
    await expect(page.locator("label.selection-card")).toHaveCount(2);
    await expect(page.locator("label.selection-card").first()).toContainText(
      "zebra.png",
    );

    await page.getByRole("link", { name: "Filename" }).click();
    await expect(page).toHaveURL(/sort=filename/);
    await expect(page.locator("label.selection-card").first()).toContainText(
      "apple.png",
    );
  } finally {
    await server.stop();
  }
});
