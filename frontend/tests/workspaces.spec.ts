import { startE2EServer } from "./support/server";
import { expect, test } from "./support/test";

test("workspace picker can switch and create workspaces", async ({ page }) => {
  const server = await startE2EServer();

  try {
    const annotatedImage = page.waitForResponse(
      (response) =>
        response.url().includes("/media/annotated/") &&
        response.status() === 200,
    );
    await page.goto(server.baseURL);
    await annotatedImage;
    await expect(page.locator(".workspace-chip-value")).toHaveText("default");

    await page.locator("summary.workspace-chip").click();
    await page.getByRole("button", { name: "archive" }).click();
    await expect(page.locator(".workspace-chip-value")).toHaveText("archive");

    await page.locator("summary.workspace-chip").click();
    await page
      .locator(
        ".workspace-switcher-panel form[action='/workspaces/create'] input",
      )
      .fill("research_2026");
    await page.getByRole("button", { name: "Create" }).click();
    await expect(page.locator(".workspace-chip-value")).toHaveText("archive");
    await page.locator("summary.workspace-chip").click();
    await expect(
      page.getByRole("button", { name: "research_2026" }),
    ).toBeVisible();
  } finally {
    await server.stop();
  }
});

test("person transfer warns on mixed images before moving", async ({
  page,
}) => {
  const server = await startE2EServer();

  try {
    await page.goto(`${server.baseURL}/people/1`);
    await page
      .locator("select[name='target_workspace']")
      .selectOption("archive");
    await page.getByRole("button", { name: "Move to workspace" }).click();

    await expect(
      page.getByText("linked image(s) also contain other people"),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Move linked images/faces" }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Copy data only" }),
    ).toBeVisible();

    await page.getByRole("button", { name: "Copy data only" }).click();
    await expect(page).toHaveURL(/\/people\/1(\?.*)?$/);
    await expect(page.locator(".workspace-chip-value")).toHaveText("archive");
    await expect(
      page.locator(".breadcrumb-item[aria-current='page']"),
    ).toHaveText("Zulu");
  } finally {
    await server.stop();
  }
});
