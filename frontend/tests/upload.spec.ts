import path from "node:path";

import { startE2EServer } from "./support/server";
import { expect, test } from "./support/test";

const fixture = (name: string): string =>
  path.join(process.cwd(), "frontend/tests/fixtures", name);

test("batch uploads send every selected file and refresh the home page", async ({
  page,
}) => {
  const server = await startE2EServer();

  try {
    await page.goto(server.baseURL);
    await page.locator("summary.upload-summary").click();

    const uploadRequest = page.waitForRequest(
      (request) =>
        request.url().endsWith("/uploads") && request.method() === "POST",
    );

    await page
      .locator("[data-upload-input]")
      .setInputFiles([fixture("zebra.png"), fixture("monkey.png")]);
    await page.getByRole("button", { name: "Upload images" }).click();

    await uploadRequest;

    await expect(page.locator("article")).toHaveCount(3);
    await expect(page.locator("article").first()).toContainText("monkey.png");

    await page.getByRole("link", { name: "Filename" }).click();
    await expect(page.locator("article").first()).toContainText("apple.png");
  } finally {
    await server.stop();
  }
});
