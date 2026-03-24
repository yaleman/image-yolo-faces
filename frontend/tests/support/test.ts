import fs from "node:fs/promises";
import path from "node:path";
import {
  test as base,
  expect,
  type Page,
  type TestInfo,
} from "@playwright/test";

const collectCoverage = process.env.PW_COLLECT_COVERAGE === "1";

async function writeCoverage(page: Page, testInfo: TestInfo): Promise<void> {
  const coverage = await page.coverage.stopJSCoverage();
  if (coverage.length === 0) {
    return;
  }

  await fs.writeFile(
    testInfo.outputPath("js-coverage.json"),
    JSON.stringify(coverage, null, 2),
    "utf-8",
  );
}

export const test = base.extend({
  page: async ({ page }, use, testInfo) => {
    const browserName = page.context().browser()?.browserType().name();
    const shouldCollectCoverage = collectCoverage && browserName === "chromium";

    if (shouldCollectCoverage) {
      await page.coverage.startJSCoverage({
        resetOnNavigation: false,
        reportAnonymousScripts: true,
      });
    }

    try {
      await use(page);
    } finally {
      if (shouldCollectCoverage) {
        try {
          await fs.mkdir(
            path.dirname(testInfo.outputPath("js-coverage.json")),
            {
              recursive: true,
            },
          );
          await writeCoverage(page, testInfo);
        } catch {
          // Ignore coverage write failures so they don't mask the test result.
        }
      }
    }
  },
});

export { expect };
