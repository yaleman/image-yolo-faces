import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "frontend/tests",
  outputDir: "output/playwright",
  reporter: [["list"]],
  use: {
    trace: "retain-on-failure",
  },
});
