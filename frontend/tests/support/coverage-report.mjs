import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import coverageLib from "istanbul-lib-coverage";
import reportLib from "istanbul-lib-report";
import reports from "istanbul-reports";
import v8ToIstanbul from "v8-to-istanbul";

const { createCoverageMap } = coverageLib;
const { createContext } = reportLib;

const repoRoot = path.resolve(
  fileURLToPath(new URL("../../..", import.meta.url)),
);
const outputDir = path.join(repoRoot, "output", "playwright");
const coverageDir = path.join(outputDir, "coverage");
const distDir = path.join(repoRoot, "image_yolo_faces", "static", "dist");
const rawFileName = "js-coverage.json";

async function walk(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const resolved = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await walk(resolved)));
    } else if (entry.name === rawFileName) {
      files.push(resolved);
    }
  }
  return files;
}

function localSourcePath(url) {
  let pathname;
  try {
    pathname = new URL(url).pathname;
  } catch {
    return null;
  }

  const prefix = "/static/dist/";
  if (!pathname.startsWith(prefix)) {
    return null;
  }

  const relativePath = pathname.slice(prefix.length);
  return path.join(distDir, relativePath);
}

async function convertEntry(entry) {
  const sourcePath = localSourcePath(entry.url);
  if (!sourcePath) {
    return null;
  }

  const converter = v8ToIstanbul(sourcePath);
  await converter.load();
  converter.applyCoverage(entry.functions);
  return converter.toIstanbul();
}

async function main() {
  const rawFiles = await walk(outputDir);
  if (rawFiles.length === 0) {
    throw new Error(
      `No Playwright coverage files found in ${outputDir}. Run with PW_COLLECT_COVERAGE=1.`,
    );
  }

  await fs.rm(coverageDir, { recursive: true, force: true });
  await fs.mkdir(coverageDir, { recursive: true });

  const coverageMap = createCoverageMap({});
  for (const rawFile of rawFiles) {
    const coverageEntries = JSON.parse(await fs.readFile(rawFile, "utf-8"));
    if (!Array.isArray(coverageEntries)) {
      continue;
    }

    for (const entry of coverageEntries) {
      if (
        !entry ||
        typeof entry !== "object" ||
        typeof entry.url !== "string" ||
        !Array.isArray(entry.functions)
      ) {
        continue;
      }

      const istanbulCoverage = await convertEntry(entry);
      if (istanbulCoverage !== null) {
        coverageMap.merge(istanbulCoverage);
      }
    }
  }

  const context = createContext({
    coverageMap,
    dir: coverageDir,
  });

  reports.create("text-summary").execute(context);
  reports.create("html").execute(context);
  reports.create("lcovonly").execute(context);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
