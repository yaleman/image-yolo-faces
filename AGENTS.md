# Agents

## Project Shape

- The scanner and web UI now operate on workspace-scoped JSON reports.
- Workspaces live under `./workspaces` by default, or under `FACES_WORKSPACES_DIR` when that environment variable is set.
- Each workspace contains its own `faces.json`, `photos/`, and `annotated/` directories.
- `default` is the fallback workspace and should be created on demand.
- The browser keeps the active workspace in a cookie so the frontend and backend stay aligned.
- Image and person edits must round-trip through the active workspace report.
- Keep the schema simple and stable. Prefer extending the existing report over adding a second datastore.

## Workflow Rules

- Use package managers to manage dependencies instead of direct file editing.
- Prefer simple, direct changes over abstractions.
- Refactor to reduce code sprawl while working.
- Do not plan for extensibility or backwards compatibility unless explicitly asked.
- Use `pnpm` instead of `npm`.
- Avoid OpenSSL at all costs.
- Run `mise check` and have it pass before considering any task complete.
- Run `pnpm test:e2e` for frontend upload, workspace, and collection-view changes before considering any UI task complete.
- Update this `AGENTS.md` file any time the system design, workflow, or required verification steps change.

## System Design

- `faces.json` is the single source of truth within each workspace for images, faces, people, and report-level configuration.
- The CLI and web UI both read and write the active workspace report rather than a second datastore.
- Shared ingestion and scan logic lives in package modules, not in the CLI or web entrypoints.
- Fresh web-loaded reports default to grouped uploads so people are created automatically unless the report explicitly turns grouping off.
- Image entries include an `added_at` Unix timestamp in nanoseconds plus a `hashes` dictionary. `hashes.sha256` is used for exact duplicate detection, and the schema is kept as a dict for future hash types.
- Uploaded images are imported into the active workspace, renamed on filename collision with a Unix-seconds suffix, scanned for faces immediately, and written back into that workspace with annotated output.
- The `/uploads` handler accepts one or more `image` parts, processes them in order, and returns per-file results so the UI can show itemized upload progress.
- The web UI shows a per-image upload queue with live progress and final status for each file instead of a single opaque submission result.
- Collection views share a `sort` query param. Image lists support `added` and `filename`; people lists support `added` and `name`; person detail pages sort their image cards by `added` or `filename`.
- Preserve `q`, `sort`, and `unnamed` when building collection links and search form submissions so view state round-trips cleanly.
- Annotated images and face previews are derived media, not primary state.
- The web UI serves server-rendered Jinja templates and loads built frontend assets from `image_yolo_faces/static/dist`.
- Frontend source lives under `frontend/` and is built with Vite, Tailwind, and TypeScript. Built assets are committed so the packaged app can run without a runtime frontend build.
- Vite emits stable committed asset names under `image_yolo_faces/static/dist` so the bundle stays diff-friendly while still using the manifest for lookup.
- Biome is the formatter/linter for frontend files, and TypeScript checks run through the repo `tsconfig.json`.
- Playwright e2e tests live under `frontend/tests/` and exercise the real browser against a lightweight test server that stubs face scanning for deterministic UI coverage.
- Playwright coverage runs in Chromium with `PW_COLLECT_COVERAGE=1 pnpm test:e2e:coverage` and writes the report under `output/playwright/coverage/`.
- Python coverage runs with `mise python-coverage` and writes the HTML report under `output/python-coverage/`.

## Python Rules

- Prefer `uv` for dependency and runtime commands.
- Keep CLI and web entrypoints thin; put shared report logic in the package modules.
- Use `click` for CLI handling and `FastAPI` for the review UI.
- The UI should load the report, show the annotated image list first, and use per-image detail pages for naming, merge, split, and workspace transfer operations.
- Render HTML with Jinja2 templates under `image_yolo_faces/templates/` rather than assembling pages manually in Python.
- Keep all CSS in standalone files under `image_yolo_faces/static/`; never use inline `style` attributes or inline `<style>` blocks.
- Keep the page title restrained and small. Treat headings as navigation and context, not as giant hero text.
- Prefer direct, simple layouts. Reuse shared base templates and static styles instead of duplicating markup or presentation logic.
- On the individual person page, previews should come from a dynamic face crop with a little surrounding context instead of the full annotated image.
- Keep frontend source in `frontend/`; treat `image_yolo_faces/static/dist` as build output generated by Vite.

## Rust Packages

- `clap` for CLI/environment/config
- `serde` for handling (de)serialization
- `sea-orm` for database-related things

## Blocked Dependencies

- `serde_yaml` is blocked. Use `yaml-rust` for YAML parsing.

## Rust Rules

- `.expect("<message>")` not `.unwrap()`
