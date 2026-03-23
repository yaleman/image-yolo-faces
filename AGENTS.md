# Agents

This repository uses a single JSON report as the source of truth for both the CLI and the web UI.

## Project Shape

- The scanner writes and updates `faces.json` by default.
- The web UI should treat `faces.json` as the implied report path when `--report` is not provided.
- Annotated images are derived output, not primary state.
- Image and person edits must round-trip through the JSON report.
- Keep the schema simple and stable. Prefer extending the existing report over adding a second datastore.

## Workflow Rules

- Use package managers to manage dependencies instead of direct file editing.
- Prefer simple, direct changes over abstractions.
- Refactor to reduce code sprawl while working.
- Do not plan for extensibility or backwards compatibility unless explicitly asked.
- Use `pnpm` instead of `npm`.
- Avoid OpenSSL at all costs.

## Python Rules

- Prefer `uv` for dependency and runtime commands.
- Keep CLI and web entrypoints thin; put shared report logic in the package modules.
- Use `click` for CLI handling and `FastAPI` for the review UI.
- `faces.json` is the default report file for the UI.
- The UI should load the report, show the annotated image list first, and use per-image detail pages for naming and merge operations.
- Render HTML with Jinja2 templates under `image_yolo_faces/templates/` rather than assembling pages manually in Python.
- Keep all CSS in standalone files under `image_yolo_faces/static/`; never use inline `style` attributes or inline `<style>` blocks.
- Keep the page title restrained and small. Treat headings as navigation and context, not as giant hero text.
- Prefer direct, simple layouts. Reuse shared base templates and static styles instead of duplicating markup or presentation logic.
- On the individual person page, previews should come from a dynamic face crop with a little surrounding context instead of the full annotated image.

## Rust Packages

- `clap` for CLI/environment/config
- `serde` for handling (de)serialization
- `sea-orm` for database-related things

## Blocked Dependencies

- `serde_yaml` is blocked. Use `yaml-rust` for YAML parsing.

## Rust Rules

- `.expect("<message>")` not `.unwrap()`
