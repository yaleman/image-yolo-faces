# image-yolo-faces

`image-yolo-faces` scans photo collections for faces, groups repeated faces into
people, and gives you a local web UI for reviewing and cleaning up the results.
It is built for personal image sets where you want a simple workspace on disk:
original photos, a JSON report, and any cropped face exports all live together.

## What You Can Do

- Scan folders of images for faces with the
  `arnabdhar/YOLOv8-Face-Detection` model from Hugging Face.
- Group repeated faces into people with InsightFace embeddings.
- Review annotated images in a browser, with face boxes and person labels drawn
  on demand.
- Name people, merge duplicate people, and split incorrectly grouped images into
  a new person.
- Upload more images from the web UI and get per-file import results.
- Search and sort images or people while keeping your current view state.
- Keep separate photo collections in workspaces such as `default`,
  `family_2026`, or `archive`.
- Copy a person into another workspace, or move linked image and face data into a
  target workspace while leaving the source workspace intact.
- Export cropped face images for a person into that workspace's `exports/`
  directory.

## How It Works

The app stores everything in workspaces. By default, workspaces live under
`workspaces/`. Set `FACES_WORKSPACES_DIR` or pass `--workspaces-dir` if you want
them somewhere else.

Each workspace contains:

- `faces.json`: the report and the single source of truth for images, faces,
  people, names, aliases, and scan configuration.
- `photos/`: imported source images.
- `exports/`: derived cropped face exports.

The scanner imports images into the active workspace, records each image by
filename, stores a SHA-256 hash for duplicate detection, and writes detections to
`faces.json`. The web UI reads and writes the same report, so CLI scans and
browser edits stay in sync.

Annotated previews and face thumbnails are generated when the browser requests
them. They are not stored as primary data. Exports are also derived files: when
you export a person, cropped face images are written to `exports/<person name>/`
without changing `faces.json`.

## Setup

This project requires Python 3.13 or newer.

```bash
uv sync
pnpm install
```

The first scan downloads the YOLO face model from Hugging Face. Person grouping
also loads an InsightFace embedding model and runs on the CPU by default.

## Scan Images From The CLI

Scan a folder into the default workspace:

```bash
uv run image-yolo-faces photos/
```

Scan into a named workspace:

```bash
uv run image-yolo-faces photos/ --workspace family_2026
```

Use a different workspace root:

```bash
uv run image-yolo-faces photos/ --workspaces-dir data/faces-workspaces
```

Group repeated faces into people:

```bash
uv run image-yolo-faces photos/ --group-by-person
```

Tune grouping if the results are too broad or too fragmented:

```bash
uv run image-yolo-faces photos/ --group-by-person --person-threshold 0.50
```

Useful options:

```bash
uv run image-yolo-faces --help
```

## Review In The Web UI

Start the local review app:

```bash
uv run image-yolo-faces-web
```

Then open the URL printed by the command. The UI starts in the active workspace
and stores the selected workspace in a browser cookie. Use the workspace menu to
switch workspaces or create a new one.

The home page shows annotated images first. From there you can:

- upload more images by choosing files or dragging them onto the page;
- open an image to compare the original and annotated versions;
- name people found in the image;
- merge a person into another person;
- delete an imported image from the dataset and filesystem;
- open the People view to review people across the whole workspace.

Person detail pages let you rename a person, merge them into another person,
split selected images into a new person, transfer data to another workspace, and
export cropped face images.

## Workspace Transfers

Workspace transfers are copy-oriented. They copy linked image and face data into
the target workspace and do not remove source images or source face records. If a
person is linked to images that also contain other people, the UI warns you
before moving linked image data so you can choose whether to copy only the person
data instead.

## Benchmark Annotated Rendering

Measure how long it takes to render annotated previews for every image in every
workspace:

```bash
uv run python scripts/benchmark_annotated.py
```

Use `--workspaces-dir` if your workspaces live somewhere other than
`workspaces/`. The script prints per-workspace and total timings, including min,
p99, and max per-image render times.

## Development

Before considering a change done, run the repo checks and fix any failures:

```bash
mise check
```

Run the browser e2e tests for frontend upload, workspace, and collection-view
changes:

```bash
pnpm test:e2e
```

Use `pnpm test:e2e:coverage` when you need the frontend coverage report.
