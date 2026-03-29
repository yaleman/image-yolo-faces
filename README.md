# image-yolo-faces

Batch face detection for image folders using the `arnabdhar/YOLOv8-Face-Detection` model from Hugging Face.

## Layout

The app now stores data in workspaces. By default they live under `./workspaces`, or you can override that with `FACES_WORKSPACES_DIR`.

Each workspace contains:

- `faces.json`
- `photos/`
- `exports/`

`faces.json` stores filenames only. The UI loads originals from `photos/` and generates annotated previews at request time from the same image.

The `default` workspace is created on demand and is used when no workspace is selected.

## Setup

```bash
uv sync
```

## CLI

Scan a folder into the default workspace:

```bash
uv run image-yolo-faces ./photos
```

Write into a named workspace:

```bash
uv run image-yolo-faces ./photos --workspace family_2026
```

Use a different workspace root:

```bash
uv run image-yolo-faces ./photos --workspaces-dir /data/faces-workspaces
```

Group faces by person using embeddings:

```bash
uv run image-yolo-faces ./photos --group-by-person
```

If the clustering is too coarse or too fragmented, tune:

```bash
uv run image-yolo-faces ./photos --group-by-person --person-threshold 0.50
```

Useful options:

```bash
uv run image-yolo-faces --help
```

## Web UI

Run the review UI against the workspaces root:

```bash
uv run image-yolo-faces-web
```

The web UI reads and writes the active workspace selected in the browser cookie. The header includes a workspace switcher and a create-workspace form. Workspace transfers on a person page can copy a person into another workspace or move the linked images/faces into another workspace, while leaving the source workspace intact and warning when mixed images are involved.

For live code reloading while developing:

```bash
uv run image-yolo-faces-web --reload
```

The web UI shows the annotated image list first. Click any image to open a review page where you can assign a name to a person, merge that cluster into an existing person, or split selected images into a new person. Person detail pages can also export cropped face images into `exports/<person name>/` within the active workspace. If an annotated preview is missing, the UI regenerates it on demand the first time it is requested.

## Benchmark

Measure how long it takes to render annotated previews for every image in every workspace:

```bash
uv run python scripts/benchmark_annotated.py
```

Use `--workspaces-dir` if your workspaces live somewhere other than `./workspaces`.
The script prints per-workspace and total timings, including min, p99, and max per-image render times.

The CLI downloads `model.pt` from the model repository on first use, loads it with `ultralytics.YOLO`, and parses the detections with `supervision.Detections.from_ultralytics(...)`, matching the model card example.

When `--group-by-person` is enabled, it uses an InsightFace ArcFace-style embedding model to generate face embeddings and clusters them by cosine similarity.

Annotated images use a stable color per person when grouping is enabled.

## Development

Before considering a change done, run the repo checks and fix any failures:

```bash
mise check
pnpm test:e2e
```

Use `pnpm test:e2e:coverage` when you need the frontend coverage report.
