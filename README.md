# image-yolo-faces

Batch face detection for image folders using the `arnabdhar/YOLOv8-Face-Detection` model from Hugging Face.

## Setup

```bash
uv sync
```

## Usage

Scan a directory of images and write a JSON report:

```bash
uv run image-yolo-faces ./photos --output faces.json
```

If `faces.json` already exists, the CLI reuses entries for image paths it has already seen and only runs the model on new images.

Write annotated images as well:

```bash
uv run image-yolo-faces ./photos --output faces.json --annotated-dir annotated
```

Group faces by person using embeddings:

```bash
uv run image-yolo-faces ./photos --output faces.json --group-by-person
```

If the clustering is too coarse or too fragmented, tune:

```bash
uv run image-yolo-faces ./photos --output faces.json --group-by-person --person-threshold 0.50
```

Useful options:

```bash
uv run image-yolo-faces --help
```

Run the review UI against the saved report:

```bash
uv run image-yolo-faces-web
```

The web UI looks for `faces.json` in the current directory by default. Pass `--report` only if your report lives somewhere else.

For live code reloading while developing:

```bash
uv run image-yolo-faces-web --reload
```

The web UI shows the annotated image list first. Click any image to open a review page where you can assign a name to a person, or merge that cluster into an existing person if it was a mismatch. If an annotated preview is missing, the UI regenerates it on demand the first time it is requested.

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
