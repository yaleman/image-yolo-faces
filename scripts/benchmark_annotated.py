from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from image_yolo_faces.ingest import render_faces_bytes
from image_yolo_faces.workspaces import (
    list_workspaces,
    resolve_workspaces_root,
    workspace_original_media_path,
    workspace_report_path,
)


def load_report(report_path: Path) -> dict[str, Any] | None:
    if not report_path.exists():
        return None

    try:
        raw_report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {report_path}: {exc}") from exc

    if not isinstance(raw_report, dict):
        raise SystemExit(f"{report_path} must contain a JSON object.")

    return raw_report


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0

    ordered_values = sorted(values)
    index = max(
        0,
        min(
            len(ordered_values) - 1,
            math.ceil((pct / 100.0) * len(ordered_values)) - 1,
        ),
    )
    return ordered_values[index]


def format_millis(value: float) -> str:
    return f"{value * 1000:.1f}ms"


def benchmark_workspace(
    workspaces_root: Path, workspace_name: str
) -> tuple[int, int, float, list[float]]:
    report_path = workspace_report_path(workspaces_root, workspace_name)
    report = load_report(report_path)
    if report is None:
        return (0, 0, 0.0, [])

    images = report.get("images", [])
    if not isinstance(images, list):
        raise SystemExit(f"{report_path} has an invalid 'images' field.")

    rendered = 0
    missing = 0
    durations: list[float] = []
    start = time.perf_counter()

    for entry in images:
        if not isinstance(entry, dict):
            continue

        image_value = entry.get("image")
        if not isinstance(image_value, str) or not image_value.strip():
            continue

        original_path = workspace_original_media_path(
            workspaces_root, workspace_name, image_value
        )
        if not original_path.exists():
            missing += 1
            continue

        faces = entry.get("faces", [])
        if not isinstance(faces, list):
            faces = []

        image_start = time.perf_counter()
        render_faces_bytes(original_path, faces)
        durations.append(time.perf_counter() - image_start)
        rendered += 1

    elapsed = time.perf_counter() - start
    return (rendered, missing, elapsed, durations)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark annotated-image generation across all workspaces."
    )
    parser.add_argument(
        "--workspaces-dir",
        type=Path,
        default=None,
        help="Path to the workspaces root. Defaults to FACES_WORKSPACES_DIR or ./workspaces.",
    )
    args = parser.parse_args()

    workspaces_root = resolve_workspaces_root(args.workspaces_dir)
    workspace_names = list_workspaces(workspaces_root)

    total_rendered = 0
    total_missing = 0
    total_elapsed = 0.0
    total_durations: list[float] = []
    benchmarked = 0

    print(f"Benchmarking annotated rendering in {workspaces_root}")
    for workspace_name in workspace_names:
        rendered, missing, elapsed, durations = benchmark_workspace(
            workspaces_root, workspace_name
        )
        if rendered == 0 and missing == 0:
            continue

        benchmarked += 1
        total_rendered += rendered
        total_missing += missing
        total_elapsed += elapsed
        total_durations.extend(durations)
        rate = rendered / elapsed if elapsed > 0 else 0.0
        min_duration = format_millis(min(durations)) if durations else "0.0ms"
        p99_duration = (
            format_millis(percentile(durations, 99.0)) if durations else "0.0ms"
        )
        max_duration = format_millis(max(durations)) if durations else "0.0ms"
        print(
            f"{workspace_name}: {rendered} image(s) in {elapsed:.3f}s "
            f"({rate:.1f} image(s)/s, {missing} missing original(s), "
            f"min={min_duration}, p99={p99_duration}, max={max_duration})"
        )

    if benchmarked == 0:
        print("No workspace reports with images were found.")
        return

    total_rate = total_rendered / total_elapsed if total_elapsed > 0 else 0.0
    min_duration = format_millis(min(total_durations)) if total_durations else "0.0ms"
    p99_duration = (
        format_millis(percentile(total_durations, 99.0))
        if total_durations
        else "0.0ms"
    )
    max_duration = format_millis(max(total_durations)) if total_durations else "0.0ms"
    print(
        f"Total: {total_rendered} image(s) across {benchmarked} workspace(s) in "
        f"{total_elapsed:.3f}s ({total_rate:.1f} image(s)/s, {total_missing} missing original(s), "
        f"min={min_duration}, p99={p99_duration}, max={max_duration})"
    )


if __name__ == "__main__":
    main()
