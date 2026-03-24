from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from image_yolo_faces import webui


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def make_fixture_image(path: Path, color: tuple[int, int, int], label: str) -> None:
    image = Image.new("RGB", (96, 96), color)
    drawer = ImageDraw.Draw(image)
    drawer.rectangle((10, 10, 86, 86), outline=(255, 255, 255), width=4)
    drawer.text((18, 40), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def seed_workspace(workspace: Path) -> Path:
    photos_dir = workspace / "photos"
    annotated_dir = workspace / "annotated"
    photos_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[3]
    seed_fixture = repo_root / "frontend" / "tests" / "fixtures" / "zebra.png"
    if not seed_fixture.exists():
        raise FileNotFoundError(seed_fixture)

    apple_image = photos_dir / "apple.png"
    zebra_image = photos_dir / "zebra.png"
    shutil.copyfile(seed_fixture, zebra_image)
    make_fixture_image(apple_image, (154, 103, 44), "apple")

    apple_annotated = annotated_dir / "apple_faces.png"
    zebra_annotated = annotated_dir / "zebra_faces.png"
    shutil.copyfile(apple_image, apple_annotated)
    shutil.copyfile(zebra_image, zebra_annotated)

    apple_bbox = [16.0, 16.0, 72.0, 72.0]
    zebra_bbox = [18.0, 18.0, 74.0, 74.0]

    apple_path = str(apple_image.resolve())
    zebra_path = str(zebra_image.resolve())
    report = {
        "group_by_person": False,
        "images": [
            {
                "image": apple_path,
                "face_count": 1,
                "faces": [
                    {
                        "bbox": apple_bbox,
                        "confidence": 0.97,
                        "person_id": 1,
                    }
                ],
                "added_at": 1_000_000_000,
                "annotated_image": "annotated/apple_faces.png",
                "hashes": {"sha256": sha256_file(apple_image)},
            },
            {
                "image": zebra_path,
                "face_count": 2,
                "faces": [
                    {
                        "bbox": zebra_bbox,
                        "confidence": 0.96,
                        "person_id": 1,
                    },
                    {
                        "bbox": zebra_bbox,
                        "confidence": 0.95,
                        "person_id": 2,
                    },
                ],
                "added_at": 2_000_000_000,
                "annotated_image": "annotated/zebra_faces.png",
                "hashes": {"sha256": sha256_file(zebra_image)},
            },
        ],
        "next_person_id": 3,
        "people": [
            {
                "person_id": 1,
                "name": "Zulu",
                "face_count": 2,
                "centroid": [1.0],
                "aliases": [],
                "faces": [
                    {
                        "image": apple_path,
                        "face_index": 0,
                        "bbox": apple_bbox,
                        "confidence": 0.97,
                        "person_id": 1,
                    },
                    {
                        "image": zebra_path,
                        "face_index": 0,
                        "bbox": zebra_bbox,
                        "confidence": 0.96,
                        "person_id": 1,
                    },
                ],
            },
            {
                "person_id": 2,
                "name": "Alpha",
                "face_count": 1,
                "centroid": [1.0],
                "aliases": [],
                "faces": [
                    {
                        "image": zebra_path,
                        "face_index": 1,
                        "bbox": zebra_bbox,
                        "confidence": 0.95,
                        "person_id": 2,
                    }
                ],
            },
        ],
    }

    report_path = workspace / "faces.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def fake_scan_image_entry(**kwargs):
    image_path = Path(kwargs["image_path"])
    annotated_path = kwargs["annotated_path"]
    if annotated_path is not None:
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image_path, annotated_path)

    return (
        {
            "image": str(image_path.resolve()),
            "face_count": 0,
            "faces": [],
            "added_at": int(kwargs["added_at_ns"]),
            "annotated_image": str(annotated_path) if annotated_path is not None else None,
            "hashes": {},
        },
        int(kwargs["next_person_id"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    report_path = seed_workspace(workspace)

    webui.scan_image_entry = fake_scan_image_entry  # type: ignore[assignment]
    app = webui.create_app(report_path)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
