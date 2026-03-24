from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from image_yolo_faces.webui import create_app
from image_yolo_faces.workspaces import (
    DEFAULT_WORKSPACE_NAME,
    ensure_workspace_layout,
    workspace_annotated_dir,
    workspace_photos_dir,
    workspace_report_path,
)


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


def seed_workspace_root(workspaces_root: Path) -> None:
    ensure_workspace_layout(workspaces_root, DEFAULT_WORKSPACE_NAME)
    ensure_workspace_layout(workspaces_root, "archive")

    photos_dir = workspace_photos_dir(workspaces_root, DEFAULT_WORKSPACE_NAME)
    annotated_dir = workspace_annotated_dir(workspaces_root, DEFAULT_WORKSPACE_NAME)

    apple_image = photos_dir / "apple.png"
    zebra_image = photos_dir / "zebra.png"
    make_fixture_image(apple_image, (154, 103, 44), "apple")
    make_fixture_image(zebra_image, (44, 89, 154), "zebra")

    apple_annotated = annotated_dir / "apple.png"
    zebra_annotated = annotated_dir / "zebra.png"
    shutil.copyfile(apple_image, apple_annotated)
    shutil.copyfile(zebra_image, zebra_annotated)

    apple_bbox = [16.0, 16.0, 72.0, 72.0]
    zebra_bbox = [18.0, 18.0, 74.0, 74.0]

    report = {
        "group_by_person": False,
        "images": [
            {
                "image": "apple.png",
                "face_count": 1,
                "faces": [
                    {
                        "bbox": apple_bbox,
                        "confidence": 0.97,
                        "person_id": 1,
                    }
                ],
                "added_at": 1_000_000_000,
                "hashes": {"sha256": sha256_file(apple_image)},
            },
            {
                "image": "zebra.png",
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
                        "image": "apple.png",
                        "face_index": 0,
                        "bbox": apple_bbox,
                        "confidence": 0.97,
                        "person_id": 1,
                    },
                    {
                        "image": "zebra.png",
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
                        "image": "zebra.png",
                        "face_index": 1,
                        "bbox": zebra_bbox,
                        "confidence": 0.95,
                        "person_id": 2,
                    }
                ],
            },
        ],
    }

    report_path = workspace_report_path(workspaces_root, DEFAULT_WORKSPACE_NAME)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def read_report(workspaces_root: Path, workspace_name: str) -> dict:
    report_path = workspace_report_path(workspaces_root, workspace_name)
    return json.loads(report_path.read_text(encoding="utf-8"))


def make_client(tmp_path: Path) -> tuple[TestClient, Path]:
    workspaces_root = tmp_path / "workspaces"
    seed_workspace_root(workspaces_root)
    return TestClient(create_app(workspaces_root)), workspaces_root


def test_workspace_cookie_defaults_and_switches(tmp_path) -> None:
    client, workspaces_root = make_client(tmp_path)

    response = client.get("/")
    assert response.status_code == 200
    assert "faces_workspace=default" in response.headers["set-cookie"]
    assert 'class="workspace-chip-value">default<' in response.text

    response = client.post(
        "/workspaces/select",
        data={"workspace": "archive"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "faces_workspace=archive" in response.headers["set-cookie"]
    client.cookies.set("faces_workspace", "archive", path="/")

    response = client.get("/")
    assert response.status_code == 200
    assert 'class="workspace-chip-value">archive<' in response.text
    assert read_report(workspaces_root, DEFAULT_WORKSPACE_NAME)["images"]


def test_workspace_create_route_creates_directories_and_sets_cookie(tmp_path) -> None:
    client, workspaces_root = make_client(tmp_path)

    response = client.post(
        "/workspaces/create",
        data={"workspace": "research_2026"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "faces_workspace=research_2026" not in response.headers.get(
        "set-cookie", ""
    )
    assert (workspaces_root / "research_2026" / "photos").is_dir()
    assert (workspaces_root / "research_2026" / "annotated").is_dir()

    response = client.get("/")
    assert response.status_code == 200
    assert 'class="workspace-chip-value">default<' in response.text


def test_person_transfer_copy_only_keeps_source_workspace_intact(tmp_path) -> None:
    client, workspaces_root = make_client(tmp_path)

    response = client.post(
        "/people/1/transfer",
        data={
            "target_workspace": "archive",
            "transfer_mode": "move",
            "transfer_choice": "copy_only",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/people/1"

    source_report = read_report(workspaces_root, DEFAULT_WORKSPACE_NAME)
    target_report = read_report(workspaces_root, "archive")

    assert len(source_report["images"]) == 2
    assert len(source_report["people"]) == 2
    assert len(target_report["images"]) == 2
    assert len(target_report["people"]) == 1
    assert target_report["people"][0]["name"] == "Zulu"


def test_person_transfer_move_rehomes_mixed_image_faces(tmp_path) -> None:
    client, workspaces_root = make_client(tmp_path)

    response = client.post(
        "/people/1/transfer",
        data={
            "target_workspace": "archive",
            "transfer_mode": "move",
            "transfer_choice": "move_linked",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/people/1"

    source_report = read_report(workspaces_root, DEFAULT_WORKSPACE_NAME)
    target_report = read_report(workspaces_root, "archive")

    assert len(source_report["images"]) == 1
    assert source_report["images"][0]["image"] == "zebra.png"
    assert len(source_report["people"]) == 1
    assert source_report["people"][0]["person_id"] == 2
    assert len(target_report["images"]) == 2
    assert len(target_report["people"]) == 1
    assert target_report["people"][0]["person_id"] == 1


def test_person_transfer_warns_before_moving_mixed_images(tmp_path) -> None:
    client, _ = make_client(tmp_path)

    response = client.post(
        "/people/1/transfer",
        data={
            "target_workspace": "archive",
            "transfer_mode": "move",
        },
    )
    assert response.status_code == 200
    assert "Move linked images/faces" in response.text
    assert "Copy data only" in response.text
    assert "linked image(s) also contain other people" in response.text
