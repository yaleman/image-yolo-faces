import pytest

from pathlib import Path

from image_yolo_faces.webui import (
    person_preview_bbox,
    preview_crop_box,
    representative_face_bbox,
    resolve_media_path,
)
from image_yolo_faces.workspaces import (
    DEFAULT_WORKSPACE_NAME,
    ensure_workspace_layout,
    normalize_report_media_paths,
    resolve_workspaces_root,
    validate_workspace_name,
    workspace_annotated_dir,
    workspace_photos_dir,
    workspace_report_path,
)


def test_resolve_media_path_handles_relative_and_absolute_paths() -> None:
    report_path = Path("/tmp/report/faces.json")

    assert resolve_media_path(report_path, "photos/example.jpg") == (
        report_path.parent / "photos/example.jpg"
    ).resolve()
    assert resolve_media_path(report_path, "/tmp/library/example.jpg") == Path(
        "/tmp/library/example.jpg"
    )


def test_representative_face_bbox_prefers_the_strongest_face() -> None:
    faces = [
        {
            "bbox": [0, 0, 12, 12],
            "confidence": 0.55,
        },
        {
            "bbox": [2, 2, 18, 18],
            "confidence": 0.92,
        },
    ]

    assert representative_face_bbox(faces) == [2.0, 2.0, 18.0, 18.0]


def test_preview_crop_box_keeps_the_crop_inside_image_bounds() -> None:
    crop = preview_crop_box([10, 20, 60, 70], image_width=120, image_height=90)

    assert crop == (0, 0, 90, 90)


def test_person_preview_bbox_uses_person_annotations_for_the_image() -> None:
    image_path = "/tmp/report/photos/Image 16.jpeg"
    entry = {
        "image": image_path,
        "faces": [
            {
                "person_id": 4,
                "bbox": [10, 10, 30, 30],
                "confidence": 0.88,
            }
        ],
    }
    person = {
        "person_id": 34,
        "faces": [
            {
                "image": "/tmp/report/photos/Image 1.jpeg",
                "bbox": [1, 1, 2, 2],
                "confidence": 0.97,
            },
            {
                "image": image_path,
                "bbox": [20, 30, 80, 90],
                "confidence": 0.62,
            },
        ],
    }

    assert person_preview_bbox(entry, 34, person) == [20.0, 30.0, 80.0, 90.0]


def test_resolve_workspaces_root_uses_the_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FACES_WORKSPACES_DIR", str(tmp_path / "spaces"))

    assert resolve_workspaces_root() == (tmp_path / "spaces").resolve()


def test_validate_workspace_name_accepts_simple_names() -> None:
    assert validate_workspace_name("team_1") == "team_1"


@pytest.mark.parametrize("workspace_name", ["team-1", " team", "", "spaces/team"])
def test_validate_workspace_name_rejects_invalid_names(workspace_name: str) -> None:
    with pytest.raises(ValueError):
        validate_workspace_name(workspace_name)


def test_ensure_workspace_layout_creates_workspace_directories(tmp_path) -> None:
    root = tmp_path / "workspaces"

    workspace_dir = ensure_workspace_layout(root, DEFAULT_WORKSPACE_NAME)

    assert workspace_dir == root / DEFAULT_WORKSPACE_NAME
    assert workspace_photos_dir(root, DEFAULT_WORKSPACE_NAME).is_dir()
    assert workspace_annotated_dir(root, DEFAULT_WORKSPACE_NAME).is_dir()
    assert workspace_report_path(root, DEFAULT_WORKSPACE_NAME) == (
        root / DEFAULT_WORKSPACE_NAME / "faces.json"
    )


def test_normalize_report_media_paths_rewrites_workspace_paths(tmp_path) -> None:
    root = tmp_path / "workspaces"
    ensure_workspace_layout(root, DEFAULT_WORKSPACE_NAME)

    image_path = workspace_photos_dir(root, DEFAULT_WORKSPACE_NAME) / "apple.png"
    image_path.write_bytes(b"image")

    report = {
        "images": [
            {
                "image": str(image_path.resolve()),
            }
        ],
        "people": [
            {
                "person_id": 1,
                "faces": [{"image": str(image_path.resolve())}],
            }
        ],
    }

    assert normalize_report_media_paths(report, root, DEFAULT_WORKSPACE_NAME) is True
    assert report["images"][0]["image"] == "apple.png"
    assert report["people"][0]["faces"][0]["image"] == "apple.png"
