from pathlib import Path

from image_yolo_faces.webui import (
    person_preview_bbox,
    preview_crop_box,
    representative_face_bbox,
    resolve_media_path,
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
