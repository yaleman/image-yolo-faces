from __future__ import annotations

import json
import colorsys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence, Tuple, cast

import click

if TYPE_CHECKING:
    from supervision import Detections
    from ultralytics import YOLO

DEFAULT_MODEL_REPO = "arnabdhar/YOLOv8-Face-Detection"
DEFAULT_MODEL_FILE = "model.pt"
DEFAULT_EMBEDDING_MODEL = "buffalo_l"
DEFAULT_PERSON_THRESHOLD = 0.45
DEFAULT_PERSON_GROUPING_STRATEGY = "full-image-iou-v1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def iter_images(paths: Sequence[Path], recursive: bool) -> Iterator[Tuple[Path, Path]]:
    for root in paths:
        if root.is_file():
            if root.suffix.lower() in IMAGE_EXTENSIONS:
                yield root.parent, root
            continue

        walker = root.rglob("*") if recursive else root.glob("*")
        for candidate in walker:
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                yield root, candidate


def normalize_image_key(image_path: Path | str) -> str:
    return str(Path(image_path).resolve())


def default_report(
    model_repo: str,
    model_file: str,
    confidence: float,
    group_by_person: bool,
    embedding_model: str,
    person_threshold: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "model_repo": model_repo,
        "model_file": model_file,
        "confidence_threshold": confidence,
        "next_person_id": 1,
        "images": [],
    }
    if group_by_person:
        report["group_by_person"] = True
        report["embedding_model"] = embedding_model
        report["person_similarity_threshold"] = person_threshold
        report["person_grouping_strategy"] = DEFAULT_PERSON_GROUPING_STRATEGY
        report["people"] = []
    return report


def load_existing_report(
    output_path: Path,
    model_repo: str,
    model_file: str,
    confidence: float,
    group_by_person: bool,
    embedding_model: str,
    person_threshold: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not output_path.exists():
        return (
            default_report(model_repo, model_file, confidence, group_by_person, embedding_model, person_threshold),
            {},
            [],
        )

    try:
        raw_report = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Existing report at {output_path} is not valid JSON: {exc}") from exc

    if not isinstance(raw_report, dict):
        raise click.ClickException(f"Existing report at {output_path} must contain a JSON object.")

    if (
        raw_report.get("model_repo") != model_repo
        or raw_report.get("model_file") != model_file
        or raw_report.get("confidence_threshold") != confidence
    ):
        return (
            default_report(model_repo, model_file, confidence, group_by_person, embedding_model, person_threshold),
            {},
            [],
        )

    if group_by_person:
        if (
            raw_report.get("group_by_person") is not True
            or raw_report.get("embedding_model") != embedding_model
            or raw_report.get("person_similarity_threshold") != person_threshold
            or raw_report.get("person_grouping_strategy") != DEFAULT_PERSON_GROUPING_STRATEGY
        ):
            return (
                default_report(
                    model_repo,
                    model_file,
                    confidence,
                    group_by_person,
                    embedding_model,
                    person_threshold,
                ),
                {},
                [],
            )

    images = raw_report.get("images", [])
    if not isinstance(images, list):
        raise click.ClickException(f"Existing report at {output_path} has an invalid 'images' field.")

    report = {
        "model_repo": model_repo,
        "model_file": model_file,
        "confidence_threshold": confidence,
        "next_person_id": 1,
        "images": images,
    }
    if group_by_person:
        report["group_by_person"] = True
        report["embedding_model"] = embedding_model
        report["person_similarity_threshold"] = person_threshold
        report["person_grouping_strategy"] = DEFAULT_PERSON_GROUPING_STRATEGY

    image_index: dict[str, dict[str, Any]] = {}
    for entry in images:
        if not isinstance(entry, dict) or "image" not in entry:
            continue
        entry["image"] = normalize_image_key(entry["image"])
        image_index[normalize_image_key(entry["image"])] = entry

    people: list[dict[str, Any]] = []
    if group_by_person:
        loaded_people = raw_report.get("people", [])
        if not isinstance(loaded_people, list):
            raise click.ClickException(f"Existing report at {output_path} has an invalid 'people' field.")
        people = loaded_people
        max_person_id = 0
        for person in people:
            try:
                max_person_id = max(max_person_id, int(person["person_id"]))
            except (KeyError, TypeError, ValueError):
                continue
        next_person_id = raw_report.get("next_person_id")
        if isinstance(next_person_id, int) and next_person_id > max_person_id:
            report["next_person_id"] = next_person_id
        else:
            report["next_person_id"] = max_person_id + 1

    return report, image_index, people


def load_model(model_repo: str, model_file: str) -> YOLO:
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    model_path = hf_hub_download(repo_id=model_repo, filename=model_file)
    return YOLO(model_path)


def load_face_encoder(embedding_model: str):
    from insightface.app import FaceAnalysis

    encoder = FaceAnalysis(name=embedding_model, providers=["CPUExecutionProvider"])
    encoder.prepare(ctx_id=-1, det_size=(640, 640))
    return encoder


def detect_faces(model: YOLO, image_path: Path, confidence: float) -> Detections:
    from PIL import Image
    from supervision import Detections

    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")

    output = model(image, verbose=False)
    detections = Detections.from_ultralytics(output[0])

    if detections.confidence is not None and len(detections.confidence):
        detections = cast(Detections, detections[detections.confidence >= confidence])

    return detections


def detections_to_faces(detections: Detections) -> list[dict[str, Any]]:
    confidences = detections.confidence if detections.confidence is not None else []
    faces: list[dict[str, Any]] = []

    for index, xyxy in enumerate(detections.xyxy):
        confidence_value = None
        if index < len(confidences):
            confidence_value = float(confidences[index])

        faces.append(
            {
                "bbox": [float(value) for value in xyxy],
                "confidence": confidence_value,
            }
        )

    return faces


def render_faces(
    image_path: Path,
    faces: Sequence[dict[str, Any]],
    annotated_path: Path,
    person_labels: Mapping[int, str] | None = None,
    force: bool = False,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if annotated_path.exists() and not force:
        return

    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")

    drawer = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for face in faces:
        bbox = face.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        left, top, right, bottom = (float(value) for value in bbox)
        confidence = face.get("confidence")
        person_id = face.get("person_id")
        label_parts = ["face"]
        outline = (255, 0, 0)
        fill = (255, 0, 0)
        text_fill = (255, 255, 255)

        if person_id is not None:
            try:
                person_number = int(person_id)
            except (TypeError, ValueError):
                person_number = None
            if person_number is not None:
                outline = person_box_color(person_number)
                fill = outline
                text_fill = contrast_text_color(outline)
                label_parts = [f"person {person_number}"]
                if person_labels is not None:
                    person_label = person_labels.get(person_number)
                    if person_label:
                        label_parts.append(person_label)

        if confidence is not None:
            label_parts.append(f"{confidence:.2f}")

        label = " ".join(label_parts)

        drawer.rectangle((left, top, right, bottom), outline=outline, width=3)
        text_box = drawer.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_left = left
        text_top = max(0.0, top - text_height - 6)
        drawer.rectangle(
            (text_left, text_top, text_left + text_width + 8, text_top + text_height + 6),
            fill=fill,
        )
        drawer.text((text_left + 4, text_top + 3), label, fill=text_fill, font=font)

    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(annotated_path)


def person_box_color(person_id: int) -> tuple[int, int, int]:
    hue = (person_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return (
        int(round(red * 255)),
        int(round(green * 255)),
        int(round(blue * 255)),
    )


def contrast_text_color(rgb: Sequence[int]) -> tuple[int, int, int]:
    red, green, blue = rgb
    luminance = (0.299 * red) + (0.587 * green) + (0.114 * blue)
    return (0, 0, 0) if luminance > 170 else (255, 255, 255)


def bbox_iou(left_bbox: Sequence[float], right_bbox: Sequence[float]) -> float:
    left_x1, top_y1, left_x2, bottom_y2 = (float(value) for value in left_bbox)
    right_x1, top_y1_b, right_x2, bottom_y2_b = (float(value) for value in right_bbox)

    intersection_x1 = max(left_x1, right_x1)
    intersection_y1 = max(top_y1, top_y1_b)
    intersection_x2 = min(left_x2, right_x2)
    intersection_y2 = min(bottom_y2, bottom_y2_b)

    if intersection_x2 <= intersection_x1 or intersection_y2 <= intersection_y1:
        return 0.0

    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, bottom_y2 - top_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, bottom_y2_b - top_y1_b)
    union_area = left_area + right_area - intersection_area
    if union_area <= 0.0:
        return 0.0
    return float(intersection_area / union_area)


def load_face_analysis_faces(face_encoder, image_path: Path) -> list[Any]:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        return []

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`rcond` parameter will change to the default of machine precision.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"`estimate` is deprecated since version 0\.26 and will be removed in version 2\.2.*",
            category=FutureWarning,
        )
        faces = face_encoder.get(image)
    return list(faces or [])


def match_embeddings_to_detections(
    face_encoder,
    image_path: Path,
    faces: Sequence[dict[str, Any]],
) -> list[list[float] | None]:
    analysis_faces = load_face_analysis_faces(face_encoder, image_path)
    matched_faces: set[int] = set()
    embeddings: list[list[float] | None] = []

    for face in faces:
        bbox = face.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            embeddings.append(None)
            continue

        best_index = None
        best_score = 0.0
        for index, analysis_face in enumerate(analysis_faces):
            if index in matched_faces:
                continue

            analysis_bbox = getattr(analysis_face, "bbox", None)
            if analysis_bbox is None:
                continue

            score = bbox_iou(bbox, analysis_bbox)
            if score > best_score:
                best_index = index
                best_score = score

        if best_index is None or best_score < 0.25:
            embeddings.append(None)
            continue

        matched_faces.add(best_index)
        embeddings.append(
            [float(value) for value in analysis_faces[best_index].normed_embedding]
        )

    return embeddings


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    import numpy as np

    left_vec = np.asarray(left, dtype=np.float32)
    right_vec = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left_vec, right_vec) / denominator)


def weighted_centroid(embedding: Sequence[float], existing_centroid: Sequence[float], existing_count: int) -> list[float]:
    import numpy as np

    new_vec = np.asarray(embedding, dtype=np.float32)
    old_vec = np.asarray(existing_centroid, dtype=np.float32)
    combined = (old_vec * existing_count + new_vec) / float(existing_count + 1)
    norm = float(np.linalg.norm(combined))
    if norm == 0.0:
        return [float(value) for value in new_vec]
    return [float(value) for value in (combined / norm)]


def assign_face_to_person(
    people: list[dict[str, Any]],
    embedding: Sequence[float],
    face_record: dict[str, Any],
    threshold: float,
) -> int:
    person_id, _ = assign_face_to_person_with_next_id(people, embedding, face_record, threshold, len(people) + 1)
    return person_id


def assign_face_to_person_with_next_id(
    people: list[dict[str, Any]],
    embedding: Sequence[float],
    face_record: dict[str, Any],
    threshold: float,
    next_person_id: int,
) -> tuple[int, int]:
    best_person_index = -1
    best_similarity = -1.0

    for index, person in enumerate(people):
        similarity = cosine_similarity(embedding, person["centroid"])
        if similarity > best_similarity:
            best_similarity = similarity
            best_person_index = index

    if best_person_index >= 0 and best_similarity >= threshold:
        person = people[best_person_index]
        person["centroid"] = weighted_centroid(embedding, person["centroid"], person["face_count"])
        person["face_count"] += 1
        person["faces"].append(face_record)
        return int(person["person_id"]), next_person_id

    person_id = next_person_id
    people.append(
        {
            "person_id": person_id,
            "face_count": 1,
            "centroid": [float(value) for value in embedding],
            "faces": [face_record],
        }
    )
    return person_id, next_person_id + 1


def annotated_output_path(root: Path, image_path: Path, annotated_dir: Path) -> Path:
    if root.is_dir():
        relative_path = image_path.relative_to(root)
    else:
        relative_path = image_path.name

    relative_path = Path(relative_path)
    return annotated_dir / relative_path.with_name(f"{relative_path.stem}_faces{relative_path.suffix}")


def build_report(
    model: YOLO,
    face_encoder,
    image_roots: Sequence[Path],
    recursive: bool,
    confidence: float,
    annotated_dir: Path | None,
    group_by_person: bool,
    person_threshold: float,
    report: dict[str, Any],
    image_index: dict[str, dict[str, Any]],
    people: list[dict[str, Any]],
) -> dict:
    if group_by_person:
        report["people"] = people
        next_person_id = int(report.get("next_person_id", len(people) + 1))
    else:
        next_person_id = 1

    for root, image_path in iter_images(image_roots, recursive):
        image_key = normalize_image_key(image_path)
        cached_entry = image_index.get(image_key)
        if cached_entry is not None:
            if annotated_dir is not None:
                cached_faces = cached_entry.get("faces", [])
                annotated_path = annotated_output_path(root, image_path, annotated_dir)
                render_faces(image_path, cached_faces, annotated_path)
                cached_entry["annotated_image"] = str(annotated_path)
            continue

        detections = detect_faces(model, image_path, confidence)
        faces = detections_to_faces(detections)
        if group_by_person:
            embeddings = match_embeddings_to_detections(face_encoder, image_path, faces)
            for face_index, (face, embedding) in enumerate(zip(faces, embeddings)):
                if embedding is None:
                    continue
                face_record = {
                    "image": normalize_image_key(image_path),
                    "face_index": face_index,
                    "bbox": face["bbox"],
                    "confidence": face["confidence"],
                }
                person_id, next_person_id = assign_face_to_person_with_next_id(
                    people,
                    embedding,
                    face_record,
                    person_threshold,
                    next_person_id,
                )
                face["person_id"] = person_id

        annotated_path = None
        if annotated_dir is not None:
            annotated_path = annotated_output_path(root, image_path, annotated_dir)
            render_faces(image_path, faces, annotated_path)

        entry = {
            "image": normalize_image_key(image_path),
            "face_count": len(faces),
            "faces": faces,
            "annotated_image": str(annotated_path) if annotated_path is not None else None,
        }
        report["images"].append(entry)
        image_index[image_key] = entry

    if group_by_person:
        report["next_person_id"] = next_person_id

    return report


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "inputs",
    nargs=-1,
    type=click.Path(path_type=Path, exists=True, readable=True),
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    help="Write the JSON report to a file instead of stdout.",
)
@click.option(
    "--annotated-dir",
    type=click.Path(path_type=Path),
    help="Write annotated images to this directory.",
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    show_default=True,
    help="Walk directories recursively when scanning for images.",
)
@click.option(
    "--confidence",
    default=0.25,
    show_default=True,
    type=click.FloatRange(min=0.0, max=1.0),
    help="Discard detections below this confidence.",
)
@click.option(
    "--model-repo",
    default=DEFAULT_MODEL_REPO,
    show_default=True,
    help="Hugging Face repository containing the model weights.",
)
@click.option(
    "--model-file",
    default=DEFAULT_MODEL_FILE,
    show_default=True,
    help="Filename inside the model repository.",
)
@click.option(
    "--group-by-person/--no-group-by-person",
    default=False,
    show_default=True,
    help="Cluster face embeddings so repeated people are grouped together.",
)
@click.option(
    "--embedding-model",
    default=DEFAULT_EMBEDDING_MODEL,
    show_default=True,
    help="InsightFace face recognition model name used to generate embeddings.",
)
@click.option(
    "--person-threshold",
    default=DEFAULT_PERSON_THRESHOLD,
    show_default=True,
    type=click.FloatRange(min=0.0, max=1.0),
    help="Minimum cosine similarity required to assign a face to an existing person group.",
)
def cli(
    inputs: Tuple[Path, ...],
    output_path: Path | None,
    annotated_dir: Path | None,
    recursive: bool,
    confidence: float,
    model_repo: str,
    model_file: str,
    group_by_person: bool,
    embedding_model: str,
    person_threshold: float,
) -> None:
    if not inputs:
        raise click.ClickException("Provide at least one image or directory.")

    model = load_model(model_repo, model_file)
    face_encoder = load_face_encoder(embedding_model) if group_by_person else None

    report, image_index, people = (
        load_existing_report(
            output_path,
            model_repo,
            model_file,
            confidence,
            group_by_person,
            embedding_model,
            person_threshold,
        )
        if output_path is not None
        else (
            default_report(
                model_repo,
                model_file,
                confidence,
                group_by_person,
                embedding_model,
                person_threshold,
            ),
            {},
            [],
        )
    )

    report = build_report(
        model=model,
        face_encoder=face_encoder,
        image_roots=inputs,
        recursive=recursive,
        confidence=confidence,
        annotated_dir=annotated_dir,
        group_by_person=group_by_person,
        person_threshold=person_threshold,
        report=report,
        image_index=image_index,
        people=people,
    )

    if not report["images"]:
        raise click.ClickException("No image files were found in the provided inputs.")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        click.echo(f"Wrote JSON report to {output_path}")
    else:
        click.echo(json.dumps(report, indent=2))

    image_count = len(report["images"])
    face_count = sum(image["face_count"] for image in report["images"])
    click.echo(f"Processed {image_count} image(s) and found {face_count} face(s).")

    if annotated_dir is not None:
        click.echo(f"Annotated images were written under {annotated_dir}")
