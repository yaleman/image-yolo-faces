from __future__ import annotations

import colorsys
import hashlib
import warnings
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

if TYPE_CHECKING:
    from supervision import Detections
    from ultralytics import YOLO

DEFAULT_MODEL_REPO = "arnabdhar/YOLOv8-Face-Detection"
DEFAULT_MODEL_FILE = "model.pt"
DEFAULT_EMBEDDING_MODEL = "buffalo_l"
DEFAULT_PERSON_THRESHOLD = 0.45
DEFAULT_PERSON_GROUPING_STRATEGY = "full-image-iou-v1"
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}


def normalize_image_key(image_path: Path | str, base_dir: Path | None = None) -> str:
    path = Path(image_path)
    if base_dir is not None:
        return path.name
    return str(path.resolve())


def stored_media_path(path: Path, storage_root: Path | None = None) -> str:
    resolved = path.resolve()
    if storage_root is None:
        return str(resolved)

    return path.name


def normalize_image_entry(entry: dict[str, Any]) -> None:
    entry.setdefault("faces", [])
    entry.setdefault("hashes", {})

    if not isinstance(entry.get("faces"), list):
        entry["faces"] = []

    hashes = entry.get("hashes")
    if not isinstance(hashes, dict):
        entry["hashes"] = {}
        return

    normalized_hashes: dict[str, str] = {}
    for key, value in hashes.items():
        if (
            isinstance(key, str)
            and isinstance(value, str)
            and key.strip()
            and value.strip()
        ):
            normalized_hashes[key.strip()] = value.strip()
    entry["hashes"] = normalized_hashes


def image_added_at(entry: dict[str, Any]) -> int:
    added_at = entry.get("added_at")
    if isinstance(added_at, int) and added_at >= 0:
        return added_at
    return 0


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hashes_for_file(path: Path) -> dict[str, str]:
    return {"sha256": sha256_file(path)}


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


def draw_faces(
    image_path: Path,
    faces: Sequence[dict[str, Any]],
    person_labels: Mapping[int, str] | None = None,
) -> Any:
    from PIL import Image, ImageDraw, ImageFont

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
            (
                text_left,
                text_top,
                text_left + text_width + 8,
                text_top + text_height + 6,
            ),
            fill=fill,
        )
        drawer.text((text_left + 4, text_top + 3), label, fill=text_fill, font=font)

    return image


def render_faces_bytes(
    image_path: Path,
    faces: Sequence[dict[str, Any]],
    person_labels: Mapping[int, str] | None = None,
) -> bytes:
    image = draw_faces(image_path, faces, person_labels=person_labels)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


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

    intersection_area = (intersection_x2 - intersection_x1) * (
        intersection_y2 - intersection_y1
    )
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


def weighted_centroid(
    embedding: Sequence[float], existing_centroid: Sequence[float], existing_count: int
) -> list[float]:
    import numpy as np

    new_vec = np.asarray(embedding, dtype=np.float32)
    old_vec = np.asarray(existing_centroid, dtype=np.float32)
    combined = (old_vec * existing_count + new_vec) / float(existing_count + 1)
    norm = float(np.linalg.norm(combined))
    if norm == 0.0:
        return [float(value) for value in new_vec]
    return [float(value) for value in (combined / norm)]


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
        person["centroid"] = weighted_centroid(
            embedding,
            person["centroid"],
            person["face_count"],
        )
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
    return annotated_dir / relative_path.with_name(
        f"{relative_path.stem}_faces{relative_path.suffix}"
    )


def scan_image_entry(
    model: YOLO,
    face_encoder,
    image_path: Path,
    confidence: float,
    added_at_ns: int,
    group_by_person: bool,
    person_threshold: float,
    people: list[dict[str, Any]],
    next_person_id: int,
    storage_root: Path | None = None,
) -> tuple[dict[str, Any], int]:
    detections = detect_faces(model, image_path, confidence)
    faces = detections_to_faces(detections)
    stored_image_value = stored_media_path(image_path, storage_root)

    if group_by_person:
        embeddings = match_embeddings_to_detections(face_encoder, image_path, faces)
        for face_index, (face, embedding) in enumerate(zip(faces, embeddings)):
            if embedding is None:
                continue
            face_record = {
                "image": stored_image_value,
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

    return (
        {
            "image": stored_image_value,
            "face_count": len(faces),
            "faces": faces,
            "added_at": added_at_ns,
            "hashes": hashes_for_file(image_path),
        },
        next_person_id,
    )
