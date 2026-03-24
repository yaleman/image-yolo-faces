from __future__ import annotations

import html
import json
import os
import threading
import time
from io import BytesIO
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote
from urllib.parse import urlencode

import click
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .ingest import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL_FILE,
    DEFAULT_MODEL_REPO,
    DEFAULT_PERSON_THRESHOLD,
    IMAGE_EXTENSIONS,
    hashes_for_file,
    load_face_encoder,
    load_model,
    normalize_image_entry,
    render_faces,
    scan_image_entry,
    sha256_bytes,
    weighted_centroid,
)

DEFAULT_REPORT_PATH = Path("faces.json")
REPORT_PATH_ENV = "IMAGE_YOLO_FACES_REPORT_PATH"


def image_key(image_path: str | Path) -> str:
    return sha1(str(Path(image_path).resolve()).encode("utf-8")).hexdigest()


def resolve_report_path(report_path: Path) -> Path:
    return report_path.expanduser().resolve()


def active_report_path(report_path: Path | None = None) -> Path:
    if report_path is not None:
        return resolve_report_path(report_path)

    env_report_path = os.environ.get(REPORT_PATH_ENV)
    if env_report_path:
        return resolve_report_path(Path(env_report_path))

    return resolve_report_path(DEFAULT_REPORT_PATH)


def resolve_media_path(report_path: Path, value: str | None) -> Path | None:
    if not value:
        return None

    path = Path(value)
    if path.is_absolute():
        return path
    return (report_path.parent / path).resolve()


def media_version(path: Path | None) -> str:
    if path is None:
        return "0"
    try:
        return str(path.stat().st_mtime_ns)
    except FileNotFoundError:
        return "0"


def media_url(path: Path | None, route: str) -> str:
    return f"{route}?v={media_version(path)}"


def placeholder_media(label: str) -> str:
    safe_label = html.escape(label[:48] or "Preview")
    svg = placeholder_svg(safe_label)
    return f"data:image/svg+xml;charset=utf-8,{quote(svg)}"


def placeholder_svg(label: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" role="img" aria-label="{label}">
  <rect width="800" height="600" fill="#efe6db"/>
  <rect x="40" y="40" width="720" height="520" rx="28" fill="#f9f4ee" stroke="#d9cbbc" stroke-width="6"/>
  <text x="50%" y="50%" text-anchor="middle" dominant-baseline="middle" font-family="sans-serif" font-size="34" fill="#6e6356">{label}</text>
</svg>"""


def placeholder_svg_bytes(label: str) -> bytes:
    return placeholder_svg(html.escape(label[:48] or "Preview")).encode("utf-8")


def media_url_for_paths(route: str, *paths: Path | None) -> str:
    if not paths:
        return f"{route}?v=0"
    version = "-".join(media_version(path) for path in paths)
    return f"{route}?v={version}"


def with_query(url: str, query: str) -> str:
    cleaned_query = query.strip()
    if not cleaned_query:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urlencode({'q': cleaned_query})}"


def report_relative_path(report_path: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(report_path.parent))
    except ValueError:
        return str(path.resolve())


def frontend_assets(static_dir: Path) -> dict[str, Any]:
    manifest_path = static_dir / "dist" / "manifest.json"
    if not manifest_path.exists():
        return {"css": [], "js": None}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest.get("frontend/main.ts")
    if not isinstance(entry, dict):
        return {"css": [], "js": None}

    css_values = entry.get("css", [])
    css_files = [
        f"/static/dist/{css_file}"
        for css_file in css_values
        if isinstance(css_file, str) and css_file
    ]
    js_file = entry.get("file")
    return {
        "css": css_files,
        "js": f"/static/dist/{js_file}"
        if isinstance(js_file, str) and js_file
        else None,
    }


def guess_image_suffix(filename: str, content: bytes) -> tuple[str, str]:
    from PIL import Image, UnidentifiedImageError

    candidate_name = Path(filename or "upload").name
    stem = Path(candidate_name).stem or "upload"
    suffix = Path(candidate_name).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return stem, suffix

    format_suffix_map = {
        "BMP": ".bmp",
        "GIF": ".gif",
        "JPEG": ".jpg",
        "PNG": ".png",
        "TIFF": ".tiff",
        "WEBP": ".webp",
    }

    try:
        with Image.open(BytesIO(content)) as image_file:
            image_format = image_file.format
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400, detail="Upload must be a supported image."
        ) from exc
    if not isinstance(image_format, str):
        raise HTTPException(status_code=400, detail="Upload must be a supported image.")

    inferred_suffix = format_suffix_map.get(image_format.upper())
    if inferred_suffix is None:
        raise HTTPException(status_code=400, detail="Upload must be a supported image.")

    return stem, inferred_suffix


def uniquify_filename(destination_dir: Path, stem: str, suffix: str) -> Path:
    candidate = destination_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate

    timestamp = str(int(time.time()))
    candidate = destination_dir / f"{stem}-{timestamp}{suffix}"
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = destination_dir / f"{stem}-{timestamp}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def face_bbox_area(bbox: list[float]) -> float:
    left, top, right, bottom = bbox
    return max(0.0, right - left) * max(0.0, bottom - top)


def representative_face_bbox(faces: list[dict[str, Any]]) -> list[float] | None:
    best_bbox: list[float] | None = None
    best_score: tuple[float, float] = (-1.0, -1.0)

    for face in faces:
        bbox = face.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        try:
            cleaned_bbox = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue

        confidence = face.get("confidence")
        confidence_value = (
            float(confidence) if isinstance(confidence, (int, float)) else -1.0
        )
        score = (confidence_value, face_bbox_area(cleaned_bbox))
        if score > best_score:
            best_score = score
            best_bbox = cleaned_bbox

    return best_bbox


def preview_crop_box(
    bbox: list[float],
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.28,
    min_padding: int = 24,
) -> tuple[int, int, int, int] | None:
    left, top, right, bottom = bbox
    face_width = max(1.0, right - left)
    face_height = max(1.0, bottom - top)
    side = max(face_width, face_height)
    padding = max(float(min_padding), side * padding_ratio)
    side += padding * 2
    side = min(side, float(min(image_width, image_height)))
    if side <= 0:
        return None

    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0

    crop_left = round(center_x - side / 2.0)
    crop_top = round(center_y - side / 2.0)
    crop_right = round(crop_left + side)
    crop_bottom = round(crop_top + side)

    if crop_left < 0:
        crop_right -= crop_left
        crop_left = 0
    if crop_top < 0:
        crop_bottom -= crop_top
        crop_top = 0
    if crop_right > image_width:
        shift = crop_right - image_width
        crop_left = max(0, crop_left - shift)
        crop_right = image_width
    if crop_bottom > image_height:
        shift = crop_bottom - image_height
        crop_top = max(0, crop_top - shift)
        crop_bottom = image_height

    if crop_right - crop_left <= 0 or crop_bottom - crop_top <= 0:
        return None

    return crop_left, crop_top, crop_right, crop_bottom


def preview_image_bytes(image_path: Path, bbox: list[float]) -> bytes | None:
    from PIL import Image, ImageOps

    try:
        with Image.open(image_path) as image_file:
            image = ImageOps.exif_transpose(image_file).convert("RGB")
    except OSError:
        return None

    crop_box = preview_crop_box(bbox, image.width, image.height)
    if crop_box is None:
        return None

    preview = image.crop(crop_box)
    preview.thumbnail((384, 384), Image.Resampling.LANCZOS)

    buffer = BytesIO()
    preview.save(buffer, format="JPEG", quality=90, optimize=True)
    return buffer.getvalue()


def normalize_person(person: dict[str, Any]) -> None:
    person.setdefault("aliases", [])
    person.setdefault("faces", [])

    if not isinstance(person.get("aliases"), list):
        person["aliases"] = []
    if not isinstance(person.get("faces"), list):
        person["faces"] = []


def normalize_report(report: dict[str, Any]) -> dict[str, Any]:
    images = report.get("images", [])
    if not isinstance(images, list):
        images = []
    report["images"] = images
    for image in images:
        if isinstance(image, dict):
            normalize_image_entry(image)

    people = report.get("people", [])
    if not isinstance(people, list):
        people = []
    report["people"] = people

    max_person_id = 0
    for person in people:
        if not isinstance(person, dict):
            continue
        normalize_person(person)
        try:
            max_person_id = max(max_person_id, int(person["person_id"]))
        except (KeyError, TypeError, ValueError):
            continue

    next_person_id = report.get("next_person_id")
    if not isinstance(next_person_id, int) or next_person_id <= max_person_id:
        report["next_person_id"] = max_person_id + 1

    return report


def load_report(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        return normalize_report({"images": [], "people": [], "next_person_id": 1})

    try:
        raw_report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"Report at {report_path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(raw_report, dict):
        raise click.ClickException(
            f"Report at {report_path} must contain a JSON object."
        )

    return normalize_report(raw_report)


def person_display_name(
    person: dict[str, Any] | None, fallback_id: int | None = None
) -> str:
    if person is None:
        return f"Person {fallback_id}" if fallback_id is not None else "Unknown person"

    name = person.get("name")
    if isinstance(name, str):
        stripped = name.strip()
        if stripped:
            return stripped

    if fallback_id is None:
        try:
            fallback_id = int(person["person_id"])
        except (KeyError, TypeError, ValueError):
            return "Unknown person"

    return f"Person {fallback_id}"


def person_option_label(person: dict[str, Any]) -> str:
    try:
        person_id = int(person["person_id"])
    except (KeyError, TypeError, ValueError):
        return "Unknown person"

    name = person_display_name(person, person_id)
    if name == f"Person {person_id}":
        return name
    return f"{name} (#{person_id})"


def person_sort_key(person_id: int, person: dict[str, Any] | None) -> tuple[str, int]:
    return (person_display_name(person, person_id).casefold(), person_id)


def person_search_text(person_id: int, person: dict[str, Any] | None) -> str:
    if person is None:
        return str(person_id)

    values = [person_display_name(person, person_id), str(person_id)]
    aliases = person.get("aliases", [])
    if isinstance(aliases, list):
        values.extend(
            str(alias) for alias in aliases if isinstance(alias, str) and alias.strip()
        )
    return " ".join(values).casefold()


def person_matches_search(
    person_id: int, person: dict[str, Any] | None, query: str
) -> bool:
    cleaned_query = query.strip().casefold()
    if not cleaned_query:
        return False
    return cleaned_query in person_search_text(person_id, person)


def image_matches_search(
    store: ReportStore,
    entry: dict[str, Any],
    query: str,
    person_lookup: dict[int, dict[str, Any]] | None = None,
) -> bool:
    cleaned_query = query.strip().casefold()
    if not cleaned_query:
        return True

    image_value = entry.get("image")
    if isinstance(image_value, str) and cleaned_query in image_value.casefold():
        return True

    image_name = (
        Path(str(image_value)).name.casefold() if isinstance(image_value, str) else ""
    )
    if image_name and cleaned_query in image_name:
        return True

    faces = entry.get("faces", [])
    if not isinstance(faces, list):
        return False

    lookup = person_lookup if person_lookup is not None else store.person_index()
    seen_person_ids: set[int] = set()
    for face in faces:
        if not isinstance(face, dict):
            continue
        try:
            person_id = int(face["person_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if person_id in seen_person_ids:
            continue
        seen_person_ids.add(person_id)
        if cleaned_query in person_search_text(person_id, lookup.get(person_id)):
            return True

    return False


def face_summary(face: dict[str, Any]) -> str:
    parts: list[str] = []
    confidence = face.get("confidence")
    if isinstance(confidence, (int, float)):
        parts.append(f"{confidence:.2f}")

    bbox = face.get("bbox", [])
    if isinstance(bbox, list) and len(bbox) == 4:
        parts.append(", ".join(f"{float(value):.0f}" for value in bbox))

    return " · ".join(parts)


@dataclass
class ReportStore:
    report_path: Path
    report: dict[str, Any]
    lock: threading.RLock = field(default_factory=threading.RLock)
    _model: Any | None = field(default=None, init=False, repr=False)
    _face_encoder: Any | None = field(default=None, init=False, repr=False)

    @classmethod
    def open(cls, report_path: Path) -> "ReportStore":
        return cls(report_path=report_path, report=load_report(report_path))

    def save(self) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(self.report, indent=2), encoding="utf-8")

    def ensure_hashes_backfilled(self) -> bool:
        changed = False
        for image in self.report["images"]:
            if not isinstance(image, dict):
                continue
            normalize_image_entry(image)
            hashes = image.get("hashes", {})
            if isinstance(hashes, dict) and isinstance(hashes.get("sha256"), str):
                continue

            image_value = image.get("image")
            if not isinstance(image_value, str):
                continue

            image_path = Path(image_value)
            if not image_path.exists():
                continue

            image["hashes"] = hashes_for_file(image_path)
            changed = True

        if changed:
            self.save()
        return changed

    def find_image_by_hash(self, algorithm: str, digest: str) -> dict[str, Any] | None:
        for image in self.report["images"]:
            if not isinstance(image, dict):
                continue
            hashes = image.get("hashes")
            if not isinstance(hashes, dict):
                continue
            if hashes.get(algorithm) == digest:
                return image
        return None

    def _common_parent_dir(self, values: list[str]) -> Path | None:
        if not values:
            return None

        resolved_paths = [str(Path(value).resolve()) for value in values]
        try:
            common = Path(os.path.commonpath(resolved_paths))
        except ValueError:
            return None

        if common == self.report_path.parent or str(common) == common.anchor:
            return None
        return common

    def inferred_images_dir(self) -> Path:
        image_values = [
            str(image["image"])
            for image in self.report["images"]
            if isinstance(image, dict) and isinstance(image.get("image"), str)
        ]
        return self._common_parent_dir(image_values) or (
            self.report_path.parent / "photos"
        )

    def inferred_annotated_dir(self) -> Path:
        annotated_values: list[str] = []
        for image in self.report["images"]:
            if not isinstance(image, dict):
                continue
            annotated_value = image.get("annotated_image")
            resolved = resolve_media_path(
                self.report_path,
                annotated_value if isinstance(annotated_value, str) else None,
            )
            if resolved is not None:
                annotated_values.append(str(resolved))
        return self._common_parent_dir(annotated_values) or (
            self.report_path.parent / "annotated"
        )

    def model_config(self) -> tuple[str, str, float]:
        model_repo = self.report.get("model_repo")
        model_file = self.report.get("model_file")
        confidence = self.report.get("confidence_threshold")
        return (
            model_repo
            if isinstance(model_repo, str) and model_repo.strip()
            else DEFAULT_MODEL_REPO,
            model_file
            if isinstance(model_file, str) and model_file.strip()
            else DEFAULT_MODEL_FILE,
            float(confidence) if isinstance(confidence, (int, float)) else 0.25,
        )

    def grouping_config(self) -> tuple[bool, str, float]:
        group_by_person = self.report.get("group_by_person") is True
        embedding_model = self.report.get("embedding_model")
        threshold = self.report.get("person_similarity_threshold")
        return (
            group_by_person,
            embedding_model
            if isinstance(embedding_model, str) and embedding_model.strip()
            else DEFAULT_EMBEDDING_MODEL,
            float(threshold)
            if isinstance(threshold, (int, float))
            else DEFAULT_PERSON_THRESHOLD,
        )

    def get_model(self):
        if self._model is None:
            model_repo, model_file, _ = self.model_config()
            self._model = load_model(model_repo, model_file)
        return self._model

    def get_face_encoder(self):
        group_by_person, embedding_model, _ = self.grouping_config()
        if not group_by_person:
            return None
        if self._face_encoder is None:
            self._face_encoder = load_face_encoder(embedding_model)
        return self._face_encoder

    def image_index(self) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for image in self.report["images"]:
            if not isinstance(image, dict):
                continue
            image_value = image.get("image")
            if not isinstance(image_value, str):
                continue
            index[image_key(image_value)] = image
        return index

    def person_index(self) -> dict[int, dict[str, Any]]:
        index: dict[int, dict[str, Any]] = {}
        for person in self.report["people"]:
            if not isinstance(person, dict):
                continue
            try:
                person_id = int(person["person_id"])
            except (KeyError, TypeError, ValueError):
                continue
            index[person_id] = person
        return index

    def image_paths_for_person(self, person_id: int) -> set[str]:
        paths: set[str] = set()
        for person in self.report["people"]:
            if not isinstance(person, dict):
                continue
            try:
                current_id = int(person["person_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if current_id != person_id:
                continue
            for face in person.get("faces", []):
                if not isinstance(face, dict):
                    continue
                image_value = face.get("image")
                if isinstance(image_value, str):
                    paths.add(image_value)
            break
        return paths

    def display_name_map(self) -> dict[int, str]:
        names: dict[int, str] = {}
        for person_id, person in self.person_index().items():
            names[person_id] = person_display_name(person, person_id)
        return names

    def person_image_groups(self, person_id: int) -> list[dict[str, Any]]:
        person = self.person_index().get(person_id)
        if person is None:
            return []

        groups: dict[str, dict[str, Any]] = {}
        for face in person.get("faces", []):
            if not isinstance(face, dict):
                continue
            image_value = face.get("image")
            if not isinstance(image_value, str):
                continue
            group = groups.setdefault(
                image_value,
                {
                    "image": image_value,
                    "image_id": image_key(image_value),
                    "faces": [],
                    "face_count": 0,
                },
            )
            group["faces"].append(face)
            group["face_count"] += 1

        return sorted(
            groups.values(),
            key=lambda group: (
                Path(group["image"]).name.lower(),
                str(group["image"]).lower(),
            ),
        )

    def re_render_images(self, image_paths: set[str]) -> None:
        if not image_paths:
            return

        image_lookup = self.image_index()
        person_labels = self.display_name_map()

        for image_value in image_paths:
            image_id = image_key(image_value)
            entry = image_lookup.get(image_id)
            if entry is None:
                continue

            annotated_value = entry.get("annotated_image")
            annotated_path = resolve_media_path(
                self.report_path,
                annotated_value if isinstance(annotated_value, str) else None,
            )
            if annotated_path is None:
                continue

            original_path = Path(image_value)
            if not original_path.exists():
                continue

            faces = entry.get("faces", [])
            if not isinstance(faces, list):
                faces = []

            render_faces(
                original_path,
                faces,
                annotated_path,
                person_labels=person_labels,
                force=True,
            )

    def ensure_annotated_image(self, entry: dict[str, Any]) -> Path | None:
        image_value = entry.get("image")
        annotated_value = entry.get("annotated_image")
        if not isinstance(image_value, str) or not isinstance(annotated_value, str):
            return None

        annotated_path = resolve_media_path(self.report_path, annotated_value)
        if annotated_path is None:
            return None
        if annotated_path.exists():
            return annotated_path

        original_path = Path(image_value)
        if not original_path.exists():
            return None

        faces = entry.get("faces", [])
        if not isinstance(faces, list):
            faces = []

        render_faces(
            original_path,
            faces,
            annotated_path,
            person_labels=self.display_name_map(),
            force=True,
        )
        if annotated_path.exists():
            return annotated_path
        return None

    def import_uploaded_image(self, filename: str, content: bytes) -> dict[str, str]:
        if not content:
            raise HTTPException(status_code=400, detail="Upload was empty.")

        stem, suffix = guess_image_suffix(filename, content)
        digest = sha256_bytes(content)
        existing = self.find_image_by_hash("sha256", digest)
        if existing is not None:
            existing_image = existing.get("image")
            if not isinstance(existing_image, str):
                raise HTTPException(
                    status_code=500, detail="Existing image is invalid."
                )
            existing_id = image_key(existing_image)
            return {
                "status": "duplicate",
                "image_id": existing_id,
                "detail_url": f"/images/{existing_id}",
            }

        destination_dir = self.inferred_images_dir()
        destination_dir.mkdir(parents=True, exist_ok=True)
        image_path = uniquify_filename(destination_dir, stem, suffix)
        image_path.write_bytes(content)

        annotated_dir = self.inferred_annotated_dir()
        annotated_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = annotated_dir / f"{image_path.stem}_faces{image_path.suffix}"

        try:
            _, _, confidence = self.model_config()
            group_by_person, _, person_threshold = self.grouping_config()
            face_encoder = self.get_face_encoder()
            people = self.report["people"] if group_by_person else []
            current_next_id = self.report.get("next_person_id")
            if not isinstance(current_next_id, int) or current_next_id < 1:
                current_next_id = max(self.person_index().keys(), default=0) + 1

            entry, next_person_id = scan_image_entry(
                model=self.get_model(),
                face_encoder=face_encoder,
                image_path=image_path,
                confidence=confidence,
                annotated_path=annotated_path,
                group_by_person=group_by_person,
                person_threshold=person_threshold,
                people=people,
                next_person_id=current_next_id,
            )
            entry["hashes"]["sha256"] = digest
            entry["annotated_image"] = report_relative_path(
                self.report_path, annotated_path
            )
            self.report["images"].append(entry)
            if group_by_person:
                self.report["people"] = people
                self.report["next_person_id"] = next_person_id

            self.save()
            image_id = image_key(entry["image"])
            return {
                "status": "imported",
                "image_id": image_id,
                "detail_url": f"/images/{image_id}",
            }
        except Exception:
            if image_path.exists():
                image_path.unlink()
            if annotated_path.exists():
                annotated_path.unlink()
            raise

    def delete_image(self, image_id: str) -> None:
        image_lookup = self.image_index()
        entry = image_lookup.get(image_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Image not found.")

        image_value = entry.get("image")
        if not isinstance(image_value, str):
            raise HTTPException(status_code=404, detail="Image path missing.")

        original_path = Path(image_value)
        annotated_value = entry.get("annotated_image")
        annotated_path = resolve_media_path(
            self.report_path,
            annotated_value if isinstance(annotated_value, str) else None,
        )

        for path in [annotated_path, original_path]:
            if path is None or not path.exists():
                continue
            try:
                path.unlink()
            except OSError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to delete file {path}.",
                ) from exc

        self.report["images"] = [
            image for image in self.report["images"] if image is not entry
        ]

        remaining_people: list[dict[str, Any]] = []
        for person in self.report["people"]:
            if not isinstance(person, dict):
                continue
            faces = person.get("faces", [])
            if not isinstance(faces, list):
                faces = []

            remaining_faces = [
                face
                for face in faces
                if not (
                    isinstance(face, dict)
                    and isinstance(face.get("image"), str)
                    and face.get("image") == image_value
                )
            ]
            if not remaining_faces:
                continue

            person["faces"] = remaining_faces
            person["face_count"] = len(remaining_faces)
            remaining_people.append(person)

        self.report["people"] = remaining_people
        current_next_id = self.report.get("next_person_id")
        max_person_id = max(self.person_index().keys(), default=0)
        if isinstance(current_next_id, int):
            self.report["next_person_id"] = max(current_next_id, max_person_id + 1)
        else:
            self.report["next_person_id"] = max_person_id + 1

        self.save()

    def rename_person(self, person_id: int, name: str) -> set[str]:
        person = self.person_index().get(person_id)
        if person is None:
            raise HTTPException(
                status_code=404, detail=f"Person {person_id} was not found."
            )

        cleaned_name = name.strip()
        if cleaned_name:
            person["name"] = cleaned_name
        else:
            person.pop("name", None)

        return self.image_paths_for_person(person_id)

    def merge_people(self, source_id: int, target_id: int) -> set[str]:
        if source_id == target_id:
            return self.image_paths_for_person(source_id)

        people = self.person_index()
        source = people.get(source_id)
        target = people.get(target_id)
        if source is None:
            raise HTTPException(
                status_code=404, detail=f"Person {source_id} was not found."
            )
        if target is None:
            raise HTTPException(
                status_code=404, detail=f"Person {target_id} was not found."
            )

        source_faces = [
            face for face in source.get("faces", []) if isinstance(face, dict)
        ]
        target_faces = [
            face for face in target.get("faces", []) if isinstance(face, dict)
        ]
        affected_images = {
            str(face["image"])
            for face in source_faces + target_faces
            if isinstance(face.get("image"), str)
        }

        source_name_value = source.get("name")
        target_name_value = target.get("name")
        source_name = (
            source_name_value.strip() if isinstance(source_name_value, str) else ""
        )
        target_name = (
            target_name_value.strip() if isinstance(target_name_value, str) else ""
        )
        if source_name:
            aliases = target.setdefault("aliases", [])
            if target_name:
                if source_name != target_name and source_name not in aliases:
                    aliases.append(source_name)
            else:
                target["name"] = source_name

        target_count = int(target.get("face_count", len(target_faces)))
        source_count = int(source.get("face_count", len(source_faces)))
        target_centroid = target.get("centroid")
        source_centroid = source.get("centroid")
        if isinstance(target_centroid, list) and isinstance(source_centroid, list):
            target["centroid"] = weighted_centroid(
                source_centroid, target_centroid, target_count
            )
        target["face_count"] = target_count + source_count
        target["faces"] = target_faces + source_faces

        for image in self.report["images"]:
            if not isinstance(image, dict):
                continue
            faces = image.get("faces", [])
            if not isinstance(faces, list):
                continue
            for face in faces:
                if isinstance(face, dict) and face.get("person_id") == source_id:
                    face["person_id"] = target_id

        self.report["people"] = [
            person for person in self.report["people"] if person is not source
        ]

        current_next_id = self.report.get("next_person_id")
        if not isinstance(current_next_id, int):
            current_next_id = 1
        self.report["next_person_id"] = max(
            current_next_id, max(self.person_index().keys(), default=0) + 1
        )

        return affected_images

    def split_person_images_to_new_person(
        self, person_id: int, selected_images: set[str], new_name: str
    ) -> int:
        people = self.person_index()
        source = people.get(person_id)
        if source is None:
            raise HTTPException(
                status_code=404, detail=f"Person {person_id} was not found."
            )

        source_faces = [
            face for face in source.get("faces", []) if isinstance(face, dict)
        ]
        source_image_paths = {
            str(face["image"])
            for face in source_faces
            if isinstance(face.get("image"), str)
        }
        selected_images = {
            image for image in selected_images if image in source_image_paths
        }
        if not selected_images:
            raise HTTPException(
                status_code=400, detail="Select at least one image to split."
            )

        moved_faces = [
            face for face in source_faces if str(face.get("image")) in selected_images
        ]
        if not moved_faces:
            raise HTTPException(
                status_code=400,
                detail="No matching faces were found for the selected images.",
            )

        remaining_faces = [
            face
            for face in source_faces
            if str(face.get("image")) not in selected_images
        ]
        current_next_id = self.report.get("next_person_id")
        if not isinstance(current_next_id, int) or current_next_id < 1:
            current_next_id = max(self.person_index().keys(), default=0) + 1

        new_person_id = current_next_id
        source_name_value = source.get("name")
        source_name = (
            source_name_value.strip() if isinstance(source_name_value, str) else ""
        )
        cleaned_new_name = new_name.strip()

        source_centroid = source.get("centroid")
        new_person: dict[str, Any] = {
            "person_id": new_person_id,
            "face_count": len(moved_faces),
            "faces": [],
            "centroid": list(source_centroid)
            if isinstance(source_centroid, list)
            else [],
        }
        if cleaned_new_name:
            new_person["name"] = cleaned_new_name
        elif not remaining_faces and source_name:
            new_person["name"] = source_name

        moved_faces_copy: list[dict[str, Any]] = []
        for face in moved_faces:
            face_copy = dict(face)
            face_copy["person_id"] = new_person_id
            moved_faces_copy.append(face_copy)
        new_person["faces"] = moved_faces_copy

        source["faces"] = remaining_faces
        source["face_count"] = len(remaining_faces)

        if not remaining_faces:
            self.report["people"] = [
                person for person in self.report["people"] if person is not source
            ]
        self.report["people"].append(new_person)
        self.report["next_person_id"] = new_person_id + 1

        for image in self.report["images"]:
            if not isinstance(image, dict):
                continue
            image_value = image.get("image")
            if not isinstance(image_value, str) or image_value not in selected_images:
                continue
            faces = image.get("faces", [])
            if not isinstance(faces, list):
                continue
            for face in faces:
                if (
                    isinstance(face, dict)
                    and face.get("person_id") == person_id
                    and str(face.get("image")) in selected_images
                ):
                    face["person_id"] = new_person_id

        return new_person_id


def build_image_context(
    store: ReportStore, entry: dict[str, Any], index: int, query: str = ""
) -> dict[str, Any]:
    image_value = entry.get("image", "")
    image_id = image_key(image_value)
    annotated_value = entry.get("annotated_image")
    image_name = Path(str(image_value)).name

    person_ids: list[int] = []
    summary_parts: list[str] = []
    faces = entry.get("faces", [])
    if isinstance(faces, list):
        for face in faces:
            if not isinstance(face, dict):
                continue
            try:
                person_id = int(face["person_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if person_id not in person_ids:
                person_ids.append(person_id)

    person_lookup = store.person_index()
    for person_id in person_ids:
        summary_parts.append(
            person_display_name(person_lookup.get(person_id), person_id)
        )

    annotated_path = resolve_media_path(
        store.report_path, annotated_value if isinstance(annotated_value, str) else None
    )

    return {
        "index": index,
        "image_id": image_id,
        "image_name": image_name,
        "image_value": image_value,
        "face_count": entry.get("face_count", 0),
        "summary": ", ".join(summary_parts) if summary_parts else "Unassigned",
        "detail_url": with_query(f"/images/{image_id}", query),
        "annotated_url": media_url(annotated_path, f"/media/annotated/{image_id}")
        if annotated_path is not None
        else placeholder_media(image_name),
        "annotated_exists": annotated_path is not None,
        "annotated_media_url": f"/media/annotated/{image_id}",
        "annotated_note": annotated_value,
    }


def build_index_context(store: ReportStore, query: str = "") -> dict[str, Any]:
    with store.lock:
        images: list[dict[str, Any]] = []
        people: list[dict[str, Any]] = []
        face_count = 0
        person_lookup = store.person_index()
        cleaned_query = query.strip()
        sorted_people = sorted(
            person_lookup.items(),
            key=lambda item: person_sort_key(item[0], item[1]),
        )

        if cleaned_query:
            for index, (person_id, person) in enumerate(sorted_people):
                if person_matches_search(person_id, person, cleaned_query):
                    people.append(
                        build_person_list_context(
                            store, person_id, person, index, cleaned_query
                        )
                    )

        for index, image in enumerate(store.report["images"]):
            if not isinstance(image, dict):
                continue
            face_value = image.get("face_count", 0)
            if isinstance(face_value, int):
                face_count += face_value
            if image_matches_search(store, image, query, person_lookup):
                images.append(build_image_context(store, image, index, query))

        total_images = len(store.report["images"])
        filtered_face_count = 0
        for image in images:
            face_value = image.get("face_count", 0)
            if isinstance(face_value, int):
                filtered_face_count += face_value

        return {
            "title": f"Annotated images - {store.report_path.name}",
            "heading": "Annotated images",
            "subtitle": "Review faces, name people, and merge mismatched clusters into the correct person.",
            "people": people,
            "images": images,
            "face_count": filtered_face_count,
            "total_face_count": face_count,
            "people_count": len(store.person_index()),
            "total_image_count": total_images,
            "search_query": cleaned_query,
            "search_active": bool(cleaned_query),
            "home_url": "/",
            "people_url": with_query("/people", cleaned_query),
        }


def build_person_list_context(
    store: ReportStore,
    person_id: int,
    person: dict[str, Any],
    index: int,
    query: str = "",
) -> dict[str, Any]:
    image_groups = store.person_image_groups(person_id)
    face_value = person.get("face_count", len(person.get("faces", [])))
    preview_src = placeholder_media(person_display_name(person, person_id))
    if image_groups:
        first_group = image_groups[0]
        preview_path = resolve_media_path(store.report_path, first_group["image"])
        preview_src = media_url_for_paths(
            f"/media/person-preview/{first_group['image_id']}/{person_id}",
            store.report_path,
            preview_path,
        )

    aliases_value = person.get("aliases", [])
    aliases = (
        [
            str(alias)
            for alias in aliases_value
            if isinstance(alias, str) and alias.strip()
        ]
        if isinstance(aliases_value, list)
        else []
    )

    return {
        "person_id": person_id,
        "name": person_display_name(person, person_id),
        "face_count": face_value
        if isinstance(face_value, int)
        else len(person.get("faces", [])),
        "image_count": len(image_groups),
        "aliases": aliases,
        "preview_src": preview_src,
        "detail_url": with_query(f"/people/{person_id}", query),
        "index": index,
    }


def build_people_context(store: ReportStore, query: str = "") -> dict[str, Any]:
    with store.lock:
        people: list[dict[str, Any]] = []
        face_count = 0
        person_lookup = store.person_index()
        for index, (person_id, person) in enumerate(
            sorted(
                person_lookup.items(),
                key=lambda item: person_sort_key(item[0], item[1]),
            )
        ):
            face_value = person.get("face_count", len(person.get("faces", [])))
            if isinstance(face_value, int):
                face_count += face_value
            people.append(
                build_person_list_context(store, person_id, person, index, query)
            )

        return {
            "title": f"People - {store.report_path.name}",
            "heading": "People",
            "subtitle": "Open a person to inspect the images associated with them, then rename, merge, or split selections into a new person.",
            "people": people,
            "face_count": face_count,
            "images_count": len(store.report["images"]),
            "search_query": query,
            "home_url": "/",
            "people_url": with_query("/people", query),
        }


def build_person_context(
    store: ReportStore, person_id: int, query: str = ""
) -> dict[str, Any]:
    with store.lock:
        person_lookup = store.person_index()
        person = person_lookup.get(person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="Person not found.")

        name = person_display_name(person, person_id)
        name_input = name if name != f"Person {person_id}" else ""
        aliases_value = person.get("aliases", [])
        aliases = (
            [
                str(alias)
                for alias in aliases_value
                if isinstance(alias, str) and alias.strip()
            ]
            if isinstance(aliases_value, list)
            else []
        )
        image_groups = store.person_image_groups(person_id)
        face_count = person.get("face_count", len(person.get("faces", [])))
        if not isinstance(face_count, int):
            face_count = len(person.get("faces", []))

        preview_src = placeholder_media(name)
        if image_groups:
            first_group = image_groups[0]
            preview_path = resolve_media_path(store.report_path, first_group["image"])
            preview_src = media_url_for_paths(
                f"/media/person-preview/{first_group['image_id']}/{person_id}",
                store.report_path,
                preview_path,
            )

        merge_options = [
            {"value": "", "label": "Keep separate"},
        ]
        for other_id, other_person in sorted(
            person_lookup.items(), key=lambda item: person_sort_key(item[0], item[1])
        ):
            if other_id == person_id:
                continue
            merge_options.append(
                {"value": str(other_id), "label": person_option_label(other_person)}
            )

        image_groups_context: list[dict[str, Any]] = []
        for group in image_groups:
            preview_path = resolve_media_path(store.report_path, group["image"])
            image_groups_context.append(
                {
                    "image": group["image"],
                    "image_id": group["image_id"],
                    "image_name": Path(group["image"]).name,
                    "face_count": group["face_count"],
                    "preview_src": media_url_for_paths(
                        f"/media/person-preview/{group['image_id']}/{person_id}",
                        store.report_path,
                        preview_path,
                    ),
                    "detail_url": with_query(f"/images/{group['image_id']}", query),
                    "checkbox_value": group["image"],
                }
            )

        return {
            "title": f"{name} - people",
            "heading": name,
            "subtitle": f"Person #{person_id}",
            "person_id": person_id,
            "face_count": face_count,
            "image_count": len(image_groups_context),
            "aliases": aliases,
            "preview_src": preview_src,
            "merge_options": merge_options,
            "images": image_groups_context,
            "name": name,
            "name_input": name_input,
            "search_query": query,
            "home_url": "/",
            "people_url": with_query("/people", query),
        }


def build_image_detail_context(
    store: ReportStore, image_id: str, query: str = ""
) -> dict[str, Any]:
    with store.lock:
        image_lookup = store.image_index()
        person_lookup = store.person_index()
        entry = image_lookup.get(image_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Image not found.")

        image_value = entry.get("image")
        if not isinstance(image_value, str):
            raise HTTPException(status_code=404, detail="Image path missing.")

        person_groups: dict[int, list[tuple[int, dict[str, Any]]]] = {}
        faces = entry.get("faces", [])
        if isinstance(faces, list):
            for face_index, face in enumerate(faces):
                if not isinstance(face, dict):
                    continue
                face_entry = cast(dict[str, Any], face)
                try:
                    person_id = int(face_entry["person_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                person_groups.setdefault(person_id, []).append((face_index, face_entry))

        original_path = Path(image_value)
        annotated_path = resolve_media_path(
            store.report_path,
            entry.get("annotated_image")
            if isinstance(entry.get("annotated_image"), str)
            else None,
        )
        image_name = Path(image_value).name
        original_exists = original_path.exists()

        person_groups_context: list[dict[str, Any]] = []
        for person_id, grouped_faces in sorted(
            person_groups.items(),
            key=lambda item: person_sort_key(item[0], person_lookup.get(item[0])),
        ):
            person = person_lookup.get(person_id)
            display_name = person_display_name(person, person_id)
            aliases_value = person.get("aliases", []) if person is not None else []
            aliases = (
                [
                    str(alias)
                    for alias in aliases_value
                    if isinstance(alias, str) and alias.strip()
                ]
                if isinstance(aliases_value, list)
                else []
            )
            merge_options = [{"value": "", "label": "Keep separate"}]
            for other_id, other_person in sorted(
                person_lookup.items(),
                key=lambda item: person_sort_key(item[0], item[1]),
            ):
                if other_id == person_id:
                    continue
                merge_options.append(
                    {"value": str(other_id), "label": person_option_label(other_person)}
                )

            faces_context = []
            for face_index, face in grouped_faces:
                bbox = face.get("bbox", [])
                preview_src = placeholder_media(f"Face {face_index + 1}")
                if (
                    original_exists
                    and isinstance(bbox, list)
                    and len(bbox) == 4
                    and all(isinstance(value, int | float) for value in bbox)
                ):
                    preview_src = media_url_for_paths(
                        f"/media/face-preview/{image_id}/{face_index}",
                        original_path,
                    )
                faces_context.append(
                    {
                        "summary": face_summary(face),
                        "confidence": face.get("confidence"),
                        "bbox": bbox,
                        "preview_src": preview_src,
                    }
                )

            person_groups_context.append(
                {
                    "person_id": person_id,
                    "name": display_name,
                    "name_input": display_name
                    if display_name != f"Person {person_id}"
                    else "",
                    "aliases": aliases,
                    "face_count": len(grouped_faces),
                    "faces": faces_context,
                    "merge_options": merge_options,
                }
            )

        return {
            "title": f"{image_name} - image review",
            "heading": image_name,
            "subtitle": image_value,
            "face_count": entry.get("face_count", 0),
            "person_count": len(person_groups_context),
            "original_src": media_url(
                original_path if original_exists else None,
                f"/media/original/{image_id}",
            )
            if original_exists
            else placeholder_media(image_name),
            "original_available": original_exists,
            "annotated_src": media_url(annotated_path, f"/media/annotated/{image_id}")
            if annotated_path is not None
            else placeholder_media(image_name),
            "annotated_available": annotated_path is not None,
            "person_groups": person_groups_context,
            "image_id": image_id,
            "search_query": query,
            "home_url": "/",
            "people_url": with_query("/people", query),
        }


def create_app(report_path: Path | None = None) -> FastAPI:
    resolved_report_path = active_report_path(report_path)
    store = ReportStore.open(resolved_report_path)
    with store.lock:
        store.ensure_hashes_backfilled()

    app = FastAPI(title="image-yolo-faces")
    templates = Jinja2Templates(directory=str(Path(__file__).with_name("templates")))
    static_dir = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    templates.env.globals["frontend_assets"] = cast(Any, frontend_assets(static_dir))
    app.state.store = store

    def update_person_profile(
        person_id: int, name: str, merge_into: str
    ) -> tuple[int, set[str]]:
        target_id = person_id
        affected_images: set[str] = set()

        merge_target = merge_into.strip()
        if merge_target:
            try:
                merge_target_id = int(merge_target)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail="Merge target must be a person ID."
                ) from exc

            if merge_target_id != person_id:
                affected_images.update(store.merge_people(person_id, merge_target_id))
                target_id = merge_target_id

        cleaned_name = name.strip()
        if cleaned_name:
            affected_images.update(store.rename_person(target_id, cleaned_name))

        return target_id, affected_images

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, q: str = "") -> HTMLResponse:
        context = build_index_context(store, q)
        return templates.TemplateResponse(request, "index.html", context)

    @app.get("/people", response_class=HTMLResponse)
    def people_index(request: Request, q: str = "") -> HTMLResponse:
        context = build_people_context(store, q)
        return templates.TemplateResponse(request, "people.html", context)

    @app.get("/people/{person_id}", response_class=HTMLResponse)
    def person_detail(request: Request, person_id: int, q: str = "") -> HTMLResponse:
        context = build_person_context(store, person_id, q)
        return templates.TemplateResponse(request, "person.html", context)

    @app.get("/images/{image_id}", response_class=HTMLResponse)
    def image_detail(request: Request, image_id: str, q: str = "") -> HTMLResponse:
        context = build_image_detail_context(store, image_id, q)
        return templates.TemplateResponse(request, "image.html", context)

    @app.get("/media/person-preview/{image_id}/{person_id}")
    def person_preview_media(image_id: str, person_id: int) -> Response:
        with store.lock:
            entry = store.image_index().get(image_id)
            if entry is None:
                raise HTTPException(status_code=404, detail="Image not found.")

            image_value = entry.get("image")
            if not isinstance(image_value, str):
                raise HTTPException(status_code=404, detail="Image path missing.")

            image_path = Path(image_value)
            person = store.person_index().get(person_id)
            person_faces: list[dict[str, Any]] = []
            faces = entry.get("faces", [])
            if isinstance(faces, list):
                for face in faces:
                    if not isinstance(face, dict):
                        continue
                    try:
                        current_person_id = int(face["person_id"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if current_person_id == person_id:
                        person_faces.append(face)

            bbox = representative_face_bbox(person_faces)
            if bbox is None or not image_path.exists():
                return Response(
                    content=placeholder_svg_bytes(
                        person_display_name(person, person_id)
                    ),
                    media_type="image/svg+xml",
                    headers={"Cache-Control": "no-store"},
                )

            preview_bytes = preview_image_bytes(image_path, bbox)
            if preview_bytes is None:
                return Response(
                    content=placeholder_svg_bytes(
                        person_display_name(person, person_id)
                    ),
                    media_type="image/svg+xml",
                    headers={"Cache-Control": "no-store"},
                )

            return Response(
                content=preview_bytes,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )

    @app.get("/media/face-preview/{image_id}/{face_index}")
    def face_preview_media(image_id: str, face_index: int) -> Response:
        with store.lock:
            entry = store.image_index().get(image_id)
            if entry is None:
                raise HTTPException(status_code=404, detail="Image not found.")

            image_value = entry.get("image")
            if not isinstance(image_value, str):
                raise HTTPException(status_code=404, detail="Image path missing.")

            faces = entry.get("faces", [])
            if not isinstance(faces, list) or face_index < 0 or face_index >= len(faces):
                return Response(
                    content=placeholder_svg_bytes("Face preview"),
                    media_type="image/svg+xml",
                    headers={"Cache-Control": "no-store"},
                )

            face = faces[face_index]
            if not isinstance(face, dict):
                return Response(
                    content=placeholder_svg_bytes("Face preview"),
                    media_type="image/svg+xml",
                    headers={"Cache-Control": "no-store"},
                )

            bbox = face.get("bbox")
            image_path = Path(image_value)
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or not all(isinstance(value, int | float) for value in bbox)
                or not image_path.exists()
            ):
                return Response(
                    content=placeholder_svg_bytes("Face preview"),
                    media_type="image/svg+xml",
                    headers={"Cache-Control": "no-store"},
                )

            preview_bytes = preview_image_bytes(image_path, [float(value) for value in bbox])
            if preview_bytes is None:
                return Response(
                    content=placeholder_svg_bytes("Face preview"),
                    media_type="image/svg+xml",
                    headers={"Cache-Control": "no-store"},
                )

            return Response(
                content=preview_bytes,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )

    @app.get("/media/original/{image_id}")
    def original_media(image_id: str) -> FileResponse:
        with store.lock:
            entry = store.image_index().get(image_id)
            if entry is None:
                raise HTTPException(status_code=404, detail="Image not found.")

            image_value = entry.get("image")
            if not isinstance(image_value, str):
                raise HTTPException(status_code=404, detail="Image path missing.")

            path = Path(image_value)
            if not path.exists():
                raise HTTPException(
                    status_code=404, detail="Original image file not found."
                )

            return FileResponse(path, headers={"Cache-Control": "no-store"})

    @app.get("/media/annotated/{image_id}")
    def annotated_media(image_id: str) -> FileResponse:
        with store.lock:
            entry = store.image_index().get(image_id)
            if entry is None:
                raise HTTPException(status_code=404, detail="Image not found.")

            path = store.ensure_annotated_image(entry)
            if path is None or not path.exists():
                raise HTTPException(
                    status_code=404, detail="Annotated image file not found."
                )

            return FileResponse(path, headers={"Cache-Control": "no-store"})

    @app.post("/uploads")
    async def upload_image(image: UploadFile = File(...)) -> JSONResponse:
        filename = image.filename or "upload"
        content = await image.read()
        with store.lock:
            result = store.import_uploaded_image(filename, content)
        return JSONResponse(result)

    @app.post("/images/{image_id}/people/{person_id}")
    def update_image_person(
        image_id: str,
        person_id: int,
        name: str = Form(""),
        merge_into: str = Form(""),
    ) -> RedirectResponse:
        with store.lock:
            if image_id not in store.image_index():
                raise HTTPException(status_code=404, detail="Image not found.")

            _, affected_images = update_person_profile(person_id, name, merge_into)
            store.save()
            store.re_render_images(affected_images)

        return RedirectResponse(url=f"/images/{image_id}", status_code=303)

    @app.post("/images/{image_id}/delete")
    def delete_image(image_id: str) -> RedirectResponse:
        with store.lock:
            store.delete_image(image_id)
        return RedirectResponse(url="/", status_code=303)

    @app.post("/people/{person_id}")
    def update_people_person(
        person_id: int,
        name: str = Form(""),
        merge_into: str = Form(""),
    ) -> RedirectResponse:
        with store.lock:
            if person_id not in store.person_index():
                raise HTTPException(status_code=404, detail="Person not found.")

            target_id, affected_images = update_person_profile(
                person_id, name, merge_into
            )
            store.save()
            store.re_render_images(affected_images)

        return RedirectResponse(url=f"/people/{target_id}", status_code=303)

    @app.post("/people/{person_id}/split")
    def split_person_images(
        person_id: int,
        selected_images: list[str] = Form(default=[]),
        new_name: str = Form(""),
    ) -> RedirectResponse:
        with store.lock:
            if person_id not in store.person_index():
                raise HTTPException(status_code=404, detail="Person not found.")

            source_image_paths = store.image_paths_for_person(person_id)
            selected_set = {
                image for image in selected_images if image in source_image_paths
            }
            if not selected_set:
                raise HTTPException(
                    status_code=400, detail="Select at least one image to split."
                )

            affected_images = set(source_image_paths)
            new_person_id = store.split_person_images_to_new_person(
                person_id, selected_set, new_name
            )
            affected_images.update(selected_set)
            affected_images.update(store.image_paths_for_person(new_person_id))
            store.save()
            store.re_render_images(affected_images)

        return RedirectResponse(url=f"/people/{new_person_id}", status_code=303)

    return app


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--report",
    "report_path",
    type=click.Path(path_type=Path, dir_okay=False, exists=False),
    default=DEFAULT_REPORT_PATH,
    show_default=True,
    help="Path to the JSON report to load and update.",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Bind host for the web server.",
)
@click.option(
    "--port",
    default=8000,
    show_default=True,
    type=click.IntRange(min=1, max=65535),
    help="Bind port for the web server.",
)
@click.option(
    "--reload/--no-reload",
    default=False,
    show_default=True,
    help="Restart the server when Python files or built frontend assets change.",
)
def main(report_path: Path, host: str, port: int, reload: bool) -> None:
    resolved_report_path = resolve_report_path(report_path)
    os.environ[REPORT_PATH_ENV] = str(resolved_report_path)
    package_dir = Path(__file__).resolve().parent
    reload_dirs = [
        str(package_dir),
        str(package_dir / "static" / "dist"),
    ]
    click.echo(f"Serving {resolved_report_path} at http://{host}:{port}")
    uvicorn.run(
        "image_yolo_faces.webui:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        reload_dirs=reload_dirs if reload else None,
    )
