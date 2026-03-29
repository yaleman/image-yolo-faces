from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

WORKSPACES_DIR_ENV = "FACES_WORKSPACES_DIR"
DEFAULT_WORKSPACES_DIR = Path("workspaces")
DEFAULT_WORKSPACE_NAME = "default"
WORKSPACE_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def resolve_workspaces_root(workspaces_dir: Path | None = None) -> Path:
    if workspaces_dir is not None:
        return workspaces_dir.expanduser().resolve()

    env_value = os.environ.get(WORKSPACES_DIR_ENV)
    if env_value:
        return Path(env_value).expanduser().resolve()

    return DEFAULT_WORKSPACES_DIR.expanduser().resolve()


def validate_workspace_name(workspace_name: str) -> str:
    cleaned_name = workspace_name
    if not cleaned_name or WORKSPACE_NAME_RE.fullmatch(cleaned_name) is None:
        raise ValueError(
            "Workspace names may only contain letters, numbers, and underscores."
        )
    return cleaned_name


def workspace_dir(workspaces_root: Path, workspace_name: str) -> Path:
    return workspaces_root / validate_workspace_name(workspace_name)


def workspace_report_path(workspaces_root: Path, workspace_name: str) -> Path:
    return workspace_dir(workspaces_root, workspace_name) / "faces.json"


def workspace_photos_dir(workspaces_root: Path, workspace_name: str) -> Path:
    return workspace_dir(workspaces_root, workspace_name) / "photos"


def workspace_exports_dir(workspaces_root: Path, workspace_name: str) -> Path:
    return workspace_dir(workspaces_root, workspace_name) / "exports"


def workspace_media_name(value: str | Path) -> str:
    return Path(value).name


def workspace_original_media_path(
    workspaces_root: Path, workspace_name: str, value: str | Path
) -> Path:
    return workspace_photos_dir(workspaces_root, workspace_name) / workspace_media_name(
        value
    )


def ensure_workspace_layout(workspaces_root: Path, workspace_name: str) -> Path:
    directory = workspace_dir(workspaces_root, workspace_name)
    directory.mkdir(parents=True, exist_ok=True)
    workspace_photos_dir(workspaces_root, workspace_name).mkdir(
        parents=True, exist_ok=True
    )
    workspace_exports_dir(workspaces_root, workspace_name).mkdir(
        parents=True, exist_ok=True
    )
    return directory


def list_workspaces(workspaces_root: Path) -> list[str]:
    if not workspaces_root.exists():
        return [DEFAULT_WORKSPACE_NAME]

    names = sorted(
        {
            child.name
            for child in workspaces_root.iterdir()
            if child.is_dir() and WORKSPACE_NAME_RE.fullmatch(child.name) is not None
        }
    )
    if DEFAULT_WORKSPACE_NAME not in names:
        names.insert(0, DEFAULT_WORKSPACE_NAME)
    return names


def resolve_workspace_media_path(
    workspaces_root: Path, workspace_name: str, value: str | None
) -> Path | None:
    if not value:
        return None

    workspace_directory = workspace_dir(workspaces_root, workspace_name)
    path = Path(value)
    if path.is_absolute():
        return path
    return (workspace_directory / path).resolve()


def workspace_relative_media_path(
    workspaces_root: Path, workspace_name: str, path: Path
) -> str:
    workspace_directory = workspace_dir(workspaces_root, workspace_name).resolve()
    resolved = path.resolve()

    try:
        return str(resolved.relative_to(workspace_directory))
    except ValueError:
        return str(resolved)


def workspace_media_key(
    workspaces_root: Path, workspace_name: str, value: str | Path
) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path.resolve())
    resolved = resolve_workspace_media_path(workspaces_root, workspace_name, str(path))
    if resolved is None:
        return ""
    return str(resolved.resolve())


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


def normalize_report_media_paths(
    report: dict[str, Any], _workspaces_root: Path, _workspace_name: str
) -> bool:
    changed = False

    def normalize_value(value: Any) -> tuple[Any, bool]:
        if not isinstance(value, str) or not value.strip():
            return value, False

        return Path(value).name, Path(value).name != value

    images = report.get("images", [])
    if isinstance(images, list):
        for image in images:
            if not isinstance(image, dict):
                continue
            image_value, image_changed = normalize_value(image.get("image"))
            if image_changed:
                image["image"] = image_value
                changed = True

    people = report.get("people", [])
    if isinstance(people, list):
        for person in people:
            if not isinstance(person, dict):
                continue
            faces = person.get("faces", [])
            if not isinstance(faces, list):
                continue
            for face in faces:
                if not isinstance(face, dict):
                    continue
                face_value, face_changed = normalize_value(face.get("image"))
                if face_changed:
                    face["image"] = face_value
                    changed = True

    return changed
