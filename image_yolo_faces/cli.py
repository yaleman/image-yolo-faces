from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Iterator, Sequence, Tuple

import click

from .ingest import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL_FILE,
    DEFAULT_MODEL_REPO,
    DEFAULT_PERSON_GROUPING_STRATEGY,
    DEFAULT_PERSON_THRESHOLD,
    FaceEncoder,
    IMAGE_EXTENSIONS,
    ModelLike,
    hashes_for_file,
    load_face_encoder,
    load_model,
    normalize_image_entry,
    scan_image_entry,
)
from .workspaces import (
    DEFAULT_WORKSPACE_NAME,
    ensure_workspace_layout,
    normalize_report_media_paths,
    resolve_workspaces_root,
    validate_workspace_name,
    uniquify_filename,
    workspace_photos_dir,
    workspace_report_path,
    WORKSPACES_DIR_ENV,
)


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
    report_path: Path,
    workspaces_root: Path,
    workspace_name: str,
    model_repo: str,
    model_file: str,
    confidence: float,
    group_by_person: bool,
    embedding_model: str,
    person_threshold: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not report_path.exists():
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

    try:
        raw_report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"Existing report at {report_path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(raw_report, dict):
        raise click.ClickException(
            f"Existing report at {report_path} must contain a JSON object."
        )

    if (
        raw_report.get("model_repo") != model_repo
        or raw_report.get("model_file") != model_file
        or raw_report.get("confidence_threshold") != confidence
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

    if group_by_person:
        if (
            raw_report.get("group_by_person") is not True
            or raw_report.get("embedding_model") != embedding_model
            or raw_report.get("person_similarity_threshold") != person_threshold
            or raw_report.get("person_grouping_strategy")
            != DEFAULT_PERSON_GROUPING_STRATEGY
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
        raise click.ClickException(
            f"Existing report at {report_path} has an invalid 'images' field."
        )

    report = raw_report
    report["model_repo"] = model_repo
    report["model_file"] = model_file
    report["confidence_threshold"] = confidence
    report["images"] = images
    if group_by_person:
        report["group_by_person"] = True
        report["embedding_model"] = embedding_model
        report["person_similarity_threshold"] = person_threshold
        report["person_grouping_strategy"] = DEFAULT_PERSON_GROUPING_STRATEGY

    people: list[dict[str, Any]] = []
    if group_by_person:
        loaded_people = raw_report.get("people", [])
        if not isinstance(loaded_people, list):
            raise click.ClickException(
                f"Existing report at {report_path} has an invalid 'people' field."
            )
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
    elif isinstance(raw_report.get("next_person_id"), int):
        report["next_person_id"] = int(raw_report["next_person_id"])
    else:
        report["next_person_id"] = 1

    normalize_report_media_paths(report, workspaces_root, workspace_name)

    image_hash_index: dict[str, dict[str, Any]] = {}
    for entry in images:
        if not isinstance(entry, dict):
            continue
        normalize_image_entry(entry)
        hashes = entry.get("hashes", {})
        if not isinstance(hashes, dict):
            continue
        digest = hashes.get("sha256")
        if isinstance(digest, str) and digest.strip():
            image_hash_index[digest] = entry

    return report, image_hash_index, people


def build_report(
    model: ModelLike,
    face_encoder: FaceEncoder | None,
    image_roots: Sequence[Path],
    recursive: bool,
    confidence: float,
    group_by_person: bool,
    person_threshold: float,
    workspaces_root: Path,
    workspace_name: str,
    report: dict[str, Any],
    image_hash_index: dict[str, dict[str, Any]],
    people: list[dict[str, Any]],
) -> dict[str, Any]:
    if group_by_person:
        report["people"] = people
        next_person_id = int(report.get("next_person_id", len(people) + 1))
    else:
        next_person_id = 1

    photos_dir = workspace_photos_dir(workspaces_root, workspace_name)
    photos_dir.mkdir(parents=True, exist_ok=True)

    for _, image_path in iter_images(image_roots, recursive):
        digest = hashes_for_file(image_path)["sha256"]
        cached_entry = image_hash_index.get(digest)
        if cached_entry is not None:
            continue

        imported_image_path = uniquify_filename(
            photos_dir, image_path.stem, image_path.suffix
        )
        shutil.copy2(image_path, imported_image_path)
        entry, next_person_id = scan_image_entry(
            model=model,
            face_encoder=face_encoder,
            image_path=imported_image_path,
            confidence=confidence,
            added_at_ns=time.time_ns(),
            group_by_person=group_by_person,
            person_threshold=person_threshold,
            people=people,
            next_person_id=next_person_id,
            storage_root=photos_dir,
        )
        entry["hashes"]["sha256"] = digest
        report["images"].append(entry)
        image_hash_index[digest] = entry

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
    "--workspaces-dir",
    "workspaces_dir",
    type=click.Path(path_type=Path, file_okay=False, exists=False),
    default=None,
    show_default=True,
    envvar=WORKSPACES_DIR_ENV,
    help="Path to the directory that contains workspace folders.",
)
@click.option(
    "--workspace",
    "workspace_name",
    default=DEFAULT_WORKSPACE_NAME,
    show_default=True,
    help="Workspace name to import images into.",
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
@click.option(
    "--show-config", is_flag=True, help="Show the resolved configuration and exit."
)
def cli(
    inputs: Tuple[Path, ...],
    workspaces_dir: Path | None,
    workspace_name: str,
    recursive: bool,
    confidence: float,
    model_repo: str,
    model_file: str,
    group_by_person: bool,
    embedding_model: str,
    person_threshold: float,
    show_config: bool,
) -> None:

    resolved_workspaces_root = resolve_workspaces_root(workspaces_dir)
    if show_config:
        click.echo("Configuration:")
        click.echo(f"  Workspaces root: {resolved_workspaces_root}")
        click.echo(f"  Workspace name: {workspace_name}")
        click.echo(f"  Recursive: {recursive}")
        click.echo(f"  Confidence threshold: {confidence}")
        click.echo(f"  Model repo: {model_repo}")
        click.echo(f"  Model file: {model_file}")
        click.echo(f"  Group by person: {group_by_person}")
        if group_by_person:
            click.echo(f"  Embedding model: {embedding_model}")
            click.echo(f"  Person similarity threshold: {person_threshold}")
        return
    if not inputs:
        raise click.ClickException("Provide at least one image or directory.")
    cleaned_workspace_name = validate_workspace_name(workspace_name)
    ensure_workspace_layout(resolved_workspaces_root, cleaned_workspace_name)
    report_path = workspace_report_path(
        resolved_workspaces_root, cleaned_workspace_name
    )

    model = load_model(model_repo, model_file)
    face_encoder = load_face_encoder(embedding_model) if group_by_person else None

    report, image_index, people = (
        load_existing_report(
            report_path,
            resolved_workspaces_root,
            cleaned_workspace_name,
            model_repo,
            model_file,
            confidence,
            group_by_person,
            embedding_model,
            person_threshold,
        )
        if report_path.exists()
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
        group_by_person=group_by_person,
        person_threshold=person_threshold,
        workspaces_root=resolved_workspaces_root,
        workspace_name=cleaned_workspace_name,
        report=report,
        image_hash_index=image_index,
        people=people,
    )

    if not report["images"]:
        raise click.ClickException("No image files were found in the provided inputs.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    click.echo(f"Wrote JSON report to {report_path}", err=True)

    image_count = len(report["images"])
    face_count = sum(image["face_count"] for image in report["images"])
    click.echo(
        f"Processed {image_count} image(s) and found {face_count} face(s).",
        err=True,
    )

    click.echo(
        f"Imported images into {workspace_photos_dir(resolved_workspaces_root, cleaned_workspace_name)}",
        err=True,
    )
