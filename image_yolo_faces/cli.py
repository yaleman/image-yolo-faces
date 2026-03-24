from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Sequence, Tuple

import click

from .ingest import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL_FILE,
    DEFAULT_MODEL_REPO,
    DEFAULT_PERSON_GROUPING_STRATEGY,
    DEFAULT_PERSON_THRESHOLD,
    IMAGE_EXTENSIONS,
    annotated_output_path,
    load_face_encoder,
    load_model,
    normalize_image_entry,
    normalize_image_key,
    render_faces,
    scan_image_entry,
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
        raw_report = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"Existing report at {output_path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(raw_report, dict):
        raise click.ClickException(
            f"Existing report at {output_path} must contain a JSON object."
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
            f"Existing report at {output_path} has an invalid 'images' field."
        )

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
        normalize_image_entry(entry)
        entry["image"] = normalize_image_key(entry["image"])
        image_index[normalize_image_key(entry["image"])] = entry

    people: list[dict[str, Any]] = []
    if group_by_person:
        loaded_people = raw_report.get("people", [])
        if not isinstance(loaded_people, list):
            raise click.ClickException(
                f"Existing report at {output_path} has an invalid 'people' field."
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

    return report, image_index, people


def build_report(
    model: Any,
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

        annotated_path = None
        if annotated_dir is not None:
            annotated_path = annotated_output_path(root, image_path, annotated_dir)
        entry, next_person_id = scan_image_entry(
            model=model,
            face_encoder=face_encoder,
            image_path=image_path,
            confidence=confidence,
            annotated_path=annotated_path,
            group_by_person=group_by_person,
            person_threshold=person_threshold,
            people=people,
            next_person_id=next_person_id,
        )
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
