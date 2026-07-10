from __future__ import annotations

import csv
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enviro_webcam_ml.annotation import task_labels
from enviro_webcam_ml.config import AppConfig


DEFAULT_EXCLUDED_LABELS: set[str] = set()


@dataclass(frozen=True)
class TrainingSetOptions:
    task_id: str
    output_path: Path
    include_labels: tuple[str, ...] = ()
    exclude_labels: tuple[str, ...] = tuple(sorted(DEFAULT_EXCLUDED_LABELS))
    min_annotators: int = 2
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    allow_missing_images: bool = False


def build_training_set(
    conn: sqlite3.Connection,
    *,
    config: AppConfig,
    options: TrainingSetOptions,
) -> dict[str, Any]:
    validate_fractions(
        options.train_fraction,
        options.val_fraction,
        options.test_fraction,
    )
    rows = candidate_rows(conn, task_id=options.task_id)
    grouped = group_latest_annotations(rows)
    configured_labels = set(task_labels(config, options.task_id))
    include_labels = set(options.include_labels) if options.include_labels else configured_labels
    exclude_labels = set(options.exclude_labels)
    selected: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for capture_id, annotations in sorted(grouped.items()):
        annotators = sorted(annotations)
        labels = {annotations[annotator]["label"] for annotator in annotators}
        first = annotations[annotators[0]]

        if len(annotators) < options.min_annotators:
            skipped["too_few_annotators"] += 1
            continue
        if len(labels) != 1:
            skipped["disagreement"] += 1
            continue

        label = next(iter(labels))
        if label not in configured_labels:
            skipped["legacy_or_unknown_label"] += 1
            continue
        if label not in include_labels:
            skipped["not_in_include_labels"] += 1
            continue
        if label in exclude_labels:
            skipped["excluded_label"] += 1
            continue

        image_path = resolve_image_path(first.get("image_path"), config.data_dir)
        image_exists = bool(image_path and image_path.exists())
        if not image_exists and not options.allow_missing_images:
            skipped["missing_image"] += 1
            continue

        selected.append(
            {
                "capture_id": capture_id,
                "camera_id": first.get("camera_id") or "",
                "captured_at_utc": first.get("captured_at_utc") or "",
                "label": label,
                "split": "",
                "image_path": str(image_path) if image_path else "",
                "image_exists": int(image_exists),
                "original_image_path": first.get("image_path") or "",
                "width": first.get("width") or "",
                "height": first.get("height") or "",
                "avg_luminance": first.get("avg_luminance") or "",
                "blur_variance": first.get("blur_variance") or "",
                "is_night": first.get("is_night") if first.get("is_night") is not None else "",
                "is_blurry": first.get("is_blurry") if first.get("is_blurry") is not None else "",
                "is_duplicate": first.get("is_duplicate") if first.get("is_duplicate") is not None else "",
                "annotators": "|".join(annotators),
                "annotator_count": len(annotators),
                "agreement_count": len(annotators),
            }
        )

    assign_chronological_splits(
        selected,
        train_fraction=options.train_fraction,
        val_fraction=options.val_fraction,
        test_fraction=options.test_fraction,
    )
    write_training_csv(selected, options.output_path)

    return {
        "output_path": str(options.output_path),
        "row_count": len(selected),
        "label_counts": dict(sorted(Counter(row["label"] for row in selected).items())),
        "split_counts": dict(sorted(Counter(row["split"] for row in selected).items())),
        "skipped": dict(sorted(skipped.items())),
    }


def candidate_rows(conn: sqlite3.Connection, *, task_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
          a.capture_id,
          COALESCE(a.annotator, '') AS annotator,
          a.label,
          a.created_at_utc AS annotated_at_utc,
          c.camera_id,
          c.captured_at_utc,
          ia.path AS image_path,
          ia.width,
          ia.height,
          fq.avg_luminance,
          fq.blur_variance,
          fq.is_night,
          fq.is_blurry,
          fq.is_duplicate
        FROM annotation a
        JOIN capture c ON c.id = a.capture_id
        JOIN image_asset ia ON ia.capture_id = a.capture_id
        LEFT JOIN frame_quality fq ON fq.capture_id = a.capture_id
        WHERE a.task_id = ?
          AND c.error IS NULL
        ORDER BY c.captured_at_utc, a.capture_id, annotator, a.created_at_utc
        """,
        (task_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def group_latest_annotations(rows: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        capture_id = int(row["capture_id"])
        annotator = row["annotator"] or ""
        existing = grouped[capture_id].get(annotator)
        if existing is None or (row.get("annotated_at_utc") or "") >= (existing.get("annotated_at_utc") or ""):
            grouped[capture_id][annotator] = row
    return grouped


def resolve_image_path(stored_path: str | None, data_dir: Path) -> Path | None:
    if not stored_path:
        return None
    path = Path(stored_path)
    if path.exists():
        return path
    if not path.is_absolute():
        return (data_dir / path).resolve()

    parts = path.parts
    data_indexes = [index for index, part in enumerate(parts) if part == "data"]
    if data_indexes:
        suffix = Path(*parts[data_indexes[-1] + 1 :])
        return (data_dir / suffix).resolve()
    return path


def assign_chronological_splits(
    rows: list[dict[str, Any]],
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> None:
    rows.sort(key=lambda row: (row["captured_at_utc"], row["camera_id"], row["capture_id"]))
    n = len(rows)
    train_end = int(n * train_fraction)
    val_end = train_end + int(n * val_fraction)

    if n >= 3:
        train_end = max(1, min(train_end, n - 2))
        val_end = max(train_end + 1, min(val_end, n - 1))
    elif n == 2:
        train_end = 1
        val_end = 1
    elif n == 1:
        train_end = 1
        val_end = 1

    for index, row in enumerate(rows):
        if index < train_end:
            row["split"] = "train"
        elif index < val_end:
            row["split"] = "val"
        else:
            row["split"] = "test"


def write_training_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "capture_id",
        "camera_id",
        "captured_at_utc",
        "split",
        "label",
        "image_path",
        "image_exists",
        "original_image_path",
        "width",
        "height",
        "avg_luminance",
        "blur_variance",
        "is_night",
        "is_blurry",
        "is_duplicate",
        "annotators",
        "annotator_count",
        "agreement_count",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_fractions(train: float, val: float, test: float) -> None:
    total = train + val + test
    if min(train, val, test) < 0:
        raise ValueError("Split fractions must be non-negative.")
    if abs(total - 1.0) > 0.0001:
        raise ValueError("Train/val/test fractions must sum to 1.0.")
