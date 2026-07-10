from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def compare_image_models(
    models_dir: Path,
    output_csv: Path,
    output_md: Path | None = None,
    *,
    camera_ids: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    metadata_files = sorted(models_dir.glob("*/metadata.json"))
    rows = [comparison_row(path, camera_ids=camera_ids) for path in metadata_files]
    rows.sort(
        key=lambda row: (
            row["test_accuracy"] is not None,
            row["test_accuracy"] if row["test_accuracy"] is not None else -1,
            row["val_accuracy"] if row["val_accuracy"] is not None else -1,
        ),
        reverse=True,
    )
    write_comparison_csv(rows, output_csv, camera_ids=camera_ids)
    if output_md is not None:
        write_comparison_markdown(rows, output_md, camera_ids=camera_ids)
    return rows


def comparison_row(metadata_path: Path, *, camera_ids: tuple[str, ...] = ()) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    detailed = metadata.get("detailed_metrics", {})
    val_overall = detailed.get("val", {}).get("overall", {})
    test_overall = detailed.get("test", {}).get("overall", {})
    val_binary = detailed.get("val", {}).get("binary", {})
    test_binary = detailed.get("test", {}).get("binary", {})
    test_by_camera = detailed.get("test", {}).get("by_camera", {})
    val_by_camera = detailed.get("val", {}).get("by_camera", {})

    if not camera_ids:
        camera_ids = tuple(
            sorted(
                set(val_by_camera.keys())
                | set(test_by_camera.keys())
            )
        )

    row = {
        "run": metadata_path.parent.name,
        "model_name": metadata.get("model_name"),
        "pretrained": metadata.get("pretrained"),
        "device": metadata.get("device"),
        "epochs": metadata.get("epochs"),
        "train_count": metadata.get("split_counts", {}).get("train", 0),
        "val_count": metadata.get("split_counts", {}).get("val", 0),
        "test_count": metadata.get("split_counts", {}).get("test", 0),
        "val_accuracy": val_overall.get("accuracy"),
        "test_accuracy": test_overall.get("accuracy"),
        "positive_label": metadata.get("positive_label") or test_binary.get("positive_label"),
        "positive_threshold": metadata.get("positive_threshold"),
        "val_ppv": val_binary.get("ppv"),
        "val_sensitivity": val_binary.get("sensitivity"),
        "val_specificity": val_binary.get("specificity"),
        "test_ppv": test_binary.get("ppv"),
        "test_sensitivity": test_binary.get("sensitivity"),
        "test_specificity": test_binary.get("specificity"),
        "metadata_path": str(metadata_path),
        "predictions_path": metadata.get("predictions_path"),
    }
    for camera_id in camera_ids:
        key = safe_column_name(camera_id)
        row[f"val_camera_{key}_accuracy"] = camera_accuracy(val_by_camera, camera_id)
        row[f"test_camera_{key}_accuracy"] = camera_accuracy(test_by_camera, camera_id)
    return row


def camera_accuracy(by_camera: dict[str, Any], camera_id: str) -> float | None:
    value = by_camera.get(camera_id, {}).get("accuracy")
    return value


def write_comparison_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    camera_ids: tuple[str, ...] = (),
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not camera_ids:
        camera_ids = camera_ids_from_rows(rows)
    camera_fields = [
        field
        for camera_id in camera_ids
        for field in (
            f"val_camera_{safe_column_name(camera_id)}_accuracy",
            f"test_camera_{safe_column_name(camera_id)}_accuracy",
        )
    ]
    fieldnames = [
        "run",
        "model_name",
        "pretrained",
        "device",
        "epochs",
        "train_count",
        "val_count",
        "test_count",
        "val_accuracy",
        "test_accuracy",
        "positive_label",
        "positive_threshold",
        "val_ppv",
        "val_sensitivity",
        "val_specificity",
        "test_ppv",
        "test_sensitivity",
        "test_specificity",
        *camera_fields,
        "metadata_path",
        "predictions_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_markdown(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    camera_ids: tuple[str, ...] = (),
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not camera_ids:
        camera_ids = camera_ids_from_rows(rows)
    camera_headers = [f"Test `{camera_id}`" for camera_id in camera_ids]
    header = [
        "Run",
        "Model",
        "Pretrained",
        "Val accuracy",
        "Test accuracy",
        "Test PPV",
        "Test sensitivity",
        "Test specificity",
        *camera_headers,
    ]
    lines = [
        "# Image model comparison",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", *(["---:"] * len(camera_ids))]) + " |",
    ]
    if rows:
        for row in rows:
            values = [
                f"`{row['run']}`",
                f"`{row['model_name']}`",
                str(row["pretrained"]),
                format_metric(row["val_accuracy"]),
                format_metric(row["test_accuracy"]),
                format_metric(row.get("test_ppv")),
                format_metric(row.get("test_sensitivity")),
                format_metric(row.get("test_specificity")),
            ]
            for camera_id in camera_ids:
                values.append(format_metric(row.get(f"test_camera_{safe_column_name(camera_id)}_accuracy")))
            lines.append("| " + " | ".join(values) + " |")
    else:
        lines.append("| No model metadata found |  |  |  |  |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def safe_column_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def camera_ids_from_rows(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    camera_ids = []
    prefix = "test_camera_"
    suffix = "_accuracy"
    for row in rows:
        for key in row:
            if key.startswith(prefix) and key.endswith(suffix):
                camera_ids.append(key[len(prefix) : -len(suffix)])
    return tuple(sorted(set(camera_ids)))
