from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def compare_image_models(models_dir: Path, output_csv: Path, output_md: Path | None = None) -> list[dict[str, Any]]:
    metadata_files = sorted(models_dir.glob("*/metadata.json"))
    rows = [comparison_row(path) for path in metadata_files]
    rows.sort(
        key=lambda row: (
            row["test_accuracy"] is not None,
            row["test_accuracy"] if row["test_accuracy"] is not None else -1,
            row["val_accuracy"] if row["val_accuracy"] is not None else -1,
        ),
        reverse=True,
    )
    write_comparison_csv(rows, output_csv)
    if output_md is not None:
        write_comparison_markdown(rows, output_md)
    return rows


def comparison_row(metadata_path: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    detailed = metadata.get("detailed_metrics", {})
    val_overall = detailed.get("val", {}).get("overall", {})
    test_overall = detailed.get("test", {}).get("overall", {})
    test_by_camera = detailed.get("test", {}).get("by_camera", {})
    val_by_camera = detailed.get("val", {}).get("by_camera", {})

    return {
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
        "val_east_accuracy": camera_accuracy(val_by_camera, "mount_tam_east_peak"),
        "val_west_accuracy": camera_accuracy(val_by_camera, "mount_tam_west_peak"),
        "test_east_accuracy": camera_accuracy(test_by_camera, "mount_tam_east_peak"),
        "test_west_accuracy": camera_accuracy(test_by_camera, "mount_tam_west_peak"),
        "metadata_path": str(metadata_path),
        "predictions_path": metadata.get("predictions_path"),
    }


def camera_accuracy(by_camera: dict[str, Any], camera_id: str) -> float | None:
    value = by_camera.get(camera_id, {}).get("accuracy")
    return value


def write_comparison_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
        "val_east_accuracy",
        "val_west_accuracy",
        "test_east_accuracy",
        "test_west_accuracy",
        "metadata_path",
        "predictions_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Image model comparison",
        "",
        "| Run | Model | Pretrained | Val accuracy | Test accuracy | Test east | Test west |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    if rows:
        for row in rows:
            lines.append(
                "| "
                f"`{row['run']}` | "
                f"`{row['model_name']}` | "
                f"{row['pretrained']} | "
                f"{format_metric(row['val_accuracy'])} | "
                f"{format_metric(row['test_accuracy'])} | "
                f"{format_metric(row['test_east_accuracy'])} | "
                f"{format_metric(row['test_west_accuracy'])} |"
            )
    else:
        lines.append("| No model metadata found |  |  |  |  |  |  |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"
