import csv
from pathlib import Path

from PIL import Image

from enviro_webcam_ml.image_training import (
    classification_metrics,
    metric_summary,
    read_training_rows,
)


def test_read_training_rows_keeps_only_existing_images(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(image)
    missing = tmp_path / "missing.jpg"
    csv_path = tmp_path / "training.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "image_exists", "label", "split"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_path": str(image),
                "image_exists": "1",
                "label": "clouds_below_peak",
                "split": "train",
            }
        )
        writer.writerow(
            {
                "image_path": str(missing),
                "image_exists": "1",
                "label": "clouds_below_peak",
                "split": "train",
            }
        )
        writer.writerow(
            {
                "image_path": str(image),
                "image_exists": "0",
                "label": "clouds_below_peak",
                "split": "train",
            }
        )

    rows = read_training_rows(csv_path)

    assert len(rows) == 1
    assert rows[0]["image_path"] == str(image)


def test_metric_summary_handles_empty_and_non_empty_counts() -> None:
    assert metric_summary(0.0, 0, 0) == {"loss": None, "accuracy": None, "count": 0}
    assert metric_summary(2.0, 3, 4) == {"loss": 0.5, "accuracy": 0.75, "count": 4}


def test_classification_metrics_include_label_and_camera_breakdowns() -> None:
    predictions = [
        {
            "camera_id": "east",
            "true_label": "clouds_below_peak",
            "pred_label": "clouds_below_peak",
            "correct": 1,
        },
        {
            "camera_id": "west",
            "true_label": "no_clouds_below_peak",
            "pred_label": "clouds_below_peak",
            "correct": 0,
        },
    ]

    metrics = classification_metrics(
        predictions,
        ["clouds_below_peak", "no_clouds_below_peak"],
    )

    assert metrics["overall"] == {"accuracy": 0.5, "count": 2}
    assert metrics["by_camera"]["east"] == {"accuracy": 1.0, "count": 1}
    assert metrics["by_camera"]["west"] == {"accuracy": 0.0, "count": 1}
    assert metrics["by_label"]["clouds_below_peak"]["recall"] == 1.0
    assert metrics["by_label"]["no_clouds_below_peak"]["recall"] == 0.0
