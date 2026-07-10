import csv
from pathlib import Path

from PIL import Image

from enviro_webcam_ml.image_training import (
    binary_metrics,
    classification_metrics,
    label_counts_by_split,
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


def test_label_counts_by_split_cross_tabs_labels() -> None:
    rows = [
        {"split": "train", "label": "positive"},
        {"split": "train", "label": "negative"},
        {"split": "train", "label": "negative"},
        {"split": "test", "label": "positive"},
    ]

    counts = label_counts_by_split(rows, ["negative", "positive"])

    assert counts == {
        "test": {"negative": 0, "positive": 1},
        "train": {"negative": 2, "positive": 1},
    }


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


def test_binary_metrics_report_ppv_sensitivity_and_specificity() -> None:
    predictions = [
        {"true_label": "positive", "pred_label": "positive"},
        {"true_label": "positive", "pred_label": "negative"},
        {"true_label": "negative", "pred_label": "positive"},
        {"true_label": "negative", "pred_label": "negative"},
        {"true_label": "other", "pred_label": "negative"},
    ]

    metrics = binary_metrics(predictions, "positive")

    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["true_negative"] == 2
    assert metrics["false_negative"] == 1
    assert metrics["ppv"] == 0.5
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 2 / 3


def test_classification_metrics_include_binary_block_for_positive_label() -> None:
    predictions = [
        {"camera_id": "east", "true_label": "positive", "pred_label": "positive", "correct": 1},
        {"camera_id": "east", "true_label": "negative", "pred_label": "positive", "correct": 0},
    ]

    metrics = classification_metrics(predictions, ["negative", "positive"], positive_label="positive")

    assert metrics["positive_label"] == "positive"
    assert metrics["binary"]["ppv"] == 0.5
