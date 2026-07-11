import csv
from pathlib import Path

import pytest
from PIL import Image

from enviro_webcam_ml import db
from enviro_webcam_ml.image_training import (
    ImageTrainingOptions,
    binary_metrics,
    classification_metrics,
    label_counts_by_split,
    metric_summary,
    read_training_rows,
    train_image_model,
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


def test_train_image_model_can_block_by_weather_hour(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    db_path = tmp_path / "weather.sqlite3"
    training_csv = tmp_path / "training.csv"
    output_dir = tmp_path / "image_model"

    write_repeated_weather_hour_training_csv(training_csv, tmp_path)
    db.init_db(db_path)
    with db.connect(db_path) as conn:
        insert_weather_rows(conn, hours=4)
        summary = train_image_model(
            ImageTrainingOptions(
                training_csv=training_csv,
                output_dir=output_dir,
                epochs=1,
                batch_size=2,
                image_size=32,
                model_name="resnet18",
                pretrained=False,
                device="cpu",
                positive_label="clouds_below_peak",
                split_strategy="weather-hour-blocked",
            ),
            conn=conn,
        )

    assert summary["split_strategy"] == "weather-hour-blocked"
    assert summary["matched_rows"] == 8
    assert summary["weather_group_leakage"]["is_blocked"] is True
    assert summary["weather_group_leakage"]["groups_spanning_multiple_splits"] == 0
    assert Path(summary["checkpoint_path"]).exists()
    assert Path(summary["metadata_path"]).exists()
    assert Path(summary["predictions_path"]).exists()

    predictions = list(csv.DictReader(Path(summary["predictions_path"]).open(encoding="utf-8")))
    groups_by_split = {}
    for row in predictions:
        groups_by_split.setdefault(row["weather_group"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups_by_split.values())
    assert {"weather_valid_at_utc", "weather_group", "weather_age_minutes"} <= set(predictions[0])


def write_repeated_weather_hour_training_csv(path: Path, tmp_path: Path) -> None:
    rows = []
    group_labels = [
        ("clouds_below_peak", 0),
        ("clouds_below_peak", 1),
        ("no_clouds_below_peak", 2),
        ("no_clouds_below_peak", 3),
    ]
    capture_id = 1
    for label, hour in group_labels:
        for minute, original_split in ((5, "train"), (10, "test")):
            image_path = tmp_path / f"frame_{capture_id}.jpg"
            color = (220, 220, 220) if label == "clouds_below_peak" else (30, 60, 90)
            Image.new("RGB", (48, 48), color=color).save(image_path)
            rows.append(
                {
                    "capture_id": str(capture_id),
                    "camera_id": "camera",
                    "captured_at_utc": f"2026-07-08T{hour:02d}:{minute:02d}:00+00:00",
                    "split": original_split,
                    "label": label,
                    "image_path": str(image_path),
                    "image_exists": "1",
                }
            )
            capture_id += 1
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "capture_id",
                "camera_id",
                "captured_at_utc",
                "split",
                "label",
                "image_path",
                "image_exists",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def insert_weather_rows(conn, *, hours: int) -> None:
    for hour in range(hours):
        cloud_cover_low = 90 if hour < 2 else 10
        db.insert_weather_records(
            conn,
            provider="test",
            camera_id="camera",
            fetched_at_utc=f"2026-07-08T{hour:02d}:10:00+00:00",
            records=[
                {
                    "valid_at_utc": f"2026-07-08T{hour:02d}:00:00+00:00",
                    "variables": {
                        "cloud_cover_low": cloud_cover_low,
                        "relative_humidity_2m": 80 if cloud_cover_low > 50 else 35,
                    },
                }
            ],
        )
