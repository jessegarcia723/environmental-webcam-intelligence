import csv
from pathlib import Path

import pytest
from PIL import Image

from enviro_webcam_ml import db
from enviro_webcam_ml.image_weather_training import (
    ImageWeatherTrainingOptions,
    train_image_weather_model,
)


pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_train_image_weather_model_writes_artifacts(tmp_path: Path) -> None:
    db_path = tmp_path / "weather.sqlite3"
    training_csv = tmp_path / "training.csv"
    output_dir = tmp_path / "image_weather"

    write_image_weather_training_csv(training_csv, tmp_path)
    db.init_db(db_path)
    with db.connect(db_path) as conn:
        insert_weather_rows(conn)
        summary = train_image_weather_model(
            conn,
            ImageWeatherTrainingOptions(
                training_csv=training_csv,
                output_dir=output_dir,
                epochs=1,
                batch_size=2,
                image_size=32,
                model_name="resnet18",
                pretrained=False,
                device="cpu",
                positive_label="clouds_below_peak",
                weather_features=("cloud_cover_low", "relative_humidity_2m"),
                weather_hidden_dim=4,
                fusion_hidden_dim=8,
            ),
        )

    assert Path(summary["checkpoint_path"]).exists()
    assert Path(summary["metadata_path"]).exists()
    assert Path(summary["predictions_path"]).exists()
    assert summary["model_type"] == "image_weather_fusion"
    assert summary["matched_rows"] == 6
    assert summary["weather_features"] == ["cloud_cover_low", "relative_humidity_2m"]
    assert summary["detailed_metrics"]["test"]["binary"]["count"] == 2

    predictions = list(csv.DictReader(Path(summary["predictions_path"]).open(encoding="utf-8")))
    assert len(predictions) == 6
    assert {"weather_valid_at_utc", "weather_group", "positive_probability"} <= set(predictions[0])


def test_train_image_weather_model_can_block_by_weather_hour(tmp_path: Path) -> None:
    db_path = tmp_path / "weather.sqlite3"
    training_csv = tmp_path / "training.csv"
    output_dir = tmp_path / "image_weather"

    write_repeated_weather_hour_training_csv(training_csv, tmp_path)
    db.init_db(db_path)
    with db.connect(db_path) as conn:
        insert_weather_rows(conn, hours=4)
        summary = train_image_weather_model(
            conn,
            ImageWeatherTrainingOptions(
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
                weather_features=("cloud_cover_low", "relative_humidity_2m"),
                weather_hidden_dim=4,
                fusion_hidden_dim=8,
            ),
        )

    assert summary["split_strategy"] == "weather-hour-blocked"
    assert summary["weather_group_leakage"]["is_blocked"] is True
    assert summary["weather_group_leakage"]["groups_spanning_multiple_splits"] == 0

    predictions = list(csv.DictReader(Path(summary["predictions_path"]).open(encoding="utf-8")))
    groups_by_split = {}
    for row in predictions:
        groups_by_split.setdefault(row["weather_group"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups_by_split.values())


def write_image_weather_training_csv(path: Path, tmp_path: Path) -> None:
    labels = [
        "clouds_below_peak",
        "clouds_below_peak",
        "clouds_below_peak",
        "no_clouds_below_peak",
        "no_clouds_below_peak",
        "no_clouds_below_peak",
    ]
    splits = ["train", "train", "test", "train", "train", "test"]
    rows = []
    for hour, (label, split) in enumerate(zip(labels, splits)):
        image_path = tmp_path / f"frame_{hour}.jpg"
        color = (220, 220, 220) if label == "clouds_below_peak" else (30, 60, 90)
        Image.new("RGB", (48, 48), color=color).save(image_path)
        rows.append(
            {
                "capture_id": str(hour + 1),
                "camera_id": "camera",
                "captured_at_utc": f"2026-07-08T{hour:02d}:05:00+00:00",
                "split": split,
                "label": label,
                "image_path": str(image_path),
                "image_exists": "1",
            }
        )
    write_rows(path, rows)


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
    write_rows(path, rows)


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
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


def insert_weather_rows(conn, *, hours: int = 6) -> None:
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
