import csv
from pathlib import Path

import pytest

from enviro_webcam_ml import db
from enviro_webcam_ml.weather_lasso import WeatherLassoOptions, train_weather_lasso


pytest.importorskip("sklearn")


def test_train_weather_lasso_writes_predictions_and_coefficients(tmp_path: Path) -> None:
    db_path = tmp_path / "weather.sqlite3"
    training_csv = tmp_path / "training.csv"
    output_dir = tmp_path / "weather_lasso"

    write_training_csv(training_csv)
    db.init_db(db_path)
    with db.connect(db_path) as conn:
        for hour, cloud_cover_low in enumerate([95, 90, 85, 5, 10, 15]):
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

        summary = train_weather_lasso(
            conn,
            WeatherLassoOptions(
                training_csv=training_csv,
                output_dir=output_dir,
                positive_label="clouds_below_peak",
                c=10.0,
            ),
        )

    assert Path(summary["model_path"]).exists()
    assert Path(summary["metadata_path"]).exists()
    assert Path(summary["predictions_path"]).exists()
    assert Path(summary["coefficients_path"]).exists()
    assert summary["matched_rows"] == 6
    assert summary["split_counts"] == {"test": 2, "train": 4}
    assert summary["split_strategy"] == "training_csv"
    assert summary["weather_group_leakage"]["is_blocked"] is True
    assert "cloud_cover_low" in summary["features"]
    assert summary["detailed_metrics"]["test"]["binary"]["count"] == 2

    coefficients = list(csv.DictReader(Path(summary["coefficients_path"]).open(encoding="utf-8")))
    assert {row["feature"] for row in coefficients} == {"cloud_cover_low", "relative_humidity_2m"}


def test_train_weather_lasso_can_block_by_weather_hour(tmp_path: Path) -> None:
    db_path = tmp_path / "weather.sqlite3"
    training_csv = tmp_path / "training.csv"
    output_dir = tmp_path / "weather_lasso"

    write_repeated_weather_hour_training_csv(training_csv)
    db.init_db(db_path)
    with db.connect(db_path) as conn:
        for hour, cloud_cover_low in enumerate([95, 90, 5, 10]):
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

        summary = train_weather_lasso(
            conn,
            WeatherLassoOptions(
                training_csv=training_csv,
                output_dir=output_dir,
                positive_label="clouds_below_peak",
                split_strategy="weather-hour-blocked",
                c=10.0,
            ),
        )

    assert summary["split_strategy"] == "weather-hour-blocked"
    assert summary["weather_group_leakage"]["is_blocked"] is True
    assert summary["weather_group_leakage"]["groups_spanning_multiple_splits"] == 0
    assert summary["blocked_split_summary"]["group_split_counts"] == {"test": 2, "train": 2}
    assert summary["blocked_split_summary"]["row_split_counts"] == {"test": 4, "train": 4}

    predictions = list(csv.DictReader(Path(summary["predictions_path"]).open(encoding="utf-8")))
    groups_by_split = {}
    for row in predictions:
        groups_by_split.setdefault(row["weather_group"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups_by_split.values())


def test_train_weather_lasso_skips_rows_without_nearby_weather(tmp_path: Path) -> None:
    db_path = tmp_path / "weather.sqlite3"
    training_csv = tmp_path / "training.csv"
    output_dir = tmp_path / "weather_lasso"

    write_training_csv(training_csv)
    db.init_db(db_path)
    with db.connect(db_path) as conn:
        db.insert_weather_records(
            conn,
            provider="test",
            camera_id="camera",
            fetched_at_utc="2026-07-08T00:00:00+00:00",
            records=[
                {
                    "valid_at_utc": "2026-07-07T00:00:00+00:00",
                    "variables": {"cloud_cover_low": 95},
                }
            ],
        )

        with pytest.raises(ValueError, match="No training rows could be matched"):
            train_weather_lasso(
                conn,
                WeatherLassoOptions(
                    training_csv=training_csv,
                    output_dir=output_dir,
                    positive_label="clouds_below_peak",
                    max_weather_age_minutes=30,
                ),
            )


def write_training_csv(path: Path) -> None:
    rows = []
    labels = [
        "clouds_below_peak",
        "clouds_below_peak",
        "clouds_below_peak",
        "no_clouds_below_peak",
        "no_clouds_below_peak",
        "no_clouds_below_peak",
    ]
    splits = ["train", "train", "test", "train", "train", "test"]
    for hour, (label, split) in enumerate(zip(labels, splits)):
        rows.append(
            {
                "capture_id": str(hour + 1),
                "camera_id": "camera",
                "captured_at_utc": f"2026-07-08T{hour:02d}:05:00+00:00",
                "split": split,
                "label": label,
            }
        )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["capture_id", "camera_id", "captured_at_utc", "split", "label"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_repeated_weather_hour_training_csv(path: Path) -> None:
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
            rows.append(
                {
                    "capture_id": str(capture_id),
                    "camera_id": "camera",
                    "captured_at_utc": f"2026-07-08T{hour:02d}:{minute:02d}:00+00:00",
                    "split": original_split,
                    "label": label,
                }
            )
            capture_id += 1

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["capture_id", "camera_id", "captured_at_utc", "split", "label"],
        )
        writer.writeheader()
        writer.writerows(rows)
