import csv
from pathlib import Path

import pytest

from enviro_webcam_ml import db
from enviro_webcam_ml.paired_events import NEGATIVE_EVENT_LABEL, POSITIVE_EVENT_LABEL
from enviro_webcam_ml.weather_classifier import (
    PairedWeatherClassifierOptions,
    WeatherClassifierOptions,
    train_paired_weather_classifier,
    train_weather_classifier,
)


pytest.importorskip("sklearn")


def test_train_extra_weather_classifiers_write_artifacts(tmp_path: Path) -> None:
    db_path = tmp_path / "weather.sqlite3"
    training_csv = tmp_path / "training.csv"
    write_training_csv(training_csv)
    db.init_db(db_path)
    with db.connect(db_path) as conn:
        insert_single_weather(conn)
        for model_kind in ("ridge_logistic", "random_forest", "hist_gradient_boosting"):
            output_dir = tmp_path / model_kind
            summary = train_weather_classifier(
                conn,
                WeatherClassifierOptions(
                    training_csv=training_csv,
                    output_dir=output_dir,
                    positive_label="clouds_below_peak",
                    model_kind=model_kind,
                    split_strategy="weather-hour-blocked",
                ),
            )

            assert summary["model_name"] == f"weather_{model_kind}"
            assert summary["model_type"] == "weather_classifier"
            assert summary["event_scope"] == "single_image"
            assert summary["matched_rows"] == 8
            assert Path(summary["model_path"]).exists()
            assert Path(summary["metadata_path"]).exists()
            assert Path(summary["predictions_path"]).exists()
            assert Path(summary["feature_importances_path"]).exists()
            assert summary["detailed_metrics"]["test"]["binary"]["count"] == 2


def test_train_paired_weather_classifier_writes_artifacts(tmp_path: Path) -> None:
    db_path = tmp_path / "weather.sqlite3"
    paired_csv = tmp_path / "paired_events.csv"
    write_paired_events_csv(paired_csv)
    db.init_db(db_path)
    with db.connect(db_path) as conn:
        insert_paired_weather(conn)
        summary = train_paired_weather_classifier(
            conn,
            PairedWeatherClassifierOptions(
                paired_events_csv=paired_csv,
                output_dir=tmp_path / "paired_weather_random_forest",
                camera_ids=("east", "west"),
                model_kind="random_forest",
            ),
        )

    assert summary["model_name"] == "weather_random_forest"
    assert summary["model_type"] == "weather_classifier"
    assert summary["event_scope"] == "paired_event"
    assert summary["matched_rows"] == 8
    assert Path(summary["model_path"]).exists()
    assert Path(summary["metadata_path"]).exists()
    assert Path(summary["predictions_path"]).exists()
    assert summary["detailed_metrics"]["test"]["binary"]["count"] == 2


def write_training_csv(path: Path) -> None:
    rows = []
    labels = [
        "clouds_below_peak",
        "clouds_below_peak",
        "clouds_below_peak",
        "clouds_below_peak",
        "no_clouds_below_peak",
        "no_clouds_below_peak",
        "no_clouds_below_peak",
        "no_clouds_below_peak",
    ]
    for hour, label in enumerate(labels):
        rows.append(
            {
                "capture_id": str(hour + 1),
                "camera_id": "camera",
                "captured_at_utc": f"2026-07-08T{hour:02d}:05:00+00:00",
                "split": "train" if hour in (0, 1, 4, 5) else "test",
                "label": label,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["capture_id", "camera_id", "captured_at_utc", "split", "label"])
        writer.writeheader()
        writer.writerows(rows)


def write_paired_events_csv(path: Path) -> None:
    rows = []
    labels = [POSITIVE_EVENT_LABEL] * 4 + [NEGATIVE_EVENT_LABEL] * 4
    for hour, label in enumerate(labels):
        rows.append(
            {
                "event_id": f"event_{hour}",
                "event_time_utc": f"2026-07-08T{hour:02d}:05:00+00:00",
                "event_time_local": "",
                "event_time_local_label": "",
                "local_date": "",
                "local_hour": "",
                "event_label": label,
                "is_both_positive": "1" if label == POSITIVE_EVENT_LABEL else "0",
                "label_pair": "",
                "east_captured_at_utc": f"2026-07-08T{hour:02d}:05:00+00:00",
                "east_image_path": "",
                "east_label": "",
                "west_captured_at_utc": f"2026-07-08T{hour:02d}:06:00+00:00",
                "west_image_path": "",
                "west_label": "",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def insert_single_weather(conn) -> None:
    for hour in range(8):
        positive_weather = hour < 4
        db.insert_weather_records(
            conn,
            provider="test",
            camera_id="camera",
            fetched_at_utc=f"2026-07-08T{hour:02d}:10:00+00:00",
            records=[
                {
                    "valid_at_utc": f"2026-07-08T{hour:02d}:00:00+00:00",
                    "variables": weather_variables(positive_weather),
                }
            ],
        )


def insert_paired_weather(conn) -> None:
    for camera_id in ("east", "west"):
        for hour in range(8):
            positive_weather = hour < 4
            db.insert_weather_records(
                conn,
                provider="test",
                camera_id=camera_id,
                fetched_at_utc=f"2026-07-08T{hour:02d}:10:00+00:00",
                records=[
                    {
                        "valid_at_utc": f"2026-07-08T{hour:02d}:00:00+00:00",
                        "variables": weather_variables(positive_weather),
                    }
                ],
            )


def weather_variables(positive_weather: bool) -> dict[str, float]:
    return {
        "cloud_cover_low": 95.0 if positive_weather else 5.0,
        "relative_humidity_2m": 85.0 if positive_weather else 30.0,
        "temperature_2m": 10.0 if positive_weather else 25.0,
    }
