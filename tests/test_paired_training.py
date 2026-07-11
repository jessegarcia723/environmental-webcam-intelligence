import csv
from pathlib import Path

import pytest
from PIL import Image

from enviro_webcam_ml import db
from enviro_webcam_ml.paired_events import NEGATIVE_EVENT_LABEL, POSITIVE_EVENT_LABEL
from enviro_webcam_ml.paired_image_training import PairedImageTrainingOptions, train_paired_image_model
from enviro_webcam_ml.paired_weather_lasso import PairedWeatherLassoOptions, train_paired_weather_lasso


pytest.importorskip("sklearn")


def test_train_paired_weather_lasso_writes_artifacts(tmp_path: Path) -> None:
    db_path = tmp_path / "weather.sqlite3"
    paired_csv = tmp_path / "paired_events.csv"
    output_dir = tmp_path / "paired_weather_lasso"
    write_paired_events_csv(paired_csv, tmp_path, with_images=False)

    db.init_db(db_path)
    with db.connect(db_path) as conn:
        insert_weather(conn, hours=6)
        summary = train_paired_weather_lasso(
            conn,
            PairedWeatherLassoOptions(
                paired_events_csv=paired_csv,
                output_dir=output_dir,
                camera_ids=("east", "west"),
                c=10.0,
            ),
        )

    assert summary["model_type"] == "paired_weather_lasso"
    assert summary["event_scope"] == "paired_event"
    assert summary["matched_rows"] == 6
    assert summary["weather_group_leakage"]["is_blocked"] is True
    assert Path(summary["model_path"]).exists()
    assert Path(summary["metadata_path"]).exists()
    assert Path(summary["predictions_path"]).exists()
    assert Path(summary["coefficients_path"]).exists()
    assert any(feature.startswith("east__") for feature in summary["features"])
    assert any(feature.startswith("west__") for feature in summary["features"])
    assert summary["detailed_metrics"]["test"]["binary"]["count"] == 2


def test_train_paired_image_model_writes_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    paired_csv = tmp_path / "paired_events.csv"
    output_dir = tmp_path / "paired_image_model"
    write_paired_events_csv(paired_csv, tmp_path, with_images=True)

    summary = train_paired_image_model(
        PairedImageTrainingOptions(
            paired_events_csv=paired_csv,
            output_dir=output_dir,
            camera_ids=("east", "west"),
            epochs=1,
            batch_size=2,
            image_size=32,
            model_name="resnet18",
            pretrained=False,
            device="cpu",
            fusion_hidden_dim=8,
        )
    )

    assert summary["model_type"] == "paired_image_fusion"
    assert summary["event_scope"] == "paired_event"
    assert summary["split_strategy"] == "event-hour-blocked"
    assert summary["blocked_split_summary"]["group_count"] == 6
    assert summary["matched_rows"] == 6
    assert Path(summary["checkpoint_path"]).exists()
    assert Path(summary["metadata_path"]).exists()
    assert Path(summary["predictions_path"]).exists()
    assert summary["detailed_metrics"]["test"]["binary"]["count"] == 2


def write_paired_events_csv(path: Path, tmp_path: Path, *, with_images: bool) -> None:
    rows = []
    labels = [POSITIVE_EVENT_LABEL, POSITIVE_EVENT_LABEL, POSITIVE_EVENT_LABEL, NEGATIVE_EVENT_LABEL, NEGATIVE_EVENT_LABEL, NEGATIVE_EVENT_LABEL]
    for hour, label in enumerate(labels):
        east_image = tmp_path / f"east_{hour}.jpg"
        west_image = tmp_path / f"west_{hour}.jpg"
        if with_images:
            color = (220, 220, 220) if label == POSITIVE_EVENT_LABEL else (20, 50, 90)
            Image.new("RGB", (48, 48), color=color).save(east_image)
            Image.new("RGB", (48, 48), color=color).save(west_image)
        rows.append(
            {
                "event_id": f"event_{hour}",
                "event_time_utc": f"2026-07-08T{hour:02d}:05:00+00:00",
                "event_time_local": f"2026-07-07T{17 + hour:02d}:05:00-07:00",
                "event_time_local_label": f"event {hour}",
                "local_date": "2026-07-07",
                "local_hour": str((17 + hour) % 24),
                "event_label": label,
                "is_both_positive": "1" if label == POSITIVE_EVENT_LABEL else "0",
                "label_pair": "clouds_below_peak|clouds_below_peak" if label == POSITIVE_EVENT_LABEL else "no_clouds_below_peak|no_clouds_below_peak",
                "east_captured_at_utc": f"2026-07-08T{hour:02d}:05:00+00:00",
                "east_image_path": str(east_image),
                "east_label": "clouds_below_peak" if label == POSITIVE_EVENT_LABEL else "no_clouds_below_peak",
                "west_captured_at_utc": f"2026-07-08T{hour:02d}:06:00+00:00",
                "west_image_path": str(west_image),
                "west_label": "clouds_below_peak" if label == POSITIVE_EVENT_LABEL else "no_clouds_below_peak",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def insert_weather(conn, *, hours: int) -> None:
    for camera_id in ("east", "west"):
        for hour in range(hours):
            positive_weather = hour < 3
            cloud_cover_low = 95 if positive_weather else 5
            db.insert_weather_records(
                conn,
                provider="test",
                camera_id=camera_id,
                fetched_at_utc=f"2026-07-08T{hour:02d}:10:00+00:00",
                records=[
                    {
                        "valid_at_utc": f"2026-07-08T{hour:02d}:00:00+00:00",
                        "variables": {
                            "cloud_cover_low": cloud_cover_low,
                            "temperature_2m": 10 if positive_weather else 25,
                        },
                    }
                ],
            )
