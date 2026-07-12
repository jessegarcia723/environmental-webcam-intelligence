import csv
import json
from pathlib import Path

from enviro_webcam_ml.config import load_config
from enviro_webcam_ml.study_report import build_study_report


def test_build_study_report_summarizes_existing_outputs_and_missing_experiments(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    training_csv = data_dir / "training" / "training.csv"
    paired_csv = data_dir / "reports" / "paired_events" / "paired_events.csv"
    models_dir = data_dir / "models" / "marine_layer_detection"
    output_dir = data_dir / "reports" / "study_report"
    config_path = tmp_path / "config.yaml"

    write_config(config_path, data_dir, training_csv, models_dir)
    write_training_csv(training_csv)
    write_paired_events_csv(paired_csv)
    write_model_metadata(
        models_dir / "image_only_blocked" / "efficientnet_b0" / "metadata.json",
        model_name="efficientnet_b0",
        category="image_only",
        accuracy=0.94,
        ppv=0.89,
        sensitivity=1.0,
        specificity=0.97,
    )
    write_weather_lasso_metadata(models_dir / "weather_lasso_blocked" / "metadata.json")

    summary = build_study_report(
        config=load_config(config_path),
        task_id="marine_layer_detection",
        output_dir=output_dir,
    )

    report = Path(summary["report_path"]).read_text(encoding="utf-8")
    assert "Best times for single-camera and paired events" in report
    assert "Weather-only predictors and performance" in report
    assert "efficientnet_b0" in report
    assert "weather_lasso" in report
    assert "No paired-image neural-network runs were found yet" in report
    assert "No separate camera-specific model runs were found" in report
    assert Path(summary["model_comparison_csv"]).exists()
    assert Path(summary["single_hour_csv"]).exists()
    assert Path(summary["hour_plot_png"]).exists()


def write_config(config_path: Path, data_dir: Path, training_csv: Path, models_dir: Path) -> None:
    config_path.write_text(
        f"""
project:
  name: test
  database_path: "{data_dir / 'envirocam.sqlite3'}"
  data_dir: "{data_dir}"
cameras:
  - id: east
    name: East
    location:
      latitude: 0
      longitude: 0
      timezone: America/Los_Angeles
    capture:
      image_url: "https://example.test/east.jpg"
    pose:
      version: initial
  - id: west
    name: West
    location:
      latitude: 0
      longitude: 0
      timezone: America/Los_Angeles
    capture:
      image_url: "https://example.test/west.jpg"
    pose:
      version: initial
weather:
  provider: open_meteo
tasks:
  - id: marine_layer_detection
    default: true
    training_csv: "{training_csv}"
    model_dir: "{models_dir}"
    labels:
      - clouds_below_peak
      - no_clouds_below_peak
    positive_label: clouds_below_peak
    comparison_groups:
      camera:
        - east
        - west
""",
        encoding="utf-8",
    )


def write_training_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "capture_id": "1",
            "camera_id": "east",
            "captured_at_utc": "2026-07-08T13:00:00+00:00",
            "split": "train",
            "label": "clouds_below_peak",
        },
        {
            "capture_id": "2",
            "camera_id": "west",
            "captured_at_utc": "2026-07-08T13:00:00+00:00",
            "split": "train",
            "label": "clouds_below_peak",
        },
        {
            "capture_id": "3",
            "camera_id": "west",
            "captured_at_utc": "2026-07-08T20:00:00+00:00",
            "split": "test",
            "label": "no_clouds_below_peak",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["capture_id", "camera_id", "captured_at_utc", "split", "label"])
        writer.writeheader()
        writer.writerows(rows)


def write_paired_events_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "event_id": "event1",
            "event_time_utc": "2026-07-08T13:00:00+00:00",
            "event_time_local": "2026-07-08T06:00:00-07:00",
            "event_time_local_label": "Wed Jul 8, 2026 · 6:00 AM PDT",
            "local_date": "2026-07-08",
            "local_hour": "6",
            "event_label": "both_cameras_clouds_below_peak",
            "is_both_positive": "1",
        }
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_model_metadata(
    path: Path,
    *,
    model_name: str,
    category: str,
    accuracy: float,
    ppv: float,
    sensitivity: float,
    specificity: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_type = "image_weather_fusion" if category.startswith("image_plus") else None
    metadata = {
        "model_name": model_name,
        "model_type": model_type,
        "split_strategy": "weather-hour-blocked",
        "positive_label": "clouds_below_peak",
        "detailed_metrics": {
            "test": {
                "overall": {"accuracy": accuracy, "count": 10},
                "binary": {
                    "ppv": ppv,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "positive_label": "clouds_below_peak",
                },
                "by_camera": {
                    "east": {"accuracy": 0.9, "count": 5},
                    "west": {"accuracy": 1.0, "count": 5},
                },
            }
        },
    }
    path.write_text(json.dumps(metadata), encoding="utf-8")


def write_weather_lasso_metadata(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": "weather_lasso_logistic",
        "split_strategy": "weather-hour-blocked",
        "positive_label": "clouds_below_peak",
        "nonzero_coefficients": [
            {"feature": "temperature_2m", "coefficient": -1.2},
            {"feature": "relative_humidity_2m", "coefficient": 0.8},
        ],
        "features": ["temperature_2m", "relative_humidity_2m"],
        "detailed_metrics": {
            "test": {
                "overall": {"accuracy": 0.8, "count": 10},
                "binary": {
                    "ppv": 0.7,
                    "sensitivity": 1.0,
                    "specificity": 0.75,
                    "positive_label": "clouds_below_peak",
                },
                "by_camera": {},
            }
        },
    }
    path.write_text(json.dumps(metadata), encoding="utf-8")
