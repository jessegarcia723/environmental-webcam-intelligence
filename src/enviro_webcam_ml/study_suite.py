from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enviro_webcam_ml.config import AppConfig
from enviro_webcam_ml.image_training import ImageTrainingOptions, train_image_model
from enviro_webcam_ml.paired_events import PairedEventOptions, build_paired_events
from enviro_webcam_ml.paired_image_training import PairedImageTrainingOptions, train_paired_image_model
from enviro_webcam_ml.paired_weather_lasso import PairedWeatherLassoOptions, train_paired_weather_lasso
from enviro_webcam_ml.study_report import build_study_report
from enviro_webcam_ml.weather_classifier import (
    SUPPORTED_WEATHER_CLASSIFIERS,
    PairedWeatherClassifierOptions,
    WeatherClassifierOptions,
    train_paired_weather_classifier,
    train_weather_classifier,
)
from enviro_webcam_ml.weather_lasso import WeatherLassoOptions, train_weather_lasso


DEFAULT_WEATHER_MODELS = ("lasso", *SUPPORTED_WEATHER_CLASSIFIERS)


@dataclass(frozen=True)
class StudySuiteOptions:
    task_id: str
    model_name: str = "efficientnet_b0"
    epochs: int = 8
    batch_size: int = 16
    learning_rate: float = 0.001
    image_size: int = 224
    num_workers: int = 0
    pretrained: bool = True
    device: str = "auto"
    max_pair_minutes: float = 3.0
    max_weather_age_minutes: float = 90.0
    lasso_c: float = 1.0
    lasso_class_weight: str = "none"
    weather_models: tuple[str, ...] = DEFAULT_WEATHER_MODELS
    paired_image_split_strategy: str = "event-hour-blocked"
    output_reports_dir: Path | None = None
    models_dir: Path | None = None
    skip_weather_only_models: bool = False
    skip_paired_weather_lasso: bool = False
    skip_paired_image_model: bool = False
    skip_camera_specific_models: bool = False


def run_study_suite(
    conn: sqlite3.Connection,
    *,
    config: AppConfig,
    options: StudySuiteOptions,
) -> dict[str, Any]:
    task_id = options.task_id
    reports_dir = options.output_reports_dir or config.data_dir / "reports"
    models_dir = options.models_dir or config.task_model_dir(task_id)
    paired_events_dir = reports_dir / "paired_events"
    study_report_dir = reports_dir / "study_report"
    camera_ids = config.task_comparison_camera_ids(task_id)
    if len(camera_ids) != 2:
        raise ValueError("Study suite needs exactly two cameras in task comparison_groups.camera.")
    positive_label = config.task_positive_label(task_id)
    if not positive_label:
        raise ValueError("Study suite requires task.positive_label.")
    crop_pixels = config.task_image_crop_pixels(task_id)

    paired_summary = build_paired_events(
        conn,
        config=config,
        options=PairedEventOptions(
            task_id=task_id,
            output_dir=paired_events_dir,
            camera_ids=(camera_ids[0], camera_ids[1]),
            positive_label=positive_label,
            max_pair_minutes=options.max_pair_minutes,
            crop_pixels=crop_pixels,
            timezone_name=config.cameras[0].location.timezone if config.cameras else "UTC",
        ),
    )
    paired_events_csv = Path(paired_summary["paths"]["events_csv"])

    weather_only_summaries = []
    paired_weather_summary = None
    if not options.skip_weather_only_models:
        weather_only_summaries = train_study_weather_only_models(
            conn,
            config=config,
            task_id=task_id,
            paired_events_csv=paired_events_csv,
            camera_ids=(camera_ids[0], camera_ids[1]),
            models_dir=models_dir,
            positive_label=positive_label,
            weather_models=options.weather_models,
            max_weather_age_minutes=options.max_weather_age_minutes,
            lasso_c=options.lasso_c,
            lasso_class_weight=options.lasso_class_weight,
            skip_paired_weather_lasso=options.skip_paired_weather_lasso,
        )
        paired_weather_summary = next(
            (
                summary for summary in weather_only_summaries
                if summary.get("model_name") == "weather_lasso_logistic"
                and summary.get("event_scope") == "paired_event"
            ),
            None,
        )

    paired_image_summary = None
    if not options.skip_paired_image_model:
        paired_image_summary = train_paired_image_model(
            PairedImageTrainingOptions(
                paired_events_csv=paired_events_csv,
                output_dir=models_dir / f"paired_image_{options.model_name}",
                camera_ids=(camera_ids[0], camera_ids[1]),
                epochs=options.epochs,
                batch_size=options.batch_size,
                learning_rate=options.learning_rate,
                image_size=options.image_size,
                num_workers=options.num_workers,
                model_name=options.model_name,
                pretrained=options.pretrained,
                device=options.device,
                crop_pixels=crop_pixels,
                split_strategy=options.paired_image_split_strategy,
            )
        )

    camera_specific_summaries = []
    if not options.skip_camera_specific_models:
        camera_specific_summaries = train_camera_specific_models(
            conn,
            config=config,
            task_id=task_id,
            camera_ids=camera_ids,
            models_dir=models_dir,
            model_name=options.model_name,
            epochs=options.epochs,
            batch_size=options.batch_size,
            learning_rate=options.learning_rate,
            image_size=options.image_size,
            num_workers=options.num_workers,
            pretrained=options.pretrained,
            device=options.device,
            crop_pixels=crop_pixels,
            positive_label=positive_label,
            max_weather_age_minutes=options.max_weather_age_minutes,
        )

    report_summary = build_study_report(
        config=config,
        task_id=task_id,
        output_dir=study_report_dir,
        paired_events_csv=paired_events_csv,
        models_dir=models_dir,
    )
    return {
        "paired_events": paired_summary,
        "paired_weather_lasso": paired_weather_summary,
        "weather_only_models": weather_only_summaries,
        "paired_image_model": paired_image_summary,
        "camera_specific_models": camera_specific_summaries,
        "study_report": report_summary,
    }


def train_study_weather_only_models(
    conn: sqlite3.Connection,
    *,
    config: AppConfig,
    task_id: str,
    paired_events_csv: Path,
    camera_ids: tuple[str, str],
    models_dir: Path,
    positive_label: str,
    weather_models: tuple[str, ...],
    max_weather_age_minutes: float,
    lasso_c: float,
    lasso_class_weight: str,
    skip_paired_weather_lasso: bool,
) -> list[dict[str, Any]]:
    validate_weather_models(weather_models)
    summaries = []
    training_csv = config.task_training_csv_path(task_id)
    for model_kind in weather_models:
        if model_kind == "lasso":
            summaries.append(
                train_weather_lasso(
                    conn,
                    WeatherLassoOptions(
                        training_csv=training_csv,
                        output_dir=models_dir / "weather_only" / "lasso",
                        positive_label=positive_label,
                        max_weather_age_minutes=max_weather_age_minutes,
                        c=lasso_c,
                        class_weight=lasso_class_weight,
                        split_strategy="weather-hour-blocked",
                    ),
                )
            )
            if not skip_paired_weather_lasso:
                summaries.append(
                    train_paired_weather_lasso(
                        conn,
                        PairedWeatherLassoOptions(
                            paired_events_csv=paired_events_csv,
                            output_dir=models_dir / "paired_weather_lasso",
                            camera_ids=camera_ids,
                            max_weather_age_minutes=max_weather_age_minutes,
                            c=lasso_c,
                            class_weight=lasso_class_weight,
                        ),
                    )
                )
            continue
        summaries.append(
            train_weather_classifier(
                conn,
                WeatherClassifierOptions(
                    training_csv=training_csv,
                    output_dir=models_dir / "weather_only" / model_kind,
                    positive_label=positive_label,
                    model_kind=model_kind,
                    max_weather_age_minutes=max_weather_age_minutes,
                    c=lasso_c,
                    class_weight=lasso_class_weight,
                    split_strategy="weather-hour-blocked",
                ),
            )
        )
        summaries.append(
            train_paired_weather_classifier(
                conn,
                PairedWeatherClassifierOptions(
                    paired_events_csv=paired_events_csv,
                    output_dir=models_dir / f"paired_weather_{model_kind}",
                    camera_ids=camera_ids,
                    model_kind=model_kind,
                    max_weather_age_minutes=max_weather_age_minutes,
                    c=lasso_c,
                    class_weight=lasso_class_weight,
                    split_strategy="weather-hour-blocked",
                ),
            )
        )
    return summaries


def validate_weather_models(weather_models: tuple[str, ...]) -> None:
    valid = set(DEFAULT_WEATHER_MODELS)
    unknown = sorted(set(weather_models) - valid)
    if unknown:
        raise ValueError(f"Unknown weather model(s): {unknown}. Choose from {sorted(valid)}.")


def train_camera_specific_models(
    conn: sqlite3.Connection,
    *,
    config: AppConfig,
    task_id: str,
    camera_ids: tuple[str, ...],
    models_dir: Path,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    image_size: int,
    num_workers: int,
    pretrained: bool,
    device: str,
    crop_pixels,
    positive_label: str,
    max_weather_age_minutes: float,
) -> list[dict[str, Any]]:
    source_csv = config.task_training_csv_path(task_id)
    rows = read_csv_rows(source_csv)
    summaries = []
    for camera_id in camera_ids:
        camera_rows = [row for row in rows if row.get("camera_id") == camera_id]
        if len({row.get("label") for row in camera_rows}) < 2:
            continue
        filtered_csv = config.data_dir / "training" / "camera_specific" / f"{task_id}_{camera_id}_training.csv"
        write_rows(filtered_csv, camera_rows)
        output_dir = models_dir / "camera_specific" / camera_id / model_name
        summaries.append(
            train_image_model(
                ImageTrainingOptions(
                    training_csv=filtered_csv,
                    output_dir=output_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    image_size=image_size,
                    num_workers=num_workers,
                    model_name=model_name,
                    pretrained=pretrained,
                    device=device,
                    crop_pixels=crop_pixels,
                    positive_label=positive_label,
                    split_strategy="weather-hour-blocked",
                    max_weather_age_minutes=max_weather_age_minutes,
                ),
                conn=conn,
            )
        )
    return summaries


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
