from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from enviro_webcam_ml import db
from enviro_webcam_ml.annotation_analysis import (
    analyze_annotations,
    write_analysis_markdown,
    write_disagreements_csv,
)
from enviro_webcam_ml.annotation import (
    AdjudicationServerOptions,
    AnnotationServerOptions,
    favorite_rows,
    serve_adjudication_app,
    serve_annotation_app,
)
from enviro_webcam_ml.backup import backup_sqlite_database
from enviro_webcam_ml.capture import capture_once
from enviro_webcam_ml.clock import ClockSanityChecker
from enviro_webcam_ml.config import AppConfig, CameraConfig, load_config
from enviro_webcam_ml.dataset import build_manifest
from enviro_webcam_ml.image_explanations import ImageExplanationOptions, explain_image_model
from enviro_webcam_ml.image_paths import resolve_image_path
from enviro_webcam_ml.image_training import ImageTrainingOptions, train_image_model
from enviro_webcam_ml.model_comparison import compare_image_models
from enviro_webcam_ml.training_dataset import TrainingSetOptions, build_training_set
from enviro_webcam_ml.training_env import training_environment_report
from enviro_webcam_ml.weather_lasso import WeatherLassoOptions, train_weather_lasso
from enviro_webcam_ml.weather.open_meteo import fetch_forecast


DEFAULT_IMAGE_MODELS = ("resnet18", "efficientnet_b0", "mobilenet_v3_small")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 - CLI should return readable failures.
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="envirocam",
        description="Environmental webcam ML data pipeline MVP.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_db = sub.add_parser("init-db", help="Create/update the SQLite schema and register configured cameras.")
    init_db.add_argument("--config", required=True)
    init_db.set_defaults(func=cmd_init_db)

    capture = sub.add_parser("capture-once", help="Capture one frame per configured camera.")
    capture.add_argument("--config", required=True)
    capture.add_argument("--camera-id", help="Limit capture to one camera.")
    capture.set_defaults(func=cmd_capture_once)

    capture_loop = sub.add_parser("capture-loop", help="Continuously capture frames using configured intervals.")
    capture_loop.add_argument("--config", required=True)
    capture_loop.add_argument("--camera-id", help="Limit capture to one camera.")
    capture_loop.add_argument(
        "--max-iterations",
        type=int,
        help="Stop after this many loop iterations; useful for smoke tests.",
    )
    capture_loop.set_defaults(func=cmd_capture_loop)

    weather = sub.add_parser("fetch-weather", help="Fetch weather forecast/past hourly values for configured cameras.")
    weather.add_argument("--config", required=True)
    weather.add_argument("--camera-id", help="Limit weather fetch to one camera.")
    weather.set_defaults(func=cmd_fetch_weather)

    collector = sub.add_parser(
        "run-collector",
        help="Continuously capture webcam frames and fetch weather on separate schedules.",
    )
    collector.add_argument("--config", required=True)
    collector.add_argument("--camera-id", help="Limit collection to one camera.")
    collector.add_argument(
        "--capture-interval-seconds",
        type=int,
        help="Override the configured capture interval; useful for testing.",
    )
    collector.add_argument(
        "--weather-interval-seconds",
        type=int,
        help="Override the configured weather fetch interval; useful for testing.",
    )
    collector.add_argument(
        "--skip-initial-weather",
        action="store_true",
        help="Do not fetch weather immediately on startup.",
    )
    collector.add_argument(
        "--disable-clock-sanity",
        action="store_true",
        help="Disable wall-clock sanity checks. Not recommended for unattended collection.",
    )
    collector.add_argument(
        "--max-iterations",
        type=int,
        help="Stop after this many scheduler iterations; useful for smoke tests.",
    )
    collector.set_defaults(func=cmd_run_collector)

    manifest = sub.add_parser("build-manifest", help="Build a CSV frame manifest for annotation/training.")
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(func=cmd_build_manifest)

    annotate = sub.add_parser("annotate", help="Run the local split-screen annotation web app.")
    annotate.add_argument("--config", required=True)
    annotate.add_argument("--host", default="127.0.0.1")
    annotate.add_argument("--port", type=int, default=8000)
    annotate.add_argument("--task-id", help="Defaults to the config task marked default: true, or the first task.")
    annotate.add_argument("--left-annotator", default="left")
    annotate.add_argument("--right-annotator", default="right")
    annotate.add_argument("--open-browser", action="store_true")
    annotate.set_defaults(func=cmd_annotate)

    adjudicate = sub.add_parser(
        "adjudicate",
        help="Run a shared adjudication app for double-labeled disagreements.",
    )
    adjudicate.add_argument("--config", required=True)
    adjudicate.add_argument("--host", default="127.0.0.1")
    adjudicate.add_argument("--port", type=int, default=8001)
    adjudicate.add_argument("--task-id", help="Defaults to the config task marked default: true, or the first task.")
    adjudicate.add_argument("--adjudicator", default="joint")
    adjudicate.add_argument(
        "--predictions",
        help="Optional model predictions CSV. Defaults to task.model_dir/<model-name>/predictions.csv.",
    )
    adjudicate.add_argument(
        "--checkpoint",
        help="Optional model checkpoint. Defaults to task.model_dir/<model-name>/model.pt for missing predictions.",
    )
    adjudicate.add_argument(
        "--model-name",
        default="efficientnet_b0",
        help="Model subdirectory used to find predictions when --predictions is omitted.",
    )
    adjudicate.add_argument("--device", default="auto", help="Device for live checkpoint inference of missing predictions.")
    adjudicate.add_argument(
        "--annotator",
        action="append",
        default=[],
        help="Only adjudicate labels from this annotator. Pass twice for the two-player workflow.",
    )
    adjudicate.add_argument(
        "--include-agreements",
        action="store_true",
        help="Adjudicate all double-labeled frames instead of disagreements only.",
    )
    adjudicate.add_argument("--open-browser", action="store_true")
    adjudicate.set_defaults(func=cmd_adjudicate)

    analysis = sub.add_parser(
        "analyze-annotations",
        help="Analyze annotation counts, multi-rater agreement, and disagreements.",
    )
    analysis.add_argument("--config", required=True)
    analysis.add_argument("--task-id", help="Defaults to the config task marked default: true, or the first task.")
    analysis.add_argument("--output", default="data/reports/annotation_analysis.md")
    analysis.add_argument("--disagreements-output", default="data/reports/disagreements.csv")
    analysis.set_defaults(func=cmd_analyze_annotations)

    favorites = sub.add_parser("export-favorites", help="Export bookmarked favorite frames to CSV.")
    favorites.add_argument("--config", required=True)
    favorites.add_argument("--task-id", help="Defaults to the config task marked default: true, or the first task.")
    favorites.add_argument("--output", default="data/reports/favorites.csv")
    favorites.set_defaults(func=cmd_export_favorites)

    backup = sub.add_parser("backup-db", help="Write a consistent SQLite database snapshot.")
    backup.add_argument("--config", required=True)
    backup.add_argument("--output", required=True)
    backup.set_defaults(func=cmd_backup_db)

    train_env = sub.add_parser("check-training-env", help="Print installed ML packages and accelerator support.")
    train_env.set_defaults(func=cmd_check_training_env)

    training_set = sub.add_parser(
        "build-training-set",
        help="Build a clean training CSV from agreed multi-rater annotations.",
    )
    training_set.add_argument("--config", required=True)
    training_set.add_argument("--task-id", help="Defaults to the config task marked default: true, or the first task.")
    training_set.add_argument("--output", help="Defaults to task.training_csv or data/training/<task>_training.csv.")
    training_set.add_argument(
        "--include-label",
        action="append",
        default=[],
        help="Label to include. Can be passed multiple times. Defaults to all configured labels.",
    )
    training_set.add_argument(
        "--exclude-label",
        action="append",
        default=[],
        help="Label to exclude. Can be passed multiple times. Defaults to task.excluded_training_labels.",
    )
    training_set.add_argument("--min-annotators", type=int, default=2)
    training_set.add_argument("--train-fraction", type=float, default=0.70)
    training_set.add_argument("--val-fraction", type=float, default=0.15)
    training_set.add_argument("--test-fraction", type=float, default=0.15)
    training_set.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Include rows even when the remapped image path does not exist.",
    )
    training_set.set_defaults(func=cmd_build_training_set)

    train_model = sub.add_parser("train-image-model", help="Train an image classifier from a training CSV.")
    train_model.add_argument("--config", help="Optional config used to derive task-specific default paths.")
    train_model.add_argument("--task-id", help="Defaults to the config task marked default: true, or the first task.")
    train_model.add_argument("--training-csv", help="Defaults to task.training_csv when --config is provided.")
    train_model.add_argument("--output-dir", help="Defaults to task.model_dir/<model-name> when --config is provided.")
    train_model.add_argument("--epochs", type=int, default=5)
    train_model.add_argument("--batch-size", type=int, default=16)
    train_model.add_argument("--learning-rate", type=float, default=0.001)
    train_model.add_argument("--image-size", type=int, default=224)
    train_model.add_argument("--num-workers", type=int, default=0)
    train_model.add_argument(
        "--model-name",
        default="resnet18",
        help="One of: resnet18, efficientnet_b0, mobilenet_v3_small.",
    )
    train_model.add_argument("--pretrained", action="store_true")
    train_model.add_argument("--device", default="auto")
    train_model.add_argument(
        "--positive-label",
        help="Positive class for PPV/sensitivity/specificity. Defaults to task.positive_label with --config.",
    )
    train_model.add_argument(
        "--positive-threshold",
        type=float,
        help="Optional probability threshold for predicting the positive class.",
    )
    train_model.add_argument(
        "--class-weight",
        action="append",
        default=[],
        help="Training class weight as LABEL=WEIGHT. Can be passed multiple times.",
    )
    train_model.set_defaults(func=cmd_train_image_model)

    train_compare = sub.add_parser(
        "train-compare-image-models",
        help="Train several image models, then write the comparison report.",
    )
    train_compare.add_argument("--config", help="Optional config used to derive task-specific default paths.")
    train_compare.add_argument("--task-id", help="Defaults to the config task marked default: true, or the first task.")
    train_compare.add_argument("--training-csv", help="Defaults to task.training_csv when --config is provided.")
    train_compare.add_argument("--models-dir", help="Defaults to task.model_dir when --config is provided.")
    train_compare.add_argument(
        "--model",
        action="append",
        default=[],
        help=f"Model to train. Can be passed multiple times. Defaults to: {', '.join(DEFAULT_IMAGE_MODELS)}.",
    )
    train_compare.add_argument("--epochs", type=int, default=5)
    train_compare.add_argument("--batch-size", type=int, default=16)
    train_compare.add_argument("--learning-rate", type=float, default=0.001)
    train_compare.add_argument("--image-size", type=int, default=224)
    train_compare.add_argument("--num-workers", type=int, default=0)
    train_compare.add_argument("--pretrained", action="store_true")
    train_compare.add_argument("--device", default="auto")
    train_compare.add_argument(
        "--positive-label",
        help="Positive class for PPV/sensitivity/specificity. Defaults to task.positive_label with --config.",
    )
    train_compare.add_argument(
        "--positive-threshold",
        type=float,
        help="Optional probability threshold for predicting the positive class.",
    )
    train_compare.add_argument(
        "--class-weight",
        action="append",
        default=[],
        help="Training class weight as LABEL=WEIGHT. Can be passed multiple times.",
    )
    train_compare.add_argument("--output-csv", help="Defaults to <models-dir>/comparison.csv.")
    train_compare.add_argument("--output-md", help="Defaults to <models-dir>/comparison.md.")
    train_compare.set_defaults(func=cmd_train_compare_image_models)

    weather_lasso = sub.add_parser(
        "train-weather-lasso",
        help="Train a weather-only L1 logistic-regression baseline for the positive class.",
    )
    weather_lasso.add_argument("--config", required=True)
    weather_lasso.add_argument("--task-id", help="Defaults to the config task marked default: true, or the first task.")
    weather_lasso.add_argument("--training-csv", help="Defaults to task.training_csv.")
    weather_lasso.add_argument("--output-dir", help="Defaults to task.model_dir/weather_lasso.")
    weather_lasso.add_argument(
        "--positive-label",
        help="Positive class to predict. Defaults to task.positive_label.",
    )
    weather_lasso.add_argument(
        "--feature",
        action="append",
        default=[],
        help="Weather variable to use. Can be passed multiple times. Defaults to all numeric weather variables.",
    )
    weather_lasso.add_argument(
        "--max-weather-age-minutes",
        type=float,
        default=90.0,
        help="Largest allowed time difference between capture and nearest hourly weather record.",
    )
    weather_lasso.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverse L1 regularization strength. Smaller values make more coefficients exactly zero.",
    )
    weather_lasso.add_argument(
        "--positive-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for predicting the positive class.",
    )
    weather_lasso.add_argument(
        "--class-weight",
        default="none",
        choices=["none", "balanced"],
        help="Use 'balanced' to upweight the minority class.",
    )
    weather_lasso.set_defaults(func=cmd_train_weather_lasso)

    compare_models = sub.add_parser(
        "compare-image-models",
        help="Compare image-model metadata across multiple training runs.",
    )
    compare_models.add_argument("--config", help="Optional config used to derive task-specific default paths/cameras.")
    compare_models.add_argument("--task-id", help="Defaults to the config task marked default: true, or the first task.")
    compare_models.add_argument("--models-dir", help="Defaults to task.model_dir when --config is provided.")
    compare_models.add_argument("--output-csv", help="Defaults to <models-dir>/comparison.csv.")
    compare_models.add_argument("--output-md", help="Defaults to <models-dir>/comparison.md.")
    compare_models.set_defaults(func=cmd_compare_image_models)

    explain_model = sub.add_parser(
        "explain-image-model",
        help="Generate Grad-CAM visual explanations for a trained image model.",
    )
    explain_model.add_argument("--config", help="Optional config used to derive task-specific model paths.")
    explain_model.add_argument("--task-id", help="Defaults to the config task marked default: true, or the first task.")
    explain_model.add_argument(
        "--model-name",
        default="resnet18",
        help="Model subdirectory to use with --config when --checkpoint is not provided.",
    )
    explain_model.add_argument("--checkpoint", help="Defaults to task.model_dir/<model-name>/model.pt with --config.")
    explain_model.add_argument(
        "--predictions",
        help="Defaults to <checkpoint-dir>/predictions.csv.",
    )
    explain_model.add_argument(
        "--output-dir",
        help="Defaults to <checkpoint-dir>/explanations.",
    )
    explain_model.add_argument("--split", default="test", help="Prediction split to explain, or 'all'.")
    explain_model.add_argument(
        "--selection",
        default="mixed",
        help="One of: mixed, incorrect, correct, high-confidence, low-confidence.",
    )
    explain_model.add_argument("--max-images", type=int, default=24)
    explain_model.add_argument(
        "--true-label",
        action="append",
        default=[],
        help="Only include rows with this true label. Can be passed multiple times.",
    )
    explain_model.add_argument(
        "--pred-label",
        action="append",
        default=[],
        help="Only include rows with this predicted label. Can be passed multiple times.",
    )
    explain_model.add_argument(
        "--target",
        default="predicted",
        help="Class to explain: predicted or true.",
    )
    explain_model.add_argument("--device", default="auto")
    explain_model.add_argument("--output-width", type=int, default=960)
    explain_model.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay opacity from 0 to 1.")
    explain_model.set_defaults(func=cmd_explain_image_model)

    return parser


def cmd_init_db(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    db.init_db(config.database_path)
    with db.connect(config.database_path) as conn:
        db.register_config(conn, config)
    print(f"Initialized database: {config.database_path}")
    return 0


def cmd_capture_once(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    db.init_db(config.database_path)
    cameras = select_cameras(config.cameras, args.camera_id)
    capture_selected(config, cameras)
    return 0


def cmd_capture_loop(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    db.init_db(config.database_path)
    cameras = select_cameras(config.cameras, args.camera_id)
    if not cameras:
        raise ValueError("No cameras selected.")

    interval_seconds = min(camera.capture.interval_seconds for camera in cameras)
    iteration = 0
    while True:
        iteration += 1
        print(f"capture-loop iteration={iteration}")
        capture_selected(config, cameras)
        if args.max_iterations is not None and iteration >= args.max_iterations:
            return 0
        time.sleep(interval_seconds)


def cmd_run_collector(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    db.init_db(config.database_path)
    cameras = select_cameras(config.cameras, args.camera_id)
    if not cameras:
        raise ValueError("No cameras selected.")

    capture_interval = positive_interval(
        args.capture_interval_seconds
        if args.capture_interval_seconds is not None
        else min(camera.capture.interval_seconds for camera in cameras),
        "capture interval",
    )
    weather_interval = positive_interval(
        args.weather_interval_seconds
        if args.weather_interval_seconds is not None
        else config.weather.fetch_interval_seconds,
        "weather interval",
    )

    next_capture_at = 0.0
    next_weather_at = (
        0.0
        if config.weather.fetch_on_start and not args.skip_initial_weather
        else time.monotonic() + weather_interval
    )
    clock_sanity = config.collector.clock_sanity
    clock_checker = None
    if clock_sanity.enabled and not args.disable_clock_sanity:
        clock_checker = ClockSanityChecker(
            max_drift_seconds=clock_sanity.max_drift_seconds,
            max_backward_seconds=clock_sanity.max_backward_seconds,
        )
    iteration = 0

    print(
        "collector started "
        f"cameras={','.join(camera.id for camera in cameras)} "
        f"capture_interval_seconds={capture_interval} "
        f"weather_interval_seconds={weather_interval} "
        f"clock_sanity={'on' if clock_checker else 'off'}"
    )

    while True:
        iteration += 1
        now = time.monotonic()

        if clock_checker:
            clock_result = clock_checker.check(wall_time=datetime.now(timezone.utc), monotonic_time=now)
            if not clock_result.ok:
                print(
                    f"[{utc_timestamp()}] clock sanity failed: {clock_result.reason}; "
                    f"wall_delta={format_seconds(clock_result.wall_delta_seconds)} "
                    f"monotonic_delta={format_seconds(clock_result.monotonic_delta_seconds)} "
                    f"drift={format_seconds(clock_result.drift_seconds)}. "
                    "Skipping capture/weather cycle."
                )
                if args.max_iterations is not None and iteration >= args.max_iterations:
                    return 0
                time.sleep(clock_sanity.retry_seconds)
                continue

        if now >= next_capture_at:
            print(f"[{utc_timestamp()}] capture cycle")
            capture_selected(config, cameras)
            next_capture_at = now + capture_interval

        if now >= next_weather_at:
            print(f"[{utc_timestamp()}] weather cycle")
            fetch_weather_selected(config, cameras)
            next_weather_at = now + weather_interval

        if args.max_iterations is not None and iteration >= args.max_iterations:
            return 0

        sleep_for = max(1.0, min(next_capture_at, next_weather_at) - time.monotonic())
        time.sleep(sleep_for)


def capture_selected(config: AppConfig, cameras: Sequence[CameraConfig]) -> None:
    for camera in cameras:
        result = capture_once(config, camera)
        if result.ok:
            print(f"captured camera={result.camera_id} capture_id={result.capture_id} path={result.path}")
        else:
            print(f"capture failed camera={result.camera_id} capture_id={result.capture_id} error={result.error}")


def cmd_fetch_weather(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    db.init_db(config.database_path)
    cameras = select_cameras(config.cameras, args.camera_id)
    fetch_weather_selected(config, cameras)
    return 0


def fetch_weather_selected(config: AppConfig, cameras: Sequence[CameraConfig]) -> None:
    if config.weather.provider != "open_meteo":
        raise ValueError(f"Unsupported weather provider: {config.weather.provider}")

    with db.connect(config.database_path) as conn:
        db.register_config(conn, config)
        for camera in cameras:
            fetch = fetch_forecast(camera, config.weather)
            db.insert_weather_payload(
                conn,
                provider=fetch.provider,
                camera_id=camera.id,
                fetched_at_utc=fetch.fetched_at_utc,
                url=fetch.url,
                payload=fetch.payload,
            )
            count = db.insert_weather_records(
                conn,
                provider=fetch.provider,
                camera_id=camera.id,
                fetched_at_utc=fetch.fetched_at_utc,
                records=fetch.records,
            )
            print(f"weather camera={camera.id} provider={fetch.provider} records={count}")


def cmd_build_manifest(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    with db.connect(config.database_path) as conn:
        count = build_manifest(conn, Path(args.output))
    print(f"Wrote {count} rows to {Path(args.output).resolve()}")
    return 0


def cmd_annotate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    task_id = config.default_task_id if args.task_id is None else args.task_id
    serve_annotation_app(
        config,
        AnnotationServerOptions(
            host=args.host,
            port=args.port,
            task_id=task_id,
            left_annotator=args.left_annotator,
            right_annotator=args.right_annotator,
            open_browser=args.open_browser,
        ),
    )
    return 0


def cmd_adjudicate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    db.init_db(config.database_path)
    task_id = config.default_task_id if args.task_id is None else args.task_id
    predictions_path = (
        Path(args.predictions)
        if args.predictions
        else config.task_model_dir(task_id) / args.model_name / "predictions.csv"
    )
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else config.task_model_dir(task_id) / args.model_name / "model.pt"
    )
    if not predictions_path.exists():
        print(f"Predictions CSV not found, continuing without ML predictions: {predictions_path}")
        predictions_path = None
    if not checkpoint_path.exists():
        print(f"Model checkpoint not found, continuing without live inference: {checkpoint_path}")
        checkpoint_path = None
    serve_adjudication_app(
        config,
        AdjudicationServerOptions(
            host=args.host,
            port=args.port,
            task_id=task_id,
            adjudicator=args.adjudicator,
            predictions_csv=predictions_path,
            checkpoint_path=checkpoint_path,
            device=args.device,
            annotators=tuple(args.annotator),
            include_agreements=args.include_agreements,
            open_browser=args.open_browser,
        ),
    )
    return 0


def cmd_analyze_annotations(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    db.init_db(config.database_path)
    task_id = config.default_task_id if args.task_id is None else args.task_id
    with db.connect(config.database_path) as conn:
        analysis = analyze_annotations(conn, config=config, task_id=task_id)

    output_path = Path(args.output)
    disagreements_output_path = Path(args.disagreements_output)
    write_analysis_markdown(analysis, output_path)
    write_disagreements_csv(analysis["disagreements"], disagreements_output_path)

    print(f"Wrote annotation analysis to {output_path.resolve()}")
    print(f"Wrote disagreements CSV to {disagreements_output_path.resolve()}")
    print(
        "Summary: "
        f"annotations={analysis['annotation_count']} "
        f"unique_captures={analysis['unique_capture_count']} "
        f"double_labeled={analysis['double_labeled_capture_count']} "
        f"disagreements={len(analysis['disagreements'])} "
        f"adjudicated={analysis['adjudication_count']} "
        f"remaining_disagreements={analysis['remaining_disagreement_count']}"
    )
    if analysis["legacy_labels"]:
        print(f"Legacy labels found: {analysis['legacy_labels']}")
    return 0


def cmd_export_favorites(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    db.init_db(config.database_path)
    task_id = config.default_task_id if args.task_id is None else args.task_id
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with db.connect(config.database_path) as conn:
        rows = favorite_rows(conn, task_id=task_id)

    fieldnames = [
        "capture_id",
        "task_id",
        "annotator",
        "camera_id",
        "captured_at_utc",
        "captured_at_pacific",
        "favorited_at_utc",
        "notes",
        "image_path",
    ]
    for row in rows:
        resolved = resolve_image_path(row.get("image_path"), config.data_dir)
        if resolved is not None:
            row["image_path"] = str(resolved)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} favorite frame(s) to {output_path.resolve()}")
    return 0


def cmd_backup_db(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output_path = Path(args.output)
    backup_sqlite_database(config.database_path, output_path)
    print(f"Wrote database backup to {output_path.resolve()}")
    return 0


def cmd_check_training_env(args: argparse.Namespace) -> int:
    report = training_environment_report()
    print(f"Python: {report['python']}")
    print(f"Platform: {report['platform']}")
    print(f"Machine: {report['machine']}")
    print("Packages:")
    for name, version in report["packages"].items():
        print(f"  {name}: {version or 'not installed'}")
    torch = report["torch"]
    print("Torch:")
    print(f"  installed: {torch['installed']}")
    print(f"  mps_available: {torch['mps_available']}")
    print(f"  cuda_available: {torch['cuda_available']}")
    print(f"  recommended_device: {torch['recommended_device'] or 'n/a'}")
    return 0


def cmd_build_training_set(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    db.init_db(config.database_path)
    task_id = config.default_task_id if args.task_id is None else args.task_id
    output_path = Path(args.output) if args.output else config.task_training_csv_path(task_id)
    exclude_labels = (
        tuple(args.exclude_label)
        if args.exclude_label
        else config.task_excluded_training_labels(task_id)
    )
    with db.connect(config.database_path) as conn:
        summary = build_training_set(
            conn,
            config=config,
            options=TrainingSetOptions(
                task_id=task_id,
                output_path=output_path,
                include_labels=tuple(args.include_label),
                exclude_labels=exclude_labels,
                min_annotators=args.min_annotators,
                train_fraction=args.train_fraction,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                allow_missing_images=args.allow_missing_images,
            ),
        )
    print(f"Wrote training set to {Path(summary['output_path']).resolve()}")
    print(f"Rows: {summary['row_count']}")
    print(f"Labels: {summary['label_counts']}")
    print(f"Splits: {summary['split_counts']}")
    print("Split labels:")
    for split, label_counts in summary["split_label_counts"].items():
        print(f"  {split}: {label_counts}")
    print(f"Skipped: {summary['skipped']}")
    return 0


def cmd_train_image_model(args: argparse.Namespace) -> int:
    config = load_config(args.config) if args.config else None
    task_id = None
    if config is not None:
        task_id = config.default_task_id if args.task_id is None else args.task_id
    training_csv = (
        Path(args.training_csv)
        if args.training_csv
        else config.task_training_csv_path(task_id) if config is not None
        else Path("data/training/training.csv")
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else config.task_model_dir(task_id) / args.model_name if config is not None
        else Path("data/models") / args.model_name
    )
    crop_pixels = config.task_image_crop_pixels(task_id) if config is not None else None
    positive_label = (
        args.positive_label
        if args.positive_label
        else config.task_positive_label(task_id) if config is not None
        else None
    )
    positive_threshold = (
        args.positive_threshold
        if args.positive_threshold is not None
        else config.task_positive_threshold(task_id) if config is not None
        else None
    )
    class_weights = config.task_class_weights(task_id) if config is not None else {}
    class_weights.update(parse_class_weight_args(args.class_weight))
    summary = train_image_model(
        ImageTrainingOptions(
            training_csv=training_csv,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            image_size=args.image_size,
            num_workers=args.num_workers,
            model_name=args.model_name,
            pretrained=args.pretrained,
            device=args.device,
            crop_pixels=crop_pixels,
            positive_label=positive_label,
            positive_threshold=positive_threshold,
            class_weights=class_weights,
        )
    )
    print_training_summary(summary)
    return 0


def cmd_train_compare_image_models(args: argparse.Namespace) -> int:
    config = load_config(args.config) if args.config else None
    task_id = None
    if config is not None:
        task_id = config.default_task_id if args.task_id is None else args.task_id
    training_csv = (
        Path(args.training_csv)
        if args.training_csv
        else config.task_training_csv_path(task_id) if config is not None
        else Path("data/training/training.csv")
    )
    models_dir = (
        Path(args.models_dir)
        if args.models_dir
        else config.task_model_dir(task_id) if config is not None
        else Path("data/models")
    )
    crop_pixels = config.task_image_crop_pixels(task_id) if config is not None else None
    positive_label = (
        args.positive_label
        if args.positive_label
        else config.task_positive_label(task_id) if config is not None
        else None
    )
    positive_threshold = (
        args.positive_threshold
        if args.positive_threshold is not None
        else config.task_positive_threshold(task_id) if config is not None
        else None
    )
    class_weights = config.task_class_weights(task_id) if config is not None else {}
    class_weights.update(parse_class_weight_args(args.class_weight))
    model_names = tuple(args.model) if args.model else DEFAULT_IMAGE_MODELS

    for model_name in model_names:
        print(f"\n=== Training {model_name} ===")
        summary = train_image_model(
            ImageTrainingOptions(
                training_csv=training_csv,
                output_dir=models_dir / model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                image_size=args.image_size,
                num_workers=args.num_workers,
                model_name=model_name,
                pretrained=args.pretrained,
                device=args.device,
                crop_pixels=crop_pixels,
                positive_label=positive_label,
                positive_threshold=positive_threshold,
                class_weights=class_weights,
            )
        )
        print_training_summary(summary)

    output_csv = Path(args.output_csv) if args.output_csv else models_dir / "comparison.csv"
    output_md = Path(args.output_md) if args.output_md else models_dir / "comparison.md"
    camera_ids = config.task_comparison_camera_ids(task_id) if config is not None else ()
    rows = compare_image_models(models_dir, output_csv, output_md, camera_ids=camera_ids)
    print(f"\nWrote comparison CSV to {output_csv.resolve()}")
    print(f"Wrote comparison Markdown to {output_md.resolve()}")
    if rows:
        best = rows[0]
        print(
            "Best run: "
            f"{best['run']} model={best['model_name']} "
            f"test_accuracy={best['test_accuracy']} "
            f"val_accuracy={best['val_accuracy']} "
            f"test_ppv={best.get('test_ppv')} "
            f"test_sensitivity={best.get('test_sensitivity')} "
            f"test_specificity={best.get('test_specificity')}"
        )
    return 0


def cmd_train_weather_lasso(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    db.init_db(config.database_path)
    task_id = config.default_task_id if args.task_id is None else args.task_id
    training_csv = Path(args.training_csv) if args.training_csv else config.task_training_csv_path(task_id)
    output_dir = Path(args.output_dir) if args.output_dir else config.task_model_dir(task_id) / "weather_lasso"
    positive_label = args.positive_label or config.task_positive_label(task_id)
    if not positive_label:
        raise ValueError("A positive label is required. Set task.positive_label or pass --positive-label.")

    with db.connect(config.database_path) as conn:
        summary = train_weather_lasso(
            conn,
            WeatherLassoOptions(
                training_csv=training_csv,
                output_dir=output_dir,
                positive_label=positive_label,
                features=tuple(args.feature),
                max_weather_age_minutes=args.max_weather_age_minutes,
                c=args.c,
                positive_threshold=args.positive_threshold,
                class_weight=args.class_weight,
            ),
        )

    print(f"Wrote weather model to {summary['model_path']}")
    print(f"Wrote metadata to {summary['metadata_path']}")
    print(f"Wrote predictions to {summary['predictions_path']}")
    print(f"Wrote coefficients to {summary['coefficients_path']}")
    print(f"Positive label: {summary['positive_label']}")
    print(f"Positive threshold: {summary['positive_threshold']}")
    print(f"L1 C: {summary['c']}")
    print(f"Class weight: {summary['class_weight']}")
    print(f"Matched rows: {summary['matched_rows']}")
    print(f"Splits: {summary['split_counts']}")
    if summary["skipped"]:
        print(f"Skipped: {summary['skipped']}")
    print("Nonzero coefficients:")
    for row in summary["nonzero_coefficients"]:
        print(f"  {row['feature']}: {row['coefficient']:.6g}")
    test_binary = summary["detailed_metrics"]["test"]["binary"]
    print(f"Test positive-class detail: {test_binary}")
    print(f"Test by camera: {summary['detailed_metrics']['test']['by_camera']}")
    return 0


def print_training_summary(summary: dict[str, Any]) -> None:
    print(f"Wrote checkpoint to {summary['checkpoint_path']}")
    print(f"Wrote metadata to {summary['metadata_path']}")
    print(f"Wrote predictions to {summary['predictions_path']}")
    print(f"Device: {summary['device']}")
    print(f"Model: {summary['model_name']}")
    print(f"Crop pixels: {summary['crop_pixels']}")
    print(f"Positive label: {summary['positive_label']}")
    print(f"Positive threshold: {summary['positive_threshold']}")
    print(f"Class weights: {summary['class_weights']}")
    print(f"Labels: {summary['labels']}")
    print(f"Splits: {summary['split_counts']}")
    print("Split labels:")
    for split, label_counts in summary["split_label_counts"].items():
        print(f"  {split}: {label_counts}")
    final = summary["history"][-1] if summary["history"] else None
    if final:
        print(f"Final train: {final['train']}")
        print(f"Final val: {final['val']}")
    print(f"Test: {summary['test']}")
    test_overall = summary["detailed_metrics"]["test"]["overall"]
    test_by_camera = summary["detailed_metrics"]["test"]["by_camera"]
    test_binary = summary["detailed_metrics"]["test"].get("binary")
    print(f"Test overall detail: {test_overall}")
    if test_binary:
        print(f"Test positive-class detail: {test_binary}")
    print(f"Test by camera: {test_by_camera}")


def cmd_compare_image_models(args: argparse.Namespace) -> int:
    config = load_config(args.config) if args.config else None
    task_id = None
    if config is not None:
        task_id = config.default_task_id if args.task_id is None else args.task_id
    models_dir = (
        Path(args.models_dir)
        if args.models_dir
        else config.task_model_dir(task_id) if config is not None
        else Path("data/models")
    )
    output_csv = Path(args.output_csv) if args.output_csv else models_dir / "comparison.csv"
    output_md = Path(args.output_md) if args.output_md else models_dir / "comparison.md"
    camera_ids = config.task_comparison_camera_ids(task_id) if config is not None else ()
    rows = compare_image_models(
        models_dir,
        output_csv,
        output_md,
        camera_ids=camera_ids,
    )
    print(f"Wrote comparison CSV to {output_csv.resolve()}")
    print(f"Wrote comparison Markdown to {output_md.resolve()}")
    if rows:
        best = rows[0]
        print(
            "Best run: "
            f"{best['run']} model={best['model_name']} "
            f"test_accuracy={best['test_accuracy']} "
            f"val_accuracy={best['val_accuracy']} "
            f"test_ppv={best.get('test_ppv')} "
            f"test_sensitivity={best.get('test_sensitivity')} "
            f"test_specificity={best.get('test_specificity')}"
        )
    else:
        print("No model metadata files found.")
    return 0


def cmd_explain_image_model(args: argparse.Namespace) -> int:
    config = load_config(args.config) if args.config else None
    task_id = None
    if config is not None:
        task_id = config.default_task_id if args.task_id is None else args.task_id

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else config.task_model_dir(task_id) / args.model_name / "model.pt" if config is not None
        else Path("data/models") / args.model_name / "model.pt"
    )
    checkpoint_dir = checkpoint_path.parent
    predictions_path = Path(args.predictions) if args.predictions else checkpoint_dir / "predictions.csv"
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / "explanations"

    summary = explain_image_model(
        ImageExplanationOptions(
            checkpoint_path=checkpoint_path,
            predictions_path=predictions_path,
            output_dir=output_dir,
            split=args.split,
            selection=args.selection,
            max_images=args.max_images,
            target=args.target,
            device=args.device,
            output_width=args.output_width,
            alpha=args.alpha,
            true_labels=tuple(args.true_label),
            pred_labels=tuple(args.pred_label),
        )
    )
    print(f"Wrote {summary['count']} Grad-CAM explanation image(s) to {Path(summary['output_dir']).resolve()}")
    print(f"Wrote gallery to {Path(summary['index_path']).resolve()}")
    print(f"Wrote summary to {Path(summary['summary_path']).resolve()}")
    print(f"Device: {summary['device']}")
    print(f"Model: {summary['model_name']}")
    print(
        "Selection: "
        f"split={summary['split']} selection={summary['selection']} target={summary['target']} "
        f"true_labels={summary['true_labels'] or 'any'} pred_labels={summary['pred_labels'] or 'any'}"
    )
    return 0


def select_cameras(cameras, camera_id: str | None):
    if camera_id is None:
        return cameras
    selected = tuple(camera for camera in cameras if camera.id == camera_id)
    if not selected:
        known = ", ".join(camera.id for camera in cameras)
        raise ValueError(f"Unknown camera_id={camera_id!r}. Known cameras: {known}")
    return selected


def positive_interval(value: int, label: str) -> int:
    if value <= 0:
        raise ValueError(f"{label} must be > 0 seconds.")
    return value


def parse_class_weight_args(values: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --class-weight {value!r}; expected LABEL=WEIGHT.")
        label, raw_weight = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Invalid --class-weight {value!r}; label is empty.")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"Invalid --class-weight {value!r}; weight must be numeric.") from exc
        if weight <= 0:
            raise ValueError(f"Invalid --class-weight {value!r}; weight must be > 0.")
        weights[label] = weight
    return weights


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}s"


if __name__ == "__main__":
    raise SystemExit(main())
