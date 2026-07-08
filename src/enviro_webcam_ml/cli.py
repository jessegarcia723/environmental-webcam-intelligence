from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from enviro_webcam_ml import db
from enviro_webcam_ml.capture import capture_once
from enviro_webcam_ml.config import AppConfig, CameraConfig, load_config
from enviro_webcam_ml.dataset import build_manifest
from enviro_webcam_ml.weather.open_meteo import fetch_forecast


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
        "--max-iterations",
        type=int,
        help="Stop after this many scheduler iterations; useful for smoke tests.",
    )
    collector.set_defaults(func=cmd_run_collector)

    manifest = sub.add_parser("build-manifest", help="Build a CSV frame manifest for annotation/training.")
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(func=cmd_build_manifest)

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
    iteration = 0

    print(
        "collector started "
        f"cameras={','.join(camera.id for camera in cameras)} "
        f"capture_interval_seconds={capture_interval} "
        f"weather_interval_seconds={weather_interval}"
    )

    while True:
        iteration += 1
        now = time.monotonic()

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


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
