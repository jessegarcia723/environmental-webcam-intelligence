from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml

from enviro_webcam_ml.image_preprocessing import PixelCrop, parse_pixel_crop


@dataclass(frozen=True)
class LocationConfig:
    latitude: float
    longitude: float
    elevation_m: float | None = None
    timezone: str = "UTC"


@dataclass(frozen=True)
class ManifestCaptureConfig:
    url: str
    frame_url_template: str
    frames_key: str = "frames"
    timestamp_source: str = "filename_epoch"
    filename_suffix: str = ".jpg"
    min_frame_spacing_seconds: int = 0
    max_frames_per_cycle: int | None = None
    skip_existing: bool = True


@dataclass(frozen=True)
class CaptureConfig:
    image_url: str
    interval_seconds: int = 300
    timeout_seconds: int = 20
    user_agent: str = "enviro-webcam-ml/0.1"
    source: str = "image_url"
    manifest: ManifestCaptureConfig | None = None


@dataclass(frozen=True)
class PoseConfig:
    version: str = "initial"
    description: str = ""


@dataclass(frozen=True)
class CameraConfig:
    id: str
    name: str
    location: LocationConfig
    capture: CaptureConfig
    pose: PoseConfig


@dataclass(frozen=True)
class WeatherConfig:
    provider: str = "open_meteo"
    timezone: str = "UTC"
    fetch_interval_seconds: int = 7200
    fetch_on_start: bool = True
    hourly_variables: tuple[str, ...] = ()
    past_days: int | None = 2
    forecast_days: int | None = 7
    past_hours: int | None = None
    forecast_hours: int | None = None


@dataclass(frozen=True)
class QualityConfig:
    night_luminance_threshold: float = 35.0
    blur_variance_threshold: float = 25.0


@dataclass(frozen=True)
class ClockSanityConfig:
    enabled: bool = True
    max_drift_seconds: float = 120.0
    max_backward_seconds: float = 1.0
    retry_seconds: int = 60


@dataclass(frozen=True)
class CollectorConfig:
    clock_sanity: ClockSanityConfig


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    database_path: Path
    data_dir: Path


@dataclass(frozen=True)
class AppConfig:
    path: Path
    project: ProjectConfig
    cameras: tuple[CameraConfig, ...]
    weather: WeatherConfig
    quality: QualityConfig
    collector: CollectorConfig
    raw: dict[str, Any]

    @property
    def database_path(self) -> Path:
        return resolve_relative(self.path.parent, self.project.database_path)

    @property
    def data_dir(self) -> Path:
        return resolve_relative(self.path.parent, self.project.data_dir)

    def task(self, task_id: str | None = None) -> dict[str, Any]:
        tasks = self.raw.get("tasks") or []
        if not tasks:
            raise ValueError("Config must include at least one task for this command.")
        if task_id is None:
            return default_task(tasks)
        for task in tasks:
            if task.get("id") == task_id:
                return task
        known = ", ".join(str(task.get("id")) for task in tasks)
        raise ValueError(f"Unknown task_id={task_id!r}. Known tasks: {known}")

    @property
    def default_task_id(self) -> str:
        return str(self.task().get("id"))

    def task_output_slug(self, task_id: str | None = None) -> str:
        task = self.task(task_id)
        return str(task.get("output_slug") or task.get("id"))

    def task_training_csv_path(self, task_id: str | None = None) -> Path:
        task = self.task(task_id)
        configured = task.get("training_csv")
        if configured:
            return resolve_relative(self.path.parent, Path(configured))
        return self.data_dir / "training" / f"{self.task_output_slug(task.get('id'))}_training.csv"

    def task_model_dir(self, task_id: str | None = None) -> Path:
        task = self.task(task_id)
        configured = task.get("model_dir")
        if configured:
            return resolve_relative(self.path.parent, Path(configured))
        return self.data_dir / "models" / self.task_output_slug(task.get("id"))

    def task_excluded_training_labels(self, task_id: str | None = None) -> tuple[str, ...]:
        task = self.task(task_id)
        return tuple(str(label) for label in task.get("excluded_training_labels", ()))

    def task_candidate_min_spacing_seconds(self, task_id: str | None = None) -> int:
        task = self.task(task_id)
        value = task.get("candidate_min_spacing_seconds", 0)
        if value in (None, ""):
            return 0
        return int(value)

    def task_comparison_camera_ids(self, task_id: str | None = None) -> tuple[str, ...]:
        task = self.task(task_id)
        groups = task.get("comparison_groups") or {}
        camera_group = groups.get("camera")
        if camera_group:
            return tuple(str(camera_id) for camera_id in camera_group)
        return tuple(camera.id for camera in self.cameras)

    def task_image_crop_pixels(self, task_id: str | None = None) -> PixelCrop | None:
        task = self.task(task_id)
        preprocessing = task.get("image_preprocessing") or {}
        return parse_pixel_crop(preprocessing.get("crop_pixels"))

    def task_positive_label(self, task_id: str | None = None) -> str | None:
        task = self.task(task_id)
        label = task.get("positive_label")
        return str(label) if label else None

    def task_positive_threshold(self, task_id: str | None = None) -> float | None:
        task = self.task(task_id)
        training = task.get("training") or {}
        value = training.get("positive_threshold")
        if value in (None, ""):
            return None
        return float(value)

    def task_class_weights(self, task_id: str | None = None) -> dict[str, float]:
        task = self.task(task_id)
        training = task.get("training") or {}
        weights = training.get("class_weights") or {}
        return {str(label): float(weight) for label, weight in weights.items()}


def resolve_relative(base: Path, path: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    project_raw = raw.get("project") or {}
    project = ProjectConfig(
        name=required(project_raw, "project.name"),
        database_path=Path(project_raw.get("database_path", "data/envirocam.sqlite3")),
        data_dir=Path(project_raw.get("data_dir", "data")),
    )

    cameras = tuple(parse_camera(item) for item in raw.get("cameras", []))
    if not cameras:
        raise ValueError("Config must include at least one camera.")

    weather_raw = raw.get("weather") or {}
    weather = WeatherConfig(
        provider=weather_raw.get("provider", "open_meteo"),
        timezone=weather_raw.get("timezone", "UTC"),
        fetch_interval_seconds=int(weather_raw.get("fetch_interval_seconds", 7200)),
        fetch_on_start=bool(weather_raw.get("fetch_on_start", True)),
        hourly_variables=tuple(weather_raw.get("hourly_variables", ())),
        past_days=maybe_int(weather_raw.get("past_days", 2)),
        forecast_days=maybe_int(weather_raw.get("forecast_days", 7)),
        past_hours=maybe_int(weather_raw.get("past_hours")),
        forecast_hours=maybe_int(weather_raw.get("forecast_hours")),
    )

    quality_raw = raw.get("quality") or {}
    quality = QualityConfig(
        night_luminance_threshold=float(quality_raw.get("night_luminance_threshold", 35.0)),
        blur_variance_threshold=float(quality_raw.get("blur_variance_threshold", 25.0)),
    )

    collector_raw = raw.get("collector") or {}
    clock_raw = collector_raw.get("clock_sanity") or {}
    collector = CollectorConfig(
        clock_sanity=ClockSanityConfig(
            enabled=as_bool(clock_raw.get("enabled", True)),
            max_drift_seconds=float(clock_raw.get("max_drift_seconds", 120.0)),
            max_backward_seconds=float(clock_raw.get("max_backward_seconds", 1.0)),
            retry_seconds=int(clock_raw.get("retry_seconds", 60)),
        )
    )

    return AppConfig(
        path=config_path,
        project=project,
        cameras=cameras,
        weather=weather,
        quality=quality,
        collector=collector,
        raw=raw,
    )


def default_task(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    defaults = [task for task in tasks if as_bool(task.get("default", False))]
    if len(defaults) > 1:
        ids = ", ".join(str(task.get("id")) for task in defaults)
        raise ValueError(f"Only one task can set default: true. Defaults found: {ids}")
    if defaults:
        return defaults[0]
    return tasks[0]


def parse_camera(raw: dict[str, Any]) -> CameraConfig:
    location_raw = raw.get("location") or {}
    capture_raw = raw.get("capture") or {}
    pose_raw = raw.get("pose") or {}

    location = LocationConfig(
        latitude=float(required(location_raw, "camera.location.latitude")),
        longitude=float(required(location_raw, "camera.location.longitude")),
        elevation_m=maybe_float(location_raw.get("elevation_m")),
        timezone=location_raw.get("timezone", "UTC"),
    )
    capture = parse_capture(capture_raw)
    pose = PoseConfig(
        version=pose_raw.get("version", "initial"),
        description=pose_raw.get("description", ""),
    )
    return CameraConfig(
        id=required(raw, "camera.id"),
        name=required(raw, "camera.name"),
        location=location,
        capture=capture,
        pose=pose,
    )


def parse_capture(raw: dict[str, Any]) -> CaptureConfig:
    source = str(raw.get("source", "image_url"))
    if source not in {"image_url", "manifest_frames"}:
        raise ValueError("camera.capture.source must be either 'image_url' or 'manifest_frames'.")
    manifest_raw = raw.get("manifest") or {}
    manifest = parse_manifest_capture(manifest_raw) if source == "manifest_frames" else None
    image_url = raw.get("image_url")
    if not image_url and manifest is not None:
        image_url = manifest.frame_url_template
    if not image_url:
        image_url = required(raw, "camera.capture.image_url")
    return CaptureConfig(
        image_url=str(image_url),
        interval_seconds=int(raw.get("interval_seconds", 300)),
        timeout_seconds=int(raw.get("timeout_seconds", 20)),
        user_agent=raw.get("user_agent", "enviro-webcam-ml/0.1"),
        source=source,
        manifest=manifest,
    )


def parse_manifest_capture(raw: dict[str, Any]) -> ManifestCaptureConfig:
    max_frames = raw.get("max_frames_per_cycle")
    parsed_max_frames = int(max_frames) if max_frames not in (None, "") else None
    if parsed_max_frames is not None and parsed_max_frames <= 0:
        raise ValueError("camera.capture.manifest.max_frames_per_cycle must be > 0.")
    min_spacing = int(raw.get("min_frame_spacing_seconds", 0))
    if min_spacing < 0:
        raise ValueError("camera.capture.manifest.min_frame_spacing_seconds must be >= 0.")
    timestamp_source = str(raw.get("timestamp_source", "filename_epoch"))
    if timestamp_source != "filename_epoch":
        raise ValueError("Only camera.capture.manifest.timestamp_source='filename_epoch' is currently supported.")
    return ManifestCaptureConfig(
        url=required(raw, "camera.capture.manifest.url"),
        frame_url_template=required(raw, "camera.capture.manifest.frame_url_template"),
        frames_key=str(raw.get("frames_key", "frames")),
        timestamp_source=timestamp_source,
        filename_suffix=str(raw.get("filename_suffix", ".jpg")),
        min_frame_spacing_seconds=min_spacing,
        max_frames_per_cycle=parsed_max_frames,
        skip_existing=as_bool(raw.get("skip_existing", True)),
    )


def required(raw: dict[str, Any], path: str) -> Any:
    key = path.split(".")[-1]
    value = raw.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required config value: {path}")
    return value


def maybe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def maybe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
