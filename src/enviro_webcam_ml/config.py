from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LocationConfig:
    latitude: float
    longitude: float
    elevation_m: float | None = None
    timezone: str = "UTC"


@dataclass(frozen=True)
class CaptureConfig:
    image_url: str
    interval_seconds: int = 300
    timeout_seconds: int = 20
    user_agent: str = "enviro-webcam-ml/0.1"


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
    hourly_variables: tuple[str, ...] = ()


@dataclass(frozen=True)
class QualityConfig:
    night_luminance_threshold: float = 35.0
    blur_variance_threshold: float = 25.0


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
    raw: dict[str, Any]

    @property
    def database_path(self) -> Path:
        return resolve_relative(self.path.parent, self.project.database_path)

    @property
    def data_dir(self) -> Path:
        return resolve_relative(self.path.parent, self.project.data_dir)


def resolve_relative(base: Path, path: Path) -> Path:
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
        hourly_variables=tuple(weather_raw.get("hourly_variables", ())),
    )

    quality_raw = raw.get("quality") or {}
    quality = QualityConfig(
        night_luminance_threshold=float(quality_raw.get("night_luminance_threshold", 35.0)),
        blur_variance_threshold=float(quality_raw.get("blur_variance_threshold", 25.0)),
    )

    return AppConfig(
        path=config_path,
        project=project,
        cameras=cameras,
        weather=weather,
        quality=quality,
        raw=raw,
    )


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
    capture = CaptureConfig(
        image_url=required(capture_raw, "camera.capture.image_url"),
        interval_seconds=int(capture_raw.get("interval_seconds", 300)),
        timeout_seconds=int(capture_raw.get("timeout_seconds", 20)),
        user_agent=capture_raw.get("user_agent", "enviro-webcam-ml/0.1"),
    )
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


def required(raw: dict[str, Any], path: str) -> Any:
    key = path.split(".")[-1]
    value = raw.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required config value: {path}")
    return value


def maybe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
