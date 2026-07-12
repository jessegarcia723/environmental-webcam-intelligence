from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

import requests

from enviro_webcam_ml.config import CameraConfig, WeatherConfig


FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


@dataclass(frozen=True)
class WeatherFetch:
    provider: str
    camera_id: str
    fetched_at_utc: str
    url: str
    payload: dict[str, Any]
    records: list[dict[str, Any]]


def fetch_forecast(camera: CameraConfig, weather: WeatherConfig) -> WeatherFetch:
    variables = weather.hourly_variables or (
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "precipitation",
        "cloud_cover",
        "cloud_cover_low",
        "pressure_msl",
        "wind_speed_10m",
        "wind_direction_10m",
    )
    params = {
        "latitude": camera.location.latitude,
        "longitude": camera.location.longitude,
        "hourly": ",".join(variables),
        "timezone": weather.timezone,
    }
    if weather.forecast_days is not None:
        params["forecast_days"] = weather.forecast_days
    if weather.past_days is not None:
        params["past_days"] = weather.past_days
    if weather.forecast_hours is not None:
        params["forecast_hours"] = weather.forecast_hours
    if weather.past_hours is not None:
        params["past_hours"] = weather.past_hours
    url = f"{FORECAST_URL}?{urlencode(params)}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    records = normalize_hourly(payload)
    return WeatherFetch(
        provider="open_meteo",
        camera_id=camera.id,
        fetched_at_utc=fetched_at,
        url=url,
        payload=payload,
        records=records,
    )


def normalize_hourly(payload: dict[str, Any]) -> list[dict[str, Any]]:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    variables = {key: value for key, value in hourly.items() if key != "time"}

    records: list[dict[str, Any]] = []
    for idx, raw_time in enumerate(times):
        valid_at = parse_open_meteo_time(raw_time)
        record_vars = {
            key: values[idx]
            for key, values in variables.items()
            if isinstance(values, list) and idx < len(values)
        }
        records.append(
            {
                "valid_at_utc": valid_at,
                "variables": record_vars,
            }
        )
    return records


def parse_open_meteo_time(raw_time: str) -> str:
    # With timezone=UTC, Open-Meteo returns strings like "2026-07-07T12:00".
    dt = datetime.fromisoformat(raw_time)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()
