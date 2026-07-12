from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from enviro_webcam_ml.config import AppConfig, CameraConfig, ManifestCaptureConfig
from enviro_webcam_ml import db
from enviro_webcam_ml.quality import assess_image_quality


@dataclass(frozen=True)
class CaptureResult:
    camera_id: str
    capture_id: int
    ok: bool
    path: Path | None
    error: str | None
    skipped: bool = False


@dataclass(frozen=True)
class FetchedImage:
    url: str
    captured_at: datetime
    image_bytes: bytes | None
    http_status: int | None
    content_type: str | None
    error: str | None


def capture_camera(config: AppConfig, camera: CameraConfig) -> list[CaptureResult]:
    if camera.capture.source == "manifest_frames":
        return capture_manifest_frames(config, camera)
    return [capture_once(config, camera)]


def capture_once(config: AppConfig, camera: CameraConfig) -> CaptureResult:
    captured_at = datetime.now(timezone.utc).replace(microsecond=0)
    fetched = fetch_image(
        url=camera.capture.image_url,
        captured_at=captured_at,
        user_agent=camera.capture.user_agent,
        timeout_seconds=camera.capture.timeout_seconds,
    )
    return store_fetched_image(config, camera, fetched)


def capture_manifest_frames(config: AppConfig, camera: CameraConfig) -> list[CaptureResult]:
    manifest = camera.capture.manifest
    if manifest is None:
        raise ValueError(f"camera {camera.id} uses source=manifest_frames but has no manifest config.")
    headers = {"User-Agent": camera.capture.user_agent}
    try:
        response = requests.get(
            manifest.url,
            headers=headers,
            timeout=camera.capture.timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        response = getattr(exc, "response", None)
        return [
            record_capture_error(
                config,
                camera,
                captured_at=datetime.now(timezone.utc).replace(microsecond=0),
                requested_url=manifest.url,
                http_status=getattr(response, "status_code", None),
                content_type=response.headers.get("content-type") if response is not None else None,
                error=f"Manifest fetch failed: {exc}",
            )
        ]

    try:
        payload = response.json()
        frame_names = manifest_frame_names(payload, manifest.frames_key)
    except ValueError as exc:
        return [
            record_capture_error(
                config,
                camera,
                captured_at=datetime.now(timezone.utc).replace(microsecond=0),
                requested_url=manifest.url,
                http_status=response.status_code,
                content_type=response.headers.get("content-type"),
                error=f"Manifest parse failed: {exc}",
            )
        ]

    frame_names = spaced_frame_names(
        sorted(frame_names, key=lambda name: frame_captured_at(name, manifest)),
        manifest=manifest,
    )
    if manifest.max_frames_per_cycle is not None:
        frame_names = frame_names[-manifest.max_frames_per_cycle :]

    results = []
    with db.connect(config.database_path) as conn:
        db.register_config(conn, config)
        existing_times = {
            frame_captured_at(frame_name, manifest).isoformat()
            for frame_name in frame_names
            if manifest.skip_existing
            and db.capture_exists(conn, camera_id=camera.id, captured_at_utc=frame_captured_at(frame_name, manifest).isoformat())
        }

    for frame_name in frame_names:
        captured_at = frame_captured_at(frame_name, manifest)
        captured_at_iso = captured_at.isoformat()
        frame_url = manifest.frame_url_template.format(frame=frame_name, camera_id=camera.id)
        if manifest.skip_existing and captured_at_iso in existing_times:
            results.append(
                CaptureResult(
                    camera_id=camera.id,
                    capture_id=0,
                    ok=True,
                    path=None,
                    error=None,
                    skipped=True,
                )
            )
            continue
        fetched = fetch_image(
            url=frame_url,
            captured_at=captured_at,
            user_agent=camera.capture.user_agent,
            timeout_seconds=camera.capture.timeout_seconds,
        )
        results.append(store_fetched_image(config, camera, fetched))
    return results


def fetch_image(*, url: str, captured_at: datetime, user_agent: str, timeout_seconds: int) -> FetchedImage:
    headers = {"User-Agent": user_agent}
    try:
        response = requests.get(url, headers=headers, timeout=timeout_seconds)
        http_status = response.status_code
        content_type = response.headers.get("content-type")
        response.raise_for_status()
        return FetchedImage(
            url=url,
            captured_at=captured_at,
            image_bytes=response.content,
            http_status=http_status,
            content_type=content_type,
            error=None,
        )
    except requests.RequestException as exc:
        response = getattr(exc, "response", None)
        return FetchedImage(
            url=url,
            captured_at=captured_at,
            image_bytes=None,
            http_status=getattr(response, "status_code", None),
            content_type=response.headers.get("content-type") if response is not None else None,
            error=str(exc),
        )


def store_fetched_image(config: AppConfig, camera: CameraConfig, fetched: FetchedImage) -> CaptureResult:
    captured_at = fetched.captured_at.astimezone(timezone.utc).replace(microsecond=0)
    captured_at_iso = captured_at.isoformat()
    image_bytes = fetched.image_bytes
    error = fetched.error

    sha = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None
    path: Path | None = None
    width: int | None = None
    height: int | None = None

    if image_bytes:
        path = image_path(config.data_dir, camera.id, captured_at, sha or "unknown")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image_bytes)
        try:
            with Image.open(path) as img:
                width, height = img.size
                img.verify()
        except Exception as exc:  # noqa: BLE001 - capture should record bad image payloads.
            error = f"Downloaded payload is not a valid image: {exc}"

    with db.connect(config.database_path) as conn:
        db.register_config(conn, config)
        capture_id = db.insert_capture(
            conn,
            camera_id=camera.id,
            pose_version=camera.pose.version,
            captured_at_utc=captured_at_iso,
            requested_url=fetched.url,
            http_status=fetched.http_status,
            content_type=fetched.content_type,
            byte_count=len(image_bytes) if image_bytes else None,
            sha256=sha,
            error=error,
        )

        if path and sha and not error:
            db.insert_image_asset(
                conn,
                capture_id=capture_id,
                path=path,
                sha256=sha,
                width=width,
                height=height,
            )
            previous_sha = db.latest_successful_sha(conn, camera.id, capture_id)
            quality = assess_image_quality(
                path,
                previous_sha=previous_sha,
                current_sha=sha,
                night_luminance_threshold=config.quality.night_luminance_threshold,
                blur_variance_threshold=config.quality.blur_variance_threshold,
            )
            db.insert_frame_quality(conn, capture_id=capture_id, **quality)

    return CaptureResult(
        camera_id=camera.id,
        capture_id=capture_id,
        ok=error is None,
        path=path,
        error=error,
    )


def record_capture_error(
    config: AppConfig,
    camera: CameraConfig,
    *,
    captured_at: datetime,
    requested_url: str,
    http_status: int | None,
    content_type: str | None,
    error: str,
) -> CaptureResult:
    with db.connect(config.database_path) as conn:
        db.register_config(conn, config)
        capture_id = db.insert_capture(
            conn,
            camera_id=camera.id,
            pose_version=camera.pose.version,
            captured_at_utc=captured_at.astimezone(timezone.utc).replace(microsecond=0).isoformat(),
            requested_url=requested_url,
            http_status=http_status,
            content_type=content_type,
            byte_count=None,
            sha256=None,
            error=error,
        )
    return CaptureResult(camera_id=camera.id, capture_id=capture_id, ok=False, path=None, error=error)


def manifest_frame_names(payload: dict[str, Any], frames_key: str) -> list[str]:
    value: Any = payload
    for part in frames_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Missing frame list at {frames_key!r}.")
        value = value[part]
    if not isinstance(value, list):
        raise ValueError(f"Manifest field {frames_key!r} is not a list.")
    frames = [str(item) for item in value if str(item)]
    if not frames:
        raise ValueError("Manifest frame list is empty.")
    return frames


def frame_captured_at(frame_name: str, manifest: ManifestCaptureConfig) -> datetime:
    if manifest.timestamp_source != "filename_epoch":
        raise ValueError(f"Unsupported manifest timestamp source: {manifest.timestamp_source}")
    stem = frame_name
    if manifest.filename_suffix and stem.endswith(manifest.filename_suffix):
        stem = stem[: -len(manifest.filename_suffix)]
    try:
        epoch_seconds = float(stem)
    except ValueError as exc:
        raise ValueError(f"Frame filename {frame_name!r} does not start with an epoch timestamp.") from exc
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).replace(microsecond=0)


def spaced_frame_names(frame_names: list[str], *, manifest: ManifestCaptureConfig) -> list[str]:
    if manifest.min_frame_spacing_seconds <= 0:
        return frame_names
    selected = []
    last_selected: datetime | None = None
    for frame_name in frame_names:
        captured_at = frame_captured_at(frame_name, manifest)
        if last_selected is None or (captured_at - last_selected).total_seconds() >= manifest.min_frame_spacing_seconds:
            selected.append(frame_name)
            last_selected = captured_at
    return selected


def image_path(data_dir: Path, camera_id: str, captured_at: datetime, sha: str) -> Path:
    suffix = sha[:12]
    return (
        data_dir
        / "raw"
        / camera_id
        / f"{captured_at:%Y}"
        / f"{captured_at:%m}"
        / f"{captured_at:%d}"
        / f"{captured_at:%Y%m%dT%H%M%SZ}_{suffix}.jpg"
    )
