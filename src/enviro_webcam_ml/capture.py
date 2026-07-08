from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image

from enviro_webcam_ml.config import AppConfig, CameraConfig
from enviro_webcam_ml import db
from enviro_webcam_ml.quality import assess_image_quality


@dataclass(frozen=True)
class CaptureResult:
    camera_id: str
    capture_id: int
    ok: bool
    path: Path | None
    error: str | None


def capture_once(config: AppConfig, camera: CameraConfig) -> CaptureResult:
    captured_at = datetime.now(timezone.utc).replace(microsecond=0)
    captured_at_iso = captured_at.isoformat()

    headers = {"User-Agent": camera.capture.user_agent}
    image_bytes: bytes | None = None
    http_status: int | None = None
    content_type: str | None = None
    error: str | None = None

    try:
        response = requests.get(
            camera.capture.image_url,
            headers=headers,
            timeout=camera.capture.timeout_seconds,
        )
        http_status = response.status_code
        content_type = response.headers.get("content-type")
        response.raise_for_status()
        image_bytes = response.content
    except requests.RequestException as exc:
        error = str(exc)

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
            requested_url=camera.capture.image_url,
            http_status=http_status,
            content_type=content_type,
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
