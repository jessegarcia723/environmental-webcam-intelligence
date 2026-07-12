from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image

from enviro_webcam_ml import db
from enviro_webcam_ml.capture import capture_camera, frame_captured_at, spaced_frame_names
from enviro_webcam_ml.config import load_config


class FakeResponse:
    def __init__(self, *, status_code: int, payload=None, content: bytes = b"", content_type: str = "application/json"):
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.headers = {"content-type": content_type}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_manifest_frame_capture_skips_existing_and_saves_new(monkeypatch, tmp_path: Path) -> None:
    config_path = write_manifest_config(tmp_path)
    config = load_config(config_path)
    db.init_db(config.database_path)
    camera = config.cameras[0]
    manifest = camera.capture.manifest
    assert manifest is not None

    existing_time = "2026-07-08T00:00:00+00:00"
    with db.connect(config.database_path) as conn:
        db.register_config(conn, config)
        db.insert_capture(
            conn,
            camera_id=camera.id,
            pose_version=camera.pose.version,
            captured_at_utc=existing_time,
            requested_url="https://example.test/old.jpg",
            http_status=200,
            content_type="image/jpeg",
            byte_count=3,
            sha256="old",
            error=None,
        )

    image_bytes = jpeg_bytes()
    calls = []

    def fake_get(url, *, headers, timeout):
        calls.append(url)
        if url.endswith("manifest.json"):
            return FakeResponse(
                status_code=200,
                payload={
                    "frames": [
                        "1783468800.000000000.jpg",
                        "1783468860.000000000.jpg",
                        "1783469100.000000000.jpg",
                    ]
                },
            )
        return FakeResponse(status_code=200, content=image_bytes, content_type="image/jpeg")

    monkeypatch.setattr("enviro_webcam_ml.capture.requests.get", fake_get)

    results = capture_camera(config, camera)

    assert [result.skipped for result in results] == [True, False]
    assert calls == [
        "https://example.test/manifest.json",
        "https://example.test/frames/1783469100.000000000.jpg",
    ]
    with db.connect(config.database_path) as conn:
        captures = conn.execute("SELECT * FROM capture ORDER BY captured_at_utc").fetchall()
        assets = conn.execute("SELECT * FROM image_asset").fetchall()
    assert len(captures) == 2
    assert captures[-1]["captured_at_utc"] == "2026-07-08T00:05:00+00:00"
    assert captures[-1]["requested_url"].endswith("/1783469100.000000000.jpg")
    assert len(assets) == 1
    assert Path(assets[0]["path"]).exists()


def test_manifest_frame_spacing_uses_epoch_filenames() -> None:
    config = load_config(Path("configs/mount_tam.yaml"))
    manifest = config.cameras[0].capture.manifest
    assert manifest is not None

    names = [
        "1783468800.000000000.jpg",
        "1783468860.000000000.jpg",
        "1783469100.000000000.jpg",
    ]

    assert frame_captured_at(names[0], manifest).isoformat() == "2026-07-08T00:00:00+00:00"
    assert spaced_frame_names(names, manifest=manifest) == [
        "1783468800.000000000.jpg",
        "1783469100.000000000.jpg",
    ]


def write_manifest_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "manifest_config.yaml"
    config_path.write_text(
        f"""
project:
  name: test_manifest_capture
  database_path: {tmp_path / "test.sqlite3"}
  data_dir: {tmp_path / "data"}

cameras:
  - id: test_camera
    name: Test Camera
    location:
      latitude: 37.0
      longitude: -122.0
      timezone: UTC
    capture:
      source: manifest_frames
      interval_seconds: 3600
      timeout_seconds: 20
      user_agent: test-agent
      manifest:
        url: "https://example.test/manifest.json"
        frame_url_template: "https://example.test/frames/{{frame}}"
        min_frame_spacing_seconds: 300
        skip_existing: true
    pose:
      version: test

weather:
  provider: open_meteo

tasks:
  - id: test
    default: true
    positive_label: positive
""",
        encoding="utf-8",
    )
    return config_path


def jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (12, 8), color=(20, 40, 80)).save(buffer, format="JPEG")
    return buffer.getvalue()
